"""Episode-local achieved-Change adaptation for Con2 deployment."""

from __future__ import annotations

import time
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy
import optax
from typing_extensions import override

from openpi.models import achieved_change_encoder
from openpi.models import model as model_lib
from openpi.policies import policy as policy_lib
from openpi.shared import nnx_utils


def _copy_tree(tree):
    return jax.tree.map(lambda value: jnp.array(value), tree)


class AchievedChangeAdaptivePolicy(base_policy.BasePolicy):
    """Wrap a frozen Con2 policy with one adapter update per completed H10 chunk."""

    def __init__(
        self,
        policy: policy_lib.Policy,
        *,
        hf_port: str,
        stage1_dir: str,
        learning_rate: float = 1e-5,
        proximal_weight: float = 1e-4,
        num_noise_samples: int = 4,
        torch_device: str = "cuda:0",
    ):
        if policy._is_pytorch_model:
            raise TypeError("Con2 online adaptation currently requires the JAX policy")
        if num_noise_samples < 1 or learning_rate <= 0 or proximal_weight < 0:
            raise ValueError("Invalid Con2 online hyperparameters")
        model = policy._model
        if not getattr(model, "use_achieved_change_adapter", False):
            raise ValueError("The checkpoint/config must include the Con2 directional adapter")

        self._input_transform = policy._input_transform
        self._output_transform = policy._output_transform
        self._sample_kwargs = dict(policy._sample_kwargs)
        self._metadata = policy.metadata
        self._num_noise_samples = num_noise_samples
        self._proximal_weight = proximal_weight
        self._rng = jax.random.key(0)
        self._pending_observation: dict[str, Any] | None = None
        self._pending_actions: np.ndarray | None = None
        self._pending_predicted_change: np.ndarray | None = None
        self._last_update: dict[str, float] | None = None

        self._change_encoder = achieved_change_encoder.AchievedChangeEncoder(
            hf_port=hf_port,
            stage1_dir=stage1_dir,
            torch_device=torch_device,
        )
        self._graphdef, state = nnx.split(model)
        adapter_filter = nnx_utils.PathRegex(".*change_to_action_(k|v)_(down|up).*")
        self._adapter_state, self._frozen_state = state.split(adapter_filter, ...)
        self._initial_adapter_state = _copy_tree(self._adapter_state)
        self._optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate),
        )
        self._optimizer_state = self._optimizer.init(self._adapter_state)

        def sample(parameters, rng, observation, noise):
            value = nnx.merge(self._graphdef, self._frozen_state, parameters)
            value.eval()
            return value.sample_action_change(
                rng,
                observation,
                noise=noise,
                **self._sample_kwargs,
            )

        self._sample = jax.jit(sample)

        def loss(parameters, rng, observation, executed_actions, achieved_change):
            value = nnx.merge(self._graphdef, self._frozen_state, parameters)
            data_loss = value.compute_online_achieved_change_loss(
                rng,
                observation,
                executed_actions,
                achieved_change,
                num_noise_samples=self._num_noise_samples,
            )
            differences = jax.tree.leaves(
                jax.tree.map(
                    lambda current, initial: jnp.sum(jnp.square(current - initial)),
                    parameters,
                    self._initial_adapter_state,
                )
            )
            proximal = jnp.sum(jnp.stack(differences))
            return data_loss + self._proximal_weight * proximal, (data_loss, proximal)

        self._loss_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))

        @jax.jit
        def apply_update(parameters, optimizer_state, gradients):
            updates, optimizer_state = self._optimizer.update(gradients, optimizer_state, parameters)
            return optax.apply_updates(parameters, updates), optimizer_state

        self._apply_update = apply_update

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def reset(self) -> None:
        """Restore the offline adapter and clear episode-local history."""
        self._adapter_state = _copy_tree(self._initial_adapter_state)
        self._optimizer_state = self._optimizer.init(self._adapter_state)
        self._pending_observation = None
        self._pending_actions = None
        self._pending_predicted_change = None
        self._last_update = None

    def _transform(self, raw: dict[str, Any], *, actions: np.ndarray | None = None):
        value = jax.tree.map(lambda item: item, raw)
        if actions is not None:
            value["actions"] = actions
        value = self._input_transform(value)
        value = jax.tree.map(lambda item: jnp.asarray(item)[None, ...], value)
        action_value = value.pop("actions", None)
        return model_lib.Observation.from_dict(value), action_value, value

    def _adapt_from_completed_chunk(self, current_raw: dict[str, Any]) -> float:
        if self._pending_observation is None or self._pending_actions is None:
            return 0.0
        started = time.monotonic()
        achieved = self._change_encoder(
            self._pending_observation["observation/image"],
            current_raw["observation/image"],
        )
        previous_observation, normalized_actions, _ = self._transform(
            self._pending_observation,
            actions=self._pending_actions,
        )
        if normalized_actions is None or normalized_actions.shape != (1, 10, 32):
            raise ValueError(f"Unexpected normalized executed-action shape: {getattr(normalized_actions, 'shape', None)}")
        self._rng, update_rng = jax.random.split(self._rng)
        (total, (data_loss, proximal)), gradients = self._loss_and_grad(
            self._adapter_state,
            update_rng,
            previous_observation,
            normalized_actions,
            jnp.asarray(achieved),
        )
        self._adapter_state, self._optimizer_state = self._apply_update(
            self._adapter_state, self._optimizer_state, gradients
        )
        self._last_update = {
            "total_loss": float(total),
            "inverse_loss": float(data_loss),
            "proximal": float(proximal),
            "gradient_norm": float(optax.global_norm(gradients)),
        }
        if self._pending_predicted_change is not None:
            self._last_update["predicted_achieved_change_mse"] = float(
                np.mean(np.square(self._pending_predicted_change - achieved[0]))
            )
        return (time.monotonic() - started) * 1000

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None, seed: int | None = None) -> dict:
        if int(obs.get("_executed_horizon", 10)) != 10:
            raise ValueError("Con2 achieved-Change adaptation requires replan_steps=10")
        if bool(obs.get("_reset_adaptation", False)):
            self.reset()
        raw = {key: value for key, value in obs.items() if not key.startswith("_")}
        adaptation_ms = self._adapt_from_completed_chunk(raw)

        observation, _, transformed = self._transform(raw)
        if seed is not None:
            sample_rng = jax.random.key(seed)
        else:
            self._rng, sample_rng = jax.random.split(self._rng)
        if noise is None:
            noise_value = jax.random.normal(sample_rng, (1, 10, 32))
        else:
            noise_value = jnp.asarray(noise)
            if noise_value.ndim == 2:
                noise_value = noise_value[None]
        started = time.monotonic()
        action, predicted_change = self._sample(
            self._adapter_state, sample_rng, observation, noise_value
        )
        inference_ms = (time.monotonic() - started) * 1000
        outputs = {
            "state": np.asarray(transformed["state"][0]),
            "actions": np.asarray(action[0]),
        }
        outputs = self._output_transform(outputs)
        physical_actions = np.asarray(outputs["actions"][:10], dtype=np.float32)
        if physical_actions.shape != (10, 7):
            raise ValueError(f"Unexpected executed-action shape: {physical_actions.shape}")
        self._pending_observation = raw
        self._pending_actions = physical_actions.copy()
        self._pending_predicted_change = np.asarray(predicted_change[0], dtype=np.float32)
        outputs["policy_timing"] = {
            "infer_ms": inference_ms,
            "adapt_ms": adaptation_ms,
        }
        if self._last_update is not None:
            outputs["con2_update"] = dict(self._last_update)
        return outputs
