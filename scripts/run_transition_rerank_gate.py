#!/usr/bin/env python3
"""Use a JEPA inverse proposal to select among frozen pi0.5 action samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import coflow
from openpi.models import model as model_lib
from openpi.models import transition_inverse
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as config_lib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--transition-params", type=Path, required=True)
    parser.add_argument("--state-params", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero_vjepa_aux")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument("--batch-samples", type=int, default=2)
    parser.add_argument("--solver-steps", type=int, default=10)
    parser.add_argument("--inverse-width", type=int, default=128)
    parser.add_argument("--inverse-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=83)
    return parser.parse_args()


def _pool_24_to_8(value: np.ndarray) -> np.ndarray:
    value = value.reshape(value.shape[0], 8, 3, 8, 3, value.shape[-1])
    return value.mean(axis=(2, 4)).reshape(value.shape[0], 64, value.shape[-1])


def _episode_split(episode_indices: np.ndarray, task_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_episodes: set[int] = set()
    validation_episodes: set[int] = set()
    for task in np.unique(task_indices):
        episodes = np.unique(episode_indices[task_indices == task])
        train_episodes.update(episodes[::2].tolist())
        validation_episodes.update(episodes[1::2].tolist())
    return (
        np.flatnonzero(np.isin(episode_indices, list(train_episodes))),
        np.flatnonzero(np.isin(episode_indices, list(validation_episodes))),
    )


def _raw_sample(samples: Any, index: int) -> dict[str, Any]:
    return {
        "observation/state": samples["states"][index],
        "observation/image": samples["current_base"][index],
        "observation/wrist_image": samples["current_wrist"][index],
        "prompt": str(samples["prompts"][index]),
        "actions": samples["action_chunks"][index],
    }


def _stack(values: list[Any]) -> Any:
    return jax.tree.map(lambda *items: np.stack(items), *values)


def _load_inverse(
    path: Path,
    mode: transition_inverse.InverseMode,
    transition: jax.Array,
    state: jax.Array,
    action_dim: int,
    args: argparse.Namespace,
) -> tuple[transition_inverse.TransitionInverseDecoder, Any]:
    model = transition_inverse.TransitionInverseDecoder(
        horizon=10,
        action_dim=action_dim,
        width=args.inverse_width,
        num_heads=args.inverse_heads,
        mode=mode,
    )
    template = model.init(jax.random.key(0), transition[:1], state[:1])["params"]
    with path.open("rb") as handle:
        params = serialization.from_bytes(template, handle.read())
    return model, params


def main() -> None:
    args = parse_args()
    if args.num_candidates < 2 or args.batch_samples < 1:
        raise ValueError("num-candidates must be >=2 and batch-samples must be positive")
    samples = np.load(args.samples, allow_pickle=False)
    _, validation = _episode_split(samples["episode_indices"], samples["task_indices"])
    policy = policy_config.create_trained_policy(config_lib.get_config(args.config), args.policy_checkpoint)
    transformed = [policy._input_transform(_raw_sample(samples, int(index))) for index in validation]
    states = jnp.asarray(np.stack([np.asarray(item["state"], dtype=np.float32) for item in transformed]))
    action_dim = int(samples["action_chunks"].shape[-1])
    experts = np.stack(
        [np.asarray(item["actions"], dtype=np.float32)[..., :action_dim] for item in transformed]
    )
    observations = [{key: value for key, value in item.items() if key != "actions"} for item in transformed]

    predicted_all = coflow.normalize_transition(
        jnp.asarray(_pool_24_to_8(np.asarray(np.load(args.prediction, mmap_mode="r"))))
    )
    predicted = predicted_all[jnp.asarray(validation)]
    transition_model, transition_params = _load_inverse(
        args.transition_params, "transition", predicted, states, action_dim, args
    )
    state_model, state_params = _load_inverse(
        args.state_params, "state", predicted, states, action_dim, args
    )
    transition_proposal = np.asarray(
        jax.jit(transition_model.apply)({"params": transition_params}, predicted, states)
    )
    state_proposal = np.asarray(jax.jit(state_model.apply)({"params": state_params}, predicted, states))

    shuffled_predicted = np.empty_like(np.asarray(predicted))
    validation_tasks = samples["task_indices"][validation]
    for task in np.unique(validation_tasks):
        positions = np.flatnonzero(validation_tasks == task)
        shuffled_predicted[positions] = np.asarray(predicted)[np.roll(positions, 1)]
    shuffled_proposal = np.asarray(
        jax.jit(transition_model.apply)(
            {"params": transition_params}, jnp.asarray(shuffled_predicted), states
        )
    )

    sample_fn = nnx_utils.module_jit(policy._model.sample_actions, static_argnames=("num_steps",))
    rng = np.random.default_rng(args.seed)
    candidate_batches: list[np.ndarray] = []
    for start in range(0, len(validation), args.batch_samples):
        stop = min(start + args.batch_samples, len(validation))
        repeated_observations = [
            observations[index]
            for index in range(start, stop)
            for _ in range(args.num_candidates)
        ]
        observation_dict = _stack(repeated_observations)
        observation = model_lib.Observation.from_dict(
            jax.tree.map(lambda value: jnp.asarray(value), observation_dict)
        )
        noise = rng.standard_normal(
            ((stop - start) * args.num_candidates, 10, 32), dtype=np.float32
        )
        candidates = sample_fn(
            jax.random.key(0),
            observation,
            noise=jnp.asarray(noise),
            num_steps=args.solver_steps,
        )
        candidate_batches.append(
            np.asarray(candidates)[..., :action_dim].reshape(
                stop - start, args.num_candidates, 10, action_dim
            )
        )
        print(json.dumps({"sampled": stop, "total": len(validation)}), flush=True)
    candidates = np.concatenate(candidate_batches, axis=0)

    def select(proposal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scores = np.mean(np.square(candidates - proposal[:, None]), axis=(2, 3))
        selected_index = np.argmin(scores, axis=1)
        return candidates[np.arange(len(candidates)), selected_index], selected_index

    selected_transition, transition_index = select(transition_proposal)
    selected_state, state_index = select(state_proposal)
    selected_shuffled, shuffled_index = select(shuffled_proposal)
    candidate_error = np.mean(np.square(candidates - experts[:, None]), axis=(2, 3))

    def mse(value: np.ndarray) -> float:
        return float(np.mean(np.square(value - experts)))

    metrics = {
        "validation_count": int(len(validation)),
        "num_candidates": args.num_candidates,
        "first_candidate_mse": mse(candidates[:, 0]),
        "mean_candidate_mse": float(np.mean(candidate_error)),
        "oracle_candidate_mse": float(np.mean(np.min(candidate_error, axis=1))),
        "state_rerank_mse": mse(selected_state),
        "transition_rerank_mse": mse(selected_transition),
        "within_task_shuffled_transition_rerank_mse": mse(selected_shuffled),
        "transition_proposal_mse": mse(transition_proposal),
        "state_proposal_mse": mse(state_proposal),
        "transition_selection_differs_from_state": float(
            np.mean(transition_index != state_index)
        ),
        "transition_selection_differs_from_shuffled": float(
            np.mean(transition_index != shuffled_index)
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    np.savez_compressed(
        args.output_dir / "selections.npz",
        validation_indices=validation,
        candidates=candidates,
        experts=experts,
        transition_proposal=transition_proposal,
        state_proposal=state_proposal,
        transition_index=transition_index,
        state_index=state_index,
        shuffled_index=shuffled_index,
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
