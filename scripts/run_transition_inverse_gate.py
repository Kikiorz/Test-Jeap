#!/usr/bin/env python3
"""Test whether JEPA transitions identify the executed action chunk.

This is the minimum information test for tracker-free inverse adaptation. A
small inverse decoder is trained on episode-disjoint expert tuples and compared
with a matched proprioception-only decoder. The decoder is evaluated both on
observed V-JEPA transitions and on current-only JEPA-WAM predictions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import coflow
from openpi.models import transition_inverse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--observed-transition", type=Path, required=True)
    parser.add_argument("--nochange-transition", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--config", default="pi05_libero_vjepa_aux")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def _pool_24_to_8(value: np.ndarray) -> np.ndarray:
    if value.ndim != 3 or value.shape[1] != 24 * 24:
        raise ValueError(f"Expected [N,576,D] transition array, got {value.shape}")
    value = value.reshape(value.shape[0], 8, 3, 8, 3, value.shape[-1])
    return value.mean(axis=(2, 4)).reshape(value.shape[0], 64, value.shape[-1])


def _episode_split(episode_indices: np.ndarray, task_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_episodes: set[int] = set()
    validation_episodes: set[int] = set()
    for task in np.unique(task_indices):
        episodes = np.unique(episode_indices[task_indices == task])
        if len(episodes) < 2:
            raise ValueError(f"Task {task} has fewer than two sampled episodes")
        train_episodes.update(episodes[::2].tolist())
        validation_episodes.update(episodes[1::2].tolist())
    return (
        np.flatnonzero(np.isin(episode_indices, list(train_episodes))),
        np.flatnonzero(np.isin(episode_indices, list(validation_episodes))),
    )


def _policy_arrays(policy: Any, samples: Any) -> tuple[np.ndarray, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    executable_width = int(samples["action_chunks"].shape[-1])
    for index in range(len(samples["states"])):
        raw = {
            "observation/state": samples["states"][index],
            "observation/image": samples["current_base"][index],
            "observation/wrist_image": samples["current_wrist"][index],
            "prompt": str(samples["prompts"][index]),
            "actions": samples["action_chunks"][index],
        }
        transformed = policy._input_transform(raw)
        states.append(np.asarray(transformed["state"], dtype=np.float32))
        actions.append(np.asarray(transformed["actions"], dtype=np.float32)[..., :executable_width])
    return np.stack(states), np.stack(actions)


def _train(
    name: str,
    decoder_mode: transition_inverse.InverseMode,
    observed: jax.Array,
    predicted: jax.Array,
    states: jax.Array,
    actions: jax.Array,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    task_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any]:
    model = transition_inverse.TransitionInverseDecoder(
        horizon=actions.shape[1],
        action_dim=actions.shape[2],
        width=args.width,
        num_heads=args.num_heads,
        mode=decoder_mode,
    )
    params = model.init(jax.random.key(args.seed), observed[:1], states[:1])["params"]
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate, 1e-4))
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(parameters, state, transition_batch, state_batch, action_batch):
        def loss_fn(value):
            estimate = model.apply({"params": value}, transition_batch, state_batch)
            return jnp.mean(jnp.square(estimate - action_batch))

        loss, gradients = jax.value_and_grad(loss_fn)(parameters)
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss

    @jax.jit
    def error(parameters, transition_batch, state_batch, action_batch):
        estimate = model.apply({"params": parameters}, transition_batch, state_batch)
        return jnp.mean(jnp.square(estimate - action_batch)), jnp.mean(
            jnp.square(estimate - action_batch), axis=(0, 1)
        )

    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        selected = rng.choice(train_indices, args.batch_size, replace=len(train_indices) < args.batch_size)
        params, optimizer_state, loss = train_step(
            params, optimizer_state, observed[selected], states[selected], actions[selected]
        )
        if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
            record = {"step": step + 1, "loss": float(loss)}
            history.append(record)
            print(json.dumps({"mode": name, **record}), flush=True)

    validation = jnp.asarray(validation_indices)
    shuffled_numpy = validation_indices.copy()
    for task in np.unique(task_indices[validation_indices]):
        positions = np.flatnonzero(task_indices[validation_indices] == task)
        if len(positions) > 1:
            shuffled_numpy[positions] = np.roll(validation_indices[positions], 1)
    shuffled = jnp.asarray(shuffled_numpy)
    observed_mse, observed_per_dim = error(
        params, observed[validation], states[validation], actions[validation]
    )
    predicted_mse, predicted_per_dim = error(
        params, predicted[validation], states[validation], actions[validation]
    )
    shuffled_observed_mse, _ = error(
        params, observed[shuffled], states[validation], actions[validation]
    )
    shuffled_predicted_mse, _ = error(
        params, predicted[shuffled], states[validation], actions[validation]
    )
    return {
        "mode": name,
        "observed_mse": float(observed_mse),
        "predicted_mse": float(predicted_mse),
        "shuffled_observed_mse": float(shuffled_observed_mse),
        "shuffled_predicted_mse": float(shuffled_predicted_mse),
        "observed_shuffle_delta": float(shuffled_observed_mse - observed_mse),
        "predicted_shuffle_delta": float(shuffled_predicted_mse - predicted_mse),
        "observed_per_dim_mse": np.asarray(observed_per_dim).tolist(),
        "predicted_per_dim_mse": np.asarray(predicted_per_dim).tolist(),
        "history": history,
    }, params


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    from openpi.policies import policy_config
    from openpi.training import config as config_lib

    samples = np.load(args.samples, allow_pickle=False)
    predicted = coflow.normalize_transition(
        jnp.asarray(_pool_24_to_8(np.asarray(np.load(args.prediction, mmap_mode="r"))))
    )
    observed = coflow.normalize_transition(
        jnp.asarray(_pool_24_to_8(np.asarray(np.load(args.observed_transition, mmap_mode="r"))))
    )
    nochange = coflow.normalize_transition(
        jnp.asarray(_pool_24_to_8(np.asarray(np.load(args.nochange_transition, mmap_mode="r"))))
    )
    policy = policy_config.create_trained_policy(config_lib.get_config(args.config), args.policy_checkpoint)
    states_numpy, actions_numpy = _policy_arrays(policy, samples)
    states = jnp.asarray(states_numpy)
    actions = jnp.asarray(actions_numpy)
    train_indices, validation_indices = _episode_split(
        samples["episode_indices"], samples["task_indices"]
    )

    mean_action = jnp.mean(actions[jnp.asarray(train_indices)], axis=0, keepdims=True)
    mean_mse = jnp.mean(jnp.square(actions[jnp.asarray(validation_indices)] - mean_action))
    metrics: dict[str, Any] = {
        "sample_count": int(len(actions)),
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "global_mean_action_mse": float(mean_mse),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "width": args.width,
        "seed": args.seed,
        "modes": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode_inputs = (
        ("state", "state", observed, predicted),
        ("nochange", "transition", nochange, nochange),
        ("transition", "transition", observed, predicted),
    )
    for name, decoder_mode, teacher_input, predicted_input in mode_inputs:
        result, params = _train(
            name,
            decoder_mode,
            teacher_input,
            predicted_input,
            states,
            actions,
            train_indices,
            validation_indices,
            samples["task_indices"],
            args,
        )
        metrics["modes"][name] = result
        with (args.output_dir / f"{name}_params.msgpack").open("wb") as handle:
            handle.write(serialization.to_bytes(params))
    metrics["transition_minus_state"] = {
        "observed_mse": metrics["modes"]["transition"]["observed_mse"]
        - metrics["modes"]["state"]["observed_mse"],
        "predicted_mse": metrics["modes"]["transition"]["predicted_mse"]
        - metrics["modes"]["state"]["predicted_mse"],
    }
    metrics["transition_minus_nochange"] = {
        "teacher_input_mse": metrics["modes"]["transition"]["observed_mse"]
        - metrics["modes"]["nochange"]["observed_mse"],
        "within_task_shuffle_delta": metrics["modes"]["transition"]["observed_shuffle_delta"],
    }
    _write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
