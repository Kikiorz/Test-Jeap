#!/usr/bin/env python3
"""Fit a small JEPA guidance field on cached frozen-pi0.5 velocities."""

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
from openpi.models import transition_guidance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--active-action-dim", type=int, default=7)
    parser.add_argument("--seed", type=int, default=71)
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def _train_mode(
    use_transition: bool,
    cache: Any,
    transitions: jax.Array,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any]:
    action_tau = jnp.asarray(cache["action_tau"])
    states = jnp.asarray(cache["states"])
    times = jnp.asarray(cache["time"])
    base_velocity = jnp.asarray(cache["base_velocity"])
    target_velocity = jnp.asarray(cache["target_velocity"])
    model = transition_guidance.TransitionGuidance(
        action_dim=action_tau.shape[-1],
        state_dim=states.shape[-1],
        width=args.width,
        depth=args.depth,
        num_heads=args.num_heads,
    )
    params = model.init(
        jax.random.key(args.seed),
        action_tau[:1],
        states[:1],
        transitions[:1],
        times[:1],
        use_transition=use_transition,
    )["params"]
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate, 1e-4))
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(parameters, state, indices):
        def loss_fn(value):
            guidance = model.apply(
                {"params": value},
                action_tau[indices],
                states[indices],
                transitions[indices],
                times[indices],
                use_transition=use_transition,
            )
            guided = base_velocity[indices] + guidance
            return jnp.mean(jnp.square(guided - target_velocity[indices]))

        loss, gradients = jax.value_and_grad(loss_fn)(parameters)
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss

    @jax.jit
    def evaluate(parameters, indices, supplied_transition):
        guidance = model.apply(
            {"params": parameters},
            action_tau[indices],
            states[indices],
            supplied_transition,
            times[indices],
            use_transition=use_transition,
        )
        base_error = jnp.square(base_velocity[indices] - target_velocity[indices])
        guided_error = jnp.square(base_velocity[indices] + guidance - target_velocity[indices])
        active = args.active_action_dim
        return {
            "base_mse": jnp.mean(base_error),
            "guided_mse": jnp.mean(guided_error),
            "base_active_mse": jnp.mean(base_error[..., :active]),
            "guided_active_mse": jnp.mean(guided_error[..., :active]),
            "guidance_rms": jnp.sqrt(jnp.mean(jnp.square(guidance))),
        }

    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        selected = rng.choice(train_indices, args.batch_size, replace=len(train_indices) < args.batch_size)
        params, optimizer_state, loss = train_step(params, optimizer_state, jnp.asarray(selected))
        if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
            record = {"step": step + 1, "loss": float(loss)}
            history.append(record)
            print(json.dumps({"use_transition": use_transition, **record}), flush=True)

    validation = jnp.asarray(validation_indices)
    matched = evaluate(params, validation, transitions[validation])
    shuffled_transition = transitions[jnp.roll(validation, 4)]
    shuffled = evaluate(params, validation, shuffled_transition)
    metrics = {key: float(value) for key, value in matched.items()}
    metrics.update({f"shuffled_{key}": float(value) for key, value in shuffled.items()})
    metrics["shuffle_active_delta"] = metrics["shuffled_guided_active_mse"] - metrics[
        "guided_active_mse"
    ]
    metrics["relative_active_improvement"] = (
        metrics["base_active_mse"] - metrics["guided_active_mse"]
    ) / metrics["base_active_mse"]
    metrics["history"] = history
    return metrics, params


def main() -> None:
    args = parse_args()
    cache = np.load(args.cache, allow_pickle=False)
    prediction = _pool_24_to_8(np.asarray(np.load(args.prediction, mmap_mode="r")))
    prediction = coflow.normalize_transition(jnp.asarray(prediction))
    transitions = prediction[jnp.asarray(cache["sample_indices"])]
    train_indices, validation_indices = _episode_split(
        cache["episode_indices"], cache["task_indices"]
    )

    result: dict[str, Any] = {
        "tuple_count": int(len(cache["time"])),
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "width": args.width,
        "depth": args.depth,
        "seed": args.seed,
        "modes": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, use_transition in (("no_transition", False), ("transition", True)):
        metrics, params = _train_mode(
            use_transition, cache, transitions, train_indices, validation_indices, args
        )
        result["modes"][name] = metrics
        with (args.output_dir / f"{name}_params.msgpack").open("wb") as handle:
            handle.write(serialization.to_bytes(params))
    result["transition_minus_no_transition"] = {
        key: result["modes"]["transition"][key] - result["modes"]["no_transition"][key]
        for key in ("guided_mse", "guided_active_mse", "relative_active_improvement")
    }
    _write_json(args.output_dir / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
