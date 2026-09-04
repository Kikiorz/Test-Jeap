#!/usr/bin/env python3
"""Train one arm of the minimal current-conditioned Action-Change Flow gate.

Both streams start from Gaussian noise and share the same rectified-flow time.
The matched independent arm has the same parameters but cannot exchange tokens.
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

from openpi.models import action_change_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--posterior", type=Path, required=True)
    parser.add_argument("--current-hidden", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("joint", "independent"), required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--change-weight", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--solver-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=431)
    return parser.parse_args()


def _episode_split(episodes: np.ndarray, tasks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_episodes: set[int] = set()
    validation_episodes: set[int] = set()
    for task in np.unique(tasks):
        task_episodes = np.unique(episodes[tasks == task])
        if len(task_episodes) < 2:
            raise ValueError(f"Task {task} has fewer than two sampled episodes")
        train_episodes.update(task_episodes[::2].tolist())
        validation_episodes.update(task_episodes[1::2].tolist())
    return (
        np.flatnonzero(np.isin(episodes, list(train_episodes))),
        np.flatnonzero(np.isin(episodes, list(validation_episodes))),
    )


def _load_norm_stats(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = json.load(handle)
    return value.get("norm_stats", value)


def _quantile_normalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    if values.shape[-1] != len(q01):
        raise ValueError(f"Value width {values.shape[-1]} does not match norm stats {len(q01)}")
    return (values.astype(np.float32) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def _per_example_mse(prediction: jax.Array, target: jax.Array) -> jax.Array:
    return jnp.mean(jnp.square(prediction - target), axis=tuple(range(1, prediction.ndim)))


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.solver_steps < 1:
        raise ValueError("steps, batch-size, and solver-steps must be positive")

    samples = np.load(args.samples, allow_pickle=False)
    posterior = np.asarray(np.load(args.posterior, mmap_mode="r", allow_pickle=False), dtype=np.float32)
    current_hidden = np.load(args.current_hidden, mmap_mode="r", allow_pickle=False)
    count = len(samples["action_chunks"])
    if posterior.shape != (count, 16, 16):
        raise ValueError(f"Expected posterior [N,16,16], got {posterior.shape}")
    if current_hidden.ndim != 3 or current_hidden.shape[0] != count:
        raise ValueError(f"Current hidden is not sample-aligned: {current_hidden.shape}")
    if int(samples["future_offset"]) != 10:
        raise ValueError(f"Expected H10 samples, got {int(samples['future_offset'])}")

    stats = _load_norm_stats(args.norm_stats)
    actions = _quantile_normalize(np.asarray(samples["action_chunks"]), stats["actions"])
    states = _quantile_normalize(np.asarray(samples["states"]), stats["state"])
    episodes = np.asarray(samples["episode_indices"])
    tasks = np.asarray(samples["task_indices"])
    train_indices, validation_indices = _episode_split(episodes, tasks)

    posterior_mean = posterior[train_indices].mean(axis=0, keepdims=True)
    posterior_scale = posterior[train_indices].std(axis=0, keepdims=True)
    posterior_scale = np.maximum(posterior_scale, 1e-3)
    posterior_normalized = (posterior - posterior_mean) / posterior_scale
    if not all(np.isfinite(value).all() for value in (actions, states, posterior_normalized, current_hidden)):
        raise ValueError("Non-finite training input")

    action_array = jnp.asarray(actions)
    change_array = jnp.asarray(posterior_normalized)
    state_array = jnp.asarray(states)
    hidden_array = jnp.asarray(np.asarray(current_hidden, dtype=np.float32))

    model = action_change_flow.ActionChangeCoFlow(
        action_dim=actions.shape[-1],
        change_dim=posterior.shape[-1],
        width=args.width,
        depth=args.depth,
        num_heads=args.num_heads,
        mode=args.mode,
    )
    init_key, action_key, change_key = jax.random.split(jax.random.key(args.seed), 3)
    action_noise = jax.random.normal(action_key, action_array[:1].shape)
    change_noise = jax.random.normal(change_key, change_array[:1].shape)
    dummy_time = jnp.full((1,), 0.5)
    action_tau, _ = action_change_flow.rectified_interpolant(action_array[:1], action_noise, dummy_time)
    change_tau, _ = action_change_flow.rectified_interpolant(change_array[:1], change_noise, dummy_time)
    params = model.init(
        init_key,
        action_tau,
        change_tau,
        hidden_array[:1],
        state_array[:1],
        dummy_time,
    )["params"]
    parameter_count = sum(int(leaf.size) for leaf in jax.tree.leaves(params))
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate, 1e-4))
    optimizer_state = optimizer.init(params)

    def objective(parameters, action_target, change_target, hidden, state, action_noise, change_noise, time):
        action_value, action_velocity = action_change_flow.rectified_interpolant(action_target, action_noise, time)
        change_value, change_velocity = action_change_flow.rectified_interpolant(change_target, change_noise, time)
        predicted_action, predicted_change = model.apply(
            {"params": parameters}, action_value, change_value, hidden, state, time
        )
        action_loss = jnp.mean(_per_example_mse(predicted_action, action_velocity))
        change_loss = jnp.mean(_per_example_mse(predicted_change, change_velocity))
        return action_loss + args.change_weight * change_loss, (action_loss, change_loss)

    @jax.jit
    def train_step(parameters, optimizer_value, action_target, change_target, hidden, state, key):
        action_noise_key, change_noise_key, time_key = jax.random.split(key, 3)
        action_noise = jax.random.normal(action_noise_key, action_target.shape)
        change_noise = jax.random.normal(change_noise_key, change_target.shape)
        time = jax.random.beta(time_key, 1.5, 1.0, (action_target.shape[0],)) * 0.999 + 0.001
        (loss, components), gradients = jax.value_and_grad(objective, has_aux=True)(
            parameters,
            action_target,
            change_target,
            hidden,
            state,
            action_noise,
            change_noise,
            time,
        )
        updates, optimizer_value = optimizer.update(gradients, optimizer_value, parameters)
        return (
            optax.apply_updates(parameters, updates),
            optimizer_value,
            loss,
            components,
            optax.global_norm(gradients),
        )

    @jax.jit
    def evaluate(parameters, action_target, change_target, hidden, state, action_noise, change_noise, time):
        action_value, action_velocity = action_change_flow.rectified_interpolant(action_target, action_noise, time)
        change_value, change_velocity = action_change_flow.rectified_interpolant(change_target, change_noise, time)
        predicted_action, predicted_change = model.apply(
            {"params": parameters}, action_value, change_value, hidden, state, time
        )
        return (
            _per_example_mse(predicted_action, action_velocity),
            _per_example_mse(predicted_change, change_velocity),
        )

    @jax.jit
    def integrate(parameters, hidden, state, action_noise, change_noise):
        dt = -1.0 / args.solver_steps

        def body(step, carry):
            action_value, change_value = carry
            time = jnp.full((action_value.shape[0],), 1.0 + step * dt)
            action_velocity, change_velocity = model.apply(
                {"params": parameters}, action_value, change_value, hidden, state, time
            )
            return action_value + dt * action_velocity, change_value + dt * change_velocity

        return jax.lax.fori_loop(0, args.solver_steps, body, (action_noise, change_noise))

    rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed + 1)
    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        selected = rng.choice(train_indices, args.batch_size, replace=len(train_indices) < args.batch_size)
        key, step_key = jax.random.split(key)
        params, optimizer_state, loss, components, gradient_norm = train_step(
            params,
            optimizer_state,
            action_array[selected],
            change_array[selected],
            hidden_array[selected],
            state_array[selected],
            step_key,
        )
        if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
            record = {
                "step": step + 1,
                "total_loss": float(loss),
                "action_loss": float(components[0]),
                "change_loss": float(components[1]),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps({"phase": "B", "mode": args.mode, **record}), flush=True)

    validation = jnp.asarray(validation_indices)
    eval_action_key, eval_change_key, eval_time_key = jax.random.split(jax.random.key(args.seed + 100), 3)
    eval_action_noise = jax.random.normal(eval_action_key, action_array[validation].shape)
    eval_change_noise = jax.random.normal(eval_change_key, change_array[validation].shape)
    eval_time = jax.random.beta(eval_time_key, 1.5, 1.0, (len(validation_indices),)) * 0.999 + 0.001
    action_flow_loss, change_flow_loss = evaluate(
        params,
        action_array[validation],
        change_array[validation],
        hidden_array[validation],
        state_array[validation],
        eval_action_noise,
        eval_change_noise,
        eval_time,
    )
    generated_action, generated_change = integrate(
        params,
        hidden_array[validation],
        state_array[validation],
        eval_action_noise,
        eval_change_noise,
    )
    action_endpoint_mse = _per_example_mse(generated_action, action_array[validation])
    change_endpoint_mse = _per_example_mse(generated_change, change_array[validation])

    metrics = {
        "phase": "B",
        "mode": args.mode,
        "definition": "(epsilon_A, epsilon_B) -> (A_star, B_R) | current VLM hidden, q",
        "sample_count": count,
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "validation_episode_count": len(np.unique(episodes[validation_indices])),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "width": args.width,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "solver_steps": args.solver_steps,
        "change_weight": args.change_weight,
        "parameter_count": parameter_count,
        "heldout_action_flow_mse": float(jnp.mean(action_flow_loss)),
        "heldout_change_flow_mse": float(jnp.mean(change_flow_loss)),
        "integrated_action_mse": float(jnp.mean(action_endpoint_mse)),
        "integrated_change_mse": float(jnp.mean(change_endpoint_mse)),
        "history": history,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.mode}_params.msgpack").write_bytes(serialization.to_bytes(params))
    np.savez(
        args.output_dir / f"{args.mode}_evaluation.npz",
        validation_indices=validation_indices,
        action_flow_loss=np.asarray(action_flow_loss, dtype=np.float32),
        change_flow_loss=np.asarray(change_flow_loss, dtype=np.float32),
        action_endpoint_mse=np.asarray(action_endpoint_mse, dtype=np.float32),
        change_endpoint_mse=np.asarray(change_endpoint_mse, dtype=np.float32),
        generated_action=np.asarray(generated_action, dtype=np.float32),
        generated_change=np.asarray(generated_change, dtype=np.float32),
        posterior_mean=posterior_mean,
        posterior_scale=posterior_scale,
    )
    with (args.output_dir / f"{args.mode}_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
