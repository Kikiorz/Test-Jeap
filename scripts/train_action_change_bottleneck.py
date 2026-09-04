#!/usr/bin/env python3
"""Train the selected Phase A model B^R = G_psi(DeltaY).

The future observation and frozen V-JEPA are used only to construct DeltaY.
The inverse decoder receives only the compact bottleneck and is training-only.
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

from openpi.models import action_change_bottleneck as bottleneck


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--realized-target", type=Path, required=True)
    parser.add_argument("--nochange-target", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-change-tokens", type=int, default=16)
    parser.add_argument("--change-token-dim", type=int, default=16)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument(
        "--validation-parity",
        type=int,
        choices=(0, 1),
        default=1,
        help="For each task's sorted sampled episodes, hold out even (0) or odd (1) episodes.",
    )
    return parser.parse_args()


def _episode_split(
    episodes: np.ndarray,
    tasks: np.ndarray,
    validation_parity: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_episodes: set[int] = set()
    validation_episodes: set[int] = set()
    for task in np.unique(tasks):
        task_episodes = np.unique(episodes[tasks == task])
        if len(task_episodes) < 2:
            raise ValueError(f"Task {task} has fewer than two sampled episodes")
        validation_episodes.update(task_episodes[validation_parity::2].tolist())
        train_episodes.update(task_episodes[1 - validation_parity :: 2].tolist())
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
    normalized = (values.astype(np.float32) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized values contain non-finite values")
    return normalized


def _quantile_denormalize(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return (values.astype(np.float32) + 1.0) * 0.5 * (q99 - q01) + q01


def _huber_per_example(prediction: jax.Array, target: jax.Array, delta: float = 1.0) -> jax.Array:
    error = jnp.abs(prediction - target)
    elementwise = jnp.where(error <= delta, 0.5 * jnp.square(error), delta * (error - 0.5 * delta))
    return jnp.mean(elementwise, axis=(1, 2))


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch-size must be positive")

    samples = np.load(args.samples, allow_pickle=False)
    realized = np.load(args.realized_target, mmap_mode="r", allow_pickle=False)
    nochange = np.load(args.nochange_target, mmap_mode="r", allow_pickle=False)
    if realized.shape != nochange.shape or realized.shape[1:] != (576, 1408):
        raise ValueError(f"Unexpected JEPA target shapes: {realized.shape}, {nochange.shape}")
    if len(realized) != len(samples["action_chunks"]):
        raise ValueError("Sample and JEPA target counts differ")
    if int(samples["future_offset"]) != 10:
        raise ValueError(f"Phase A requires H10 samples, got {int(samples['future_offset'])}")

    norm_stats = _load_norm_stats(args.norm_stats)
    actions = _quantile_normalize(np.asarray(samples["action_chunks"]), norm_stats["actions"])
    tasks = np.asarray(samples["task_indices"])
    episodes = np.asarray(samples["episode_indices"])
    train_indices, validation_indices = _episode_split(episodes, tasks, args.validation_parity)

    model = bottleneck.PhaseAModel(
        num_tokens=args.num_change_tokens,
        token_dim=args.change_token_dim,
        width=args.width,
        depth=args.depth,
        num_heads=args.num_heads,
        horizon=actions.shape[1],
        action_dim=actions.shape[2],
    )
    example_delta = bottleneck.latent_displacement(
        jnp.asarray(np.asarray(realized[:1], dtype=np.float32)),
        jnp.asarray(np.asarray(nochange[:1], dtype=np.float32)),
    )
    params = model.init(jax.random.key(args.seed), example_delta)["params"]
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate, 1e-4))
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(parameters, state, realized_batch, nochange_batch, action_batch):
        displacement = bottleneck.latent_displacement(realized_batch, nochange_batch)

        def loss_fn(value):
            _, prediction = model.apply({"params": value}, displacement)
            return jnp.mean(_huber_per_example(prediction, action_batch))

        loss, gradients = jax.value_and_grad(loss_fn)(parameters)
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss, optax.global_norm(gradients)

    @jax.jit
    def evaluate(parameters, realized_batch, nochange_batch, action_batch):
        displacement = bottleneck.latent_displacement(realized_batch, nochange_batch)
        change, prediction = model.apply({"params": parameters}, displacement)
        return change, prediction, _huber_per_example(prediction, action_batch)

    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        selected = rng.choice(train_indices, args.batch_size, replace=len(train_indices) < args.batch_size)
        params, optimizer_state, loss, gradient_norm = train_step(
            params,
            optimizer_state,
            jnp.asarray(np.asarray(realized[selected], dtype=np.float32)),
            jnp.asarray(np.asarray(nochange[selected], dtype=np.float32)),
            jnp.asarray(actions[selected]),
        )
        if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
            record = {
                "step": step + 1,
                "loss": float(loss),
                "gradient_norm": float(gradient_norm),
            }
            history.append(record)
            print(json.dumps({"phase": "A", **record}), flush=True)

    def encode(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        changes: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        losses: list[np.ndarray] = []
        for start in range(0, len(indices), args.batch_size):
            selected = indices[start : start + args.batch_size]
            change, prediction, loss = evaluate(
                params,
                jnp.asarray(np.asarray(realized[selected], dtype=np.float32)),
                jnp.asarray(np.asarray(nochange[selected], dtype=np.float32)),
                jnp.asarray(actions[selected]),
            )
            changes.append(np.asarray(change, dtype=np.float32))
            predictions.append(np.asarray(prediction, dtype=np.float32))
            losses.append(np.asarray(loss, dtype=np.float32))
        return np.concatenate(changes), np.concatenate(predictions), np.concatenate(losses)

    all_indices = np.arange(len(actions), dtype=np.int32)
    posterior, _, _ = encode(all_indices)
    _, _, train_loss = encode(train_indices)
    _, validation_prediction, validation_loss = encode(validation_indices)
    validation_target = actions[validation_indices]
    mean_template = actions[train_indices].mean(axis=0, keepdims=True)
    mean_prediction = np.broadcast_to(mean_template, validation_target.shape)
    mean_huber = np.asarray(_huber_per_example(jnp.asarray(mean_prediction), jnp.asarray(validation_target)))
    normalized_error = validation_prediction - validation_target
    mean_error = mean_prediction - validation_target
    prediction_physical = _quantile_denormalize(validation_prediction, norm_stats["actions"])
    target_physical = np.asarray(samples["action_chunks"])[validation_indices].astype(np.float32)
    physical_mae_by_dim = np.abs(prediction_physical - target_physical).mean(axis=(0, 1))
    baseline_huber = float(mean_huber.mean())
    heldout_huber = float(validation_loss.mean())
    summary = {
        "phase": "A",
        "definition": "B_R = G_psi(DeltaY)",
        "sample_count": len(actions),
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "validation_episode_count": len(np.unique(episodes[validation_indices])),
        "validation_parity": args.validation_parity,
        "task_count": len(np.unique(tasks)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_change_tokens": args.num_change_tokens,
        "change_token_dim": args.change_token_dim,
        "train_huber": float(train_loss.mean()),
        "heldout_huber": heldout_huber,
        "heldout_normalized_mae": float(np.abs(normalized_error).mean()),
        "heldout_normalized_rmse": float(np.sqrt(np.square(normalized_error).mean())),
        "mean_template_huber": baseline_huber,
        "mean_template_normalized_rmse": float(np.sqrt(np.square(mean_error).mean())),
        "relative_huber_improvement_percent": 100.0 * (baseline_huber - heldout_huber) / baseline_huber,
        "physical_mae_by_action_dim": physical_mae_by_dim.tolist(),
        "history": history,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "phase_a_change_only_params.msgpack").write_bytes(serialization.to_bytes(params))
    np.save(args.output_dir / "posterior_b_r.npy", posterior, allow_pickle=False)
    np.savez(
        args.output_dir / "heldout_predictions.npz",
        validation_indices=validation_indices,
        episode_indices=episodes[validation_indices],
        task_indices=tasks[validation_indices],
        prediction_normalized=validation_prediction,
        target_normalized=validation_target,
        prediction_physical=prediction_physical,
        target_physical=target_physical,
        per_example_huber=validation_loss,
        mean_template_huber=mean_huber,
    )
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
