#!/usr/bin/env python3
"""Run the expert-only Phase A falsification test for Con1.

The script learns a compact inverse-dynamics-aligned representation from a
no-change-referenced V-JEPA displacement. It trains matched raw-pair and
current-only controls, but never uses state, language, or intervention data as
model inputs.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Literal

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import action_change_bottleneck as bottleneck

RepresentationMode = Literal["delta", "raw_pair", "current_only"]


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
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=401)
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


def _within_task_shuffle(indices: np.ndarray, tasks: np.ndarray) -> np.ndarray:
    shuffled = indices.copy()
    for task in np.unique(tasks[indices]):
        positions = np.flatnonzero(tasks[indices] == task)
        if len(positions) < 2:
            raise ValueError(f"Task {task} has fewer than two validation samples")
        shuffled[positions] = np.roll(indices[positions], 1)
    if np.any(shuffled == indices):
        raise RuntimeError("Within-task shuffle left one or more examples unchanged")
    return shuffled


def _normalize_actions(actions: np.ndarray, norm_stats_path: Path) -> np.ndarray:
    with norm_stats_path.open() as handle:
        value = json.load(handle)
    stats = value.get("norm_stats", value)["actions"]
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    if actions.shape[-1] != len(q01):
        raise ValueError(f"Action width {actions.shape[-1]} does not match norm stats {len(q01)}")
    normalized = (actions.astype(np.float32) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized actions contain non-finite values")
    return normalized


def _representation(mode: RepresentationMode, realized: jax.Array, nochange: jax.Array) -> jax.Array:
    if mode == "delta":
        return bottleneck.latent_displacement(realized, nochange)
    if mode == "raw_pair":
        return bottleneck.l2_normalize_tokens(realized)
    if mode == "current_only":
        return bottleneck.l2_normalize_tokens(nochange)
    raise ValueError(f"Unknown representation mode: {mode}")


def _huber_per_example(prediction: jax.Array, target: jax.Array, delta: float = 1.0) -> jax.Array:
    error = jnp.abs(prediction - target)
    elementwise = jnp.where(error <= delta, 0.5 * jnp.square(error), delta * (error - 0.5 * delta))
    return jnp.mean(elementwise, axis=(1, 2))


def _bootstrap_episode_gap(
    per_example_gap: np.ndarray,
    validation_indices: np.ndarray,
    episodes: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float]:
    by_episode: dict[int, list[float]] = defaultdict(list)
    for index, gap in zip(validation_indices.tolist(), per_example_gap.tolist(), strict=True):
        by_episode[int(episodes[index])].append(float(gap))
    episode_means = np.asarray([np.mean(values) for values in by_episode.values()], dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = rng.choice(episode_means, size=(replicates, len(episode_means)), replace=True).mean(axis=1)
    return {
        "mean": float(episode_means.mean()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "episode_count": len(episode_means),
    }


def _train_mode(
    mode: RepresentationMode,
    realized: Any,
    nochange: Any,
    actions: np.ndarray,
    tasks: np.ndarray,
    episodes: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    shuffled_validation: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any, bottleneck.PhaseAModel]:
    model = bottleneck.PhaseAModel(
        num_tokens=args.num_change_tokens,
        token_dim=args.change_token_dim,
        width=args.width,
        depth=args.depth,
        num_heads=args.num_heads,
        horizon=actions.shape[1],
        action_dim=actions.shape[2],
    )
    example_realized = jnp.asarray(np.asarray(realized[:1], dtype=np.float32))
    example_nochange = jnp.asarray(np.asarray(nochange[:1], dtype=np.float32))
    example_representation = _representation(mode, example_realized, example_nochange)
    params = model.init(jax.random.key(args.seed), example_representation)["params"]
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate, 1e-4))
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(parameters, state, realized_batch, nochange_batch, action_batch):
        def loss_fn(value):
            representation = _representation(mode, realized_batch, nochange_batch)
            _, prediction = model.apply({"params": value}, representation)
            return jnp.mean(_huber_per_example(prediction, action_batch))

        loss, gradients = jax.value_and_grad(loss_fn)(parameters)
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss, optax.global_norm(gradients)

    @jax.jit
    def evaluate(parameters, realized_batch, nochange_batch, action_batch):
        representation = _representation(mode, realized_batch, nochange_batch)
        change, prediction = model.apply({"params": parameters}, representation)
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
            print(json.dumps({"phase": "A", "mode": mode, **record}), flush=True)

    validation_realized = jnp.asarray(np.asarray(realized[validation_indices], dtype=np.float32))
    validation_nochange = jnp.asarray(np.asarray(nochange[validation_indices], dtype=np.float32))
    _, _, correct_loss = evaluate(
        params,
        validation_realized,
        validation_nochange,
        jnp.asarray(actions[validation_indices]),
    )
    _, _, shuffled_loss = evaluate(
        params,
        jnp.asarray(np.asarray(realized[shuffled_validation], dtype=np.float32)),
        jnp.asarray(np.asarray(nochange[shuffled_validation], dtype=np.float32)),
        jnp.asarray(actions[validation_indices]),
    )
    correct_numpy = np.asarray(correct_loss)
    shuffled_numpy = np.asarray(shuffled_loss)
    metrics: dict[str, Any] = {
        "mode": mode,
        "correct_huber": float(correct_numpy.mean()),
        "shuffled_huber": float(shuffled_numpy.mean()),
        "shuffle_gap": float((shuffled_numpy - correct_numpy).mean()),
        "shuffle_gap_bootstrap": _bootstrap_episode_gap(
            shuffled_numpy - correct_numpy,
            validation_indices,
            episodes,
            args.bootstrap_replicates,
            args.seed + 100,
        ),
        "history": history,
    }
    task_gaps: dict[str, float] = {}
    for task in np.unique(tasks[validation_indices]):
        mask = tasks[validation_indices] == task
        task_gaps[str(int(task))] = float(np.mean(shuffled_numpy[mask] - correct_numpy[mask]))
    metrics["task_shuffle_gaps"] = task_gaps
    metrics["positive_task_fraction"] = float(np.mean(np.asarray(list(task_gaps.values())) > 0.0))

    if mode == "delta":
        zeros = jnp.zeros_like(validation_realized)
        _, _, zero_loss = evaluate(
            params,
            zeros,
            zeros,
            jnp.asarray(actions[validation_indices]),
        )
        metrics["zero_huber"] = float(np.asarray(zero_loss).mean())
    return metrics, params, model


def _encode_all(
    model: bottleneck.PhaseAModel,
    params: Any,
    realized: Any,
    nochange: Any,
    batch_size: int,
) -> np.ndarray:
    @jax.jit
    def encode(realized_batch, nochange_batch):
        representation = bottleneck.latent_displacement(realized_batch, nochange_batch)
        change, _ = model.apply({"params": params}, representation)
        return change

    outputs: list[np.ndarray] = []
    for start in range(0, len(realized), batch_size):
        stop = min(start + batch_size, len(realized))
        outputs.append(
            np.asarray(
                encode(
                    jnp.asarray(np.asarray(realized[start:stop], dtype=np.float32)),
                    jnp.asarray(np.asarray(nochange[start:stop], dtype=np.float32)),
                ),
                dtype=np.float32,
            )
        )
    return np.concatenate(outputs, axis=0)


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.bootstrap_replicates < 1:
        raise ValueError("steps, batch-size, and bootstrap-replicates must be positive")
    samples = np.load(args.samples, allow_pickle=False)
    realized = np.load(args.realized_target, mmap_mode="r", allow_pickle=False)
    nochange = np.load(args.nochange_target, mmap_mode="r", allow_pickle=False)
    if realized.shape != nochange.shape or realized.shape[1:] != (576, 1408):
        raise ValueError(f"Unexpected JEPA target shapes: {realized.shape}, {nochange.shape}")
    if len(realized) != len(samples["action_chunks"]):
        raise ValueError("Sample and JEPA target counts differ")
    if int(samples["future_offset"]) != 10:
        raise ValueError(f"Phase A requires H10 samples, got {int(samples['future_offset'])}")

    actions = _normalize_actions(np.asarray(samples["action_chunks"]), args.norm_stats)
    tasks = np.asarray(samples["task_indices"])
    episodes = np.asarray(samples["episode_indices"])
    train_indices, validation_indices = _episode_split(episodes, tasks)
    shuffled_validation = _within_task_shuffle(validation_indices, tasks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    trained: dict[str, tuple[Any, bottleneck.PhaseAModel]] = {}
    for mode in ("delta", "raw_pair", "current_only"):
        metrics, params, model = _train_mode(
            mode,
            realized,
            nochange,
            actions,
            tasks,
            episodes,
            train_indices,
            validation_indices,
            shuffled_validation,
            args,
        )
        results[mode] = metrics
        trained[mode] = (params, model)

    delta = results["delta"]
    raw = results["raw_pair"]
    conditions = {
        "correct_better_than_shuffled": delta["correct_huber"] < delta["shuffled_huber"],
        "correct_better_than_zero": delta["correct_huber"] < delta["zero_huber"],
        "episode_bootstrap_ci_above_zero": delta["shuffle_gap_bootstrap"]["ci95_low"] > 0.0,
        "delta_gap_better_than_raw_pair": delta["shuffle_gap"] > raw["shuffle_gap"],
    }
    summary = {
        "phase": "A",
        "sample_count": len(actions),
        "train_count": len(train_indices),
        "validation_count": len(validation_indices),
        "task_count": len(np.unique(tasks)),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_change_tokens": args.num_change_tokens,
        "change_token_dim": args.change_token_dim,
        "results": results,
        "conditions": conditions,
        "passed_all_numeric_conditions": bool(all(conditions.values())),
    }
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    delta_params, delta_model = trained["delta"]
    (args.output_dir / "phase_a_delta_params.msgpack").write_bytes(serialization.to_bytes(delta_params))
    posterior = _encode_all(delta_model, delta_params, realized, nochange, args.batch_size)
    np.save(args.output_dir / "posterior_b_r.npy", posterior)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
