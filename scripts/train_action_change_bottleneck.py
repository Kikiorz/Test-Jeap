#!/usr/bin/env python3
"""Run the expert-only Phase A incremental-value test for Con1.

The change bottleneck is computed only from a no-change-referenced V-JEPA
displacement. Current robot state is supplied separately to the training-only
inverse decoder. Four matched models test whether change adds held-out action
information beyond state and whether delta is preferable to the raw pair.
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

RepresentationMode = Literal["state_change", "change_only", "state_only", "state_raw_pair"]


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


def _load_norm_stats(norm_stats_path: Path) -> dict[str, Any]:
    with norm_stats_path.open() as handle:
        value = json.load(handle)
    return value.get("norm_stats", value)


def _quantile_normalize(values: np.ndarray, stats: dict[str, Any], name: str) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    if values.shape[-1] != len(q01):
        raise ValueError(f"{name} width {values.shape[-1]} does not match norm stats {len(q01)}")
    normalized = (values.astype(np.float32) - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
    if not np.isfinite(normalized).all():
        raise ValueError(f"Normalized {name} contains non-finite values")
    return normalized


def _representation(mode: RepresentationMode, realized: jax.Array, nochange: jax.Array) -> jax.Array:
    if mode in ("state_change", "change_only"):
        return bottleneck.latent_displacement(realized, nochange)
    if mode == "state_raw_pair":
        return bottleneck.l2_normalize_tokens(realized)
    if mode == "state_only":
        return jnp.zeros_like(realized, dtype=jnp.float32)
    raise ValueError(f"Unknown representation mode: {mode}")


def _state_condition(mode: RepresentationMode, state: jax.Array) -> jax.Array:
    if mode == "change_only":
        return jnp.zeros_like(state, dtype=jnp.float32)
    return state.astype(jnp.float32)


def _huber_per_example(prediction: jax.Array, target: jax.Array, delta: float = 1.0) -> jax.Array:
    error = jnp.abs(prediction - target)
    elementwise = jnp.where(error <= delta, 0.5 * jnp.square(error), delta * (error - 0.5 * delta))
    return jnp.mean(elementwise, axis=(1, 2))


def _bootstrap_episode_improvement(
    per_example_improvement: np.ndarray,
    validation_indices: np.ndarray,
    episodes: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    by_episode: dict[int, list[float]] = defaultdict(list)
    for index, improvement in zip(
        validation_indices.tolist(), per_example_improvement.tolist(), strict=True
    ):
        by_episode[int(episodes[index])].append(float(improvement))
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
    states: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], np.ndarray, Any, bottleneck.PhaseAModel]:
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
    example_state = _state_condition(mode, jnp.asarray(states[:1]))
    params = model.init(jax.random.key(args.seed), example_representation, example_state)["params"]
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate, 1e-4))
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(parameters, optimizer_value, realized_batch, nochange_batch, state_batch, action_batch):
        def loss_fn(value):
            representation = _representation(mode, realized_batch, nochange_batch)
            state_condition = _state_condition(mode, state_batch)
            _, prediction = model.apply({"params": value}, representation, state_condition)
            return jnp.mean(_huber_per_example(prediction, action_batch))

        loss, gradients = jax.value_and_grad(loss_fn)(parameters)
        updates, optimizer_value = optimizer.update(gradients, optimizer_value, parameters)
        return (
            optax.apply_updates(parameters, updates),
            optimizer_value,
            loss,
            optax.global_norm(gradients),
        )

    @jax.jit
    def evaluate(parameters, realized_batch, nochange_batch, state_batch, action_batch):
        representation = _representation(mode, realized_batch, nochange_batch)
        state_condition = _state_condition(mode, state_batch)
        change, prediction = model.apply({"params": parameters}, representation, state_condition)
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
            jnp.asarray(states[selected]),
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
        jnp.asarray(states[validation_indices]),
        jnp.asarray(actions[validation_indices]),
    )
    correct_numpy = np.asarray(correct_loss)
    metrics: dict[str, Any] = {
        "mode": mode,
        "heldout_huber": float(correct_numpy.mean()),
        "history": history,
    }
    return metrics, correct_numpy, params, model


def _encode_all(
    model: bottleneck.PhaseAModel,
    params: Any,
    realized: Any,
    nochange: Any,
    states: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    @jax.jit
    def encode(realized_batch, nochange_batch, state_batch):
        representation = bottleneck.latent_displacement(realized_batch, nochange_batch)
        change, _ = model.apply({"params": params}, representation, state_batch)
        return change

    outputs: list[np.ndarray] = []
    for start in range(0, len(realized), batch_size):
        stop = min(start + batch_size, len(realized))
        outputs.append(
            np.asarray(
                encode(
                    jnp.asarray(np.asarray(realized[start:stop], dtype=np.float32)),
                    jnp.asarray(np.asarray(nochange[start:stop], dtype=np.float32)),
                    jnp.asarray(states[start:stop]),
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

    norm_stats = _load_norm_stats(args.norm_stats)
    actions = _quantile_normalize(np.asarray(samples["action_chunks"]), norm_stats["actions"], "actions")
    states = _quantile_normalize(np.asarray(samples["states"]), norm_stats["state"], "state")
    tasks = np.asarray(samples["task_indices"])
    episodes = np.asarray(samples["episode_indices"])
    train_indices, validation_indices = _episode_split(episodes, tasks)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    validation_losses: dict[str, np.ndarray] = {}
    trained: dict[str, tuple[Any, bottleneck.PhaseAModel]] = {}
    for mode in ("state_only", "change_only", "state_change", "state_raw_pair"):
        metrics, per_example_loss, params, model = _train_mode(
            mode,
            realized,
            nochange,
            actions,
            states,
            train_indices,
            validation_indices,
            args,
        )
        results[mode] = metrics
        validation_losses[mode] = per_example_loss
        trained[mode] = (params, model)

    comparisons: dict[str, Any] = {}
    for control_name in ("state_only", "change_only", "state_raw_pair"):
        improvement = validation_losses[control_name] - validation_losses["state_change"]
        task_improvements: dict[str, float] = {}
        for task in np.unique(tasks[validation_indices]):
            mask = tasks[validation_indices] == task
            task_improvements[str(int(task))] = float(np.mean(improvement[mask]))
        comparisons[f"state_change_vs_{control_name}"] = {
            "mean_huber_improvement": float(np.mean(improvement)),
            "bootstrap": _bootstrap_episode_improvement(
                improvement,
                validation_indices,
                episodes,
                args.bootstrap_replicates,
                args.seed + 100,
            ),
            "positive_task_fraction": float(
                np.mean(np.asarray(list(task_improvements.values())) > 0.0)
            ),
            "task_improvements": task_improvements,
        }

    conditions = {
        "state_change_better_than_state_only": (
            results["state_change"]["heldout_huber"] < results["state_only"]["heldout_huber"]
        ),
        "state_increment_bootstrap_ci_above_zero": (
            comparisons["state_change_vs_state_only"]["bootstrap"]["ci95_low"] > 0.0
        ),
        "state_change_better_than_change_only": (
            results["state_change"]["heldout_huber"] < results["change_only"]["heldout_huber"]
        ),
        "state_change_better_than_state_raw_pair": (
            results["state_change"]["heldout_huber"] < results["state_raw_pair"]["heldout_huber"]
        ),
        "delta_vs_raw_pair_bootstrap_ci_above_zero": (
            comparisons["state_change_vs_state_raw_pair"]["bootstrap"]["ci95_low"] > 0.0
        ),
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
        "comparisons": comparisons,
        "conditions": conditions,
        "passed_all_numeric_conditions": bool(all(conditions.values())),
    }
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    delta_params, delta_model = trained["state_change"]
    (args.output_dir / "phase_a_delta_params.msgpack").write_bytes(serialization.to_bytes(delta_params))
    posterior = _encode_all(delta_model, delta_params, realized, nochange, states, args.batch_size)
    np.save(args.output_dir / "posterior_b_r.npy", posterior)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
