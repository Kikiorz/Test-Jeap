#!/usr/bin/env python3
"""Test whether actions predict distinct outcomes in the JEPA transition space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import action_consequence
import run_transition_inverse_gate as inverse_gate
from run_transition_energy_gate import _actions_from_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--nochange", type=Path, required=True)
    parser.add_argument("--proposals", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=191)
    return parser.parse_args()


def _pool(value: np.ndarray) -> jax.Array:
    pooled = inverse_gate._pool_24_to_8(value)
    return action_consequence.normalize(jnp.asarray(pooled))


def _bootstrap_interval(delta: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(10_000, len(delta)))
    means = delta[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    samples = np.load(args.samples, allow_pickle=False)
    cache = np.load(args.cache, allow_pickle=False)
    observed = _pool(np.asarray(np.load(args.observed, mmap_mode="r")))
    predicted = _pool(np.asarray(np.load(args.predicted, mmap_mode="r")))
    current = _pool(np.asarray(np.load(args.nochange, mmap_mode="r")))
    actions = _actions_from_cache(cache, len(observed))
    tasks = np.asarray(samples["task_indices"])
    episodes = np.asarray(samples["episode_indices"])
    train_indices, validation_indices = inverse_gate._episode_split(episodes, tasks)
    train_by_task = {
        int(task): train_indices[tasks[train_indices] == task]
        for task in np.unique(tasks[train_indices])
    }
    validation_by_task = {
        int(task): validation_indices[tasks[validation_indices] == task]
        for task in np.unique(tasks[validation_indices])
    }

    model = action_consequence.ActionConditionedConsequence(
        horizon=actions.shape[1],
        action_dim=actions.shape[2],
        transition_dim=observed.shape[-1],
        width=args.width,
        num_heads=args.num_heads,
    )
    params = model.init(
        jax.random.key(args.seed), current[:2], actions[:2], observed[:2]
    )["params"]
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate, 1e-4))
    optimizer_state = optimizer.init(params)

    def fixed_state_action_loss(parameters, current_batch, action_batch, target_batch):
        batch = current_batch.shape[0]
        candidate_count = action_batch.shape[0]
        repeated_current = jnp.repeat(current_batch[:, None], candidate_count, axis=1)
        tiled_action = jnp.broadcast_to(
            action_batch[None], (batch, candidate_count, *action_batch.shape[1:])
        )
        repeated_target = jnp.repeat(target_batch[:, None], candidate_count, axis=1)
        consequence, encoded_target, scale = model.apply(
            {"params": parameters},
            repeated_current.reshape(batch * candidate_count, *current_batch.shape[1:]),
            tiled_action.reshape(batch * candidate_count, *action_batch.shape[1:]),
            repeated_target.reshape(batch * candidate_count, *target_batch.shape[1:]),
        )
        consequence = consequence.reshape(batch, candidate_count, -1)
        encoded_target = encoded_target.reshape(batch, candidate_count, -1)[:, 0]
        logits = scale * jnp.einsum("bkd,bd->bk", consequence, encoded_target)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, jnp.arange(batch)
        ).mean()

    @jax.jit
    def train_step(parameters, state, current_batch, action_batch, target_batch):
        loss, gradients = jax.value_and_grad(fixed_state_action_loss)(
            parameters, current_batch, action_batch, target_batch
        )
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss

    rng = np.random.default_rng(args.seed)
    task_order = np.asarray(sorted(train_by_task))
    history = []
    for step in range(args.steps):
        task = int(task_order[step % len(task_order)])
        indices = train_by_task[task].copy()
        rng.shuffle(indices)
        params, optimizer_state, loss = train_step(
            params, optimizer_state, current[indices], actions[indices], observed[indices]
        )
        if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
            item = {"step": step + 1, "loss": float(loss)}
            history.append(item)
            print(json.dumps(item), flush=True)

    @jax.jit
    def score_candidates(current_batch, action_candidates, target_batch):
        batch, candidate_count = action_candidates.shape[:2]
        repeated_current = jnp.repeat(current_batch[:, None], candidate_count, axis=1)
        repeated_target = jnp.repeat(target_batch[:, None], candidate_count, axis=1)
        consequence, encoded_target, scale = model.apply(
            {"params": params},
            repeated_current.reshape(batch * candidate_count, *current_batch.shape[1:]),
            action_candidates.reshape(batch * candidate_count, *action_candidates.shape[2:]),
            repeated_target.reshape(batch * candidate_count, *target_batch.shape[1:]),
        )
        return (scale * jnp.sum(consequence * encoded_target, axis=-1)).reshape(
            batch, candidate_count
        )

    def action_sensitivity(target: jax.Array) -> dict[str, float | int]:
        correct = 0
        total = 0
        margins = []
        for indices in validation_by_task.values():
            count = len(indices)
            candidate_actions = jnp.broadcast_to(
                actions[indices][None], (count, count, *actions.shape[1:])
            )
            scores = np.asarray(
                score_candidates(current[indices], candidate_actions, target[indices])
            )
            labels = np.arange(count)
            correct += int(np.sum(np.argmax(scores, axis=1) == labels))
            total += count
            negative = scores.copy()
            negative[labels, labels] = -np.inf
            margins.extend((scores[labels, labels] - np.max(negative, axis=1)).tolist())
        margins = np.asarray(margins)
        return {
            "top1": correct / total,
            "mean_margin": float(margins.mean()),
            "median_margin": float(np.median(margins)),
            "fraction_positive_margin": float(np.mean(margins > 0)),
            "count": total,
        }

    result = {
        "config": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "width": args.width,
            "seed": args.seed,
            "train_count": int(len(train_indices)),
            "validation_count": int(len(validation_indices)),
        },
        "history": history,
        "action_sensitivity": {
            "realized_target": action_sensitivity(observed),
            "current_only_desired_target": action_sensitivity(predicted),
            "nochange_target": action_sensitivity(current),
        },
    }

    if args.proposals is not None:
        proposal_data = np.load(args.proposals, allow_pickle=False)
        proposal_indices = np.asarray(proposal_data["validation_indices"])
        if not np.array_equal(proposal_indices, validation_indices):
            raise ValueError("Proposal validation indices do not match the episode split")
        candidates = jnp.asarray(proposal_data["candidates"])
        experts = jnp.asarray(proposal_data["experts"])
        candidate_mse = np.asarray(
            jnp.mean(jnp.square(candidates - experts[:, None]), axis=(2, 3))
        )
        base = candidate_mse[:, 0]
        task_values = tasks[validation_indices]
        shuffled_indices = validation_indices.copy()
        for task in np.unique(task_values):
            positions = np.flatnonzero(task_values == task)
            shuffled_indices[positions] = np.roll(validation_indices[positions], 1)
        targets = {
            "predicted": predicted[validation_indices],
            "realized_oracle": observed[validation_indices],
            "nochange": current[validation_indices],
            "within_task_shuffled_predicted": predicted[shuffled_indices],
        }
        rerank = {}
        rows = np.arange(len(validation_indices))
        for offset, (name, target) in enumerate(targets.items()):
            scores = np.asarray(
                score_candidates(current[validation_indices], candidates, target)
            )
            selected = np.argmax(scores, axis=1)
            selected_mse = candidate_mse[rows, selected]
            delta = selected_mse - base
            rerank[name] = {
                "mse": float(selected_mse.mean()),
                "mean_mse_delta": float(delta.mean()),
                "bootstrap_95ci_mean_delta": _bootstrap_interval(delta, args.seed + offset),
                "fraction_better_than_first": float(np.mean(delta < 0)),
                "fraction_changed_selection": float(np.mean(selected != 0)),
            }
        result["proposal_rerank"] = {
            "first_candidate_mse": float(base.mean()),
            "oracle_candidate_mse": float(candidate_mse.min(axis=1).mean()),
            "targets": rerank,
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "consequence_params.msgpack").write_bytes(serialization.to_bytes(params))
    temporary = args.output_dir / ".metrics.json.tmp"
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output_dir / "metrics.json")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
