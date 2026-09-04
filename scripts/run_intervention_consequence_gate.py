#!/usr/bin/env python3
"""Learn and test action consequences from same-state LIBERO interventions."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--realized", type=Path, required=True)
    parser.add_argument("--nochange", type=Path, required=True)
    parser.add_argument("--desired", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=223)
    parser.add_argument(
        "--load-params",
        type=Path,
        help="Evaluate an existing consequence model without retraining it.",
    )
    return parser.parse_args()


def _pool(value: np.ndarray) -> jax.Array:
    pooled = inverse_gate._pool_24_to_8(value)
    return action_consequence.normalize(jnp.asarray(pooled))


def _state_split(state_ids: np.ndarray, tasks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train_states = []
    validation_states = []
    for task in np.unique(tasks):
        task_states = np.unique(state_ids[tasks == task])
        train_states.extend(task_states[::2].tolist())
        validation_states.extend(task_states[1::2].tolist())
    return np.asarray(train_states), np.asarray(validation_states)


def _cosine_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first = first.astype(np.float32)
    second = second.astype(np.float32)
    first /= np.maximum(np.linalg.norm(first, axis=-1, keepdims=True), 1e-6)
    second /= np.maximum(np.linalg.norm(second, axis=-1, keepdims=True), 1e-6)
    return np.mean(1.0 - np.sum(first * second, axis=-1), axis=-1)


def _bootstrap_interval(delta: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(10_000, len(delta)))
    means = delta[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    samples = np.load(args.samples, allow_pickle=False)
    realized_raw = np.asarray(np.load(args.realized, mmap_mode="r"))
    desired_raw = np.asarray(np.load(args.desired, mmap_mode="r"))
    current = _pool(np.asarray(np.load(args.nochange, mmap_mode="r")))
    realized = _pool(realized_raw)
    desired = _pool(desired_raw)
    actions = jnp.asarray(samples["action_chunks"])
    state_ids = np.asarray(samples["episode_indices"])
    candidate_ids = np.asarray(samples["candidate_indices"])
    tasks = np.asarray(samples["task_indices"])
    train_states, validation_states = _state_split(state_ids, tasks)

    groups = {int(state): np.flatnonzero(state_ids == state) for state in np.unique(state_ids)}
    candidate_count = len(next(iter(groups.values())))
    if any(len(indices) != candidate_count for indices in groups.values()):
        raise ValueError("Every intervention state must contain the same number of candidates")
    if any(not np.array_equal(np.sort(candidate_ids[indices]), np.arange(candidate_count)) for indices in groups.values()):
        raise ValueError("Candidate IDs must be exactly 0..K-1 within every state")

    model = action_consequence.ActionConditionedConsequence(
        horizon=actions.shape[1],
        action_dim=actions.shape[2],
        transition_dim=realized.shape[-1],
        width=args.width,
        num_heads=args.num_heads,
    )
    params = model.init(jax.random.key(args.seed), current[:2], actions[:2], realized[:2])["params"]
    if args.load_params is not None:
        params = serialization.from_bytes(params, args.load_params.read_bytes())

    def consequence_loss(parameters, current_batch, action_batch, realized_batch):
        consequence, target, scale = model.apply(
            {"params": parameters}, current_batch, action_batch, realized_batch
        )
        logits = scale * consequence @ target.T
        labels = jnp.arange(logits.shape[0])
        forward = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        reverse = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels).mean()
        return 0.5 * (forward + reverse)

    @jax.jit
    def train_step(parameters, state, current_batch, action_batch, realized_batch):
        loss, gradients = jax.value_and_grad(consequence_loss)(
            parameters, current_batch, action_batch, realized_batch
        )
        updates, state = optimizer.update(gradients, state, parameters)
        return optax.apply_updates(parameters, updates), state, loss

    history = []
    if args.load_params is None:
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate, 1e-4))
        optimizer_state = optimizer.init(params)
        rng = np.random.default_rng(args.seed)
        for step in range(args.steps):
            state = int(train_states[step % len(train_states)])
            indices = groups[state].copy()
            rng.shuffle(indices)
            params, optimizer_state, loss = train_step(
                params, optimizer_state, current[indices], actions[indices], realized[indices]
            )
            if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
                item = {"step": step + 1, "loss": float(loss)}
                history.append(item)
                print(json.dumps(item), flush=True)

    @jax.jit
    def embeddings(current_batch, action_batch, target_batch):
        return model.apply({"params": params}, current_batch, action_batch, target_batch)[:2]

    sensitivity_rows = []
    selection_rows = []
    for state in validation_states:
        indices = groups[int(state)]
        consequence, target = embeddings(current[indices], actions[indices], realized[indices])
        scores = np.asarray(consequence @ target.T)
        labels = np.arange(candidate_count)
        predicted_match = np.argmax(scores, axis=1)
        negative = scores.copy()
        negative[labels, labels] = -np.inf
        sensitivity_rows.append(
            {
                "state": int(state),
                "correct": int(np.sum(predicted_match == labels)),
                "count": candidate_count,
                "margins": (scores[labels, labels] - np.max(negative, axis=1)).tolist(),
            }
        )

        repeated_desired = jnp.repeat(desired[indices[:1]], candidate_count, axis=0)
        predicted_consequence, encoded_desired = embeddings(
            current[indices], actions[indices], repeated_desired
        )
        desired_scores = np.asarray(jnp.sum(predicted_consequence * encoded_desired, axis=-1))
        selected = int(np.argmax(desired_scores))
        raw_quality = _cosine_distance(
            realized_raw[indices], np.repeat(desired_raw[indices[:1]], candidate_count, axis=0)
        )
        selection_rows.append(
            {
                "state": int(state),
                "first_quality": float(raw_quality[0]),
                "selected_quality": float(raw_quality[selected]),
                "oracle_quality": float(np.min(raw_quality)),
                "selected_index": selected,
                "oracle_index": int(np.argmin(raw_quality)),
            }
        )

    margins = np.concatenate([np.asarray(row["margins"]) for row in sensitivity_rows])
    correct = sum(row["correct"] for row in sensitivity_rows)
    total = sum(row["count"] for row in sensitivity_rows)
    first_quality = np.asarray([row["first_quality"] for row in selection_rows])
    selected_quality = np.asarray([row["selected_quality"] for row in selection_rows])
    oracle_quality = np.asarray([row["oracle_quality"] for row in selection_rows])
    selection_delta = selected_quality - first_quality

    pair_distances = []
    for state in validation_states:
        indices = groups[int(state)]
        values = realized_raw[indices]
        for first in range(candidate_count):
            for second in range(first + 1, candidate_count):
                pair_distances.append(float(_cosine_distance(values[first : first + 1], values[second : second + 1])[0]))

    result = {
        "config": {
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "width": args.width,
            "seed": args.seed,
            "train_states": train_states.tolist(),
            "validation_states": validation_states.tolist(),
            "candidate_count": candidate_count,
            "loaded_params": str(args.load_params) if args.load_params is not None else None,
        },
        "history": history,
        "realized_action_sensitivity": {
            "top1": correct / total,
            "chance": 1.0 / candidate_count,
            "mean_margin": float(margins.mean()),
            "median_margin": float(np.median(margins)),
            "fraction_positive_margin": float(np.mean(margins > 0)),
            "mean_pairwise_realized_distance": float(np.mean(pair_distances)),
            "count": total,
        },
        "desired_consequence_selection": {
            "first_quality": float(first_quality.mean()),
            "selected_quality": float(selected_quality.mean()),
            "oracle_quality": float(oracle_quality.mean()),
            "mean_delta_vs_first": float(selection_delta.mean()),
            "bootstrap_95ci_delta_vs_first": _bootstrap_interval(selection_delta, args.seed),
            "fraction_better_than_first": float(np.mean(selection_delta < 0)),
            "fraction_oracle_selected": float(
                np.mean(
                    [row["selected_index"] == row["oracle_index"] for row in selection_rows]
                )
            ),
            "state_count": len(selection_rows),
        },
        "sensitivity_per_state": sensitivity_rows,
        "selection_per_state": selection_rows,
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
