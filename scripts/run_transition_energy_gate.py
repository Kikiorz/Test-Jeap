#!/usr/bin/env python3
"""Fit and falsify a single JEPA transition-action compatibility energy.

The model is trained only on episode-split expert tuples.  Every contrastive
batch contains examples from one task, preventing task identity alone from
solving the objective.  The core gate is retrieval of the matching action from
current-only JEPA-WAM transition predictions on held-out episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models import transition_energy
import run_transition_inverse_gate as inverse_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--nochange", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=137)
    return parser.parse_args()


def _pool_and_normalize(value: np.ndarray) -> jax.Array:
    return transition_energy.normalize(jnp.asarray(inverse_gate._pool_24_to_8(value)))


def _actions_from_cache(cache: np.lib.npyio.NpzFile, sample_count: int) -> jax.Array:
    first: dict[int, int] = {}
    for tuple_index, sample_index in enumerate(np.asarray(cache["sample_indices"])):
        first.setdefault(int(sample_index), tuple_index)
    if set(first) != set(range(sample_count)):
        raise ValueError("Cache does not contain exactly the requested samples")
    indices = np.asarray([first[index] for index in range(sample_count)])
    return jnp.asarray(cache["actions"][indices, :, :7])


def main() -> None:
    args = parse_args()
    samples = np.load(args.samples, allow_pickle=False)
    cache = np.load(args.cache, allow_pickle=False)
    observed = _pool_and_normalize(np.asarray(np.load(args.observed, mmap_mode="r")))
    predicted = _pool_and_normalize(np.asarray(np.load(args.predicted, mmap_mode="r")))
    nochange = _pool_and_normalize(np.asarray(np.load(args.nochange, mmap_mode="r")))
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
    if any(len(indices) < 2 for indices in train_by_task.values()):
        raise ValueError("Every task needs at least two train examples for within-task negatives")

    model = transition_energy.TransitionActionEnergy(
        horizon=actions.shape[1],
        action_dim=actions.shape[2],
        transition_dim=observed.shape[-1],
        width=args.width,
        num_heads=args.num_heads,
    )
    params = model.init(jax.random.key(args.seed), observed[:2], actions[:2])["params"]
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(args.learning_rate, 1e-4))
    optimizer_state = optimizer.init(params)

    def contrastive_loss(parameters, transition, action):
        transition_embedding, action_embedding, scale = model.apply(
            {"params": parameters}, transition, action
        )
        logits = scale * transition_embedding @ action_embedding.T
        labels = jnp.arange(logits.shape[0])
        transition_to_action = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        action_to_transition = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels).mean()
        return 0.5 * (transition_to_action + action_to_transition)

    @jax.jit
    def train_step(parameters, state, observed_batch, predicted_batch, action_batch):
        def objective(value):
            # One action embedding space must explain both the realized JEPA
            # transition used online and the current-only prediction used for
            # action guidance.
            return 0.5 * (
                contrastive_loss(value, observed_batch, action_batch)
                + contrastive_loss(value, predicted_batch, action_batch)
            )

        loss, gradients = jax.value_and_grad(objective)(parameters)
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
            params, optimizer_state, observed[indices], predicted[indices], actions[indices]
        )
        if step == 0 or (step + 1) % max(args.steps // 10, 1) == 0:
            item = {"step": step + 1, "loss": float(loss)}
            history.append(item)
            print(json.dumps(item), flush=True)

    @jax.jit
    def embeddings(parameters, transition, action):
        return model.apply({"params": parameters}, transition, action)[:2]

    def evaluate(name: str, transition: jax.Array) -> dict[str, float | int]:
        correct = 0
        total = 0
        margins = []
        reciprocal_ranks = []
        for indices in validation_by_task.values():
            transition_embedding, action_embedding = embeddings(
                params, transition[indices], actions[indices]
            )
            scores = np.asarray(transition_embedding @ action_embedding.T)
            labels = np.arange(len(indices))
            correct += int(np.sum(np.argmax(scores, axis=1) == labels))
            total += len(indices)
            negative = scores.copy()
            negative[labels, labels] = -np.inf
            margins.extend((scores[labels, labels] - np.max(negative, axis=1)).tolist())
            order = np.argsort(-scores, axis=1)
            reciprocal_ranks.extend(
                [1.0 / (int(np.flatnonzero(order[row] == row)[0]) + 1) for row in labels]
            )
        margins_array = np.asarray(margins)
        return {
            "name": name,
            "top1": correct / total,
            "mean_reciprocal_rank": float(np.mean(reciprocal_ranks)),
            "mean_positive_margin": float(np.mean(margins_array)),
            "median_positive_margin": float(np.median(margins_array)),
            "fraction_positive_margin": float(np.mean(margins_array > 0)),
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
            "negatives": "within_task_only",
        },
        "history": history,
        "observed": evaluate("realized_vjepa_transition", observed),
        "predicted": evaluate("current_only_jepawam_prediction", predicted),
        "nochange": evaluate("current_only_nochange", nochange),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "energy_params.msgpack").open("wb") as handle:
        handle.write(serialization.to_bytes(params))
    temporary = args.output_dir / ".metrics.json.tmp"
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output_dir / "metrics.json")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
