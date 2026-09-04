#!/usr/bin/env python3
"""Rerank frozen pi0.5 action proposals with a learned transition-action energy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import transition_energy
import run_transition_inverse_gate as inverse_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--nochange", type=Path, required=True)
    parser.add_argument("--energy-params", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=157)
    return parser.parse_args()


def _pool(value: np.ndarray) -> jax.Array:
    return transition_energy.normalize(jnp.asarray(inverse_gate._pool_24_to_8(value)))


def _bootstrap_interval(delta: np.ndarray, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(10_000, len(delta)))
    means = delta[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main() -> None:
    args = parse_args()
    samples = np.load(args.samples, allow_pickle=False)
    proposal_data = np.load(args.proposals, allow_pickle=False)
    validation_indices = np.asarray(proposal_data["validation_indices"])
    candidates = jnp.asarray(proposal_data["candidates"])
    experts = jnp.asarray(proposal_data["experts"])

    _, expected_indices = inverse_gate._episode_split(
        samples["episode_indices"], samples["task_indices"]
    )
    if not np.array_equal(validation_indices, expected_indices):
        raise ValueError("Proposal validation indices do not match the episode split")

    observed = _pool(np.asarray(np.load(args.observed, mmap_mode="r")))
    predicted = _pool(np.asarray(np.load(args.predicted, mmap_mode="r")))
    nochange = _pool(np.asarray(np.load(args.nochange, mmap_mode="r")))
    validation = jnp.asarray(validation_indices)
    tasks = np.asarray(samples["task_indices"])[validation_indices]
    shuffled_indices = validation_indices.copy()
    for task in np.unique(tasks):
        positions = np.flatnonzero(tasks == task)
        shuffled_indices[positions] = np.roll(validation_indices[positions], 1)

    model = transition_energy.TransitionActionEnergy(
        horizon=candidates.shape[2],
        action_dim=candidates.shape[3],
        transition_dim=observed.shape[-1],
        width=args.width,
        num_heads=args.num_heads,
    )
    template = model.init(jax.random.key(args.seed), observed[:2], candidates[:2, 0])["params"]
    params = serialization.from_bytes(template, args.energy_params.read_bytes())

    @jax.jit
    def scores(transition: jax.Array, action_candidates: jax.Array) -> jax.Array:
        batch, count = action_candidates.shape[:2]
        tiled_transition = jnp.repeat(transition[:, None], count, axis=1)
        transition_embedding, action_embedding, scale = model.apply(
            {"params": params},
            tiled_transition.reshape(batch * count, *transition.shape[1:]),
            action_candidates.reshape(batch * count, *action_candidates.shape[2:]),
        )
        return (scale * jnp.sum(transition_embedding * action_embedding, axis=-1)).reshape(
            batch, count
        )

    candidate_mse = jnp.mean(jnp.square(candidates - experts[:, None]), axis=(2, 3))
    first_mse = np.asarray(candidate_mse[:, 0])
    result = {
        "config": {"count": int(len(validation_indices)), "candidates": int(candidates.shape[1])},
        "first_candidate_mse": float(first_mse.mean()),
        "oracle_candidate_mse": float(np.asarray(jnp.min(candidate_mse, axis=1)).mean()),
        "transitions": {},
    }
    transition_inputs = {
        "observed": observed[validation],
        "predicted": predicted[validation],
        "nochange": nochange[validation],
        "within_task_shuffled_predicted": predicted[jnp.asarray(shuffled_indices)],
    }
    rows = jnp.arange(len(validation_indices))
    for offset, (name, transition) in enumerate(transition_inputs.items()):
        compatibility = scores(transition, candidates)
        selected_index = jnp.argmax(compatibility, axis=1)
        selected_mse = np.asarray(candidate_mse[rows, selected_index])
        delta = selected_mse - first_mse
        result["transitions"][name] = {
            "mse": float(selected_mse.mean()),
            "mean_mse_delta": float(delta.mean()),
            "median_mse_delta": float(np.median(delta)),
            "bootstrap_95ci_mean_delta": _bootstrap_interval(delta, args.seed + offset),
            "fraction_better_than_first": float(np.mean(delta < 0)),
            "fraction_changed_selection": float(np.mean(np.asarray(selected_index) != 0)),
            "selected_counts": np.bincount(
                np.asarray(selected_index), minlength=candidates.shape[1]
            ).tolist(),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    with temporary.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
