#!/usr/bin/env python3
"""Compare paired held-out errors from joint and independent A-B flows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--independent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=531)
    return parser.parse_args()


def _episode_bootstrap(
    improvement: np.ndarray,
    validation_indices: np.ndarray,
    episodes: np.ndarray,
    replicates: int,
    seed: int,
) -> dict[str, float | int]:
    unique = np.unique(episodes[validation_indices])
    means = np.asarray(
        [improvement[episodes[validation_indices] == episode].mean() for episode in unique],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(means, (replicates, len(means)), replace=True).mean(axis=1)
    return {
        "mean_improvement": float(means.mean()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "episode_count": len(means),
    }


def main() -> None:
    args = parse_args()
    samples = np.load(args.samples, allow_pickle=False)
    joint = np.load(args.joint, allow_pickle=False)
    independent = np.load(args.independent, allow_pickle=False)
    validation_indices = np.asarray(joint["validation_indices"])
    if not np.array_equal(validation_indices, independent["validation_indices"]):
        raise ValueError("Validation indices differ between arms")
    episodes = np.asarray(samples["episode_indices"])

    fields = (
        "action_flow_loss",
        "change_flow_loss",
        "action_endpoint_mse",
        "change_endpoint_mse",
    )
    comparisons = {}
    for offset, field in enumerate(fields):
        improvement = np.asarray(independent[field]) - np.asarray(joint[field])
        comparisons[field] = _episode_bootstrap(
            improvement,
            validation_indices,
            episodes,
            args.bootstrap_replicates,
            args.seed + offset,
        )
        comparisons[field]["positive_sample_fraction"] = float(np.mean(improvement > 0.0))

    conditions = {f"joint_improves_{field}": comparisons[field]["ci95_low"] > 0.0 for field in fields}
    result = {
        "comparison": "joint minus matched independent, reported as independent_error - joint_error",
        "comparisons": comparisons,
        "conditions": conditions,
        "passed_all_conditions": bool(all(conditions.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
