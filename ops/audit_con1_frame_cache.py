#!/usr/bin/env python3
"""Read-only cache/split audit, emitting JSON for monitoring."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from openpi.training.orthogonal_targets import episode_split_indices


def audit(dataset_root, cache_root):
    with (dataset_root / "meta/episodes.jsonl").open() as handle:
        episodes = [json.loads(line) for line in handle]
    with (dataset_root / "meta/tasks.jsonl").open() as handle:
        tasks = [json.loads(line) for line in handle]
    names = {row["task"]: int(row["task_index"]) for row in tasks}
    episode_ids = np.array([int(row["episode_index"]) for row in episodes])
    task_ids = np.array([names[row["tasks"][0]] for row in episodes])
    train = set(episode_split_indices(task_ids, episode_ids, split="train").tolist())
    val = set(episode_split_indices(task_ids, episode_ids, split="validation").tolist())
    if train & val or len(train) + len(val) != len(episodes):
        raise RuntimeError("Invalid episode partition")
    with (cache_root / "manifest.json").open() as handle:
        manifest = json.load(handle)
    suite_names = ("libero_10", "libero_goal", "libero_object", "libero_spatial")
    suites = {name: Counter() for name in suite_names}
    errors = []
    aggregate_diagnostics = {f"h{h}": Counter() for h in (1, 5, 10)}
    for position, row in enumerate(episodes):
        episode = int(row["episode_index"])
        length = int(row["length"])
        suite = suites[suite_names[task_ids[position] // 10]]
        suite["total_episodes"] += 1
        suite["total_frames"] += length
        split = "train" if position in train else "validation"
        suite[f"{split}_episodes"] += 1
        suite[f"{split}_frames"] += length
        path = (cache_root / "states" / f"chunk-{episode // manifest['chunks_size']:03d}"
                / f"episode_{episode:06d}.npy")
        if not path.with_suffix(".json").exists():
            continue
        try:
            states = np.load(path, mmap_mode="r", allow_pickle=False)
            if states.shape != (length, manifest["state_dim"]) or states.dtype != np.float16:
                raise ValueError("Unexpected cache shape or dtype")
            with path.with_suffix(".json").open() as handle:
                metadata = json.load(handle)
            if metadata["episode"] != episode or metadata["frames"] != length:
                raise ValueError("Unexpected episode identity")
            if metadata.get("task_index", int(task_ids[position])) != int(task_ids[position]):
                raise ValueError("Cache task disagrees with dataset")
            suite["complete_episodes"] += 1
            suite["complete_frames"] += length
            for horizon, aggregate in aggregate_diagnostics.items():
                item = metadata["diagnostics"].get(horizon)
                if item is None:
                    continue
                aggregate["count"] += item["count"]
                for key in ("state_cosine_mean", "delta_norm_mean", "delta_zero_predictor_mse"):
                    aggregate[key] += item[key] * item["count"]
                aggregate["orthogonality_max_abs"] = max(aggregate["orthogonality_max_abs"], item["orthogonality_max_abs"])
        except (OSError, ValueError, KeyError) as error:
            errors.append({"episode": episode, "error": str(error)})
    for aggregate in aggregate_diagnostics.values():
        for key in ("state_cosine_mean", "delta_norm_mean", "delta_zero_predictor_mse"):
            aggregate[key] /= max(aggregate["count"], 1)
    return {
        "total_frames": sum(int(row["length"]) for row in episodes),
        "complete_frames": sum(suite["complete_frames"] for suite in suites.values()),
        "complete_episodes": sum(suite["complete_episodes"] for suite in suites.values()),
        "total_episodes": len(episodes), "suites": suites,
        "train_episode_ids": [int(episode_ids[i]) for i in sorted(train)],
        "validation_episode_ids": [int(episode_ids[i]) for i in sorted(val)],
        "diagnostics": aggregate_diagnostics, "errors": errors,
        "complete": not errors and sum(s["complete_episodes"] for s in suites.values()) == len(episodes),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset_root, args.cache_root)
    # Keep full episode lists available via the importable audit() API.
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("episode_ids")}, indent=2))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
