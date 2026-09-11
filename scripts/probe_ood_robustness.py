#!/usr/bin/env python3
"""Which representation keeps predicting the action under perturbation?

Reads two ``dump_ood_features.py`` outputs (policy VLM tokens and the V-JEPA
2-AC latent) and reports, for each condition, the held-out R^2 of a ridge fitted
on **clean** data. Features are reduced with a PCA fitted on the clean training
split so the fit is well posed and the comparison across representations and
conditions is like for like.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def previous_actions(dataset: Path, episode_label, frame_label):
    """The previous action is available at deployment, so it is a legitimate
    feature - and it is the baseline everything else has to beat."""
    cache = {}
    values = []
    for episode, frame in zip(episode_label, frame_label, strict=True):
        episode = int(episode)
        if episode not in cache:
            chunk = episode // 1000
            path = dataset / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
            table = pq.read_table(path, columns=["actions"])
            cache[episode] = np.asarray(table["actions"].to_pylist(), dtype=np.float32)
        values.append(cache[episode][max(int(frame) - 1, 0)])
    return np.asarray(values, dtype=np.float32)


def load(path: Path):
    data = np.load(path, allow_pickle=False)
    conditions = [str(c) for c in data["conditions"]]
    return data, conditions


def r2(prediction, target):
    residual = ((prediction - target) ** 2).sum()
    total = ((target - target.mean(axis=0)) ** 2).sum()
    return 1.0 - residual / max(total, 1e-12)


def evaluate(data, conditions, feature_key, components, previous, ridge_values=(1e-2, 1e-1, 1e0, 1e1, 1e2)):
    # The dump writes one block per condition, in order, so slice the labels the
    # same way instead of indexing them with condition-global positions.
    per_condition = len(data["chunk"]) // len(conditions)
    index = {c: i * per_condition for i, c in enumerate(conditions)}
    chunk = {c: data["chunk"][start:start + per_condition] for c, start in index.items()}
    prev = {c: previous[start:start + per_condition] for c, start in index.items()}
    features = {c: data[f"features_{c}"] for c in conditions}
    mean = features["clean"].mean(0)
    # PCA basis from the clean split.
    centred = features["clean"] - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    basis = vt[:components].T
    reduced = {c: (features[c] - mean) @ basis for c in conditions}
    # Standardise the previous action too; it dominates the raw scale otherwise.
    prev_mean, prev_std = prev["clean"].mean(0), prev["clean"].std(0) + 1e-6
    prev = {c: (value - prev_mean) / prev_std for c, value in prev.items()}

    # Split frames, not conditions, into train/eval halves of the clean set.
    half = per_condition // 2
    train_index, eval_index = np.arange(half), np.arange(half, per_condition)
    results = {}
    for c in conditions:
        if c == "clean":
            rows = eval_index
        else:
            rows = np.arange(per_condition)
        results[c] = {"frames": int(len(rows))}
        for label, blocks in (("prev_only", [prev]), ("prev_plus_repr", [prev, reduced])):
            x_train = np.concatenate([b["clean"][train_index] for b in blocks]
                                     + [np.ones((len(train_index), 1))], axis=1)
            x_eval = np.concatenate([b[c][rows] for b in blocks]
                                    + [np.ones((len(rows), 1))], axis=1)
            best = None
            for value in ridge_values:
                weight = np.linalg.solve(x_train.T @ x_train + value * np.eye(x_train.shape[1]),
                                         x_train.T @ chunk["clean"][train_index])
                score = r2(x_eval @ weight, chunk[c][rows])
                if best is None or score > best[1]:
                    best = (value, score)
            results[c][label] = {"best_ridge": best[0], "heldout_r2": best[1]}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--ac", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = {"components": args.components, "representations": {}}
    for name, path in (("policy_vlm_tokens", args.policy), ("vjepa_ac_latent", args.ac)):
        data, conditions = load(path)
        report["conditions"] = conditions
        previous = previous_actions(args.dataset, data["episode"], data["frame"])
        report["representations"][name] = evaluate(data, conditions, name, args.components, previous)
        print(json.dumps({name: report["representations"][name]}), flush=True)
    text = json.dumps(report, indent=2, sort_keys=True)
    print("RESULT", text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
