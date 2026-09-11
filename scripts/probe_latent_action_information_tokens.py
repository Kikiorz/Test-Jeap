#!/usr/bin/env python3
"""Token-level action information (does the spatial mean pool destroy it?).

The pooled latent explains almost none of the action (held-out R^2 0.135 for the
chunk; 0.047 for the transition, versus 0.705 for the previous action alone).
The mean pool is the obvious suspect: it averages away exactly the local
geometry the action depends on. This uses the AC patch-token cache and a dual
ridge (N < D, so the sample-space solution is the cheap one) to ask whether the
*flattened token grid* carries action information the mean does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_actions(dataset: Path, episode: int) -> np.ndarray:
    chunk = episode // 1000
    path = dataset / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(path, columns=["actions"])
    return np.asarray(table["actions"].to_pylist(), dtype=np.float32)


def collect(args, episodes, rng, limit):
    rows = {"tokens_t": [], "tokens_next": [], "mean_t": [], "prev_action": [], "chunk": []}
    for episode in episodes:
        tokens_path = args.cache / "tokens" / f"episode_{episode:06d}.npy"
        if not tokens_path.exists():
            continue
        tokens = np.load(tokens_path, mmap_mode="r")
        try:
            actions = load_actions(args.dataset, episode)
        except FileNotFoundError:
            continue
        length = min(len(tokens), len(actions))
        if length < args.horizon + 3:
            continue
        index = rng.choice(np.arange(1, length - args.horizon - 1),
                           size=min(args.frames_per_episode, length - args.horizon - 2), replace=False)
        for frame in index:
            frame = int(frame)
            rows["tokens_t"].append(np.asarray(tokens[frame], dtype=np.float16))
            rows["tokens_next"].append(np.asarray(tokens[frame + 1], dtype=np.float16))
            rows["mean_t"].append(np.asarray(tokens[frame], dtype=np.float32).mean(0))
            rows["prev_action"].append(actions[frame - 1])
            rows["chunk"].append(actions[frame:frame + args.horizon].reshape(-1))
        if limit and len(rows["chunk"]) >= limit:
            break
    return {k: np.asarray(v) for k, v in rows.items()}


def dual_ridge(x_train, y_train, x_test, y_test, ridge_values=(1e-2, 1e-1, 1e0, 1e1)):
    x_train = x_train.astype(np.float64)
    x_test = x_test.astype(np.float64)
    y_train = y_train.astype(np.float64)
    gram = x_train @ x_train.T
    best = None
    for value in ridge_values:
        alpha = np.linalg.solve(gram + value * np.eye(len(gram)), y_train)
        prediction = (x_test @ x_train.T) @ alpha
        residual = ((prediction - y_test) ** 2).sum()
        total = ((y_test - y_test.mean(0)) ** 2).sum()
        score = 1.0 - residual / max(total, 1e-12)
        if best is None or score > best[1]:
            best = (value, score)
    return best


def r2_from_ridge(x_train, y_train, x_test, y_test, ridge_values=(1e-1, 1e1, 1e3, 1e5)):
    x_train = np.concatenate([x_train.astype(np.float64), np.ones((len(x_train), 1))], 1)
    x_test = np.concatenate([x_test.astype(np.float64), np.ones((len(x_test), 1))], 1)
    best = None
    for value in ridge_values:
        weight = np.linalg.solve(x_train.T @ x_train + value * np.eye(x_train.shape[1]), x_train.T @ y_train)
        prediction = x_test @ weight
        residual = ((prediction - y_test) ** 2).sum()
        total = ((y_test - y_test.mean(0)) ** 2).sum()
        score = 1.0 - residual / max(total, 1e-12)
        if best is None or score > best[1]:
            best = (value, score)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/workspace/artifacts/con2/ac_tokens_440"))
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    parser.add_argument("--train-episodes", type=int, nargs="+", default=list(range(0, 120, 2)))
    parser.add_argument("--test-episodes", type=int, nargs="+", default=list(range(400, 440)))
    parser.add_argument("--frames-per-episode", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=2400)
    parser.add_argument("--test-limit", type=int, default=800)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    train = collect(args, args.train_episodes, rng, args.train_limit)
    test = collect(args, args.test_episodes, rng, args.test_limit)
    print(json.dumps({"train": len(train["chunk"]), "test": len(test["chunk"])}), flush=True)
    y_train, y_test = train["chunk"], test["chunk"]

    mean_train = np.log1p(np.abs(train["mean_t"])) * np.sign(train["mean_t"])  # keep it linear-friendly
    mean_test = np.log1p(np.abs(test["mean_t"])) * np.sign(test["mean_t"])
    prev = r2_from_ridge(train["prev_action"], y_train, test["prev_action"], y_test)
    pooled = r2_from_ridge(np.concatenate([train["prev_action"], mean_train], 1), y_train,
                           np.concatenate([test["prev_action"], mean_test], 1), y_test)

    flat_train = train["tokens_t"].astype(np.float32).reshape(len(y_train), -1)
    flat_test = test["tokens_t"].astype(np.float32).reshape(len(y_test), -1)
    diff_train = (train["tokens_next"].astype(np.float32) - train["tokens_t"].astype(np.float32)).reshape(len(y_train), -1)
    diff_test = (test["tokens_next"].astype(np.float32) - test["tokens_t"].astype(np.float32)).reshape(len(y_test), -1)

    report = {
        "train_samples": int(len(y_train)), "test_samples": int(len(y_test)),
        "prev_action_only_r2": prev[1],
        "prev_action_plus_pooled_r2": pooled[1],
    }
    for name, xtr, xte in (("tokens_flat", flat_train, flat_test),
                           ("tokens_flat_diff", diff_train, diff_test)):
        score = dual_ridge(xtr, y_train, xte, y_test)
        report[name] = {"best_ridge": score[0], "heldout_r2": score[1]}
        print(json.dumps({name: report[name]}), flush=True)
    text = json.dumps(report, indent=2, sort_keys=True)
    print("RESULT", text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
