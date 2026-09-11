#!/usr/bin/env python3
"""Which latent target is worth coupling to the action: JEPA pooled, VLM pooled, or VLM tokens?

Everything is read from the existing anchored feature cache, which stores, for
every frame of every episode, ``z`` (2816-dim pooled V-JEPA 2.1) and ``r``
(64 x 2048 PI0.5 VLM predictive tokens). That makes the future target
``Delta = f[t + j] - f[t]`` free for any horizon.

For each candidate target the script reports the three numbers that decide
whether a Con1/Con2 coupling can work at all:

  * ``predictability`` - can a linear head (anchored on the current
    representation) predict the true delta? Direction cosine and normalised MSE
    against the "copy the anchor" trivial baseline.
  * ``r2_prev`` / ``r2_prev_true`` - held-out R^2 of the action chunk from the
    previous action alone, and from the previous action plus the *true* future
    delta. The gap is the ceiling any TTT correction could ever recover.
  * ``r2_prev_pred`` - the same but with the *predicted* delta. If this already
    sits at the ceiling, there is nothing left for Con2 to correct; if it sits
    at ``r2_prev``, the head (not the TTT rule) is the bottleneck.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_actions(dataset: Path, episode: int) -> np.ndarray | None:
    chunk = episode // 1000
    path = dataset / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    if not path.exists():
        return None
    table = pq.read_table(path, columns=["actions"])
    return np.asarray(table["actions"].to_pylist(), dtype=np.float32)


def collect(args, episodes, rng):
    rows = {
        "anchor_pool": [],      # [z_t ; mean_t r_t]
        "prev_action": [],
        "chunk": [],
        "delta_jepa": [],       # z[t+j] - z[t]
        "delta_vlm_pool": [],   # mean_t (r[t+j] - r[t])
        "delta_vlm_token": [],  # flatten(r[t+j] - r[t])
        "absolute_vlm_pool": [],  # mean_t r[t+j] - control
    }
    for episode in episodes:
        r_path = args.cache / "episodes" / f"{episode:06d}_r.npy"
        z_path = args.cache / "episodes" / f"{episode:06d}_z.npy"
        if not r_path.exists() or not z_path.exists():
            continue
        actions = load_actions(args.dataset, episode)
        if actions is None:
            continue
        r = np.load(r_path, mmap_mode="r")
        z = np.load(z_path, mmap_mode="r")
        length = min(len(r), len(z), len(actions))
        if length < args.horizon + 3:
            continue
        candidates = np.arange(1, length - args.horizon - 1)
        take = min(args.frames_per_episode, len(candidates))
        for frame in np.sort(rng.choice(candidates, size=take, replace=False)):
            frame = int(frame)
            r_t = np.asarray(r[frame], dtype=np.float32)
            r_f = np.asarray(r[frame + args.horizon], dtype=np.float32)
            z_t = np.asarray(z[frame], dtype=np.float32)
            z_f = np.asarray(z[frame + args.horizon], dtype=np.float32)
            rows["anchor_pool"].append(np.concatenate([z_t, r_t.mean(0)]))
            rows["prev_action"].append(actions[frame - 1])
            rows["chunk"].append(actions[frame:frame + args.horizon].reshape(-1))
            rows["delta_jepa"].append(z_f - z_t)
            rows["delta_vlm_pool"].append((r_f - r_t).mean(0))
            rows["delta_vlm_token"].append((r_f - r_t).reshape(-1))
            rows["absolute_vlm_pool"].append(r_f.mean(0))
        if args.limit and len(rows["chunk"]) >= args.limit:
            break
    return {k: np.asarray(v, dtype=np.float32) for k, v in rows.items()}


def standardise(x_train, x_test):
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    return (x_train - mean) / std, (x_test - mean) / std


def dual_ridge(x_train, y_train, x_test, y_test, ridge_values):
    """Sample-space ridge, for feature blocks wider than the sample count."""
    x_train, x_test = standardise(x_train, x_test)
    y_train = y_train.astype(np.float64)
    y_test = y_test.astype(np.float64)
    gram = x_train @ x_train.T
    best = None
    for value in ridge_values:
        alpha = np.linalg.solve(gram + value * np.eye(len(gram)), y_train)
        pred = (x_test @ x_train.T) @ alpha
        residual = ((pred - y_test) ** 2).sum()
        total = ((y_test - y_test.mean(0)) ** 2).sum()
        score = 1.0 - residual / max(total, 1e-12)
        if best is None or score > best[1]:
            best = (value, score, pred, (gram @ alpha))
    return best


def fit_delta(x_train, d_train, x_test, d_test, ridge_values):
    """Return the held-out metrics plus in-sample/held-out delta predictions.

    The action ridge has to be fitted on *train* rows, so it needs the head's
    train-side prediction as well as the held-out one.
    """
    score = dual_ridge(x_train, d_train, x_test, d_test, ridge_values)
    prediction = score[2]
    prediction_train = score[3]
    d_test = d_test.astype(np.float64)
    # Direction cosine, averaged per sample.
    numerator = (prediction * d_test).sum(1)
    denominator = np.linalg.norm(prediction, axis=1) * np.linalg.norm(d_test, axis=1) + 1e-12
    cosine = float(np.mean(numerator / denominator))
    nmse_vs_copy = float(((prediction - d_test) ** 2).sum() / max((d_test ** 2).sum(), 1e-12))
    metrics = {"best_ridge": score[0], "nmse_vs_copy": nmse_vs_copy, "direction_cosine": cosine}
    return metrics, prediction_train, prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path,
                        default=Path("/workspace/artifacts/con1/anchored_40k_features_v1"))
    parser.add_argument("--dataset", type=Path,
                        default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    parser.add_argument("--train-episodes", type=int, nargs="+", default=list(range(0, 300)))
    parser.add_argument("--test-episodes", type=int, nargs="+", default=list(range(400, 460)))
    parser.add_argument("--frames-per-episode", type=int, default=6)
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 10])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    ridge_delta = (1e-1, 1e1, 1e3, 1e5)
    ridge_action = (1e-1, 1e1, 1e3, 1e5, 1e7)
    report = {}

    for horizon in args.horizon:
        args.horizon = horizon
        rng = np.random.default_rng(horizon)
        train = collect(args, args.train_episodes, rng)
        test = collect(args, args.test_episodes, rng)
        print(json.dumps({"horizon": horizon, "train": len(train["chunk"]),
                          "test": len(test["chunk"])}), flush=True)
        y_train, y_test = train["chunk"], test["chunk"]
        a_train, a_test = train["prev_action"], test["prev_action"]

        prev_only = dual_ridge(a_train, y_train, a_test, y_test, ridge_action)
        settings = {
            "jepa_pooled": ("delta_jepa", "anchor_pool"),
            "vlm_pooled": ("delta_vlm_pool", "anchor_pool"),
            "vlm_token": ("delta_vlm_token", "anchor_pool"),
            "vlm_absolute_control": ("absolute_vlm_pool", "anchor_pool"),
        }
        rows = {}
        for name, (delta_key, anchor_key) in settings.items():
            d_train, d_test = train[delta_key], test[delta_key]
            predictability, delta_hat_train, delta_hat = fit_delta(
                train[anchor_key], d_train, test[anchor_key], d_test, ridge_delta)
            with_true = dual_ridge(np.concatenate([a_train, d_train], 1), y_train,
                                   np.concatenate([a_test, d_test], 1), y_test, ridge_action)
            with_pred = dual_ridge(np.concatenate([a_train, delta_hat_train.astype(np.float32)], 1), y_train,
                                   np.concatenate([a_test, delta_hat.astype(np.float32)], 1), y_test,
                                   ridge_action)
            rows[name] = {
                "dim": int(d_train.shape[1]),
                "predictability": predictability,
                "r2_prev_only": prev_only[1],
                "r2_prev_true_delta": with_true[1],
                "r2_prev_pred_delta": with_pred[1],
                "ttt_headroom": with_true[1] - with_pred[1],
            }
            print(json.dumps({name: rows[name]}), flush=True)
        report[f"horizon_{horizon}"] = rows

    text = json.dumps(report, indent=2, sort_keys=True)
    print("RESULT", text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
