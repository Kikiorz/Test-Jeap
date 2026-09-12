#!/usr/bin/env python3
"""How predictable is the RoboTwin future-latent delta at all?

The Con1 head reaches ``con1_delta_nmse`` ~= 0.86 on RoboTwin, where the metric
is ``mse / mse(predicting zero)``, i.e. the head explains ~14% of the delta
variance. Con2 refines that delta, so if the head is underfitting a signal that
is actually there, a stronger head fixes Con2; if the delta is close to
unpredictable from the observation, Con2's premise has to change instead.

This measures the ceiling with ridge regression on the cached features:
  z_only  : the current latent
  r_only  : the pooled 64 V-JEPA / VLM predictive tokens
  z_r     : both
It reports the same NMSE the training loop reports, so the numbers are directly
comparable, and runs on CPU so it does not contend with training for the GPUs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path,
                   default=Path("/workspace/artifacts/con1/robotwin_clean20_19999_features_v1"))
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--train-episodes", type=int, default=500)
    p.add_argument("--test-episodes", type=int, default=150)
    p.add_argument("--train-stride", type=int, default=5)
    p.add_argument("--train-offset", type=int, default=0)
    p.add_argument("--test-stride", type=int, default=17)
    p.add_argument("--test-offset", type=int, default=3)
    p.add_argument("--lambdas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    p.add_argument("--out", type=Path)
    return p.parse_args()


def episode_ids(root: Path, stride: int, offset: int, limit: int) -> list[int]:
    available = sorted(int(p.stem.split("_")[0]) for p in root.glob("*_z.npy"))
    selected = available[offset::stride]
    return selected[:limit]


def load_episode(root: Path, index: int, max_horizon: int):
    z = np.asarray(np.load(root / f"{index:06d}_z.npy", mmap_mode="r"), np.float32)
    r = np.asarray(np.load(root / f"{index:06d}_r.npy", mmap_mode="r"), np.float32)
    usable = z.shape[0] - max_horizon
    if usable <= 1:
        return None
    return z[:usable], r[:usable].mean(axis=1)


class Design:
    """Ridge normal equations for one feature subset, accumulated per episode."""

    def __init__(self, dim: int, target_dim: int):
        self.xtx = np.zeros((dim, dim), np.float64)
        self.xty = np.zeros((dim, target_dim), np.float64)
        self.n = 0

    def add(self, features: np.ndarray, targets: np.ndarray) -> None:
        x = features.astype(np.float64)
        y = targets.astype(np.float64)
        self.xtx += x.T @ x
        self.xty += x.T @ y
        self.n += x.shape[0]

    def solve(self, lam: float) -> np.ndarray:
        reg = lam * max(self.n, 1)
        return np.linalg.solve(self.xtx + reg * np.eye(self.xtx.shape[0]), self.xty)


def squared_error(features: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> tuple[float, float, int]:
    prediction = features.astype(np.float64) @ weights
    sse = float(np.sum((prediction - targets.astype(np.float64)) ** 2))
    zero = float(np.sum(targets.astype(np.float64) ** 2))
    return sse, zero, features.shape[0]


def main() -> None:
    args = parse()
    root = args.features / "episodes"
    max_horizon = max(args.horizons)
    train_ids = episode_ids(root, args.train_stride, args.train_offset, args.train_episodes)
    test_ids = episode_ids(root, args.test_stride, args.test_offset, args.test_episodes)
    print(f"train episodes: {len(train_ids)}  test episodes: {len(test_ids)}", flush=True)

    # Pass 1: feature means and variances, so the ridge is scale-free.
    count = 0
    sum_z = np.zeros(4224, np.float64)
    sum_r = np.zeros(2048, np.float64)
    sum_z2 = np.zeros(4224, np.float64)
    sum_r2 = np.zeros(2048, np.float64)
    for index in train_ids:
        loaded = load_episode(root, index, max_horizon)
        if loaded is None:
            continue
        z, pooled = loaded
        sum_z += z.sum(axis=0, dtype=np.float64)
        sum_r += pooled.sum(axis=0, dtype=np.float64)
        sum_z2 += (z.astype(np.float64) ** 2).sum(axis=0)
        sum_r2 += (pooled.astype(np.float64) ** 2).sum(axis=0)
        count += z.shape[0]
    mean_z, mean_r = sum_z / count, sum_r / count
    std_z = np.sqrt(np.maximum(sum_z2 / count - mean_z**2, 1e-12))
    std_r = np.sqrt(np.maximum(sum_r2 / count - mean_r**2, 1e-12))

    # Pass 2: normal equations per horizon and feature subset.
    designs = {
        horizon: {name: Design(dim, 4224) for name, dim in
                  (("z_only", 4224), ("r_only", 2048), ("z_r", 4224 + 2048))}
        for horizon in args.horizons
    }
    for index in train_ids:
        loaded = load_episode(root, index, max_horizon)
        if loaded is None:
            continue
        z, pooled = loaded
        zn = (z - mean_z) / std_z
        rn = (pooled - mean_r) / std_r
        for horizon in args.horizons:
            targets = z[horizon:] - z[:-horizon]
            zn_in, rn_in = zn[:-horizon], rn[:-horizon]
            designs[horizon]["z_only"].add(zn_in, targets)
            designs[horizon]["r_only"].add(rn_in, targets)
            designs[horizon]["z_r"].add(np.concatenate([zn_in, rn_in], axis=1), targets)

    # Test pass: for every lambda, solve and score on episodes never seen.
    report: dict = {"train_episodes": len(train_ids), "test_episodes": len(test_ids),
                    "horizons": {}, "metric": "mse / mse(predicting zero); 1.0 = no skill"}
    for horizon in args.horizons:
        weights = {name: designs[horizon][name].solve(lam)
                   for name in ("z_only", "r_only", "z_r") for lam in args.lambdas}
        sse = {key: 0.0 for key in weights}
        zero_total = 0.0
        count_test = 0
        for index in test_ids:
            loaded = load_episode(root, index, max_horizon)
            if loaded is None:
                continue
            z, pooled = loaded
            zn = (z - mean_z) / std_z
            rn = (pooled - mean_r) / std_r
            targets = z[horizon:] - z[:-horizon]
            zn_in, rn_in = zn[:-horizon], rn[:-horizon]
            features = {"z_only": zn_in, "r_only": rn_in,
                        "z_r": np.concatenate([zn_in, rn_in], axis=1)}
            for name, matrix in features.items():
                for lam in args.lambdas:
                    error, zero, _ = squared_error(matrix, targets, weights[(name, lam)])
                    sse[(name, lam)] += error
                    if name == "z_only" and lam == args.lambdas[-1]:
                        zero_total += zero
            count_test += targets.shape[0]
        entry = {"samples": count_test, "predict_zero_nmse": 1.0}
        for name in ("z_only", "r_only", "z_r"):
            for lam in args.lambdas:
                entry[f"{name}_nmse_lambda_{lam:g}"] = sse[(name, lam)] / max(zero_total, 1e-12)
        report["horizons"][str(horizon)] = entry
        best = min((entry[f"{name}_nmse_lambda_{lam:g}"], name, lam)
                   for name in ("z_only", "r_only", "z_r") for lam in args.lambdas)
        print(f"h={horizon:2d} samples={count_test:6d}  "
              + "  ".join(f"{name}*{lam:g}={entry[f'{name}_nmse_lambda_{lam:g}']:.4f}"
                          for lam in args.lambdas for name in ("z_only", "r_only", "z_r"))
              + f"   best={best[1]}@{best[2]:g}={best[0]:.4f}", flush=True)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
