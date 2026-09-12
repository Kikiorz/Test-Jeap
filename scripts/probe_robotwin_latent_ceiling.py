#!/usr/bin/env python3
"""How predictable is the RoboTwin future-latent delta at all?

The Con1 head reaches ``con1_delta_nmse`` ~= 0.86 on RoboTwin, where the metric
is ``mse / mse(predicting zero)``, i.e. the head explains ~14% of the delta
variance, and Con2 refines that delta. So either the head is underfitting a
signal that is really there (a stronger head fixes Con2) or the delta is close
to unpredictable from the observation (Con2's premise has to change).

This measures the ceiling with ridge regression on the cached features:
  z_only : the current latent
  r_only : the pooled 64 predictive tokens
  z_r    : both

The design is capped by *sample count* rather than episode count: 6272 features
are already well determined by ~15k samples, and the gram matrix dominates the
cost. Everything runs on CPU so training keeps the GPUs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

Z_DIM = 4224
R_DIM = 2048
SUBSETS = (("z_only", Z_DIM), ("r_only", R_DIM), ("z_r", Z_DIM + R_DIM))


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", type=Path,
                   default=Path("/workspace/artifacts/con1/robotwin_clean20_19999_features_v1"))
    p.add_argument("--horizons", type=int, nargs="+", default=[4, 16])
    p.add_argument("--max-train-samples", type=int, default=15000)
    p.add_argument("--max-test-samples", type=int, default=5000)
    p.add_argument("--train-stride", type=int, default=5)
    p.add_argument("--test-stride", type=int, default=17)
    p.add_argument("--lambdas", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    p.add_argument("--out", type=Path)
    return p.parse_args()


def collect(root: Path, stride: int, offset: int, cap: int, horizons: list[int]):
    """Gather up to `cap` samples as (features, {horizon: delta}) in memory."""
    max_horizon = max(horizons)
    available = sorted(int(p.stem.split("_")[0]) for p in root.glob("*_z.npy"))
    features: list[np.ndarray] = []
    deltas: dict[int, list[np.ndarray]] = {h: [] for h in horizons}
    total = 0
    for index in available[offset::stride]:
        z = np.asarray(np.load(root / f"{index:06d}_z.npy", mmap_mode="r"), np.float32)
        if z.shape[0] <= max_horizon + 1:
            continue
        r = np.asarray(np.load(root / f"{index:06d}_r.npy", mmap_mode="r"), np.float32)
        pooled = r.mean(axis=1)
        cut = z.shape[0] - max_horizon
        anchor = z[:cut]
        features.append(np.concatenate([anchor, pooled[:cut]], axis=1))
        for horizon in horizons:
            deltas[horizon].append(z[horizon:horizon + cut] - anchor)
        total += cut
        if total >= cap:
            break
    if not features:
        raise SystemExit(f"no usable episodes under {root}")
    x = np.concatenate(features, axis=0)[:cap]
    y = {h: np.concatenate(deltas[h], axis=0)[:cap] for h in horizons}
    return x, y


def main() -> None:
    args = parse()
    root = args.features / "episodes"
    started = time.monotonic()

    train_x, train_y = collect(root, args.train_stride, 0, args.max_train_samples, args.horizons)
    test_x, test_y = collect(root, args.test_stride, 3, args.max_test_samples, args.horizons)
    print(f"train samples {train_x.shape[0]}  test samples {test_x.shape[0]}  "
          f"({time.monotonic() - started:.0f}s to load)", flush=True)

    # Standardise with train statistics only; ridge is scale sensitive.
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(train_x.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    train_x = (train_x - mean) / std
    test_x = (test_x - mean) / std

    spans = {"z_only": slice(0, Z_DIM), "r_only": slice(Z_DIM, Z_DIM + R_DIM),
             "z_r": slice(0, Z_DIM + R_DIM)}
    grams: dict[str, np.ndarray] = {}
    cross: dict[str, dict[int, np.ndarray]] = {}
    for name, _ in SUBSETS:
        block = train_x[:, spans[name]].astype(np.float64)
        grams[name] = block.T @ block
        cross[name] = {h: block.T @ train_y[h].astype(np.float64) for h in args.horizons}
        print(f"gram {name}: {grams[name].shape} ({time.monotonic() - started:.0f}s)", flush=True)

    report: dict = {"train_samples": int(train_x.shape[0]), "test_samples": int(test_x.shape[0]),
                    "metric": "mse / mse(predicting zero); 1.0 means no skill", "horizons": {}}
    for horizon in args.horizons:
        target = test_y[horizon].astype(np.float64)
        zero = float((target ** 2).sum())
        entry: dict = {"predict_zero_nmse": 1.0}
        for name, _ in SUBSETS:
            block = test_x[:, spans[name]].astype(np.float64)
            gram = grams[name]
            for lam in args.lambdas:
                reg = lam * max(train_x.shape[0], 1)
                weights = np.linalg.solve(gram + reg * np.eye(gram.shape[0]), cross[name][horizon])
                error = float(((block @ weights - target) ** 2).sum())
                entry[f"{name}_lambda_{lam:g}"] = error / max(zero, 1e-12)
        report["horizons"][str(horizon)] = entry
        cells = "  ".join(f"{name}@{lam:g}={entry[f'{name}_lambda_{lam:g}']:.4f}"
                          for name, _ in SUBSETS for lam in args.lambdas)
        best = min((entry[f"{name}_lambda_{lam:g}"], name, lam)
                   for name, _ in SUBSETS for lam in args.lambdas)
        print(f"h={horizon:2d}  {cells}   best={best[1]}@{best[2]:g}={best[0]:.4f}", flush=True)

    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
