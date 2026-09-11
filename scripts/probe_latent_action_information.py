#!/usr/bin/env python3
"""How much *action* information does the latent actually carry?

Motivation: the Con1 coupling tries to let a future-latent prediction influence
the action. That can only work if the latent carries information about the
action that the policy's own observation encoding does not already have. This
probe measures that directly, with held-out episodes and three baselines that
matter:

  * the zero predictor (R^2 = 0 by definition),
  * previous-action autocorrelation (a_t from a_{t-1}) - in manipulation the
    action is highly autocorrelated, so a latent that only reproduces that adds
    nothing,
  * the current latent alone (does the *transition* add anything?).

Latents: the frozen V-JEPA 2.1 pooled teacher latent used by Con1, or the
V-JEPA 2-AC token mean. Targets: the immediate action and the 10-step chunk.
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


def load_latents(source: str, cache: Path, manifest, episode: int) -> np.ndarray:
    if source == "anchored":
        entry = next(e for e in manifest["episodes"] if e["id"] == episode)
        return np.load(cache / entry["z"], allow_pickle=False).astype(np.float32)
    tokens = np.load(cache / "tokens" / f"episode_{episode:06d}.npy", mmap_mode="r")
    return np.asarray(tokens, dtype=np.float32).mean(axis=1)


def ridge_fit(x, y, ridge):
    x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    weight = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    return weight


def r2(weight, x, y):
    x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    prediction = x @ weight
    residual = ((prediction - y) ** 2).sum()
    total = ((y - y.mean(axis=0)) ** 2).sum()
    return 1.0 - residual / max(total, 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["anchored", "ac"], default="anchored")
    parser.add_argument("--cache", type=Path,
                        default=Path("/workspace/artifacts/con1/anchored_40k_features_v1"))
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    parser.add_argument("--train-episodes", type=int, nargs="+", default=list(range(0, 200, 2)))
    parser.add_argument("--test-episodes", type=int, nargs="+", default=list(range(400, 440)))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--ridges", type=float, nargs="+", default=[1e-2, 1e0, 1e2, 1e4])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifest = {"episodes": []}
    if args.source == "anchored":
        manifest = json.loads((args.cache / "manifest.json").read_text())

    def collect(episodes):
        rows = {"z_t": [], "z_next": [], "diff": [], "prev_action": [],
                "action": [], "chunk": []}
        for episode in episodes:
            try:
                latents = load_latents(args.source, args.cache, manifest, episode)
                actions = load_actions(args.dataset, episode)
            except (FileNotFoundError, StopIteration):
                continue
            length = min(len(latents), len(actions))
            for frame in range(1, length - args.horizon - 1):
                rows["z_t"].append(latents[frame])
                rows["z_next"].append(latents[frame + 1])
                rows["diff"].append(latents[frame + 1] - latents[frame])
                rows["prev_action"].append(actions[frame - 1])
                rows["action"].append(actions[frame])
                rows["chunk"].append(actions[frame:frame + args.horizon].reshape(-1))
        return {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}

    train = collect(args.train_episodes)
    test = collect(args.test_episodes)
    print(json.dumps({"train_samples": len(train["action"]), "test_samples": len(test["action"])}), flush=True)

    # Standardise on the training split.
    means = {k: train[k].mean(axis=0) for k in ("z_t", "z_next", "diff", "prev_action")}
    scales = {k: train[k].std(axis=0) + 1e-6 for k in means}

    def standardise(values, key):
        return (values[key] - means[key]) / scales[key]

    features = {
        "prev_action": standardise(train, "prev_action"),
        "z_t": standardise(train, "z_t"),
        "z_next": standardise(train, "z_next"),
        "diff": standardise(train, "diff"),
        "z_t+z_next": np.concatenate([standardise(train, "z_t"), standardise(train, "z_next")], axis=1),
        "z_t+z_next+prev": np.concatenate([standardise(train, "z_t"), standardise(train, "z_next"),
                                           standardise(train, "prev_action")], axis=1),
    }
    test_features = {
        "prev_action": standardise(test, "prev_action"),
        "z_t": standardise(test, "z_t"),
        "z_next": standardise(test, "z_next"),
        "diff": standardise(test, "diff"),
        "z_t+z_next": np.concatenate([standardise(test, "z_t"), standardise(test, "z_next")], axis=1),
        "z_t+z_next+prev": np.concatenate([standardise(test, "z_t"), standardise(test, "z_next"),
                                           standardise(test, "prev_action")], axis=1),
    }

    report = {"source": args.source, "train_samples": len(train["action"]),
              "test_samples": len(test["action"]), "results": {}}
    for target in ("action", "chunk"):
        report["results"][target] = {}
        for name, x in features.items():
            best = None
            for ridge in args.ridges:
                weight = ridge_fit(x, train[target], ridge)
                value = r2(weight, test_features[name], test[target])
                if best is None or value > best[1]:
                    best = (ridge, value)
            report["results"][target][name] = {"best_ridge": best[0], "heldout_r2": best[1]}
            print(json.dumps({"target": target, "features": name, "ridge": best[0],
                              "r2": round(best[1], 4)}), flush=True)
    text = json.dumps(report, indent=2, sort_keys=True)
    print("RESULT", text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
