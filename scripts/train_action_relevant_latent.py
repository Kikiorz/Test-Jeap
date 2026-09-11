#!/usr/bin/env python3
"""Can the latent be *shaped* to carry action information?

Measured first (scripts/probe_latent_action_information.py): the latent
transition Delta z carries almost no action information (held-out R^2 0.047 in
the V-JEPA 2.1 pooled space, 0.009 in the AC space) while the previous action
alone explains 0.96 / 0.70 of it. So no coupling that routes Delta z into the
action can work - the quantity being routed is not informative.

This script trains a small projection ``h = P_psi(z_t, z_{t+1})`` with an
inverse-dynamics objective (predict the executed action), which is exactly what
the VLM-based policy does not have, and measures three things on **held-out
episodes**:

  * R^2 of the action predicted from h,
  * the *incremental* R^2 over the previous-action baseline (the number that
    decides whether a coupling can help at all),
  * the same under a simulated deploy-time TTT (adapt on the first half of an
    episode with the executed actions, evaluate on the second half).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


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


def collect(args, episodes, manifest):
    features, targets, previous, episode_ids, order = [], [], [], [], []
    for episode in episodes:
        try:
            latents = load_latents(args.source, args.cache, manifest, episode)
            actions = load_actions(args.dataset, episode)
        except (FileNotFoundError, StopIteration):
            continue
        length = min(len(latents), len(actions))
        for frame in range(1, length - args.horizon - 1):
            features.append(np.concatenate([latents[frame], latents[frame + 1] - latents[frame]]))
            if args.target == "delta":
                # Control-relevant target: what the action does *beyond* the
                # current trend. The absolute action is dominated by temporal
                # continuity (previous action alone explains R^2 0.96), so it is
                # the wrong target for asking whether the latent adds anything.
                targets.append(actions[frame] - actions[frame - 1])
            elif args.target == "immediate":
                targets.append(actions[frame])
            else:
                targets.append(actions[frame:frame + args.horizon].reshape(-1))
            previous.append(actions[frame - 1])
            episode_ids.append(episode)
            order.append(frame)
    return (np.asarray(features, np.float32), np.asarray(targets, np.float32),
            np.asarray(previous, np.float32), np.asarray(episode_ids), np.asarray(order))


def r2(prediction, target):
    residual = ((prediction - target) ** 2).sum()
    total = ((target - target.mean(axis=0)) ** 2).sum()
    return 1.0 - residual / max(total, 1e-12)


def ridge(x, y, x_test, y_test, ridge_values=(1e-1, 1e1, 1e3, 1e5)):
    x = np.concatenate([x, np.ones((len(x), 1), np.float64)], 1)
    x_test = np.concatenate([x_test, np.ones((len(x_test), 1), np.float64)], 1)
    best = None
    for value in ridge_values:
        weight = np.linalg.solve(x.T @ x + value * np.eye(x.shape[1]), x.T @ y)
        score = r2(x_test @ weight, y_test)
        if best is None or score > best[1]:
            best = (value, score)
    return best


class Projection(torch.nn.Module):
    def __init__(self, width_in, hidden, width_out):
        super().__init__()
        self.body = torch.nn.Sequential(
            torch.nn.Linear(width_in, hidden), torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden), torch.nn.GELU())
        self.head = torch.nn.Linear(hidden, width_out)
        self.width = hidden

    def forward(self, x):
        return self.head(self.body(x))

    def embed(self, x):
        return self.body(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["anchored", "ac"], default="anchored")
    parser.add_argument("--cache", type=Path,
                        default=Path("/workspace/artifacts/con1/anchored_40k_features_v1"))
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    parser.add_argument("--train-episodes", type=int, nargs="+", default=list(range(0, 200, 2)))
    parser.add_argument("--test-episodes", type=int, nargs="+", default=list(range(400, 440)))
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--target", choices=["chunk", "immediate", "delta"], default="chunk")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ttt-steps", type=int, default=30)
    parser.add_argument("--ttt-learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    manifest = {"episodes": []}
    if args.source == "anchored":
        manifest = json.loads((args.cache / "manifest.json").read_text())
    device = torch.device(args.device)

    x_train, y_train, prev_train, _, _ = collect(args, args.train_episodes, manifest)
    x_test, y_test, prev_test, episode_test, order_test = collect(args, args.test_episodes, manifest)
    print(json.dumps({"train": len(x_train), "test": len(x_test)}), flush=True)

    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    prev_mean, prev_std = prev_train.mean(0), prev_train.std(0) + 1e-6
    target_mean, target_std = y_train.mean(0), y_train.std(0) + 1e-6

    def prep(x):
        return torch.from_numpy((x - mean) / std).to(device)

    model = Projection(x_train.shape[1], args.hidden, y_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = np.random.default_rng(0)
    target_t = torch.from_numpy((y_train - target_mean) / target_std).to(device)
    x_train_t = prep(x_train)
    for step in range(args.steps):
        index = torch.from_numpy(generator.integers(0, len(x_train), args.batch_size)).to(device)
        loss = torch.nn.functional.mse_loss(model(x_train_t[index]), target_t[index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 500 == 0:
            print(json.dumps({"step": step, "loss": float(loss)}), flush=True)

    with torch.no_grad():
        x_test_t = prep(x_test)
        prediction = model(x_test_t).cpu().numpy() * target_std + target_mean
        embedding = model.embed(x_test_t).cpu().numpy()
        train_embedding = model.embed(prep(x_train)).cpu().numpy()
    # Fitted on TRAIN, evaluated on held-out episodes - an in-sample fit here
    # would flatter every feature set, and the whole question is whether the
    # embedding transfers.
    prev_baseline = ridge(prev_train, y_train, prev_test, y_test)
    combined = ridge(np.concatenate([prev_train, train_embedding], 1), y_train,
                     np.concatenate([prev_test, embedding], 1), y_test)
    embedding_only = ridge(train_embedding, y_train, embedding, y_test)
    raw_latent = ridge(np.concatenate([prev_train, x_train], 1), y_train,
                       np.concatenate([prev_test, x_test], 1), y_test)

    # Deploy-time TTT simulation: adapt the head on the first half of each
    # held-out episode using only the executed actions, score the second half.
    ttt_before, ttt_after = [], []
    for episode in np.unique(episode_test):
        mask = episode_test == episode
        index = np.where(mask)[0]
        split = index[len(index) // 2]
        first, second = index[index < split], index[index >= split]
        if len(first) < 5 or len(second) < 5:
            continue
        head = torch.nn.Linear(model.width, y_train.shape[1]).to(device)
        with torch.no_grad():
            head.weight.copy_(model.head.weight)
            head.bias.copy_(model.head.bias)
        body = model.body
        params = list(head.parameters()) + list(body.parameters())
        optimizer_ttt = torch.optim.AdamW(params, lr=args.ttt_learning_rate, weight_decay=0.0)
        first_x = prep(x_train[:0])  # placeholder to keep dtype/device consistent
        first_features = prep(x_test[first])
        first_target = torch.from_numpy((y_test[first] - target_mean) / target_std).to(device)
        second_features = prep(x_test[second])
        second_target = y_test[second]
        with torch.no_grad():
            before = head(body(second_features)).cpu().numpy() * target_std + target_mean
        for _ in range(args.ttt_steps):
            optimizer_ttt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(head(body(first_features)), first_target)
            loss.backward()
            optimizer_ttt.step()
        with torch.no_grad():
            after = head(body(second_features)).cpu().numpy() * target_std + target_mean
        ttt_before.append(r2(before, second_target))
        ttt_after.append(r2(after, second_target))

    report = {
        "source": args.source,
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "model_holdout_r2": r2(prediction, y_test),
        "prev_action_only_r2": prev_baseline[1],
        "embedding_only_ridge_r2": embedding_only[1],
        "prev_action_plus_embedding_r2": combined[1],
        "prev_action_plus_raw_latent_r2": raw_latent[1],
        "incremental_over_prev": combined[1] - prev_baseline[1],
        "incremental_raw_latent_over_prev": raw_latent[1] - prev_baseline[1],
        "ttt_episodes": len(ttt_before),
        "ttt_mean_r2_before": float(np.mean(ttt_before)) if ttt_before else None,
        "ttt_mean_r2_after": float(np.mean(ttt_after)) if ttt_after else None,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print("RESULT", text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
