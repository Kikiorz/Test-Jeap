#!/usr/bin/env python3
"""Is the frozen V-JEPA 2-AC latent space predictable on LIBERO at all?

Fits a ridge readout on frozen features and reports NMSE against the copy
baseline on held-out episodes. This separates two failure modes:

  * the released AC *predictor* does not transfer, but the space is fine, or
  * the frozen AC *features* carry no usable next-frame signal for LIBERO,
    in which case fine-tuning the predictor alone cannot help.

Sample split is by episode, so no frame leaks between train and eval.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

VJEPA_ROOT = Path("/workspace/vjepa2")
sys.path.insert(0, str(VJEPA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openpi.con2.ac_world_model import (  # noqa: E402
    ACWorldModel,
    build_models,
    frame_transform,
    map_libero_action,
    map_libero_state,
)
from probe_vjepa_ac_libero import load_episode  # noqa: E402


def ridge_fit(x, y, ridge):
    x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    gram = x.T @ x
    gram += ridge * np.eye(gram.shape[0])
    return np.linalg.solve(gram, x.T @ y)


def ridge_predict(weight, x):
    x = np.concatenate([x, np.ones((len(x), 1), dtype=np.float64)], axis=1)
    return x @ weight


def build_features(episodes, args, model):
    rows = {"train": [], "eval": []}
    for episode in episodes:
        frames, state, action = load_episode(args.dataset, episode, args.camera)
        tokens = model.encode(frames[::args.stride], chunk=args.encode_chunk)
        pose = map_libero_state(state[::args.stride])
        act = map_libero_action(action[::args.stride], args.action_variant)
        pooled = F.layer_norm(tokens, (tokens.shape[-1],)).mean(1).numpy()  # [T, D]
        length = len(pooled)
        if length < args.horizon + 2:
            continue
        split = "train" if episode in args.train_episodes else "eval"
        for k in range(length - args.horizon):
            context = np.concatenate([pooled[k], act[k], pose[k]])
            target = pooled[k + args.horizon] - pooled[k]
            rows[split].append((context, target))
    out = {}
    for split, values in rows.items():
        out[split] = (np.stack([v[0] for v in values]).astype(np.float64),
                      np.stack([v[1] for v in values]).astype(np.float64))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=VJEPA_ROOT / "vjepa2-ac-vitg.pt")
    parser.add_argument("--dataset", type=Path,
                        default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    parser.add_argument("--train-episodes", type=int, nargs="+",
                        default=list(range(0, 40)))
    parser.add_argument("--eval-episodes", type=int, nargs="+",
                        default=list(range(40, 50)))
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--camera", choices=["agentview", "wrist"], default="agentview")
    parser.add_argument("--action-variant", choices=["raw", "robosuite"], default="raw")
    parser.add_argument("--encoder-key", choices=["encoder", "target_encoder"], default="encoder")
    parser.add_argument("--encode-chunk", type=int, default=8)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    args.train_episodes = set(args.train_episodes)

    device = torch.device(args.device)
    encoder, _ = build_models(args.checkpoint, encoder_key=args.encoder_key, root=VJEPA_ROOT, device=device)
    model = ACWorldModel(predictor=None, encoder=encoder, transform=frame_transform(VJEPA_ROOT), device=device)

    episodes = sorted(args.train_episodes) + sorted(set(args.eval_episodes) - args.train_episodes)
    features = build_features(episodes, args, model)
    x_train, y_train = features["train"]
    x_eval, y_eval = features["eval"]

    results = {}
    error_zero = float((y_eval ** 2).mean())
    results["copy_current_nmse"] = 1.0
    results["predict_zero_nmse"] = float(((y_eval - 0) ** 2).mean() / error_zero)
    for ridge in (1e-2, 1e0, 1e2, 1e4, 1e6):
        weight = ridge_fit(x_train, y_train, ridge)
        prediction = ridge_predict(weight, x_eval)
        mse = float(((prediction - y_eval) ** 2).mean())
        results[f"ridge_{ridge:g}"] = {
            "heldout_nmse": mse / error_zero,
            "train_nmse": float(((ridge_predict(weight, x_train) - y_train) ** 2).mean() / (y_train ** 2).mean()),
        }
    report = {
        "camera": args.camera,
        "stride": args.stride,
        "fps_effective": 10.0 / args.stride,
        "horizon_frames": args.horizon,
        "action_variant": args.action_variant,
        "encoder_key": args.encoder_key,
        "train_windows": len(x_train),
        "eval_windows": len(x_eval),
        "feature_dim": int(x_train.shape[1]),
        "results": results,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
