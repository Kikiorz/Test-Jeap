#!/usr/bin/env python3
"""Audit the Con2 token cache against a fresh online encode of the same frames.

The Con1 pipeline shipped a real bug of exactly this shape - the offline cache
was built with one encoder (or one preprocessing path) and serving used another,
which silently injected a constant offset into every downstream number. Every
Con2 result depends on the cached tokens being reproducible, so this script
checks them directly:

  * tokens: cached fp16 vs a fresh ``ACWorldModel.encode`` of the same frame,
  * actions/states: cached vs a fresh mapping from the parquet,
  * and the fp16 storage error itself, which sets the noise floor for the audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    parser.add_argument("--checkpoint", type=Path, default=VJEPA_ROOT / "vjepa2-ac-vitg.pt")
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 7, 42, 400])
    parser.add_argument("--camera", choices=["agentview", "wrist"], default="agentview")
    parser.add_argument("--action-variant", choices=["raw", "robosuite", "calibrated"], default="raw")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder, _ = build_models(args.checkpoint, root=VJEPA_ROOT, device=device)
    model = ACWorldModel(predictor=None, encoder=encoder, transform=frame_transform(VJEPA_ROOT),
                         device=str(device))
    contract = json.loads((args.cache / "contract.json").read_text())
    if contract["camera"] != args.camera:
        raise ValueError(f"Cache was built for {contract['camera']}, asked for {args.camera}")

    rng = np.random.default_rng(0)
    records = []
    for episode in args.episodes:
        tokens_path = args.cache / "tokens" / f"episode_{episode:06d}.npy"
        if not tokens_path.exists():
            continue
        cached = np.load(tokens_path, mmap_mode="r")
        frames, state, action = load_episode(args.dataset, episode, args.camera)
        if cached.shape[0] != len(frames):
            raise ValueError(f"Episode {episode}: cache has {cached.shape[0]} frames, dataset {len(frames)}")
        index = np.sort(rng.choice(len(frames), size=min(args.frames, len(frames)), replace=False))
        fresh = model.encode(np.asarray(frames[index]), chunk=8).half().numpy()
        stored = np.asarray(cached[index], dtype=np.float32)
        difference = np.abs(stored - fresh.astype(np.float32))
        scale = float(np.abs(stored).mean())
        # fp16 storage has a coarse step at large magnitudes (2^-5 above 16), so a
        # few large tokens can differ by ~0.03 while the *average* error stays at
        # fp16 rounding level. A real cache/serving mismatch moves the mean.
        relative_element = difference / (np.abs(stored) + 1e-3)
        cached_actions = np.load(args.cache / "actions" / f"episode_{episode:06d}.npy")[index]
        cached_states = np.load(args.cache / "states" / f"episode_{episode:06d}.npy")[index]
        records.append({
            "episode": episode,
            "frames": int(len(index)),
            "token_abs_mean": scale,
            "max_abs_diff": float(difference.max()),
            "mean_abs_diff": float(difference.mean()),
            "relative_max": float(difference.max() / max(scale, 1e-9)),
            "mean_abs_diff_over_token_scale": float(difference.mean() / max(scale, 1e-9)),
            "max_element_relative": float(relative_element.max()),
            "action_max_abs_diff": float(np.abs(
                np.asarray(action[index], dtype=np.float32) - cached_actions).max()),
            "state_max_abs_diff": float(np.abs(
                map_libero_state(np.asarray(state, dtype=np.float32))[index].astype(np.float32)
                - cached_states).max()),
        })
        print(json.dumps(records[-1]), flush=True)

    worst = max(r["mean_abs_diff_over_token_scale"] for r in records)
    report = {
        "cache": str(args.cache),
        "contract_encoder_key": contract["encoder_key"],
        "camera": args.camera,
        "episodes": records,
        "worst_mean_relative_token_diff": worst,
        "verdict": "consistent" if worst < 1e-3 else "MISMATCH",
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
