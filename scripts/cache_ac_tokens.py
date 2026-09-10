#!/usr/bin/env python3
"""Cache frozen V-JEPA 2-AC frame tokens for LIBERO (Con2 training input).

The encoder is frozen, so its patch tokens are a fixed function of the frame and
can be written once instead of recomputed eight times per training window.

Layout (one directory per shard, merged by ``--merge``):

    <out>/tokens/episode_XXXXXX.npy   fp16 [T, 256, 1408]
    <out>/actions/episode_XXXXXX.npy  fp32 [T, 7]
    <out>/states/episode_XXXXXX.npy   fp32 [T, 7]   (mapped to the AC pose format)
    <out>/manifest.json               episode list + contract hash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

VJEPA_ROOT = Path("/workspace/vjepa2")
sys.path.insert(0, str(VJEPA_ROOT))
sys.path.insert(0, "/workspace/ts_JEPA_con1_clean/scripts")

import src.hub.backbones as hub  # noqa: E402
from app.vjepa_droid.transforms import make_transforms  # noqa: E402
from probe_vjepa_ac_libero import map_action, map_state  # noqa: E402


def atomic_npy(path: Path, value: np.ndarray):
    temp = path.with_suffix(".tmp.npy")
    np.save(temp, value)
    os.replace(temp, path)


def atomic_json(path: Path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 ** 2), b""):
            h.update(block)
    return h.hexdigest()


def encode(encoder, transform, episode, dataset_root, args, device) -> np.ndarray:
    from probe_vjepa_ac_libero import load_episode

    frames, _, _ = load_episode(dataset_root, episode, args.camera)
    chunks = []
    for start in range(0, len(frames), args.encode_chunk):
        clip = np.ascontiguousarray(frames[start:start + args.encode_chunk])
        batch = transform(clip).unsqueeze(0)
        b, c, t, h, w = batch.shape
        batch = batch.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        with torch.no_grad():
            tokens = encoder(batch.to(device))
        chunks.append(tokens.reshape(t, -1, tokens.shape[-1]).half().cpu().numpy())
    return np.concatenate(chunks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    parser.add_argument("--checkpoint", type=Path, default=VJEPA_ROOT / "vjepa2-ac-vitg.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--camera", choices=["agentview", "wrist"], default="agentview")
    parser.add_argument("--action-variant", choices=["raw", "robosuite"], default="raw")
    parser.add_argument("--encoder-key", choices=["encoder", "target_encoder"], default="encoder")
    parser.add_argument("--encode-chunk", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder, _ = hub._make_vjepa2_ac_model(pretrained=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(hub._clean_backbone_key(dict(state[args.encoder_key])), strict=True)
    del state
    encoder = encoder.to(device).eval()
    transform = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
                                random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False,
                                motion_shift=False, crop_size=256)

    shard_episodes = args.episodes[args.shard::args.shards]
    for name in ("tokens", "actions", "states"):
        (args.output / name).mkdir(parents=True, exist_ok=True)
    contract = {
        "schema": "con2-ac-tokens-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": digest(args.checkpoint),
        "encoder_key": args.encoder_key,
        "camera": args.camera,
        "action_variant": args.action_variant,
        "dataset": str(args.dataset),
        "token_dim": 1408,
        "tokens_per_frame": 256,
    }
    if args.shard == 0:
        atomic_json(args.output / "contract.json", contract)
    status_path = args.output / f"status-shard{args.shard}.json"
    started = 0
    for index, episode in enumerate(shard_episodes):
        from probe_vjepa_ac_libero import load_episode

        frames, state_array, action_array = load_episode(args.dataset, episode, args.camera)
        tokens_path = args.output / "tokens" / f"episode_{episode:06d}.npy"
        if not tokens_path.exists():
            tokens = encode(encoder, transform, episode, args.dataset, args, device)
            atomic_npy(tokens_path, tokens)
            atomic_npy(args.output / "actions" / f"episode_{episode:06d}.npy",
                       map_action(action_array, args.action_variant).astype(np.float32))
            atomic_npy(args.output / "states" / f"episode_{episode:06d}.npy",
                       map_state(state_array).astype(np.float32))
        started += 1
        atomic_json(status_path, {"state": "encoding", "shard": args.shard, "completed": started,
                                  "total": len(shard_episodes), "episode": episode,
                                  "frames": int(len(frames))})
    atomic_json(status_path, {"state": "complete", "shard": args.shard, "completed": started,
                              "total": len(shard_episodes)})


if __name__ == "__main__":
    main()
