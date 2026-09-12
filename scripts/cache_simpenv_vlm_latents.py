#!/usr/bin/env python3
"""Cache pi0.5's own VLM pooled latents for the SimplerEnv (Bridge) Con1 branch.

The SimpENV branch does not use the JEPA predictive tokens (pi0.5 was never
trained with R_t on this data), so the Con1 latent target is the mask-weighted
mean of the whole VLM prefix of a frozen pi0.5. The cached ``z`` is what the
training pipeline reads as ``con1_current_latent`` / ``con1_future_latents``;
the ``r`` file is a shape placeholder because the delta head now receives the
live prefix tokens instead of a cached R_t.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import imageio.v3 as iio
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq

from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_bridge")
    parser.add_argument("--checkpoint", type=Path, default=Path("/workspace/models/pi05_base"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/workspace/data/bridge_view0"))
    parser.add_argument("--output-root", type=Path, default=Path("/workspace/data/bridge_vlm_latents"))
    parser.add_argument("--image-key", default="observation.images.image_1")
    parser.add_argument("--wrist-key", default="observation.images.image_0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


def episode_paths(root: Path, image_key: str, wrist_key: str, episode: int):
    chunk = episode // 1000
    data = root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    video = root / "videos" / f"chunk-{chunk:03d}" / image_key / f"episode_{episode:06d}.mp4"
    wrist = root / "videos" / f"chunk-{chunk:03d}" / wrist_key / f"episode_{episode:06d}.mp4"
    return data, video, wrist


def main() -> None:
    args = parse_args()
    config = _config.get_config(args.config)
    policy = policy_config.create_trained_policy(config, str(args.checkpoint))
    model = policy._model  # noqa: SLF001
    pooled_fn = nnx_utils.module_jit(model.extract_pooled_prefix)
    transform = policy._input_transform  # noqa: SLF001

    episodes = sorted(int(p.stem.split("_")[1]) for p in (args.dataset_root / "data").rglob("episode_*.parquet"))
    episodes = [e for i, e in enumerate(episodes) if i % args.shards == args.shard]
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]
    if not episodes:
        raise SystemExit("no episodes in this shard")

    out = args.output_root
    (out / "episodes").mkdir(parents=True, exist_ok=True)
    entries = []
    for episode in episodes:
        data_path, video_path, wrist_path = episode_paths(
            args.dataset_root, args.image_key, args.wrist_key, episode)
        if not data_path.exists() or not video_path.exists() or not wrist_path.exists():
            continue
        z_path = out / "episodes" / f"{episode:06d}_z.npy"
        # The task id lives in the parquet; read it in both branches so the
        # train/validation split can group by task.
        table = pq.read_table(data_path, columns=["task_index"])
        task_id = int(table["task_index"][0].as_py())
        if z_path.exists():
            # Already cached (the view grows as more Bridge chunks land).
            z = np.load(z_path, mmap_mode="r")
            entries.append({
                "id": episode,
                "task_id": task_id,
                "length": int(len(z)),
                "z": f"episodes/{episode:06d}_z.npy",
                "r": f"episodes/{episode:06d}_r.npy",
            })
            continue
        table = pq.read_table(data_path, columns=["observation.state"])
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        frames = iio.imread(video_path, plugin="pyav")
        wrists = iio.imread(wrist_path, plugin="pyav")
        length = min(len(states), len(frames))
        features = []
        for start in range(0, length, args.batch_size):
            batch_frames = frames[start:start + args.batch_size]
            batch_wrists = wrists[start:start + args.batch_size]
            batch_states = states[start:start + args.batch_size]
            observations = []
            for image, wrist, state in zip(batch_frames, batch_wrists, batch_states, strict=True):
                observations.append(transform({
                    "observation/image": np.asarray(image[..., :3], dtype=np.uint8),
                    "observation/wrist_image": np.asarray(wrist[..., :3], dtype=np.uint8),
                    "observation/state": state,
                    "prompt": "do the task",
                }))
            values = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *observations)
            observation = _model.Observation.from_dict(values)
            observation = dataclasses.replace(observation, vjepa_target=None)
            features.append(np.asarray(pooled_fn(observation), dtype=np.float16))
        z = np.concatenate(features, axis=0)
        np.save(out / "episodes" / f"{episode:06d}_z.npy", z)
        # Placeholder tokens: the head reads the live prefix, not this file.
        np.save(out / "episodes" / f"{episode:06d}_r.npy", np.zeros((len(z), 1, 1), dtype=np.float16))
        entries.append({
            "id": episode,
            "task_id": task_id,
            "length": int(len(z)),
            "z": f"episodes/{episode:06d}_z.npy",
            "r": f"episodes/{episode:06d}_r.npy",
        })
        if len(entries) % 50 == 0:
            print(json.dumps({"episodes": len(entries), "last": episode}), flush=True)

    manifest = {
        "schema": "con1-anchored-features-v1",
        "anchor_source": "current_only_frozen_teacher",
        "latent_space": "pi05_vlm_pooled_prefix",
        "latent_dim": int(z.shape[-1]),
        "r_shape": [1, 1],
        "chunks_size": 1000,
        "complete": True,
        "episodes": entries,
    }
    (out / f"manifest.shard{args.shard}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("SHARD_DONE", args.shard, len(entries))


if __name__ == "__main__":
    main()
