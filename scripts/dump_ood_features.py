#!/usr/bin/env python3
"""Dump clean and perturbed features for the OOD-robustness comparison.

Two representations, same frames, same conditions:
  * ``policy`` - the frozen PI0.5 VLM prefix tokens (the features the policy's
    action head actually consumes),
  * ``ac``     - the V-JEPA 2-AC encoder's frame tokens (mean-pooled).

The action chunk at each frame is the label. Because the action is invariant to
an image-space perturbation, the held-out R^2 of a ridge fitted on *clean* data
and evaluated on perturbed frames measures how OOD-robust each representation is
for predicting the action.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ood_perturbations import CONDITIONS, perturb  # noqa: E402
from probe_vjepa_ac_libero import load_episode  # noqa: E402


def collect_frames(args):
    """Return a list of dicts with the images, state, prompt and action chunk."""
    rng = np.random.default_rng(args.seed)
    samples = []
    for episode in args.episodes:
        try:
            frames, state, action = load_episode(args.dataset, episode, "agentview")
            wrist = load_episode(args.dataset, episode, "wrist")[0]
        except FileNotFoundError:
            continue
        length = min(len(frames), len(action))
        if length < args.horizon + 3:
            continue
        index = np.sort(rng.choice(np.arange(1, length - args.horizon - 1),
                                   size=min(args.frames_per_episode, length - args.horizon - 2), replace=False))
        for frame in index:
            frame = int(frame)
            samples.append({"episode": episode, "frame": frame,
                            "image": frames[frame], "wrist": wrist[frame],
                            "state": state[frame].astype(np.float32),
                            "chunk": action[frame:frame + args.horizon].reshape(-1).astype(np.float32)})
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["policy", "ac"], required=True)
    parser.add_argument("--config", default="pi05_libero_vjepa_aux")
    parser.add_argument("--checkpoint", default=(
        "/workspace/artifacts/models/jepa_wam_pi05_robot_sweep/checkpoints/openpi/pi05_libero_vjepa_aux/"
        "pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/40000"))
    parser.add_argument("--vjepa-checkpoint", type=Path, default=Path("/workspace/vjepa2/vjepa2-ac-vitg.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--frames-per-episode", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    samples = collect_frames(args)
    print(json.dumps({"samples": len(samples)}), flush=True)
    rng = np.random.default_rng(args.seed)

    features = {}
    labels = {"chunk": [], "episode": [], "frame": [], "condition": []}
    if args.mode == "ac":
        import torch
        sys.path.insert(0, "/workspace/vjepa2")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from openpi.con2.ac_world_model import ACWorldModel, build_models, frame_transform
        encoder, _ = build_models(args.vjepa_checkpoint, root="/workspace/vjepa2", device=args.device)
        model = ACWorldModel(predictor=None, encoder=encoder,
                             transform=frame_transform("/workspace/vjepa2"), device=args.device)
        for condition in args.conditions:
            images = [perturb(sample["image"], condition, rng) for sample in samples]
            with torch.no_grad():
                tokens = model.encode(np.stack(images))
            features[condition] = tokens.mean(1).float().numpy()
            for sample in samples:
                labels["chunk"].append(sample["chunk"])
                labels["episode"].append(sample["episode"])
                labels["frame"].append(sample["frame"])
                labels["condition"].append(condition)
            print(json.dumps({"mode": "ac", "condition": condition, "frames": len(images)}), flush=True)
    else:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as model_lib
        from openpi.policies import policy_config
        from openpi.shared import nnx_utils
        from openpi.training import config as _configs
        config = _configs.get_config(args.config)
        policy = policy_config.create_trained_policy(config, args.checkpoint)
        extract = nnx_utils.module_jit(policy._model.extract_predictive_tokens)
        for condition in args.conditions:
            pooled = []
            batch = 8
            for start in range(0, len(samples), batch):
                chunk = samples[start:start + batch]
                transformed = [policy._input_transform({
                    "observation/image": perturb(sample["image"], condition, rng),
                    "observation/wrist_image": perturb(sample["wrist"], condition, rng),
                    "observation/state": sample["state"], "prompt": "do the task",
                }) for sample in chunk]
                values = jax.tree.map(lambda *xs: jnp.asarray(np.stack(xs)), *transformed)
                observation = model_lib.Observation.from_dict(values)
                observation = dataclasses.replace(observation, vjepa_target=None)
                tokens = np.asarray(extract(observation))  # [B, 64, 2048]
                pooled.append(tokens.mean(1))
            features[condition] = np.concatenate(pooled, axis=0)
            for sample in samples:
                labels["chunk"].append(sample["chunk"])
                labels["episode"].append(sample["episode"])
                labels["frame"].append(sample["frame"])
                labels["condition"].append(condition)
            print(json.dumps({"mode": "policy", "condition": condition, "frames": len(samples)}), flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, conditions=np.asarray(args.conditions),
             chunk=np.asarray(labels["chunk"]), episode=np.asarray(labels["episode"]),
             frame=np.asarray(labels["frame"]), condition=np.asarray(labels["condition"]),
             **{f"features_{key}": value for key, value in features.items()})
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
