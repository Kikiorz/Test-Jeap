#!/usr/bin/env python3
"""Control probe: run the AC predictor on the official Franka trajectory.

Same metric as the LIBERO probe (NMSE against copying the current latent) but on
an in-distribution trajectory shipped with the V-JEPA 2 repo. This separates "my
probe code is wrong" from "LIBERO is out of distribution for this checkpoint".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

VJEPA_ROOT = Path("/workspace/vjepa2")
sys.path.insert(0, str(VJEPA_ROOT))
sys.path.insert(0, str(VJEPA_ROOT / "notebooks"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openpi.con2.ac_world_model import (  # noqa: E402
    TOKENS_PER_FRAME,
    ACWorldModel,
    build_models,
    frame_transform,
    normalize_reps,
)
from utils.mpc_utils import poses_to_diff  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=VJEPA_ROOT / "vjepa2-ac-vitg.pt")
    parser.add_argument("--trajectory", type=Path, default=VJEPA_ROOT / "notebooks/franka_example_traj.npz")
    parser.add_argument("--stride", type=int, default=1, help="Subsample the trajectory (DROID is 4 fps).")
    parser.add_argument("--rollout", type=int, default=4, help="Autoregressive steps to score.")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder, predictor = build_models(args.checkpoint, root=VJEPA_ROOT, device=device)
    model = ACWorldModel(predictor=predictor, encoder=encoder,
                         transform=frame_transform(VJEPA_ROOT), device=device)

    trajectory = np.load(args.trajectory)
    clips = trajectory["observations"]
    states = trajectory["states"]
    n_frames = clips.shape[1]
    print(json.dumps({"frames": int(n_frames), "clip_shape": list(clips.shape),
                      "state_shape": list(states.shape)}))

    index = list(range(0, n_frames, args.stride))
    pose = torch.tensor(states[0][index], dtype=torch.float32)
    actions = torch.stack([poses_to_diff(pose[i], pose[i + 1]) for i in range(len(index) - 1)]).float()
    frames = np.stack([np.asarray(clips[0][i], dtype=np.uint8) for i in index])

    tokens = normalize_reps(model.encode(frames))
    n_tokens = tokens.shape[1]

    records = {}
    for step in range(1, args.rollout + 1):
        errors, copies = [], []
        for k in range(len(index) - step):
            # The released notebook rolls the context forward one frame at a time
            # while repeating the action and the last pose; that is model.rollout.
            prediction = model.rollout(tokens[k:k + 1], actions[k:k + 1], pose[k:k + 1], steps=step)
            target = tokens[k + step].to(device)[None]
            errors.append(float((prediction[0] - target[0]).pow(2).sum(-1).mean()))
            copies.append(float((tokens[k].to(device) - target[0]).pow(2).sum(-1).mean()))
        records[f"rollout_{step}"] = {
            "windows": len(errors),
            "mse_model": float(np.mean(errors)),
            "mse_copy_current": float(np.mean(copies)),
            "nmse_vs_copy_current": float(np.mean(errors) / np.mean(copies)),
        }

    report = {"stride": args.stride, "frames_used": len(index), "results": records}
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
