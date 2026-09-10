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

import src.hub.backbones as hub  # noqa: E402
from app.vjepa_droid.transforms import make_transforms  # noqa: E402
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
    encoder, predictor = hub._make_vjepa2_ac_model(pretrained=False)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    encoder.load_state_dict(hub._clean_backbone_key(dict(state["encoder"])), strict=True)
    predictor.load_state_dict(hub._clean_backbone_key(dict(state["predictor"])), strict=True)
    del state
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    transform = make_transforms(random_horizontal_flip=False, random_resize_aspect_ratio=(1.0, 1.0),
                                random_resize_scale=(1.0, 1.0), reprob=0.0, auto_augment=False,
                                motion_shift=False, crop_size=256)

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

    tokens = []
    for start in range(0, len(frames), 8):
        batch = transform(np.ascontiguousarray(frames[start:start + 8])).unsqueeze(0)
        b, c, t, h, w = batch.shape
        batch = batch.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        with torch.no_grad():
            out = encoder(batch.to(device))
        tokens.append(out.reshape(t, -1, out.shape[-1]).float().cpu())
    tokens = torch.cat(tokens)
    tokens = F.layer_norm(tokens, (tokens.shape[-1],))
    n_tokens = tokens.shape[1]

    records = {}
    for step in range(1, args.rollout + 1):
        errors, copies = [], []
        for k in range(len(index) - step):
            z = tokens[k:k + 1].reshape(1, n_tokens, -1).to(device)
            a = actions[k:k + 1].unsqueeze(0).to(device)
            s = pose[k:k + 1].unsqueeze(0).to(device)
            for _ in range(step):
                with torch.no_grad():
                    prediction = predictor(z, a, s)[:, -n_tokens:]
                    prediction = F.layer_norm(prediction, (prediction.shape[-1],))
                z = torch.cat([z, prediction], dim=1)
                a = torch.cat([a, a[:, -1:]], dim=1)
                s = torch.cat([s, s[:, -1:]], dim=1)
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
