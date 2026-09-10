#!/usr/bin/env python3
"""Zero-shot probe: pretrained V-JEPA 2-AC as an action-conditioned future predictor.

No training, no fitting. Answers one question on LIBERO demonstrations:

    does  z_hat_{k+1} = LN(P(LN(h_k), a_k, s_k))  beat "copy the current latent"?

Protocol follows the official code so the probe cannot be accused of using the
released weights in an off-distribution way:

  * ``app/vjepa_droid/train.py`` - clips ``[B,C,T,H,W]``, actions ``[B,T-1,7]``,
    states ``[B,T,7]``; the predictor output block ``k`` predicts frame ``k+1``;
    both sides are layer-normed over the feature dim when ``normalize_reps``.
  * ``notebooks/energy_landscape_example.ipynb`` - how ``vjepa2_ac_vit_giant`` is
    driven: single-frame context repeated over ``tubelet_size``, representation
    layer-normed, predictor read from its last frame block.
  * ``configs/train/vitg16/droid-256px-8f.yaml`` - 256 px, 8-frame context, 4 fps.

LIBERO here is the LeRobot conversion at 10 fps, so ``--stride 2`` matches the
4-5 fps the AC model was trained at, and ``--stride 1`` is the 10 fps ceiling.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image

VJEPA_ROOT = Path("/workspace/vjepa2")
sys.path.insert(0, str(VJEPA_ROOT))

import src.hub.backbones as hub  # noqa: E402
from app.vjepa_droid.transforms import make_transforms  # noqa: E402

LIBERO_STATE_DIM = 8
AC_STATE_DIM = 7
AC_ACTION_DIM = 7


def decode_image(value, dataset_root: Path) -> Image.Image:
    raw = value.get("bytes")
    if raw is not None:
        with Image.open(io.BytesIO(raw)) as image:
            return image.convert("RGB").copy()
    path = Path(value["path"])
    if not path.is_absolute():
        path = dataset_root / path
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def load_episode(dataset_root: Path, episode: int, camera: str):
    chunk = episode // 1000
    source = dataset_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
    table = pq.read_table(source, columns=["image", "wrist_image", "state", "actions", "frame_index"])
    rows = table.to_pylist()
    key = "image" if camera == "agentview" else "wrist_image"
    frames = np.stack([np.asarray(decode_image(row[key], dataset_root), dtype=np.uint8) for row in rows])
    state = np.asarray([row["state"] for row in rows], dtype=np.float32)
    action = np.asarray([row["actions"] for row in rows], dtype=np.float32)
    if state.shape[1] != LIBERO_STATE_DIM or action.shape[1] != AC_ACTION_DIM:
        raise ValueError(f"Unexpected LIBERO shapes: state {state.shape}, action {action.shape}")
    return frames, state, action


def map_state(state: np.ndarray) -> np.ndarray:
    """LIBERO 8-d state -> the 7-d ``[pos, euler-xyz, gripper]`` pose the AC model expects.

    The released checkpoint encodes state with ``Linear(7, D)`` and the notebook
    feeds ``[xyz(3), euler_xyz(3), gripper(1)]``. LIBERO stores
    ``[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]``; axis-angle is treated as
    euler-xyz (identical for small rotations) and the two finger positions are
    collapsed to a closedness in ``[0, 1]`` against the robosuite travel limit.

    The two finger joints are mirrored (``qpos[1] ~ -qpos[0]``), so the opening
    fraction is the mean of their absolute values. Averaging them with sign
    cancels to ~0 and would pin the gripper channel at 1.0 for every frame.
    """
    pose = np.empty((len(state), AC_STATE_DIM), dtype=np.float32)
    pose[:, :6] = state[:, :6]
    opening = np.abs(state[:, 6:8]).mean(axis=1) / 0.04
    pose[:, 6] = np.clip(1.0 - opening, 0.0, 1.0)
    return pose


def map_action(action: np.ndarray, variant: str) -> np.ndarray:
    """LIBERO OSC_POSE action -> the metric deltas DROID was trained on.

    robosuite's OSC_POSE controller scales its normalised input by
    ``output_max`` before applying it, so ``raw`` and ``robosuite`` differ by
    that constant. Both are reported because the gripper convention is the part
    that cannot be read off from the checkpoint.
    """
    mapped = action.copy()
    if variant == "robosuite":
        mapped[:, :3] *= 0.05
        mapped[:, 3:6] *= 0.5
    return mapped


def build_models(checkpoint: Path, device: torch.device, encoder_key: str):
    encoder, predictor = hub._make_vjepa2_ac_model(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # Training consumption used the EMA target encoder's features
    # (``target_encoder`` in app/vjepa_droid/train.py) while the released
    # notebook drives the predictor with the online ``encoder``; both are
    # self-consistent, so the probe reports either.
    encoder_state = hub._clean_backbone_key(dict(state[encoder_key]))
    predictor_state = hub._clean_backbone_key(dict(state["predictor"]))
    encoder_missing, encoder_unexpected = encoder.load_state_dict(encoder_state, strict=True)
    predictor_missing, predictor_unexpected = predictor.load_state_dict(predictor_state, strict=True)
    del state
    encoder = encoder.to(device).eval()
    predictor = predictor.to(device).eval()
    audit = {
        "encoder_key": encoder_key,
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "predictor_parameters": sum(p.numel() for p in predictor.parameters()),
        "encoder_missing_keys": len(encoder_missing),
        "encoder_unexpected_keys": len(encoder_unexpected),
        "predictor_missing_keys": len(predictor_missing),
        "predictor_unexpected_keys": len(predictor_unexpected),
    }
    return encoder, predictor, audit


def encode_frames(encoder, transform, frames: np.ndarray, device, chunk: int = 8) -> torch.Tensor:
    """Return ``[T, N, D]`` raw (not yet layer-normed) tokens for every frame."""
    outputs = []
    for start in range(0, len(frames), chunk):
        clip = np.ascontiguousarray(frames[start:start + chunk])
        batch = transform(clip).unsqueeze(0)  # [1, C, t, H, W]
        b, c, t, h, w = batch.shape
        batch = batch.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        with torch.no_grad():
            tokens = encoder(batch.to(device))
        if isinstance(tokens, list):
            tokens = tokens[-1]
        outputs.append(tokens.reshape(t, -1, tokens.shape[-1]).float().cpu())
    return torch.cat(outputs)


class Recorder:
    def __init__(self):
        self.error_model = 0.0
        self.error_copy = 0.0
        self.error_zero = 0.0
        self.cosine = 0.0
        self.count = 0

    def add(self, predicted, current, target):
        dim = predicted.shape[-1]
        self.error_model += float((predicted - target).pow(2).sum(-1).mean())
        self.error_copy += float((current - target).pow(2).sum(-1).mean())
        self.error_zero += float(target.pow(2).sum(-1).mean())
        delta_hat = predicted - current
        delta = target - current
        cos = F.cosine_similarity(delta_hat, delta, dim=-1).mean()
        self.cosine += float(cos)
        self.count += 1

    def report(self):
        if not self.count:
            return None
        model = self.error_model / self.count
        copy = self.error_copy / self.count
        zero = self.error_zero / self.count
        return {
            "windows": self.count,
            "mse_model": model,
            "mse_copy_current": copy,
            "mse_zero_predictor": zero,
            "nmse_vs_copy_current": model / copy,
            "nmse_vs_zero_predictor": model / zero,
            "delta_cosine": self.cosine / self.count,
        }


def probe(args, encoder, predictor, transform, device):
    episodes = args.episodes.split(",") if isinstance(args.episodes, str) else args.episodes
    episodes = [int(e) for e in episodes]
    records = {name: Recorder() for name in ("model", "shuffled_action", "zero_action")}
    windows = []
    for episode in episodes:
        frames, state, action = load_episode(args.dataset, episode, args.camera)
        tokens = encode_frames(encoder, transform, frames[::args.stride], device, args.encode_chunk)
        pose = map_state(state[::args.stride])
        act = map_action(action[::args.stride], args.action_variant)
        n_tokens = tokens.shape[1]
        length = tokens.shape[0]
        if length < args.context + 2:
            continue
        for k in range(args.context - 1, length - 1):
            lo = k - args.context + 1
            context = tokens[lo:k + 1].reshape(1, args.context * n_tokens, -1).to(device)
            state_window = torch.from_numpy(pose[lo:k + 1]).unsqueeze(0).to(device)
            action_window = torch.from_numpy(act[lo:k + 1]).unsqueeze(0).to(device)
            with torch.no_grad():
                predicted = predictor(context, action_window, state_window)[:, -n_tokens:]
                predicted = F.layer_norm(predicted, (predicted.shape[-1],))
                # Shuffle the action chunk along time (dim 0 is the batch axis).
                shuffled = action_window.flip(1)
                predicted_shuffled = predictor(context, shuffled, state_window)[:, -n_tokens:]
                predicted_shuffled = F.layer_norm(predicted_shuffled, (predicted_shuffled.shape[-1],))
                zero_action = torch.zeros_like(action_window)
                predicted_zero = predictor(context, zero_action, state_window)[:, -n_tokens:]
                predicted_zero = F.layer_norm(predicted_zero, (predicted_zero.shape[-1],))
            current = F.layer_norm(tokens[k].to(device), (tokens.shape[-1],))[None]
            target = F.layer_norm(tokens[k + 1].to(device), (tokens.shape[-1],))[None]
            records["model"].add(predicted, current, target)
            records["shuffled_action"].add(predicted_shuffled, current, target)
            records["zero_action"].add(predicted_zero, current, target)
            windows.append(k)
    return {name: recorder.report() for name, recorder in records.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=VJEPA_ROOT / "vjepa2-ac-vitg.pt")
    parser.add_argument("--dataset", type=Path,
                        default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    parser.add_argument("--episodes", default="0,1,2,3")
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--context", type=int, default=8, help="Context frames; training used 8.")
    parser.add_argument("--camera", choices=["agentview", "wrist"], default="agentview")
    parser.add_argument("--action-variant", choices=["raw", "robosuite"], default="raw")
    parser.add_argument("--encoder-key", choices=["encoder", "target_encoder"], default="encoder",
                        help="Checkpoint to encode frames with; training consumed the EMA target encoder.")
    parser.add_argument("--encode-chunk", type=int, default=8)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder, predictor, audit = build_models(args.checkpoint, device, args.encoder_key)
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=256,
    )
    result = probe(args, encoder, predictor, transform, device)
    report = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "episodes": args.episodes,
        "stride": args.stride,
        "fps_effective": 10.0 / args.stride,
        "context_frames": args.context,
        "camera": args.camera,
        "action_variant": args.action_variant,
        "audit": audit,
        "results": result,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
