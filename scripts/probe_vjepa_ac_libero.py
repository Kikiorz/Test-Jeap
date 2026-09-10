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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openpi.con2.ac_world_model import (  # noqa: E402
    ACTION_DIM,
    FEATURE_DIM,
    LIBERO_STATE_DIM,
    STATE_DIM,
    TOKENS_PER_FRAME,
    ACWorldModel,
    build_models,
    frame_transform,
    map_libero_action,
    map_libero_state,
)

VJEPA_ROOT = Path("/workspace/vjepa2")
sys.path.insert(0, str(VJEPA_ROOT))

map_action = map_libero_action
map_state = map_libero_state


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
    if state.shape[1] != LIBERO_STATE_DIM or action.shape[1] != ACTION_DIM:
        raise ValueError(f"Unexpected LIBERO shapes: state {state.shape}, action {action.shape}")
    return frames, state, action


def build(device, encoder_key, checkpoint):
    """Model pair plus the audit record the report carries."""
    encoder, predictor = build_models(checkpoint, encoder_key=encoder_key, root=VJEPA_ROOT, device=device)
    audit = {
        "encoder_key": encoder_key,
        "encoder_parameters": sum(p.numel() for p in encoder.parameters()),
        "predictor_parameters": sum(p.numel() for p in predictor.parameters()),
    }
    return encoder, predictor, audit


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
    model = ACWorldModel(predictor=predictor, encoder=encoder, transform=transform, device=device)
    records = {name: Recorder() for name in ("model", "shuffled_action", "zero_action")}
    windows = []
    for episode in episodes:
        frames, state, action = load_episode(args.dataset, episode, args.camera)
        tokens = model.encode(frames[::args.stride], chunk=args.encode_chunk)
        pose = map_state(state[::args.stride])
        act = map_action(action[::args.stride], args.action_variant)
        n_tokens = tokens.shape[1]
        length = tokens.shape[0]
        if length < args.context + 2:
            continue
        for k in range(args.context - 1, length - 1):
            lo = k - args.context + 1
            context = tokens[lo:k + 1]
            state_window = pose[lo:k + 1]
            action_window = act[lo:k + 1]
            predicted = model.predict(context, action_window, state_window)
            predicted_shuffled = model.predict(context, action_window[::-1], state_window)
            predicted_zero = model.predict(context, np.zeros_like(action_window), state_window)
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
    encoder, predictor, audit = build(device, args.encoder_key, args.checkpoint)
    transform = frame_transform(VJEPA_ROOT)
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
