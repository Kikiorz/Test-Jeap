#!/usr/bin/env python3
"""Cache independent frozen V-JEPA states for paper Con1 orthogonal targets.

Phi(o) = concat_views(mean_spatial(VJEPA([o, o]))).  Neither a future frame
nor another episode enters Phi(o).  The loader normalizes the cached vector
and constructs all H displacements relative to the same current anchor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow.parquet as pq

from precompute_vjepa_displacement_targets import ensure_manifest
from precompute_vjepa_pair_targets import (
    decode_image,
    input_path,
    load_target_encoder,
    preprocess_image,
    read_episodes,
    read_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vjepa-source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-keys", nargs="+", default=["image", "wrist_image"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-free-gib", type=float, default=16)
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def state_path(root, episode, chunks_size):
    return root / "states" / f"chunk-{episode // chunks_size:03d}" / f"episode_{episode:06d}.npy"


def valid_state(path, length, dim):
    if not path.is_file() or not path.with_suffix(".json").is_file():
        return False
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        return values.shape == (length, dim) and values.dtype == np.float16
    except (ValueError, OSError):
        return False


def make_contract(args, info, episodes):
    digest = hashlib.sha256()
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl"):
        digest.update(name.encode())
        digest.update((args.dataset_root / "meta" / name).read_bytes())
    checkpoint = args.checkpoint.stat()
    source_revision = subprocess.check_output(
        ["git", "-C", str(args.vjepa_source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "format_version": 1,
        "kind": "con1_independent_vjepa_frame_states",
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_metadata_sha256": digest.hexdigest(),
        "dataset_total_frames": sum(int(row["length"]) for row in episodes),
        "dataset_total_episodes": len(episodes),
        "dataset_fps": info["fps"],
        "chunks_size": int(info.get("chunks_size", 1000)),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_size": checkpoint.st_size,
        "checkpoint_mtime_ns": checkpoint.st_mtime_ns,
        "source_revision": source_revision,
        "teacher": "V-JEPA2.1 ViT-g/384 target_encoder, frozen, eval, bfloat16",
        "state_operator": "concatenate_spatial_mean_per_view_of_same_frame_pair",
        "temporal_input": "[o_k,o_k]; never [o_t,o_future]",
        "image_keys": list(args.image_keys),
        "teacher_shape_per_view": [576, 1408],
        "state_dim": 1408 * len(args.image_keys),
        "state_dtype": "float16",
        "channel_projection": "none",
        "normalization": "loader_l2_normalizes_concatenated_vector_in_float32",
        "delta": "u[t+j] - dot(u[t],u[t+j])*u[t], j=1..H",
        "tail_policy": "mask_out_of_episode_future_positions; no_endpoint_repetition",
    }


def write_json(path, value):
    temporary = path.with_name(f".{path.name}.pid-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def state_diagnostics(states):
    values = np.asarray(states, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Teacher produced non-finite frame features")
    norm = np.linalg.norm(values, axis=-1)
    if np.any(norm < 1e-6):
        raise ValueError("Teacher produced zero frame features")
    unit = values / norm[:, None]
    result = {"state_norm_mean": float(norm.mean()), "state_norm_min": float(norm.min())}
    for horizon in (1, 5, 10):
        if len(unit) <= horizon:
            continue
        anchor, future = unit[:-horizon], unit[horizon:]
        cosine = np.sum(anchor * future, axis=-1)
        delta = future - cosine[:, None] * anchor
        result[f"h{horizon}"] = {
            "count": len(delta),
            "state_cosine_mean": float(cosine.mean()),
            "delta_norm_mean": float(np.linalg.norm(delta, axis=-1).mean()),
            "delta_zero_predictor_mse": float(np.square(delta).mean()),
            "orthogonality_max_abs": float(np.abs(np.sum(anchor * delta, axis=-1)).max()),
        }
    return result


def process_episode(args, info, episode, model, device):
    import torch

    index, length = int(episode["episode_index"]), int(episode["length"])
    dim = 1408 * len(args.image_keys)
    path = state_path(args.output_root, index, int(info.get("chunks_size", 1000)))
    if valid_state(path, length, dim):
        print(json.dumps({"event": "skip", "episode": index}), flush=True)
        return
    # Complete data with missing sidecar can follow a crash between two atomic commits.
    if path.is_file():
        existing = np.load(path, allow_pickle=False)
        if existing.shape != (length, dim) or existing.dtype != np.float16:
            raise ValueError(f"Refusing to overwrite malformed cache {path}")
        write_json(path.with_suffix(".json"), {"episode": index, "frames": length,
                                             "diagnostics": state_diagnostics(existing)})
        return
    table = pq.read_table(input_path(args.dataset_root, index),
                          columns=[*args.image_keys, "frame_index", "episode_index", "index", "task_index"])
    if table.num_rows != length or not np.array_equal(table["frame_index"].to_numpy(), np.arange(length)):
        raise ValueError(f"Episode {index}: inconsistent frame indices")
    if not np.all(table["episode_index"].to_numpy() == index):
        raise ValueError(f"Episode {index}: cross-episode rows")
    if not np.array_equal(table["index"].to_numpy(), np.arange(episode["global_start"], episode["global_start"] + length)):
        raise ValueError(f"Episode {index}: inconsistent global indices")
    task_ids = np.unique(table["task_index"].to_numpy())
    if len(task_ids) != 1:
        raise ValueError(f"Episode {index}: multiple task ids")
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    states = np.empty((length, dim), dtype=np.float16)
    with torch.inference_mode():
        for view, key in enumerate(args.image_keys):
            rows = table[key].to_pylist()
            for start in range(0, length, args.batch_size):
                end = min(start + args.batch_size, length)
                frames = np.stack([preprocess_image(decode_image(row, args.dataset_root)) for row in rows[start:end]])
                video = torch.from_numpy(np.stack((frames, frames), axis=2)).to(device, dtype=torch.bfloat16)
                output = model(video)
                if isinstance(output, list):
                    output = output[-1]
                if tuple(output.shape) != (end - start, 576, 1408):
                    raise ValueError(f"Unexpected teacher shape {output.shape}")
                feature = output.float().mean(dim=1)
                if not torch.isfinite(feature).all():
                    raise ValueError(f"Non-finite features in episode {index}")
                states[start:end, view * 1408:(view + 1) * 1408] = feature.cpu().to(torch.float16).numpy()
    diagnostics = state_diagnostics(states)
    temporary = path.with_name(f".{path.name}.pid-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, states, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    elapsed = time.monotonic() - started
    report = {"event": "complete", "episode": index, "task_index": int(task_ids[0]),
              "frames": length, "seconds": elapsed, "frames_per_second": length / elapsed,
              "peak_gpu_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
              "diagnostics": diagnostics}
    write_json(path.with_suffix(".json"), report)
    print(json.dumps(report), flush=True)


def status(args, info, episodes):
    complete = [row for row in episodes if valid_state(
        state_path(args.output_root, int(row["episode_index"]), int(info.get("chunks_size", 1000))),
        int(row["length"]), 1408 * len(args.image_keys))]
    return {"complete_episodes": len(complete), "total_episodes": len(episodes),
            "complete_frames": sum(int(row["length"]) for row in complete),
            "total_frames": sum(int(row["length"]) for row in episodes)}


def main():
    args = parse_args()
    if args.batch_size < 1 or args.world_size < 1 or not 0 <= args.worker_rank < args.world_size:
        raise ValueError("Invalid batch size or worker assignment")
    info = read_json(args.dataset_root / "meta/info.json")
    episodes = read_episodes(args.dataset_root / "meta/episodes.jsonl")
    contract = make_contract(args, info, episodes)
    if args.plan_only or args.status_only:
        print(json.dumps({**(contract if args.plan_only else {}), **status(args, info, episodes)}, indent=2))
        return
    ensure_manifest(args.output_root, contract)
    assigned = [row for row in episodes if int(row["episode_index"]) % args.world_size == args.worker_rank]
    if args.max_episodes is not None:
        assigned = assigned[:args.max_episodes]
    missing = [row for row in assigned if not valid_state(
        state_path(args.output_root, int(row["episode_index"]), contract["chunks_size"]),
        int(row["length"]), contract["state_dim"])]
    if missing:
        model, device = load_target_encoder(args)
        for episode in missing:
            process_episode(args, info, episode, model, device)
    print(json.dumps({"event": "worker_done", "rank": args.worker_rank, "assigned": len(assigned)}), flush=True)


if __name__ == "__main__":
    main()
