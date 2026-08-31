#!/usr/bin/env python3
"""Precompute compact V-JEPA pair targets for ACTR.

The teacher remains the frozen V-JEPA 2.1 encoder.  Its 24x24x1408 output is
unit-normalized, average-pooled to the released JEPA-WAM query grid (8x8), and
mapped to a fixed 256-D Johnson--Lindenstrauss space.  This keeps the temporal
supervision while avoiding hundreds of GiB of redundant upsampled targets.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import time
from typing import Any

# Projection generation must not initialize a CUDA JAX backend in this
# PyTorch teacher process.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from precompute_vjepa_pair_targets import IMAGENET_MEAN, IMAGENET_STD, preprocess_image


TEACHER_GRID = 24
TEACHER_DIM = 1408
COMPACT_GRID = 8
TARGET_DTYPE = np.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--hf-port", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--future-offset", type=int, default=31)
    parser.add_argument("--compact-dim", type=int, default=256)
    parser.add_argument("--projection-seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def read_episodes(root: Path) -> list[dict[str, Any]]:
    jsonl = root / "meta" / "episodes.jsonl"
    if jsonl.is_file():
        episodes = []
        global_start = 0
        with jsonl.open() as handle:
            for line in handle:
                row = json.loads(line)
                row["global_start"] = global_start
                global_start += int(row["length"])
                episodes.append(row)
        return episodes

    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError("Dataset has neither meta/episodes.jsonl nor meta/episodes/*.parquet")
    table = pa.concat_tables([pq.read_table(path) for path in paths])
    episodes = sorted(table.to_pylist(), key=lambda row: int(row["episode_index"]))
    for row in episodes:
        row["global_start"] = int(row.get("dataset_from_index", 0))
    return episodes


def target_path(root: Path, episode_index: int, chunks_size: int) -> Path:
    return root / "targets" / f"chunk-{episode_index // chunks_size:03d}" / f"episode_{episode_index:06d}.npy"


def decode_image(value: dict[str, Any], dataset_root: Path) -> Image.Image:
    if value.get("bytes") is not None:
        with Image.open(io.BytesIO(value["bytes"])) as image:
            return image.convert("RGB").copy()
    path = Path(value["path"])
    if not path.is_absolute():
        path = dataset_root / path
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def episode_data_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    return root / template.format(
        episode_chunk=episode_index // int(info.get("chunks_size", 1000)),
        episode_index=episode_index,
    )


def fixed_projection(dim: int, seed: int) -> np.ndarray:
    projection = jax.random.rademacher(
        jax.random.key(seed), (TEACHER_DIM, dim), dtype=jax.numpy.float32
    ) * (dim**-0.5)
    return np.asarray(projection)


def compact_target(output, projection: np.ndarray) -> np.ndarray:
    import torch

    value = output.to(device="cpu", dtype=torch.float32).numpy()
    value = value.reshape(value.shape[0], TEACHER_GRID, TEACHER_GRID, TEACHER_DIM)
    value /= np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-6)
    value = value.reshape(value.shape[0], COMPACT_GRID, 3, COMPACT_GRID, 3, TEACHER_DIM).mean(axis=(2, 4))
    value /= np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-6)
    value = np.einsum("bhwd,dc->bhwc", value, projection, optimize=True)
    value /= np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-6)
    return value.reshape(value.shape[0], COMPACT_GRID**2, projection.shape[1]).astype(TARGET_DTYPE)


def manifest(args: argparse.Namespace, info: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "format_version": 2,
        "teacher_kind": "vjepa2.1_hf_bit_exact_port",
        "hf_port": str(args.hf_port.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_total_episodes": len(episodes),
        "dataset_total_frames": sum(int(row["length"]) for row in episodes),
        "chunks_size": int(info.get("chunks_size", 1000)),
        "image_key": args.image_key,
        "future_offset": args.future_offset,
        "tail_policy": "clamp_future_to_last_frame",
        "teacher_shape": [TEACHER_GRID**2, TEACHER_DIM],
        "target_shape": [COMPACT_GRID**2, args.compact_dim],
        "target_dtype": "float16",
        "spatial_reduction": "unit_normalize_then_average_pool_3x3_then_unit_normalize",
        "feature_reduction": "fixed_rademacher_jl_then_unit_normalize",
        "projection_seed": args.projection_seed,
        "normalization_mean": IMAGENET_MEAN.tolist(),
        "normalization_std": IMAGENET_STD.tolist(),
    }


def ensure_manifest(root: Path, contract: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        with path.open() as handle:
            existing = json.load(handle)
        if existing != contract:
            raise RuntimeError(f"Existing compact-target manifest differs: {path}") from None
        return
    with os.fdopen(fd, "w") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
        handle.write("\n")


def valid_target(path: Path, length: int, dim: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        return value.shape == (length, COMPACT_GRID**2, dim) and value.dtype == TARGET_DTYPE
    except (OSError, ValueError):
        return False


def load_teacher(args: argparse.Namespace):
    import torch
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V-JEPA compact-target generation")
    device = torch.device(args.device)
    model = AutoModel.from_pretrained(
        args.hf_port,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.requires_grad_(False)
    model.eval().to(device)
    return model, device


def process_episode(args, info, episode, model, device, projection) -> None:
    import torch

    index = int(episode["episode_index"])
    length = int(episode["length"])
    chunks_size = int(info.get("chunks_size", 1000))
    output_path = target_path(args.output_root, index, chunks_size)
    if valid_target(output_path, length, args.compact_dim):
        print(f"skip episode={index:06d}", flush=True)
        return
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite invalid target: {output_path}")

    table = pq.read_table(
        episode_data_path(args.dataset_root, info, index),
        columns=[args.image_key, "frame_index", "episode_index", "index"],
    )
    if table.num_rows != length:
        raise ValueError(f"Episode {index} length mismatch: {table.num_rows} != {length}")
    if not np.array_equal(table["frame_index"].to_numpy(), np.arange(length)):
        raise ValueError(f"Episode {index} frame indices are not contiguous")
    images = [decode_image(value, args.dataset_root) for value in table[args.image_key].to_pylist()]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(f".{output_path.name}.tmp-r{args.worker_rank}-p{os.getpid()}")
    targets = np.lib.format.open_memmap(
        temp, mode="w+", dtype=TARGET_DTYPE, shape=(length, COMPACT_GRID**2, args.compact_dim)
    )
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, length, args.batch_size):
                end = min(start + args.batch_size, length)
                current = np.stack([preprocess_image(images[i]) for i in range(start, end)])
                futures = [min(i + args.future_offset, length - 1) for i in range(start, end)]
                future = np.stack([preprocess_image(images[i]) for i in futures])
                pair = torch.from_numpy(np.stack((current, future), axis=2)).to(
                    device=device, dtype=torch.bfloat16
                )
                output = model(pixel_values_videos=pair, skip_predictor=True).last_hidden_state
                expected = (end - start, TEACHER_GRID**2, TEACHER_DIM)
                if tuple(output.shape) != expected:
                    raise ValueError(f"Unexpected V-JEPA output {tuple(output.shape)}, expected {expected}")
                targets[start:end] = compact_target(output, projection)
                del pair, output
        targets.flush()
        del targets
        os.replace(temp, output_path)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
    print(f"complete episode={index:06d} rows={length} elapsed={time.monotonic()-started:.1f}s", flush=True)


def status(args, info, episodes) -> dict[str, Any]:
    complete = 0
    frames = 0
    size = 0
    for row in episodes:
        path = target_path(args.output_root, int(row["episode_index"]), int(info.get("chunks_size", 1000)))
        if valid_target(path, int(row["length"]), args.compact_dim):
            complete += 1
            frames += int(row["length"])
            size += path.stat().st_size
    total_frames = sum(int(row["length"]) for row in episodes)
    return {
        "complete_episodes": complete,
        "total_episodes": len(episodes),
        "complete_frames": frames,
        "total_frames": total_frames,
        "complete_gib": size / 1024**3,
        "expected_gib": total_frames * COMPACT_GRID**2 * args.compact_dim * 2 / 1024**3,
    }


def main() -> None:
    args = parse_args()
    if args.future_offset < 0 or args.compact_dim < 1 or args.batch_size < 1:
        raise ValueError("future-offset, compact-dim and batch-size must be valid positive values")
    if args.world_size < 1 or not 0 <= args.worker_rank < args.world_size:
        raise ValueError("worker-rank must be in [0, world-size)")
    with (args.dataset_root / "meta" / "info.json").open() as handle:
        info = json.load(handle)
    episodes = read_episodes(args.dataset_root)
    contract = manifest(args, info, episodes)
    if args.plan_only:
        print(json.dumps({**contract, **status(args, info, episodes)}, indent=2))
        return
    if args.status_only:
        print(json.dumps(status(args, info, episodes), indent=2))
        return
    ensure_manifest(args.output_root, contract)
    assigned = [row for row in episodes if int(row["episode_index"]) % args.world_size == args.worker_rank]
    if args.max_episodes is not None:
        assigned = assigned[: args.max_episodes]
    projection = fixed_projection(args.compact_dim, args.projection_seed)
    model, device = load_teacher(args)
    for episode in assigned:
        process_episode(args, info, episode, model, device, projection)
    print(f"worker complete rank={args.worker_rank}/{args.world_size}", flush=True)


if __name__ == "__main__":
    main()
