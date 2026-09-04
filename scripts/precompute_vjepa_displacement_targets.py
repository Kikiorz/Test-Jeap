#!/usr/bin/env python3
"""Precompute JL-free no-change-referenced V-JEPA2 displacement for Con1.

For every valid H-step transition this script encodes [o_t, o_(t+H)] and
[o_t, o_t] with the same frozen Hugging Face V-JEPA2 port, normalizes and
average-pools both 24x24 grids to 8x8, then stores their 1408-D difference.
Only one displacement cache is written; no learned or random channel
projection is used.
"""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from precompute_vjepa_pair_targets import IMAGENET_MEAN, IMAGENET_STD, preprocess_image


TEACHER_GRID = 24
TEACHER_DIM = 1408
OUTPUT_GRID = 8
TARGET_DTYPE = np.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--hf-port", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-key", default="image")
    parser.add_argument("--future-offset", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--worker-rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--status-only", action="store_true")
    return parser.parse_args()


def read_episodes(root: Path) -> list[dict[str, Any]]:
    paths = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if paths:
        table = pa.concat_tables([pq.read_table(path) for path in paths])
        episodes = sorted(table.to_pylist(), key=lambda row: int(row["episode_index"]))
        for row in episodes:
            row["global_start"] = int(row.get("dataset_from_index", 0))
        return episodes

    path = root / "meta" / "episodes.jsonl"
    if path.is_file():
        episodes = []
        global_start = 0
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                row["global_start"] = global_start
                global_start += int(row["length"])
                episodes.append(row)
        return episodes
    raise FileNotFoundError("Dataset has neither meta/episodes/*.parquet nor meta/episodes.jsonl")


def target_path(root: Path, episode: int, chunks_size: int) -> Path:
    return root / "targets" / f"chunk-{episode // chunks_size:03d}" / f"episode_{episode:06d}.npy"


def episode_data_path(root: Path, info: dict[str, Any], episode: int) -> Path:
    template = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    return root / template.format(
        episode_chunk=episode // int(info.get("chunks_size", 1000)),
        episode_index=episode,
    )


def decode_image(value: dict[str, Any], dataset_root: Path) -> Image.Image:
    if value.get("bytes") is not None:
        with Image.open(io.BytesIO(value["bytes"])) as image:
            return image.convert("RGB").copy()
    path = Path(value["path"])
    if not path.is_absolute():
        path = dataset_root / path
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def decode_video_episode(
    dataset_root: Path,
    info: dict[str, Any],
    episode: dict[str, Any],
    image_key: str,
) -> list[Image.Image]:
    import av

    metadata_key = f"videos/observation.images.{image_key}"
    if f"{metadata_key}/file_index" not in episode:
        raise KeyError(f"Episode metadata does not define {metadata_key}")
    chunk = int(episode[f"{metadata_key}/chunk_index"])
    file_index = int(episode[f"{metadata_key}/file_index"])
    fps = float(info["fps"])
    start = round(float(episode[f"{metadata_key}/from_timestamp"]) * fps)
    length = int(episode["length"])
    path = (
        dataset_root
        / "videos"
        / f"observation.images.{image_key}"
        / f"chunk-{chunk:03d}"
        / f"file-{file_index:03d}.mp4"
    )
    decoded: dict[int, Image.Image] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if stream.time_base is None:
            raise ValueError(f"Video stream has no time base: {path}")
        seek_pts = int((start / fps) / float(stream.time_base))
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_index = round(float(frame.pts * stream.time_base) * fps)
            if frame_index < start:
                continue
            if frame_index >= start + length:
                break
            decoded.setdefault(frame_index, Image.fromarray(frame.to_ndarray(format="rgb24")))
    expected = list(range(start, start + length))
    missing = [index for index in expected if index not in decoded]
    if missing:
        raise ValueError(f"Video segment {path} is missing frames: {missing[:8]}")
    return [decoded[index] for index in expected]


def spatial_feature(output):
    """Apply the shared parameter-free S(.) operator on GPU."""
    import torch.nn.functional as functional

    value = output.float().reshape(output.shape[0], TEACHER_GRID, TEACHER_GRID, TEACHER_DIM)
    value = functional.normalize(value, dim=-1, eps=1e-6)
    value = value.reshape(value.shape[0], OUTPUT_GRID, 3, OUTPUT_GRID, 3, TEACHER_DIM).mean(dim=(2, 4))
    return functional.normalize(value, dim=-1, eps=1e-6)


def manifest(args: argparse.Namespace, info: dict[str, Any], episodes: list[dict[str, Any]]) -> dict[str, Any]:
    valid_samples = sum(max(0, int(row["length"]) - args.future_offset) for row in episodes)
    return {
        "format_version": 3,
        "kind": "vjepa2_nochange_referenced_displacement",
        "teacher_kind": "vjepa2.1_hf_bit_exact_port",
        "hf_port": str(args.hf_port.resolve()),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_total_episodes": len(episodes),
        "dataset_total_frames": sum(int(row["length"]) for row in episodes),
        "valid_sample_count": valid_samples,
        "chunks_size": int(info.get("chunks_size", 1000)),
        "image_key": args.image_key,
        "future_offset": args.future_offset,
        "tail_policy": "exclude_t_plus_h_outside_episode",
        "teacher_shape": [TEACHER_GRID**2, TEACHER_DIM],
        "target_shape": [OUTPUT_GRID**2, TEACHER_DIM],
        "target_dtype": "float16",
        "spatial_operator": "l2norm_then_nonoverlap_avgpool_3x3_then_l2norm",
        "feature_reduction": "none",
        "displacement": "S(pair)-S(same_frame)",
        "normalization_mean": IMAGENET_MEAN.tolist(),
        "normalization_std": IMAGENET_STD.tolist(),
    }


def ensure_manifest(root: Path, contract: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    temp = root / f".manifest-{os.getpid()}.json"
    with temp.open("w") as handle:
        json.dump(contract, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temp, path)
    except FileExistsError:
        pass
    finally:
        temp.unlink(missing_ok=True)
    with path.open() as handle:
        existing = json.load(handle)
    if existing != contract:
        raise RuntimeError(f"Existing displacement manifest differs: {path}")


def valid_target(path: Path, length: int, offset: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        return value.shape == (length - offset, OUTPUT_GRID**2, TEACHER_DIM) and value.dtype == TARGET_DTYPE
    except (OSError, ValueError):
        return False


def load_teacher(args: argparse.Namespace):
    import torch
    from transformers import AutoModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V-JEPA displacement generation")
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


def encode(model, videos):
    output = model(pixel_values_videos=videos, skip_predictor=True).last_hidden_state
    expected = (videos.shape[0], TEACHER_GRID**2, TEACHER_DIM)
    if tuple(output.shape) != expected:
        raise ValueError(f"Unexpected V-JEPA output {tuple(output.shape)}, expected {expected}")
    return spatial_feature(output)


def process_episode(args, info, episode, model, device) -> None:
    import torch

    index = int(episode["episode_index"])
    length = int(episode["length"])
    valid_length = length - args.future_offset
    if valid_length <= 0:
        return
    chunks_size = int(info.get("chunks_size", 1000))
    output_path = target_path(args.output_root, index, chunks_size)
    if valid_target(output_path, length, args.future_offset):
        print(f"skip episode={index:06d}", flush=True)
        return
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite invalid target: {output_path}")

    parquet_path = episode_data_path(args.dataset_root, info, index)
    if parquet_path.is_file() and args.image_key in pq.read_schema(parquet_path).names:
        table = pq.read_table(parquet_path, columns=[args.image_key, "frame_index"])
        if table.num_rows != length or not np.array_equal(table["frame_index"].to_numpy(), np.arange(length)):
            raise ValueError(f"Episode {index} has inconsistent parquet rows")
        images = [decode_image(value, args.dataset_root) for value in table[args.image_key].to_pylist()]
    else:
        images = decode_video_episode(args.dataset_root, info, episode, args.image_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(f".{output_path.name}.tmp-r{args.worker_rank}-p{os.getpid()}")
    targets = np.lib.format.open_memmap(
        temp,
        mode="w+",
        dtype=TARGET_DTYPE,
        shape=(valid_length, OUTPUT_GRID**2, TEACHER_DIM),
    )
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for start in range(0, valid_length, args.batch_size):
                end = min(start + args.batch_size, valid_length)
                current = np.stack([preprocess_image(images[i]) for i in range(start, end)])
                future = np.stack([preprocess_image(images[i + args.future_offset]) for i in range(start, end)])
                pair = torch.from_numpy(np.stack((current, future), axis=2)).to(device=device, dtype=torch.bfloat16)
                pair_feature = encode(model, pair)
                del pair
                same = torch.from_numpy(np.stack((current, current), axis=2)).to(device=device, dtype=torch.bfloat16)
                same_feature = encode(model, same)
                displacement = (pair_feature - same_feature).reshape(end - start, OUTPUT_GRID**2, TEACHER_DIM)
                targets[start:end] = displacement.to(device="cpu", dtype=torch.float16).numpy()
                del same, pair_feature, same_feature, displacement
        targets.flush()
        del targets
        os.replace(temp, output_path)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise
    print(
        f"complete episode={index:06d} rows={valid_length} elapsed={time.monotonic()-started:.1f}s",
        flush=True,
    )


def status(args, info, episodes) -> dict[str, Any]:
    complete = 0
    frames = 0
    size = 0
    for row in episodes:
        path = target_path(args.output_root, int(row["episode_index"]), int(info.get("chunks_size", 1000)))
        if valid_target(path, int(row["length"]), args.future_offset):
            complete += 1
            frames += max(0, int(row["length"]) - args.future_offset)
            size += path.stat().st_size
    total = sum(max(0, int(row["length"]) - args.future_offset) for row in episodes)
    return {
        "complete_episodes": complete,
        "total_episodes": len(episodes),
        "complete_samples": frames,
        "total_samples": total,
        "complete_gib": size / 1024**3,
        "expected_gib": total * OUTPUT_GRID**2 * TEACHER_DIM * 2 / 1024**3,
    }


def main() -> None:
    args = parse_args()
    if args.future_offset < 1 or args.batch_size < 1:
        raise ValueError("future-offset and batch-size must be positive")
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
    model, device = load_teacher(args)
    print(
        f"worker start rank={args.worker_rank}/{args.world_size} episodes={len(assigned)} "
        f"offset={args.future_offset} batch={args.batch_size}",
        flush=True,
    )
    for episode in assigned:
        process_episode(args, info, episode, model, device)
    print(f"worker complete rank={args.worker_rank}/{args.world_size}", flush=True)


if __name__ == "__main__":
    main()
