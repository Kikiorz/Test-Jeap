#!/usr/bin/env python3
"""Build a LeRobot v2.1 dataset from the RoboTwin 2.0 randomized archives.

The official ``TianxingChen/RoboTwin2.0`` LeRobot release only carries the
50-task *clean* demonstrations (2500 episodes). The randomized demonstrations
exist only as per-task archives
(``dataset/<task>/<robot>_randomized_500.zip``) holding one HDF5 per episode.

This script turns those archives into the same v2.1 layout openpi consumes,
with the frames stored inline as JPEG bytes (the LIBERO-style layout the
``*_inline`` RoboTwin datasets use), so training never decodes video. The
archive is read member-by-member and never unpacked to disk.

Conventions, verified against the official v2.1 release
(see ``docs_ROBOTWIN_RANDOM_FT.md``):

* ``observation.state[t] = joint_action/vector[t]``  (14 = 6+1 + 6+1)
* ``action[t] = joint_action/vector[t+1]``, last frame repeated
  (the official release stores exactly this: ``|a[t] - s[t+1]| == 0``)
* ``fps = 15``
* ``head_camera -> cam_high``, ``left_camera -> cam_left_wrist``,
  ``right_camera -> cam_right_wrist``
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = [
    ("head_camera", "observation.images.cam_high"),
    ("left_camera", "observation.images.cam_left_wrist"),
    ("right_camera", "observation.images.cam_right_wrist"),
]

FPS = 15
CHUNK_SIZE = 1000

MOTOR_NAMES = [
    "left_joint_0",
    "left_joint_1",
    "left_joint_2",
    "left_joint_3",
    "left_joint_4",
    "left_joint_5",
    "left_joint_6",
    "right_joint_0",
    "right_joint_1",
    "right_joint_2",
    "right_joint_3",
    "right_joint_4",
    "right_joint_5",
    "right_joint_6",
]

IMAGE_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

# ``load_dataset("parquet", ...)`` infers features from the arrow schema, and a
# plain ``struct<bytes, path>`` column would come back as a struct rather than a
# PIL image. HuggingFace records the declared feature types in this schema
# metadata key, so the inline frames decode exactly like the LIBERO dataset's.
HF_FEATURES_METADATA = json.dumps({
    "info": {
        "features": {
            "observation.state": {"_type": "Sequence", "length": 14,
                                  "feature": {"_type": "Value", "dtype": "float32"}},
            "action": {"_type": "Sequence", "length": 14,
                       "feature": {"_type": "Value", "dtype": "float32"}},
            **{target: {"_type": "Image"} for _, target in CAMERAS},
            "timestamp": {"_type": "Value", "dtype": "float32"},
            "frame_index": {"_type": "Value", "dtype": "int64"},
            "episode_index": {"_type": "Value", "dtype": "int64"},
            "index": {"_type": "Value", "dtype": "int64"},
            "task_index": {"_type": "Value", "dtype": "int64"},
        }
    }
})

_ZIP: zipfile.ZipFile | None = None


def _open_zip(path: str) -> zipfile.ZipFile:
    """One handle per worker process; archives are far too big to reopen per member."""
    global _ZIP
    if _ZIP is None or _ZIP.filename != str(path):
        _ZIP = zipfile.ZipFile(path)
    return _ZIP


def _encode_jpeg(frame: np.ndarray, size: int, quality: int) -> bytes:
    from PIL import Image

    image = Image.fromarray(frame)
    if image.size != (size, size):
        image = image.resize((size, size), resample=Image.BICUBIC)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _read_episode(job):
    """Decode one raw episode; runs in a worker process."""
    zip_path, prefix, episode, size, quality, instruction = job

    import h5py
    from PIL import Image

    handle = _open_zip(zip_path)
    with handle.open(f"{prefix}/data/episode{episode}.hdf5") as member:
        payload = member.read()

    with h5py.File(io.BytesIO(payload), "r") as data:
        state = np.asarray(data["joint_action/vector"][:], dtype=np.float32)
        if state.ndim != 2 or state.shape[1] != 14:
            raise ValueError(f"episode {episode}: unexpected vector shape {state.shape}")
        # The recorded vector is the joint state measured at each frame; the
        # command that produced frame t+1 is what the release stores as a[t].
        action = np.concatenate([state[1:], state[-1:]], axis=0)

        images = {}
        for source, target in CAMERAS:
            raws = data[f"observation/{source}/rgb"][:]
            frames = []
            for raw in raws:
                frame = np.asarray(Image.open(io.BytesIO(bytes(raw))).convert("RGB"))
                frames.append(_encode_jpeg(frame, size, quality))
            images[target] = frames

    return {
        "episode": episode,
        "state": state,
        "action": action,
        "images": images,
        "instruction": instruction,
    }


def _episode_jobs(args, zip_path: Path, prefix: str, episodes: list[int], seed: int):
    # Instructions are read in the parent: they are tiny and the resulting
    # strings have to be collected anyway to build the task index.
    handle = zipfile.ZipFile(zip_path)
    jobs = []
    for episode in episodes:
        with handle.open(f"{prefix}/instructions/episode{episode}.json") as stream:
            payload = json.loads(stream.read().decode("utf-8"))
        pool = list(payload.get("seen", [])) + list(payload.get("unseen", []))
        if not pool:
            raise ValueError(f"episode {episode}: no instructions in archive")
        import random

        instruction = random.Random(seed * 100_003 + episode).choice(pool)
        jobs.append((str(zip_path), prefix, episode, args.size, args.quality, instruction))
    handle.close()
    return jobs


def _stats(values: np.ndarray) -> dict:
    return {
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "count": np.array([len(values)]),
    }


def _image_stats(frames: list[np.ndarray]) -> dict:
    """Channel statistics over a sample of frames, shape (3, 1, 1) as lerobot wants."""
    stack = np.stack(frames).astype(np.float32) / 255.0  # (N, H, W, 3)
    axes = (0, 1, 2)
    keep = {"min": np.min(stack, axis=axes, keepdims=True),
            "max": np.max(stack, axis=axes, keepdims=True),
            "mean": np.mean(stack, axis=axes, keepdims=True),
            "std": np.std(stack, axis=axes, keepdims=True)}
    return {
        "min": keep["min"].reshape(3, 1, 1),
        "max": keep["max"].reshape(3, 1, 1),
        "mean": keep["mean"].reshape(3, 1, 1),
        "std": keep["std"].reshape(3, 1, 1),
        "count": np.array([len(frames)]),
    }


def _jsonify(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _aggregate(per_episode: list[dict], key: str) -> dict:
    entries = [entry["stats"][key] for entry in per_episode]
    counts = np.stack([entry["count"] for entry in entries]).astype(np.float64)
    total = counts.sum(axis=0)

    def weighted(stat: str) -> np.ndarray:
        stacked = np.stack([entry[stat] for entry in entries])
        counts_expanded = counts
        while counts_expanded.ndim < stacked.ndim:
            counts_expanded = np.expand_dims(counts_expanded, axis=-1)
        return (stacked * counts_expanded).sum(axis=0) / total

    means = weighted("mean")
    second_moment = weighted("mean") ** 2 + weighted("std") ** 2
    variances = np.maximum(second_moment - means**2, 0.0)
    return {
        "min": np.stack([e["min"] for e in entries]).min(axis=0),
        "max": np.stack([e["max"] for e in entries]).max(axis=0),
        "mean": means,
        "std": np.sqrt(variances),
        "count": np.array([int(total[0])]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zips", action="append", required=True,
                        help="task_name=/path/to/<robot>_randomized_500.zip (repeatable)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes-per-task", type=int, default=200)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-image-stats-frames", type=int, default=4000)
    args = parser.parse_args()

    output = args.output
    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    tasks: dict[str, int] = {}
    episode_records: list[dict] = []
    episode_stats: list[dict] = []
    image_samples: dict[str, list[np.ndarray]] = {target: [] for _, target in CAMERAS}
    global_index = 0

    for spec in args.zips:
        task, _, zip_path = spec.partition("=")
        zip_path = Path(zip_path)
        with zipfile.ZipFile(zip_path) as handle:
            names = handle.namelist()
        prefix = names[0].split("/")[0]
        available = sorted(
            int(Path(name).stem.replace("episode", ""))
            for name in names
            if name.startswith(f"{prefix}/data/") and name.endswith(".hdf5")
        )
        episodes = available[: args.episodes_per_task]
        print(f"[{task}] {len(episodes)}/{len(available)} episodes from {zip_path.name}", flush=True)

        jobs = _episode_jobs(args, zip_path, prefix, episodes, args.seed)
        done = 0
        with mp.Pool(args.workers) as pool:
            for result in pool.imap(_read_episode, jobs, chunksize=1):
                episode = result["episode"]
                state, action = result["state"], result["action"]
                length = state.shape[0]

                instruction = result["instruction"]
                if instruction not in tasks:
                    tasks[instruction] = len(tasks)
                task_index = tasks[instruction]

                gid = len(episode_records)
                frames = np.arange(length)
                table = pa.table({
                    "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
                    "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
                    "timestamp": pa.array((frames / FPS).astype(np.float32), type=pa.float32()),
                    "frame_index": pa.array(frames, type=pa.int64()),
                    "episode_index": pa.array(np.full(length, gid), type=pa.int64()),
                    "index": pa.array(np.arange(global_index, global_index + length), type=pa.int64()),
                    "task_index": pa.array(np.full(length, task_index), type=pa.int64()),
                })
                for _, target in CAMERAS:
                    table = table.append_column(
                        target,
                        pa.array([{"bytes": blob, "path": None} for blob in result["images"][target]],
                                 type=IMAGE_STRUCT),
                    )
                table = table.replace_schema_metadata({"huggingface": HF_FEATURES_METADATA})

                chunk = gid // CHUNK_SIZE
                (output / "data" / f"chunk-{chunk:03d}").mkdir(parents=True, exist_ok=True)
                pq.write_table(
                    table,
                    output / "data" / f"chunk-{chunk:03d}" / f"episode_{gid:06d}.parquet",
                )
                global_index += length

                record_stats = {
                    "observation.state": _stats(state),
                    "action": _stats(action),
                    # Scalar bookkeeping columns; the release stores stats for
                    # them too and some LeRobot paths assume every feature has one.
                    "timestamp": _stats((frames / FPS).astype(np.float32)[:, None]),
                    "frame_index": _stats(frames.astype(np.float64)[:, None]),
                    "episode_index": _stats(np.full((length, 1), gid, dtype=np.float64)),
                    "index": _stats(np.arange(global_index, global_index + length,
                                               dtype=np.float64)[:, None]),
                    "task_index": _stats(np.full((length, 1), task_index, dtype=np.float64)),
                }
                for _, target in CAMERAS:
                    if len(image_samples[target]) < args.max_image_stats_frames:
                        from PIL import Image

                        blob = result["images"][target][length // 2]
                        image_samples[target].append(
                            np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"))
                        )
                    record_stats[target] = None
                episode_stats.append({"episode_index": gid, "stats": record_stats})
                episode_records.append({
                    "episode_index": gid,
                    "tasks": [instruction],
                    "length": length,
                })

                done += 1
                if done % 25 == 0:
                    print(f"[{task}] {done}/{len(episodes)} episodes, {global_index} frames",
                          flush=True)
        print(f"[{task}] done: {done} episodes, {global_index} frames total", flush=True)

    total_frames = sum(record["length"] for record in episode_records)
    features = {
        "observation.state": {"dtype": "float32", "shape": [14], "names": [MOTOR_NAMES]},
        "action": {"dtype": "float32", "shape": [14], "names": [MOTOR_NAMES]},
    }
    for _, target in CAMERAS:
        features[target] = {"dtype": "image", "shape": [args.size, args.size, 3],
                            "names": ["height", "width", "channels"]}
    for key, dtype in (("timestamp", "float32"), ("frame_index", "int64"),
                       ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
        features[key] = {"dtype": dtype, "shape": [1], "names": None}

    info = {
        "codebase_version": "v2.1",
        "robot_type": "unified_robot",
        "total_episodes": len(episode_records),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": 0,
        "total_chunks": (len(episode_records) + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": FPS,
        "splits": {"train": f"0:{len(episode_records)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": features,
    }
    (output / "meta" / "info.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    with (output / "meta" / "tasks.jsonl").open("w") as stream:
        for instruction, task_index in sorted(tasks.items(), key=lambda item: item[1]):
            stream.write(json.dumps({"task_index": task_index, "task": instruction}) + "\n")
    with (output / "meta" / "episodes.jsonl").open("w") as stream:
        for record in episode_records:
            stream.write(json.dumps(record) + "\n")
    for _, target in CAMERAS:
        stats = _image_stats(image_samples[target]) if image_samples[target] else None
        for record in episode_stats:
            record["stats"][target] = stats
    with (output / "meta" / "episodes_stats.jsonl").open("w") as stream:
        for record in episode_stats:
            stream.write(json.dumps(_jsonify(record)) + "\n")

    global_stats = {
        "observation.state": _aggregate(episode_stats, "observation.state"),
        "action": _aggregate(episode_stats, "action"),
    }
    global_stats.update({
        key: _aggregate(episode_stats, key)
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")
    })
    for _, target in CAMERAS:
        global_stats[target] = _image_stats(image_samples[target])
    (output / "meta" / "stats.json").write_text(
        json.dumps(_jsonify(global_stats), indent=2, sort_keys=True) + "\n"
    )
    print(f"DONE {len(episode_records)} episodes / {total_frames} frames / {len(tasks)} instructions"
          f" -> {output}", flush=True)


if __name__ == "__main__":
    main()
