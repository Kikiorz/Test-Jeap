#!/usr/bin/env python3
"""Convert the official RoboTwin 2.0 LeRobot v3 dataset to the v2.1 layout.

openpi pins a LeRobot revision with ``CODEBASE_VERSION = v2.1`` (per-episode
parquet, ``meta/tasks.jsonl``, ``meta/episodes.jsonl``, one mp4 per episode), so
the v3 release cannot be consumed as-is.

The conversion is cheap because nothing is re-encoded: the v3 videos pack many
episodes into one mp4 per chunk file, and the v3 episode metadata already records
which file each episode lives in plus its ``from_timestamp``. We

  * rewrite each frame's ``timestamp`` to the *global* time inside that mp4,
  * symlink ``videos/chunk-XXX/{camera}/episode_XXXXXX.mp4`` to the right v3 file
    (so the v2.1 path template resolves), and
  * write the v2.1 meta files, reusing the per-episode statistics that v3 stores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def scalar(value):
    """v3 stores statistics as length-1 lists; v2.1 wants plain numbers."""
    while isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return 0.0
        value = value[0]
    return value.item() if isinstance(value, np.generic) else value


def vector(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [scalar(item) if not isinstance(item, (list, tuple, np.ndarray)) else vector(item)
                for item in value]
    return value.item() if isinstance(value, np.generic) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source, output = args.source, args.output
    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (output / "videos" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = json.loads((source / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    tasks = pq.read_table(source / "meta" / "tasks.parquet").to_pandas()
    episodes = pq.read_table(source / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
    frames = pq.read_table(source / "data" / "chunk-000" / "file-000.parquet")
    frame_index = np.asarray(frames["episode_index"].to_pylist())

    with (output / "meta" / "tasks.jsonl").open("w") as handle:
        for task, row in tasks.iterrows():
            handle.write(json.dumps({"task_index": int(row["task_index"]), "task": str(task)}) + "\n")

    episodes_lines, stats_lines = [], []
    offset = 0
    for _, row in episodes.iterrows():
        episode = int(row["episode_index"])
        length = int(row["length"])
        where = np.flatnonzero(frame_index == episode)
        if len(where) != length:
            raise ValueError(f"episode {episode}: parquet has {len(where)} frames, meta says {length}")
        table = frames.take(pa.array(where))
        camera_meta = {}
        for camera in CAMERAS:
            chunk = int(scalar(row[f"videos/{camera}/chunk_index"]))
            file_index = int(scalar(row[f"videos/{camera}/file_index"]))
            start = float(scalar(row[f"videos/{camera}/from_timestamp"]))
            camera_meta[camera] = (chunk, file_index, start)
            target_dir = output / "videos" / "chunk-000" / camera
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"episode_{episode:06d}.mp4"
            if not target.exists():
                origin = source / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
                target.symlink_to(origin.resolve())
        # v2.1 reads frames by timestamp, so shift them into the shared mp4's clock.
        stamps = np.round(camera_meta[CAMERAS[0]][2] + np.arange(length) / fps, 6)
        table = table.set_column(table.schema.get_field_index("timestamp"), "timestamp",
                                 pa.array(stamps.astype(np.float32)))
        pq.write_table(table, output / "data" / "chunk-000" / f"episode_{episode:06d}.parquet")
        offset += length

        episodes_lines.append({"episode_index": episode, "tasks": list(row["tasks"]), "length": length})
        entry = {"episode_index": episode}
        for key in ("observation.state", "action"):
            entry[key] = {
                stat: (vector(row[f"stats/{key}/{stat}"]) if stat in ("mean", "std", "min", "max", "q01", "q99")
                       else int(scalar(row[f"stats/{key}/{stat}"])))
                for stat in ("min", "max", "mean", "std", "count")
            }
        for camera in CAMERAS:
            entry[camera] = {stat: vector(row[f"stats/{camera}/{stat}"]) if stat in ("mean", "std", "min", "max")
                             else int(scalar(row[f"stats/{camera}/{stat}"]))
                             for stat in ("min", "max", "mean", "std", "count")}
        stats_lines.append(entry)
        if episode % 250 == 0:
            print(json.dumps({"converted": episode + 1}), flush=True)

    with (output / "meta" / "episodes.jsonl").open("w") as handle:
        for line in episodes_lines:
            handle.write(json.dumps(line) + "\n")
    with (output / "meta" / "episodes_stats.jsonl").open("w") as handle:
        for line in stats_lines:
            handle.write(json.dumps(line) + "\n")

    merged = json.loads((source / "meta" / "stats.json").read_text())
    (output / "meta" / "stats.json").write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")

    features = {
        "observation.state": {"dtype": "float32", "shape": [14],
                              "names": info["features"]["observation.state"].get("names")},
        "action": {"dtype": "float32", "shape": [14],
                   "names": info["features"]["action"].get("names")},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "frame_index": {"dtype": "int64", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
        "index": {"dtype": "int64", "shape": [1]},
        "task_index": {"dtype": "int64", "shape": [1]},
    }
    for camera in CAMERAS:
        shape = info["features"][camera].get("shape", [480, 640, 3])
        features[camera] = {"dtype": "video", "shape": shape,
                            "names": ["height", "width", "channels"],
                            "video_info": {"video.fps": fps, "video.codec": "h264",
                                           "video.pix_fmt": "yuv420p", "video.is_depth_map": False,
                                           "has_audio": False}}
    new_info = {
        "codebase_version": "v2.1",
        "robot_type": info.get("robot_type", "aloha"),
        "total_episodes": len(episodes_lines),
        "total_frames": int(sum(e["length"] for e in episodes_lines)),
        "total_tasks": len(tasks),
        "total_videos": len(episodes_lines) * len(CAMERAS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes_lines)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (output / "meta" / "info.json").write_text(json.dumps(new_info, indent=2, sort_keys=True) + "\n")
    print("WROTE", output)


if __name__ == "__main__":
    main()
