#!/usr/bin/env python3
"""Split the packed RoboTwin v3 mp4s into one file per episode and camera.

The v3 release packs many episodes into one mp4 per chunk file, and the three
cameras use *different* file boundaries, while LeRobot v2.1 has a single
``timestamp`` column shared by all cameras. The only consistent layout is one
video per (episode, camera) whose clock starts at zero, which is what this
produces with a stream copy (no re-encode).
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CAMERAS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def scalar(value):
    while isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return 0.0
        value = value[0]
    return value.item() if isinstance(value, np.generic) else value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="v3 dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="v2.1 dataset directory")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    source, output = args.source, args.output
    info = json.loads((source / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    episodes = pq.read_table(source / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()

    jobs = []
    for _, row in episodes.iterrows():
        episode = int(row["episode_index"])
        length = int(row["length"])
        for camera in CAMERAS:
            chunk = int(scalar(row[f"videos/{camera}/chunk_index"]))
            file_index = int(scalar(row[f"videos/{camera}/file_index"]))
            start = float(scalar(row[f"videos/{camera}/from_timestamp"]))
            origin = source / "videos" / camera / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
            target_dir = output / "videos" / f"chunk-{episode // 1000:03d}" / camera
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"episode_{episode:06d}.mp4"
            jobs.append((origin, start, length / fps, target))

    print(json.dumps({"jobs": len(jobs)}), flush=True)

    def run(job):
        origin, start, duration, target = job
        if target.exists() and target.stat().st_size > 0 and not target.is_symlink():
            return True
        if target.is_symlink():
            target.unlink()
        command = [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-i", str(origin),
            "-c", "copy", "-an", str(target),
        ]
        return subprocess.run(command, capture_output=True).returncode == 0

    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for ok in pool.map(run, jobs):
            done += 1
            if not ok:
                print(json.dumps({"failed": done}), flush=True)
            if done % 500 == 0:
                print(json.dumps({"split": done}), flush=True)
    print("DONE", done)


if __name__ == "__main__":
    main()
