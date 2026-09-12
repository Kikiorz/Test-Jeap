#!/usr/bin/env python3
"""Pack ffmpeg-extracted JPEG frames into the dataset as inline Image columns.

Second half of the inline-image materialisation: `extract_robotwin_frames.sh`
writes `<camera>/episode_XXXXXX/NNNNN.jpg`, and this copies those bytes into the
per-episode parquet as HuggingFace `Image` structs (the layout the LIBERO
dataset uses, which needs no video decoding at training time).
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]
IMAGE_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    source, frames_root, output = args.source, args.frames, args.output
    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    info = json.loads((source / "meta" / "info.json").read_text())
    episodes = [json.loads(line) for line in (source / "meta" / "episodes.jsonl").read_text().splitlines()]

    def pack(entry):
        episode = int(entry["episode_index"])
        chunk = episode // 1000
        target = output / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
        if target.exists() and target.stat().st_size > 0:
            return episode
        table = pq.read_table(source / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet")
        for camera in CAMERAS:
            directory = frames_root / camera / f"episode_{episode:06d}"
            files = sorted(directory.glob("*.jpg"))
            expected = int(entry["length"])
            if abs(len(files) - expected) > 1:
                raise ValueError(f"{directory}: {len(files)} frames, expected {expected}")
            payload = [{"bytes": path.read_bytes(), "path": None} for path in files]
            if len(files) < expected:
                # The release's metadata occasionally counts one frame more than
                # the video actually holds; repeat the last frame rather than
                # dropping the episode.
                payload.append(dict(payload[-1]))
            table = table.append_column(
                camera,
                pa.array(payload, type=IMAGE_TYPE),
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target)
        return episode

    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in pool.map(pack, episodes):
            done += 1
            if done % 250 == 0:
                print(json.dumps({"packed": done}), flush=True)

    for name in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl", "stats.json"):
        shutil.copyfile(source / "meta" / name, output / "meta" / name)
    features = dict(info["features"])
    for camera in CAMERAS:
        features[camera] = {"dtype": "image", "shape": [args.size, args.size, 3],
                            "names": ["height", "width", "channels"]}
    (output / "meta" / "info.json").write_text(
        json.dumps(dict(info, features=features, total_videos=0), indent=2, sort_keys=True) + "\n")
    print("DONE", done, flush=True)


if __name__ == "__main__":
    main()
