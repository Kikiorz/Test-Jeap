#!/usr/bin/env python3
"""Materialise the RoboTwin camera streams as inline images (LIBERO-style).

Training decodes three AV1 videos per sample, and LeRobot does not cache decoded
frames, so the training loop measured ~6-7 s/step at batch 8 with the GPU idle:
it is entirely decode-bound. This writes a *sibling* dataset whose episodes carry
the frames inline as HuggingFace ``Image`` columns (the layout the LIBERO dataset
uses), leaving the video dataset untouched for the Con1 cache pipeline (whose
contract hashes the dataset metadata).

Only the image representation changes: episodes, lengths, prompts, state and
actions are copied verbatim.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CAMERAS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--size", type=int, default=384)
    args = parser.parse_args()

    source, output = args.source, args.output
    (output / "meta").mkdir(parents=True, exist_ok=True)
    (output / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    info = json.loads((source / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    episodes = [json.loads(line) for line in (source / "meta" / "episodes.jsonl").read_text().splitlines()]
    if args.max_episodes:
        episodes = episodes[: args.max_episodes]

    from lerobot.common.datasets.video_utils import decode_video_frames

    def _encode(frame: np.ndarray, size: int) -> bytes:
        """Store at the V-JEPA input size (384) as JPEG; the policy resizes to 224."""
        import io

        from PIL import Image

        image = Image.fromarray(frame)
        if image.size != (size, size):
            image = image.resize((size, size), resample=Image.Resampling.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        return buffer.getvalue()

    def convert(entry):
        episode = int(entry["episode_index"])
        length = int(entry["length"])
        chunk = episode // 1000
        path = source / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
        target = output / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
        if target.exists() and target.stat().st_size > 0:
            return episode  # resumable: already materialised
        table = pq.read_table(path)
        columns = {}
        for camera in CAMERAS:
            video = source / "videos" / f"chunk-{chunk:03d}" / camera / f"episode_{episode:06d}.mp4"
            frames = decode_video_frames(video, [index / fps for index in range(length)], tolerance_s=0.1)
            array = (frames.permute(0, 2, 3, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
            columns[camera] = pa.array(
                [{"bytes": _encode(frame, args.size), "path": None} for frame in array],
                type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
        for camera, column in columns.items():
            table = table.append_column(camera, column)
        pq.write_table(table, target)
        return episode

    done = 0
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for episode in pool.map(convert, episodes):
            done += 1
            if done % 100 == 0:
                print(json.dumps({"converted": done}), flush=True)

    for name in ("tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl", "stats.json"):
        shutil.copyfile(source / "meta" / name, output / "meta" / name)
    features = dict(info["features"])
    for camera in CAMERAS:
        features[camera] = {"dtype": "image", "shape": [args.size, args.size, 3],
                            "names": ["height", "width", "channels"]}
    new_info = dict(info, features=features, total_videos=0)
    (output / "meta" / "info.json").write_text(json.dumps(new_info, indent=2, sort_keys=True) + "\n")
    print("DONE", done, flush=True)


if __name__ == "__main__":
    main()
