#!/usr/bin/env python3
"""Rewrite cached AC states after the gripper-mapping fix (tokens untouched).

Reads only the parquet ``state`` column, so this costs seconds per episode
instead of re-encoding the frames.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from openpi.con2.ac_world_model import map_libero_state  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    args = parser.parse_args()

    episodes = sorted(int(p.stem.split("_")[1]) for p in (args.cache / "tokens").glob("episode_*.npy"))
    changed = 0
    for episode in episodes:
        path = args.cache / "states" / f"episode_{episode:06d}.npy"
        chunk = episode // 1000
        source = args.dataset / "data" / f"chunk-{chunk:03d}" / f"episode_{episode:06d}.parquet"
        table = pq.read_table(source, columns=["state"])
        state = np.asarray(table["state"].to_pylist(), dtype=np.float32)
        mapped = map_libero_state(state).astype(np.float32)
        temp = path.with_suffix(".tmp.npy")
        np.save(temp, mapped)
        os.replace(temp, path)
        changed += 1
    print(f"repaired {changed} episodes in {args.cache}")


if __name__ == "__main__":
    main()
