#!/usr/bin/env python3
"""Rewrite the converted RoboTwin dataset's actions into per-frame deltas.

The released data stores absolute joint targets (``action[t] == state[t+1]``),
while the RoboTwin JEPA-WAM base was trained on per-frame joint velocities:
measuring the base's own flow-matching loss against five candidate conventions
and a time-scale sweep puts ``action[t] - state[t]`` an order of magnitude below
every alternative (0.0224 vs 0.19-0.36) — see scripts/probe_robotwin_convention.py.

Because ``action[t] == state[t+1]``, that difference is exactly
``state[t+1] - state[t]``; writing it into the column lets the training pipeline
read the chunk unchanged (openpi's own ``DeltaActions`` subtracts the *query*
state from every chunk step, which is a different — and much worse — convention
here, so it stays disabled).
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(glob.glob(str(args.dataset / "data" / "chunk-*" / "episode_*.parquet")))
    changed = 0
    for path in files:
        table = pq.read_table(path)
        actions = np.asarray(table["actions"].to_pylist(), np.float32)
        state = np.asarray(table["observation.state"].to_pylist(), np.float32)
        delta = actions - state
        if args.dry_run:
            if changed == 0:
                print("first episode: |action| mean %.4f -> |delta| mean %.4f"
                      % (np.abs(actions).mean(), np.abs(delta).mean()))
            changed += 1
            continue
        column = table.schema.get_field_index("actions")
        table = table.set_column(column, "actions", pa.array(delta.tolist()))
        pq.write_table(table, path)
        changed += 1
        if changed % 500 == 0:
            print(f"rewrote {changed}", flush=True)
    print("DONE", changed)


if __name__ == "__main__":
    main()
