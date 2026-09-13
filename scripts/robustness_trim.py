#!/usr/bin/env python3
"""Are the paired conclusions driven by a handful of extreme batches?

Every statistic in this session is a paired mean over probe batches. A handful of
pathological episodes could produce (or destroy) several of them, and that is
exactly the failure mode a reviewer would probe. This recomputes each comparison
after trimming the most extreme batches, two ways:

  * trim by baseline flow - drops the hardest batches overall. **This is the
    informative one.**
  * trim by |paired difference| - reported only to show how badly it misleads:
    dropping the batches that contribute most to the mean is a winner's-curse
    filter and systematically inflates |t|. Never quote that column as evidence
    that a result is robust.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.stats

COMPARISONS = (
    ("a", "base"),
    ("b", "base"),
    ("d", "base"),
    ("a_only_residual", "a"),
    ("a_grafted", "a"),
)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("/workspace/artifacts/con2/hp"))
    p.add_argument("--trim", type=float, default=0.05)
    return p.parse_args()


def flows(directory: Path, name: str):
    path = directory / f"robotwin_ab_{name}.json"
    if name == "a_grafted":
        path = directory / "robotwin_ab_a_graft.json"
    if not path.exists():
        return None
    records = json.loads(path.read_text())["records"]
    records.sort(key=lambda r: r["batch"])
    return np.asarray([r["flow_b0.05_true"] for r in records], dtype=float)


def paired(left, right, keep):
    diff = (left - right)[keep]
    n = len(diff)
    mean = diff.mean()
    se = diff.std(ddof=1) / math.sqrt(n)
    t = mean / se if se > 0 else float("nan")
    return mean, t, n


def main() -> None:
    args = parse()
    print(f"directory {args.dir}   trimming {args.trim:.0%}")
    print(f"{'comparison':>24} {'full t':>8} {'trim |d| t':>12} {'trim flow t':>13} {'n':>5}")
    for left_name, right_name in COMPARISONS:
        left, right = flows(args.dir, left_name), flows(args.dir, right_name)
        if left is None or right is None or len(left) != len(right):
            print(f"{left_name + ' vs ' + right_name:>24}   (missing)")
            continue
        n = len(left)
        cut = int(n * args.trim)
        diff = left - right

        keep_diff = np.isin(np.arange(n), np.argsort(np.abs(diff))[: n - cut])
        keep_flow = np.isin(np.arange(n), np.argsort(right)[: n - cut])
        _, t_full, _ = paired(left, right, np.ones(n, bool))
        _, t_diff, n_diff = paired(left, right, keep_diff)
        _, t_flow, _ = paired(left, right, keep_flow)
        print(f"{left_name + ' vs ' + right_name:>24} {t_full:>8.2f} "
              f"{t_diff:>12.2f} {t_flow:>13.2f} {n_diff:>5}")
    print()
    print("Read the 'trim flow' column, not 'trim |d|': the latter removes the")
    print("largest contributors to the mean and therefore inflates |t| by")
    print("construction. A conclusion is outlier-driven if 'trim flow' collapses.")


if __name__ == "__main__":
    main()
