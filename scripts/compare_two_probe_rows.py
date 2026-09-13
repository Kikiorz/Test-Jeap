#!/usr/bin/env python3
"""Paired comparison of two probe JSONs produced from the same batches.

Two runs of probe_con1_budget_and_conditioning.py with the same seed, batch size
and batch count walk the identical shuffled sequence, so batch i is the same
batch in both files and a paired t statistic is meaningful.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left", type=Path, help="treatment file")
    p.add_argument("right", type=Path, help="baseline file")
    p.add_argument("--labels", nargs=2, default=None)
    return p.parse_args()


def flows(entry):
    records = entry.get("records") or []
    key = next((k for k in records[0] if k.startswith("flow_") and k.endswith("_true")), None)
    return [record[key] for record in records]


def main() -> None:
    args = parse()
    left = json.loads(args.left.read_text())
    right = json.loads(args.right.read_text())
    lv, rv = flows(left), flows(right)
    if len(lv) != len(rv):
        raise SystemExit(f"batch counts differ: {len(lv)} vs {len(rv)}")

    def mean(values):
        return sum(values) / len(values)

    diff = [a - b for a, b in zip(lv, rv)]
    n = len(diff)
    m = mean(diff)
    var = sum((d - m) ** 2 for d in diff) / (n - 1)
    se = math.sqrt(var / n)
    t = m / se if se > 0 else float("nan")
    lm, rm = mean(lv), mean(rv)

    left_label, right_label = args.labels or (args.left.stem, args.right.stem)
    print(f"{left_label:38s} flow = {lm:.6f}")
    print(f"{right_label:38s} flow = {rm:.6f}")
    print(f"{'difference (treatment - baseline)':38s} = {m:+.3e}  "
          f"({(lm / rm - 1) * 100:+.2f}%)")
    print(f"{'paired t (n batches)':38s} = {t:+.2f}  (n = {n})")
    print()
    if abs(t) >= 3:
        print("=> the difference is above the |t| >= 3 bar used in this project")
    else:
        print("=> below the |t| >= 3 bar; treat as not resolved")


if __name__ == "__main__":
    main()
