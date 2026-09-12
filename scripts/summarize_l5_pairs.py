#!/usr/bin/env python3
"""Paired LIBERO-Plus L5 comparison: baseline (official 40k) vs Con1 adapter.

Reads the per-shard episode journals of both sides, pairs on
``(task_id, episode_idx)`` and reports, per perturbation category, the success
rates plus the discordant counts that drive the difference.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

CATEGORIES = [
    ("Background Textures", "Background_Textures"),
    ("Camera Viewpoints", "Camera_Viewpoints"),
    ("Language Instructions", "Language_Instructions"),
    ("Light Conditions", "Light_Conditions"),
    ("Objects Layout", "Objects_Layout"),
    ("Robot Initial States", "Robot_Initial_States"),
    ("Sensor Noise", "Sensor_Noise"),
]


def load(root: Path, side: str, slug: str) -> dict:
    records = {}
    pattern = root / f"{side}_L5_{slug}" / "plus-libero_10.shard-*.jsonl"
    for path in sorted(glob.glob(str(pattern))):
        try:
            handle = open(path, encoding="utf-8")
        except OSError:
            continue
        with handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("record_type") != "episode":
                    continue
                key = (record["task_id"], record.get("episode_idx"))
                records[key] = bool(record.get("success"))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/libero-eval"))
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--side", default="adapter",
                        help="Run-id prefix of the candidate model (default: adapter).")
    parser.add_argument("--baseline", default="baseline",
                        help="Run-id prefix of the reference model.")
    args = parser.parse_args()

    report = {}
    print(f"{'category':24s} {'paired':>6s} {'adapter':>9s} {'baseline':>9s} "
          f"{'A-only':>6s} {'B-only':>6s} {'both':>5s} {'neither':>7s} {'delta':>7s}")
    for category, slug in CATEGORIES:
        adapter = load(args.root, args.side, slug)
        baseline = load(args.root, args.baseline, slug)
        keys = sorted(set(adapter) & set(baseline))
        if not keys:
            print(f"{category:24s} {0:6d}   (no paired episodes yet)")
            report[category] = {"paired": 0, "adapter_records": len(adapter),
                                "baseline_records": len(baseline)}
            continue
        a_hits = sum(adapter[k] for k in keys)
        b_hits = sum(baseline[k] for k in keys)
        a_only = sum(1 for k in keys if adapter[k] and not baseline[k])
        b_only = sum(1 for k in keys if baseline[k] and not adapter[k])
        both = sum(1 for k in keys if adapter[k] and baseline[k])
        neither = sum(1 for k in keys if not adapter[k] and not baseline[k])
        paired = len(keys)
        print(f"{category:24s} {paired:6d} {100 * a_hits / paired:8.1f}% "
              f"{100 * b_hits / paired:8.1f}% {a_only:6d} {b_only:6d} {both:5d} "
              f"{neither:7d} {100 * (a_hits - b_hits) / paired:+6.1f}pp")
        report[category] = {
            "paired": paired,
            "adapter_success": a_hits,
            "baseline_success": b_hits,
            "adapter_rate": a_hits / paired,
            "baseline_rate": b_hits / paired,
            "adapter_only": a_only,
            "baseline_only": b_only,
            "both": both,
            "neither": neither,
            "adapter_records": len(adapter),
            "baseline_records": len(baseline),
        }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("WROTE", args.json)


if __name__ == "__main__":
    main()
