#!/usr/bin/env python3
"""Four-suite LIBERO-Plus summary for the Con1/Con2 candidate.

Reads every shard journal of every suite, keeps the best record per
``(task_id, episode_idx)`` (infrastructure failures never overwrite a genuine
evaluation), and reports success rates per suite, per perturbation category and
per difficulty level, plus the category-macro average.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

SUITES = ["libero_10", "libero_spatial", "libero_object", "libero_goal"]
TOTAL_TASKS = {"libero_10": 2519, "libero_spatial": 2402,
               "libero_object": 2518, "libero_goal": 2591}


def load(root: Path, side: str, suite: str) -> dict:
    records: dict = {}
    for path in glob.glob(str(root / f"{side}_plus_full" / f"plus-{suite}.shard-*.jsonl")):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("record_type") != "episode":
                    continue
                key = (record["task_id"], record.get("episode_idx"))
                previous = records.get(key)
                if previous is not None and previous.get("error") and not record.get("error"):
                    records[key] = record
                elif previous is not None and not previous.get("error") and record.get("error"):
                    continue
                else:
                    records[key] = record
    return {k: v for k, v in records.items() if not v.get("error")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/libero-eval"))
    parser.add_argument("--side", default="con1con2")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    report: dict = {}
    categories: dict = collections.defaultdict(lambda: [0, 0])
    difficulties: dict = collections.defaultdict(lambda: [0, 0])
    print(f"{'suite':16s} {'evaluated':>9s} {'total':>6s} {'success':>8s} {'macro(cat)':>11s}")
    for suite in SUITES:
        records = load(args.root, args.side, suite)
        if not records:
            continue
        hits = sum(bool(r.get("success")) for r in records.values())
        cat = collections.defaultdict(lambda: [0, 0])
        dif = collections.defaultdict(lambda: [0, 0])
        for record in records.values():
            ok = bool(record.get("success"))
            cat[record.get("category")][0] += ok
            cat[record.get("category")][1] += 1
            dif[record.get("difficulty_level")][0] += ok
            dif[record.get("difficulty_level")][1] += 1
            categories[record.get("category")][0] += ok
            categories[record.get("category")][1] += 1
            difficulties[record.get("difficulty_level")][0] += ok
            difficulties[record.get("difficulty_level")][1] += 1
        macro = sum(ok / n for ok, n in cat.values()) / len(cat)
        print(f"{suite:16s} {len(records):9d} {TOTAL_TASKS[suite]:6d} "
              f"{100 * hits / len(records):7.1f}% {100 * macro:10.1f}%")
        report[suite] = {
            "evaluated": len(records),
            "total_tasks": TOTAL_TASKS[suite],
            "success": hits,
            "rate": hits / len(records),
            "macro_category_rate": macro,
            "categories": {k: {"n": v[1], "rate": v[0] / v[1]} for k, v in cat.items()},
            "difficulties": {str(k): {"n": v[1], "rate": v[0] / v[1]} for k, v in dif.items()},
        }

    print()
    print(f"{'category':24s} {'n':>6s} {'success':>8s}")
    for name, (ok, n) in sorted(categories.items(), key=lambda kv: -kv[1][1]):
        print(f"{name:24s} {n:6d} {100 * ok / n:7.1f}%")
    print()
    print(f"{'difficulty':16s} {'n':>6s} {'success':>8s}")
    for level in sorted(difficulties, key=lambda x: (x is None, x)):
        ok, n = difficulties[level]
        print(f"L{level:<15} {n:6d} {100 * ok / n:7.1f}%")

    total_ok = sum(v[0] for v in categories.values())
    total_n = sum(v[1] for v in categories.values())
    print()
    print(f"pooled across suites: {total_n} tasks, {100 * total_ok / total_n:.1f}% success")
    report["pooled"] = {"n": total_n, "success": total_ok, "rate": total_ok / total_n,
                        "categories": {k: {"n": v[1], "rate": v[0] / v[1]} for k, v in categories.items()},
                        "difficulties": {str(k): {"n": v[1], "rate": v[0] / v[1]}
                                         for k, v in difficulties.items()}}
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("WROTE", args.json)


if __name__ == "__main__":
    main()
