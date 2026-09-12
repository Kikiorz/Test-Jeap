#!/usr/bin/env python3
"""Paired LIBERO-Plus summary over the whole suite.

Pairs the candidate and reference journals on ``(task_id, episode_idx)`` and
reports the success rate per perturbation category, per difficulty level, and
overall, with the discordant counts that drive each difference.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from math import comb
from pathlib import Path


def load(root: Path, side: str, suite: str) -> dict:
    records = {}
    pattern = root / f"{side}_plus_full" / f"plus-{suite}.shard-*.jsonl"
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
                records[(record["task_id"], record.get("episode_idx"))] = record
    return records


def mcnemar(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def group(keys, cand, ref, field):
    buckets = defaultdict(list)
    for key in keys:
        buckets[cand[key].get(field) or "unknown"].append(key)
    return buckets


def line(label, keys, cand, ref, width=34):
    if not keys:
        return None
    a = sum(bool(cand[k].get("success")) for k in keys)
    b = sum(bool(ref[k].get("success")) for k in keys)
    a_only = sum(1 for k in keys if cand[k].get("success") and not ref[k].get("success"))
    b_only = sum(1 for k in keys if ref[k].get("success") and not cand[k].get("success"))
    n = len(keys)
    return (f"{label:<{width}} {n:5d} {100 * a / n:8.1f}% {100 * b / n:8.1f}% "
            f"{100 * (a - b) / n:+8.1f}pp {a_only:5d}/{b_only:<5d} {mcnemar(a_only, b_only):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/libero-eval"))
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--candidate", default="con1con2")
    parser.add_argument("--reference", default="baseline")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    cand = load(args.root, args.candidate, args.suite)
    ref = load(args.root, args.reference, args.suite)
    keys = sorted(set(cand) & set(ref))
    if not keys:
        print("no paired episodes yet")
        return

    header = f"{'group':<34} {'n':>5} {'cand':>9} {'ref':>9} {'delta':>9} {'+/-':>11} {'p':>7}"
    print(header)
    print("-" * len(header))
    report = {"candidate": args.candidate, "reference": args.reference,
              "paired": len(keys), "categories": {}, "difficulties": {}}
    for label, field in (("category", "category"), ("difficulty", "difficulty_level")):
        for name, bucket in sorted(group(keys, cand, ref, field).items(), key=lambda kv: str(kv[0])):
            text = line(f"{label}: {name}", bucket, cand, ref)
            if text:
                print(text)
                report[f"{label}s"][str(name)] = {
                    "n": len(bucket),
                    "candidate": sum(bool(cand[k].get("success")) for k in bucket),
                    "reference": sum(bool(ref[k].get("success")) for k in bucket),
                }
        print()
    text = line("OVERALL", keys, cand, ref)
    print(text)
    report["overall"] = {
        "n": len(keys),
        "candidate": sum(bool(cand[k].get("success")) for k in keys),
        "reference": sum(bool(ref[k].get("success")) for k in keys),
    }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print("WROTE", args.json)


if __name__ == "__main__":
    main()
