"""Paired LIBERO-Plus comparison between two served checkpoints.

Reads the incremental result journals written by examples/libero/main.py and
reports, per run pair: task counts, successes, and the discordant tasks
(adapter-only / baseline-only). Both sides must use the same task panel, seed
and replan interval; the only intended difference is the policy.

Usage:
    python scripts/paired_libero_plus_summary.py RUN_A RUN_B [--suite libero_10]
where RUN_A/RUN_B are RUN_IDs under data/libero-eval/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_journal(path: Path) -> dict[int, bool]:
    results: dict[int, bool] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        task_id = record.get("task_id", record.get("task_index"))
        if task_id is None:
            continue
        results[int(task_id)] = bool(record.get("success"))
    return results


def journal_for(root: Path, run_id: str, suite: str, mode: str) -> Path:
    return root / run_id / f"{mode}-{suite}.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--root", default="data/libero-eval")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--mode", default="plus")
    args = parser.parse_args()

    root = Path(args.root)
    a_path = journal_for(root, args.run_a, args.suite, args.mode)
    b_path = journal_for(root, args.run_b, args.suite, args.mode)
    a, b = load_journal(a_path), load_journal(b_path)
    common = sorted(set(a) & set(b))
    a_only = [t for t in common if a[t] and not b[t]]
    b_only = [t for t in common if b[t] and not a[t]]
    both = [t for t in common if a[t] and b[t]]
    neither = [t for t in common if not a[t] and not b[t]]
    report = {
        "run_a": args.run_a,
        "run_b": args.run_b,
        "suite": args.suite,
        "records": {"run_a": len(a), "run_b": len(b)},
        "common": len(common),
        "successes": {"run_a": sum(a[t] for t in common), "run_b": sum(b[t] for t in common)},
        "rate": {k: (v / len(common) if common else None) for k, v in
                 (("run_a", sum(a[t] for t in common)), ("run_b", sum(b[t] for t in common)))},
        "paired": {"a_only": a_only, "b_only": b_only, "both": len(both), "neither": len(neither)},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
