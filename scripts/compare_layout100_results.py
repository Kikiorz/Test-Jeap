#!/usr/bin/env python3
"""Compare two sealed Layout100 evaluations with paired statistics."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
from collections import defaultdict
from typing import Any


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def _load_run(root: pathlib.Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    records: dict[tuple[str, int, int], dict[str, Any]] = {}
    for suite in SUITES:
        path = root / f"{suite}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing result journal: {path}")
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                if record.get("record_type") != "episode":
                    continue
                key = (record["task_suite_name"], record["task_id"], record["episode_idx"])
                if key in records:
                    raise ValueError(f"Duplicate episode {key} in {path}:{line_number}")
                if record["status"] == "error":
                    raise ValueError(f"Evaluation error for {key}: {record.get('error')}")
                records[key] = record
    if len(records) != 100:
        raise ValueError(f"Expected exactly 100 Layout episodes in {root}, found {len(records)}")
    return records


def _mcnemar_exact(baseline_only: int, method_only: int) -> float:
    discordant = baseline_only + method_only
    if discordant == 0:
        return 1.0
    smaller = min(baseline_only, method_only)
    lower_tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * lower_tail)


def _bootstrap_ci(differences: list[int], replicates: int, seed: int) -> tuple[float, float]:
    generator = random.Random(seed)
    n = len(differences)
    samples = sorted(
        sum(differences[generator.randrange(n)] for _ in range(n)) / n for _ in range(replicates)
    )
    low = samples[int(0.025 * (replicates - 1))]
    high = samples[int(0.975 * (replicates - 1))]
    return low, high


def _group_summary(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    baseline_successes = sum(bool(left["success"]) for left, _ in pairs)
    method_successes = sum(bool(right["success"]) for _, right in pairs)
    total = len(pairs)
    return {
        "episodes": total,
        "baseline_successes": baseline_successes,
        "method_successes": method_successes,
        "baseline_success_rate": baseline_successes / total,
        "method_success_rate": method_successes / total,
        "paired_difference": (method_successes - baseline_successes) / total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline_root", type=pathlib.Path)
    parser.add_argument("method_root", type=pathlib.Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    baseline = _load_run(args.baseline_root)
    method = _load_run(args.method_root)
    if baseline.keys() != method.keys():
        missing_method = sorted(baseline.keys() - method.keys())
        missing_baseline = sorted(method.keys() - baseline.keys())
        raise ValueError(
            f"Paired episode keys differ: missing_method={missing_method}, missing_baseline={missing_baseline}"
        )

    pairs = [(baseline[key], method[key]) for key in sorted(baseline)]
    for left, right in pairs:
        paired_fields = ("task_name", "task_description", "difficulty_level", "episode_seed", "max_steps")
        for field in paired_fields:
            if left.get(field) != right.get(field):
                raise ValueError(f"Pair mismatch for {field}: {left.get(field)!r} != {right.get(field)!r}")

    baseline_only = sum(bool(left["success"]) and not bool(right["success"]) for left, right in pairs)
    method_only = sum(not bool(left["success"]) and bool(right["success"]) for left, right in pairs)
    differences = [int(bool(right["success"])) - int(bool(left["success"])) for left, right in pairs]
    ci_low, ci_high = _bootstrap_ci(differences, args.bootstrap_replicates, args.seed)

    by_suite: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    by_difficulty: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        by_suite[pair[0]["task_suite_name"]].append(pair)
        by_difficulty[str(pair[0]["difficulty_level"])].append(pair)

    report = {
        "baseline_root": str(args.baseline_root.resolve()),
        "method_root": str(args.method_root.resolve()),
        "overall": _group_summary(pairs),
        "discordant_pairs": {"baseline_only": baseline_only, "method_only": method_only},
        "mcnemar_exact_two_sided_p": _mcnemar_exact(baseline_only, method_only),
        "paired_bootstrap_95_ci": [ci_low, ci_high],
        "bootstrap_replicates": args.bootstrap_replicates,
        "by_suite": {name: _group_summary(group) for name, group in sorted(by_suite.items())},
        "by_difficulty": {name: _group_summary(group) for name, group in sorted(by_difficulty.items())},
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
