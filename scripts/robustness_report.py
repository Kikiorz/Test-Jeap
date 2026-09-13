#!/usr/bin/env python3
"""Holm-Bonferroni over the pre-registered RoboTwin comparisons.

The probe table reports ~ten paired comparisons, and a bare 't = -2.99' is not a
claim once that many tests are run. This recomputes each one from the stored
per-batch flow losses and applies a Holm correction, so the surviving statements
are the ones that can be written down without qualification.

The family is the set of comparisons named in docs_ROBOTWIN_READING_THE_TABLE.md,
fixed before the numbers arrived.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import scipy.stats

FAMILY = (
    ("a", "base"),
    ("b", "base"),
    ("c", "base"),
    ("d", "base"),
    ("b", "a"),
    ("d", "a"),
    ("a_graft", "a"),
    ("a_only_residual", "a"),
    ("a_only_adapter", "a"),
)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("/workspace/artifacts/con2/hp"))
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def flows(path: Path):
    if not path.exists():
        return None
    entry = json.loads(path.read_text())
    records = entry.get("records") or []
    if not records:
        return None
    key = next((k for k in records[0] if k.startswith("flow_") and k.endswith("_true")), None)
    return [record[key] for record in records]


def main() -> None:
    args = parse()
    series = {}
    for name in {n for pair in FAMILY for n in pair}:
        values = flows(args.dir / f"robotwin_ab_{name}.json")
        if values is not None:
            series[name] = values

    results = []
    for left, right in FAMILY:
        if left not in series or right not in series:
            continue
        lv, rv = series[left], series[right]
        if len(lv) != len(rv):
            continue
        diff = [a - b for a, b in zip(lv, rv)]
        n = len(diff)
        mean = sum(diff) / n
        var = sum((d - mean) ** 2 for d in diff) / (n - 1)
        se = math.sqrt(var / n)
        t = mean / se if se > 0 else float("nan")
        p = 2 * scipy.stats.t.sf(abs(t), df=n - 1)
        results.append({"pair": f"{left} vs {right}", "t": t, "p": p,
                        "relative": (sum(lv) / sum(rv) - 1) * 100, "n": n})

    results.sort(key=lambda r: r["p"])
    m = len(results)
    print(f"family size {m}, alpha {args.alpha}, Holm-Bonferroni")
    print(f"{'comparison':>22} {'rel':>8} {'t':>7} {'p':>10} {'Holm thresh':>12} verdict")
    rejected = 0
    for index, row in enumerate(results):
        threshold = args.alpha / (m - index)
        ok = row["p"] < threshold
        if ok and rejected == index:
            rejected += 1
            verdict = "survives"
        else:
            verdict = "not significant"
        print(f"{row['pair']:>22} {row['relative']:>7.2f}% {row['t']:>7.2f} "
              f"{row['p']:>10.5f} {threshold:>12.5f} {verdict}")
    print()
    print(f"{rejected} of {m} comparisons survive the correction; n = {results[0]['n']} batches")


if __name__ == "__main__":
    main()
