#!/usr/bin/env python3
"""Final Con1-vs-official LIBERO-Plus L5 report with paired significance.

McNemar exact test on the discordant pairs per category, plus a pooled figure
that also reports the category-macro average so the bigger suites (Camera,
Sensor Noise) do not silently dominate.
"""

from __future__ import annotations

import json
from math import comb
from pathlib import Path

from summarize_l5_pairs import CATEGORIES, load


def mcnemar_exact(a_only: int, b_only: int) -> float:
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    root = Path("data/libero-eval")
    rows = {}
    print(f"{'category':24s} {'n':>5s} {'adapter':>9s} {'baseline':>9s} {'delta':>8s} "
          f"{'+/-':>9s} {'p(McNemar)':>11s}")
    for category, slug in CATEGORIES:
        adapter = load(root, "adapter", slug)
        baseline = load(root, "baseline", slug)
        keys = sorted(set(adapter) & set(baseline))
        if not keys:
            continue
        a = sum(adapter[k] for k in keys)
        b = sum(baseline[k] for k in keys)
        a_only = sum(1 for k in keys if adapter[k] and not baseline[k])
        b_only = sum(1 for k in keys if baseline[k] and not adapter[k])
        p = mcnemar_exact(a_only, b_only)
        n = len(keys)
        rows[category] = {"n": n, "adapter": a, "baseline": b, "adapter_only": a_only,
                          "baseline_only": b_only, "p": p}
        print(f"{category:24s} {n:5d} {100 * a / n:8.1f}% {100 * b / n:8.1f}% "
              f"{100 * (a - b) / n:+7.1f}pp {a_only:4d}/{b_only:<4d} {p:11.4f}")

    total_n = sum(r["n"] for r in rows.values())
    total_a = sum(r["adapter"] for r in rows.values())
    total_b = sum(r["baseline"] for r in rows.values())
    ao = sum(r["adapter_only"] for r in rows.values())
    bo = sum(r["baseline_only"] for r in rows.values())
    macro = sum((r["adapter"] - r["baseline"]) / r["n"] for r in rows.values()) / len(rows)
    print()
    print(f"pooled  n={total_n}  adapter {100 * total_a / total_n:.1f}%  "
          f"baseline {100 * total_b / total_n:.1f}%  "
          f"delta {100 * (total_a - total_b) / total_n:+.2f}pp  "
          f"discordant {ao}/{bo}  p={mcnemar_exact(ao, bo):.4f}")
    print(f"category-macro delta {100 * macro:+.2f}pp")
    Path("/workspace/artifacts/con2/l5_pairs_final.json").write_text(
        json.dumps({"categories": rows, "pooled": {
            "n": total_n, "adapter": total_a, "baseline": total_b,
            "adapter_only": ao, "baseline_only": bo,
            "p": mcnemar_exact(ao, bo), "macro_delta": macro}}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
