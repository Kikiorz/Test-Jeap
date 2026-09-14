#!/usr/bin/env python3
"""Turn a RoboTwin closed-loop suite summary into a PACE-table comparison.

The suite (`scripts/robotwin_eval_suite.sh`) appends one line per task/config to
`summary_<exp>_<step>.tsv`:

    adjust_bottle\tdemo_clean\tFinal batch success rate: 12/25 = 48.0%
    adjust_bottle\tdemo_randomized\tFAILED (see /path/to/client.log)

This prints the 20-task x {Clean, Random} table next to the JEPA-WAM paper's
published pi0.5 column, which is what the numbers are meant to be read against
(`result/PACE.md` in the paper worktree, average pi0.5 75.1 Clean / 36.7 Random).

Usage:
    python scripts/robotwin_eval_report.py <summary.tsv> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# (task key as the simulator names it, label, pi0.5 Clean, pi0.5 Random)
# pi0.5 is trained on the Clean demonstrations only; the Random column is the
# generalisation number the randomized finetune is trying to move.
PACE_PI05 = [
    ("adjust_bottle", "Adjust Bottle", 97, 26),
    ("beat_block_hammer", "Beat Block Hammer", 76, 9),
    ("click_alarmclock", "Click Alarmclock", 90, 63),
    ("click_bell", "Click Bell", 98, 58),
    ("dump_bin_bigbin", "Dump Bin Bigbin", 95, 42),
    ("grab_roller", "Grab Roller", 92, 64),
    ("handover_mic", "Handover Mic", 84, 8),
    ("lift_pot", "Lift Pot", 63, 4),
    ("place_bread_basket", "Place Bread Basket", 51, 27),
    ("place_bread_skillet", "Place Bread Skillet", 56, 20),
    ("place_burger_fries", "Place Burger Fries", 83, 54),
    ("place_cans_plasticbox", "Place Cans Plasticbox", 36, 42),
    ("place_empty_cup", "Place Empty Cup", 74, 59),
    ("place_object_basket", "Place Object Basket", 66, 8),
    ("place_shoe", "Place Shoe", 29, 15),
    ("press_stapler", "Press Stapler", 67, 22),
    ("shake_bottle_horizontally", "Shake Bottle Horizontally", 100, 61),
    ("shake_bottle", "Shake Bottle", 99, 82),
    ("stack_bowls_three", "Stack Bowls Three", 59, 29),
    ("stack_bowls_two", "Stack Bowls Two", 87, 40),
]

RATE_RE = re.compile(r"([0-9]+)\s*/\s*([0-9]+)\s*=\s*([0-9.]+)%")
CONFIG_KEYS = {"demo_clean": "clean", "demo_randomized": "random"}


def parse_summary(path: Path) -> dict[tuple[str, str], tuple[int, int]]:
    """(task, clean|random) -> (successes, trials); failures are left out."""
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for line in path.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        task, cfg, result = parts[0], parts[1], parts[2]
        key = CONFIG_KEYS.get(cfg)
        if key is None:
            continue
        m = RATE_RE.search(result)
        if m:
            out[(task, key)] = (int(m.group(1)), int(m.group(2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    got = parse_summary(args.summary)
    if not got:
        raise SystemExit(f"no parsable rows in {args.summary}")

    rows = []
    print(f"{'task':<26} {'ours Clean':>11} {'pi0.5 Clean':>11} {'ours Random':>11} {'pi0.5 Random':>12}")
    for task, label, p_clean, p_rand in PACE_PI05:
        c = got.get((task, "clean"))
        r = got.get((task, "random"))
        c_txt = f"{100 * c[0] / c[1]:.0f}% ({c[0]}/{c[1]})" if c else "-"
        r_txt = f"{100 * r[0] / r[1]:.0f}% ({r[0]}/{r[1]})" if r else "-"
        print(f"{label:<26} {c_txt:>11} {str(p_clean) + '%':>11} {r_txt:>11} {str(p_rand) + '%':>12}")
        rows.append(
            {
                "task": task,
                "label": label,
                "ours_clean": 100 * c[0] / c[1] if c else None,
                "pi05_clean": p_clean,
                "ours_random": 100 * r[0] / r[1] if r else None,
                "pi05_random": p_rand,
            }
        )

    # Paired means: a partial run must not compare our 15 finished tasks against
    # the paper's mean over all 20, so the baseline is averaged over exactly the
    # tasks where we have a number.
    def paired(ours_key: str, base_key: str) -> tuple[float | None, float | None, int]:
        rows_with = [r for r in rows if r[ours_key] is not None]
        if not rows_with:
            return None, None, 0
        ours = sum(r[ours_key] for r in rows_with) / len(rows_with)
        base = sum(r[base_key] for r in rows_with) / len(rows_with)
        return ours, base, len(rows_with)

    mc, pc, nc = paired("ours_clean", "pi05_clean")
    mr, pr, nr = paired("ours_random", "pi05_random")
    print()
    print(f"mean Clean : ours {mc:.1f}  vs pi0.5 {pc:.1f}  ({mc - pc:+.1f})  over {nc} tasks" if mc else "mean Clean : n/a")
    print(f"mean Random: ours {mr:.1f}  vs pi0.5 {pr:.1f}  ({mr - pr:+.1f})  over {nr} tasks" if mr else "mean Random: n/a")
    n_clean = sum(c[1] for (t, k), c in got.items() if k == "clean")
    n_rand = sum(c[1] for (t, k), c in got.items() if k == "random")
    print(f"episodes   : clean {n_clean}, random {n_rand}, tasks with rows {len({t for t, _ in got})}/20")

    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "mean_clean": mc, "mean_random": mr}, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
