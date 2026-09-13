#!/usr/bin/env python3
"""One line per arm: the latent quality the pre-registered rule asks about.

The rule fixed before the numbers arrived was "arm D's mean con1_delta_nmse is at
least 0.05 below arm A's". This prints that quantity for every arm at the shared
probe step, plus the best it ever reached, so the audit can be completed without
re-reading four metrics files by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_ARMS = {
    "A": "pi05_robotwin_con1_livecross_20k/robotwin_a_con1",
    "B": "pi05_robotwin_con1con2_ctx_20k/robotwin_b_full",
    "C": "pi05_robotwin_con1_actcond_20k/robotwin_c_actcond",
    "D": "pi05_robotwin_con1_headlr_20k/robotwin_d_headlr",
    "E": "pi05_robotwin_con1_ctxheadlr_20k/robotwin_e_ctxheadlr",
}


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, default=Path("/workspace/artifacts/checkpoints"))
    p.add_argument("--step", type=int, default=9000, help="shared probe step")
    p.add_argument("--window", type=int, default=50)
    return p.parse_args()


def main() -> None:
    args = parse()
    print(f"shared probe step {args.step}")
    print(f"{'arm':>3} {'step':>6} {'delta_nmse@step':>16} {'best':>8} {'flow_loss':>10}")
    values = {}
    for arm, rel in DEFAULT_ARMS.items():
        path = args.base / rel / "metrics.jsonl"
        if not path.exists():
            print(f"{arm:>3}  (no metrics)")
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        near = [r for r in rows if abs(r["completed_updates"] - args.step) <= args.window]
        row = near[-1] if near else rows[-1]
        best = min(r["con1_delta_nmse"] for r in rows)
        values[arm] = row["con1_delta_nmse"]
        print(f"{arm:>3} {row['completed_updates']:>6} {row['con1_delta_nmse']:>16.4f} "
              f"{best:>8.4f} {row['flow_loss']:>10.5f}")

    if "A" in values:
        print()
        for arm in ("B", "C", "D"):
            if arm in values:
                gain = values["A"] - values[arm]
                verdict = "meets the >=0.05 rule" if gain >= 0.05 else "below the 0.05 rule"
                print(f"{arm} vs A: {gain:+.4f} NMSE  ({verdict})")


if __name__ == "__main__":
    main()
