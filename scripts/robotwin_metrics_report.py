#!/usr/bin/env python3
"""Summarise the RoboTwin arm training metrics that matter for Con1/Con2.

Reads the per-arm `metrics.jsonl` written next to the checkpoints and prints a
sparse table so we can tell whether the latent head itself is converging
(`con1_delta_nmse`) and what that does to the action flow loss.
"""
import argparse
import json
from pathlib import Path

DEFAULT_ARMS = (
    "pi05_robotwin_con1_livecross_20k/robotwin_a_con1",
    "pi05_robotwin_con1con2_ctx_20k/robotwin_b_full",
)
COLUMNS = (
    "con1_alpha",
    "con1_delta_loss",
    "con1_delta_nmse",
    "con1_residual_energy",
    "flow_loss",
    "grad_norm",
    "loss",
)


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-base-dir", type=Path,
                   default=Path("/workspace/artifacts/checkpoints"))
    p.add_argument("--arms", nargs="*", default=list(DEFAULT_ARMS))
    p.add_argument("--rows", type=int, default=12)
    return p.parse_args()


def main() -> int:
    args = parse()
    for arm in args.arms:
        path = args.checkpoint_base_dir / arm / "metrics.jsonl"
        if not path.exists():
            print(f"== {arm}: no metrics.jsonl")
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            print(f"== {arm}: empty")
            continue
        present = [c for c in COLUMNS if c in rows[-1]]
        stride = max(1, len(rows) // args.rows)
        print(f"== {arm}  ({len(rows)} records, steps {rows[0].get('step')}..{rows[-1].get('step')})")
        header = f"{'step':>7} " + " ".join(f"{c:>12}" for c in present)
        print(header)
        for row in rows[::stride] + [rows[-1]]:
            values = " ".join(f"{row.get(c, float('nan')):>12.5f}" for c in present)
            print(f"{row.get('step', -1):>7} {values}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
