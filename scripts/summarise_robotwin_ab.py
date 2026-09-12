#!/usr/bin/env python3
"""Print the RoboTwin arm comparison table from the probe JSON files.

Kept separate from the probe runner so the table can be regenerated from stored
results without touching the GPUs, and so it can be checked against a run whose
answer is already known.

Each probe file stores one [mean, std] pair per residual budget under "flow" and
"correction_rms", plus one record per paired batch. The same 48 batches, the same
loader and the same seed are used for every row, so the differences can be tested
with a paired t statistic over the per-batch flow losses.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LABELS = {
    "basetrue": "released base",
    "base": "base (Con1 silenced)",
    "a": "arm A (Con1)",
    "b": "arm B (Con1+Con2+ctx)",
    "c": "arm C (Con1+action-cond)",
    "d": "arm D (Con1+head x5 LR)",
    "a_graft": "arm A + offline head (diagnostic)",
}
ORDER = ("basetrue", "base", "a", "a_graft", "b", "c", "d")
COMPARISONS = (("a", "base"), ("b", "base"), ("b", "a"), ("c", "base"), ("c", "a"),
               ("d", "base"), ("d", "a"), ("a_graft", "a"), ("a_graft", "base"),
               ("a", "basetrue"), ("base", "basetrue"))


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("/workspace/artifacts/con2"),
                   help="directory holding robotwin_ab_<arm>.json")
    p.add_argument("--suffix", default="", help="read robotwin_ab_<arm><suffix>.json instead")
    p.add_argument("--json-out", type=Path, help="also write the parsed table as JSON")
    return p.parse_args()


def series(entry, field):
    if not entry:
        return None, None
    per_budget = entry.get(field) or {}
    if not per_budget:
        return None, None
    key = next(iter(per_budget))
    return per_budget[key][0], per_budget[key][1]


def per_batch(entry):
    records = (entry or {}).get("records") or []
    if not records:
        return []
    key = next((k for k in records[0] if k.startswith("flow_") and k.endswith("_true")), None)
    if key is None:
        key = next((k for k in records[0] if k.startswith("flow_")), None)
    return [record.get(key) for record in records]


def main() -> None:
    args = parse()
    entries = {}
    steps = {}
    for name in ORDER:
        path = args.dir / f"robotwin_ab_{name}{args.suffix}.json"
        if not path.exists():
            entries[name] = None
            continue
        entries[name] = json.loads(path.read_text())
        steps[name] = entries[name].get("restored_step")

    flows = {}
    print("=" * 72)
    for name in ORDER:
        mean, std = series(entries[name], "flow")
        if mean is None:
            print(f"{LABELS[name]:26s} missing")
            continue
        rms, _ = series(entries[name], "correction_rms")
        flows[name] = mean
        step = steps.get(name)
        print(f"{LABELS[name]:26s} flow={mean:.6f} +-{std:.6f}  "
              f"corr_rms={rms:.4f}  step={step}")

    for left, right in COMPARISONS:
        if left in flows and right in flows:
            print(f"{LABELS[left]:26s} vs {LABELS[right]:26s}: "
                  f"{(flows[left] / flows[right] - 1) * 100:+.2f}%")

    print("-" * 72)
    paired = {}
    for left, right in COMPARISONS:
        lv, rv = per_batch(entries[left]), per_batch(entries[right])
        if not lv or not rv or len(lv) != len(rv):
            continue
        diff = [float(x) - float(y) for x, y in zip(lv, rv)]
        n = len(diff)
        mean = sum(diff) / n
        var = sum((d - mean) ** 2 for d in diff) / (n - 1) if n > 1 else 0.0
        se = math.sqrt(var / n) if var > 0 else 0.0
        t = mean / se if se > 0 else float("nan")
        paired[f"{left}_vs_{right}"] = {"delta": mean, "t": t, "n": n}
        print(f"paired {LABELS[left]:26s} - {LABELS[right]:26s}: "
              f"d={mean:+.2e} t={t:+.2f} (n={n})")
    print("=" * 72)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"flows": flows, "steps": steps, "paired": paired,
             "relative": {f"{l}_vs_{r}": flows[l] / flows[r] - 1
                          for l, r in COMPARISONS if l in flows and r in flows}},
            indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
