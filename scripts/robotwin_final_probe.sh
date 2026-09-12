#!/usr/bin/env bash
# Probe the final checkpoint of every RoboTwin arm once training has stopped.
#
# The arm continuation script saves every KEEP_PERIOD steps and writes one last
# checkpoint whose number is the final zero-indexed training step, so the newest
# numeric directory is the converged checkpoint we want to compare.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
CKPT_BASE="${CKPT_BASE:-/workspace/artifacts/checkpoints}"
OUT_DIR="${OUT_DIR:-/workspace/artifacts/con2}"
LOG="${LOG:-/workspace/robotwin_arms_continue.log}"
BATCHES="${BATCHES:-48}"
BATCH_SIZE="${BATCH_SIZE:-8}"

A_DIR="$CKPT_BASE/pi05_robotwin_con1_livecross_20k/robotwin_a_con1"
B_DIR="$CKPT_BASE/pi05_robotwin_con1con2_ctx_20k/robotwin_b_full"
C_DIR="$CKPT_BASE/pi05_robotwin_con1_actcond_20k/robotwin_c_actcond"
D_DIR="$CKPT_BASE/pi05_robotwin_con1_headlr_20k/robotwin_d_headlr"

latest_step() {
  ls -1 "$1" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1
}

# Every arm that produced a checkpoint contributes to the comparison, so probe
# the largest step that all of them share rather than each arm's own last step.
common_step() {
  local shared=""
  local dir arm_steps
  for dir in "$@"; do
    [[ -d "$dir" ]] || continue
    # `comm` needs lexicographic order, not `sort -n`; only the final pick is
    # numeric. Using `sort -n` here silently produced wrong intersections.
    arm_steps=$(ls -1 "$dir" 2>/dev/null | grep -E '^[0-9]+$' | sort)
    [[ -n "$arm_steps" ]] || continue
    if [[ -z "$shared" ]]; then
      shared="$arm_steps"
    else
      shared=$(comm -12 <(printf '%s\n' "$shared") <(printf '%s\n' "$arm_steps"))
    fi
  done
  printf '%s\n' "$shared" | grep -E '^[0-9]+$' | sort -n | tail -1
}

# Fallback when the arms share no save step at all: compare at the earliest of
# their last steps, so a short arm degrades the comparison instead of skipping it.
fallback_step() {
  local dir step
  local best=""
  for dir in "$@"; do
    [[ -d "$dir" ]] || continue
    step=$(latest_step "$dir")
    [[ -n "$step" ]] || continue
    if [[ -z "$best" || "$step" -lt "$best" ]]; then
      best="$step"
    fi
  done
  printf '%s\n' "$best"
}

wait_for_finish() {
  # The runner hands over from arm A to arm B with a short gap, so require the
  # runner itself to be gone as well and then re-check after a settle delay.
  while pgrep -f "scripts/train.py" >/dev/null 2>&1 \
     || pgrep -f "run_robotwin_arms_continue.sh" >/dev/null 2>&1; do
    sleep 60
  done
  sleep 30
  if pgrep -f "scripts/train.py" >/dev/null 2>&1; then
    wait_for_finish
  fi
}

wait_for_finish

A_STEP=$(latest_step "$A_DIR")
B_STEP=$(latest_step "$B_DIR")
C_STEP=$(latest_step "$C_DIR")
D_STEP=$(latest_step "$D_DIR")
echo "[$(date -u +%H:%M:%S)] training stopped; final steps A=$A_STEP B=$B_STEP " \
     "C=${C_STEP:-none} D=${D_STEP:-none}" | tee -a "$LOG"

STEP=$(common_step "$A_DIR" "$B_DIR" "$C_DIR" "$D_DIR")
if [[ -z "$STEP" ]]; then
  STEP=$(fallback_step "$A_DIR" "$B_DIR" "$C_DIR" "$D_DIR")
  echo "[$(date -u +%H:%M:%S)] arms share no save step; falling back to $STEP" | tee -a "$LOG"
fi
echo "[$(date -u +%H:%M:%S)] probing every arm at the shared step $STEP" | tee -a "$LOG"
if [[ -z "$STEP" ]]; then
  echo "[$(date -u +%H:%M:%S)] no checkpoints found; skipping probe" | tee -a "$LOG"
  exit 1
fi

STEPS="$STEP" BATCHES="$BATCHES" BATCH_SIZE="$BATCH_SIZE" OUT_DIR="$OUT_DIR" \
  bash "$ROOT/scripts/run_robotwin_ab_probe.sh" 2>&1 | tee -a "$LOG"

python3 - "$OUT_DIR" <<'PY' | tee -a "$LOG"
import json
import math
import sys
from pathlib import Path

out = Path(sys.argv[1])
entries = {}
for name in ("basetrue", "base", "a", "b", "c", "d"):
    path = out / f"robotwin_ab_{name}.json"
    entries[name] = json.loads(path.read_text()) if path.exists() else None

# A probe file stores one [mean, std] pair per budget under "flow" and
# "correction_rms"; there is a single 0.05 budget in these runs.
def series(entry, field):
    if not entry:
        return None, None
    per_budget = entry.get(field) or {}
    if not per_budget:
        return None, None
    key = next(iter(per_budget))
    mean, std = per_budget[key]
    return mean, std

label = {
    "basetrue": "released base",
    "base": "base (Con1 silenced)",
    "a": "arm A (Con1)",
    "b": "arm B (Con1+Con2+ctx)",
    "c": "arm C (Con1+action-cond)",
    "d": "arm D (Con1+head 5x LR)",
}
flows = {}
print("=" * 68)
for name in ("basetrue", "base", "a", "b", "c", "d"):
    mean, std = series(entries[name], "flow")
    rms, _ = series(entries[name], "correction_rms")
    if mean is None:
        print(f"{label[name]:26s} missing")
        continue
    flows[name] = mean
    print(f"{label[name]:26s} flow={mean:.6f} +-{std:.6f}  corr_rms={rms:.4f}")
if "base" in flows and "a" in flows:
    print(f"arm A vs base        : {(flows['a'] / flows['base'] - 1) * 100:+.2f}%")
if "basetrue" in flows and "base" in flows:
    print(f"base vs released     : {(flows['base'] / flows['basetrue'] - 1) * 100:+.2f}%")
if "basetrue" in flows and "a" in flows:
    print(f"arm A vs released    : {(flows['a'] / flows['basetrue'] - 1) * 100:+.2f}%")
if "base" in flows and "b" in flows:
    print(f"arm B vs base        : {(flows['b'] / flows['base'] - 1) * 100:+.2f}%")
if "a" in flows and "b" in flows:
    print(f"arm B vs arm A       : {(flows['b'] / flows['a'] - 1) * 100:+.2f}%")
if "base" in flows and "c" in flows:
    print(f"arm C vs base        : {(flows['c'] / flows['base'] - 1) * 100:+.2f}%")
if "a" in flows and "c" in flows:
    print(f"arm C vs arm A       : {(flows['c'] / flows['a'] - 1) * 100:+.2f}%")
if "base" in flows and "d" in flows:
    print(f"arm D vs base        : {(flows['d'] / flows['base'] - 1) * 100:+.2f}%")
if "a" in flows and "d" in flows:
    print(f"arm D vs arm A       : {(flows['d'] / flows['a'] - 1) * 100:+.2f}%")

# Paired t statistic over the shared per-batch records, which is what decides
# whether the difference is real at this budget. Each record carries one entry
# per budget under names such as "flow_b0.05_true".
def per_batch(entry):
    if not entry:
        return []
    records = entry.get("records") or []
    if not records:
        return []
    key = next((k for k in records[0] if k.startswith("flow_") and k.endswith("_true")), None)
    if key is None:
        key = next((k for k in records[0] if k.startswith("flow_")), None)
    return [record.get(key) for record in records]

for left, right in (("a", "base"), ("b", "base"), ("b", "a"), ("c", "base"), ("c", "a"),
                    ("d", "base"), ("d", "a"), ("a", "basetrue"), ("base", "basetrue")):
    lv, rv = per_batch(entries[left]), per_batch(entries[right])
    if not lv or not rv or len(lv) != len(rv):
        continue
    diff = [x - y for x, y in zip(lv, rv)]
    n = len(diff)
    mean = sum(diff) / n
    var = sum((d - mean) ** 2 for d in diff) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if var > 0 else 0.0
    t = mean / se if se > 0 else float("nan")
    print(f"paired {label[left]} - {label[right]}: d={mean:+.2e} t={t:.2f} (n={n})")
print("=" * 68)
PY

echo "[$(date -u +%H:%M:%S)] final probe done" | tee -a "$LOG"
