#!/usr/bin/env bash
# Train arms C and D, then run the full ten-row probe at their shared step.
#
# Separate from run_robotwin_post.sh because that script also runs the early A/B
# probe first, and those results already exist in con2/step15k. Launching this
# after C and D have been cleaned up is the fast path to the complete table.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
STEPS="${STEPS:-9000}"
LOG="${LOG:-/workspace/robotwin_cd.log}"

cd "$ROOT"
echo "[$(date -u +%H:%M:%S)] starting arm C ($STEPS steps)" | tee -a "$LOG"
STEPS="$STEPS" bash "$ROOT/scripts/run_robotwin_arm_c.sh" \
  || echo "[$(date -u +%H:%M:%S)] arm C failed; continuing" | tee -a "$LOG"

echo "[$(date -u +%H:%M:%S)] starting arm D ($STEPS steps)" | tee -a "$LOG"
STEPS="$STEPS" bash "$ROOT/scripts/run_robotwin_arm_d.sh" \
  || echo "[$(date -u +%H:%M:%S)] arm D failed; continuing" | tee -a "$LOG"

# Wait for the last trainer to release its devices before the probe builds a
# model of its own; the first attempt died on a 143 MB allocation otherwise.
for _ in $(seq 1 20); do
  busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
  (( busy == 0 )) && break
  echo "[$(date -u +%H:%M:%S)] $busy GPU(s) still busy; waiting" | tee -a "$LOG"
  sleep 30
done

echo "[$(date -u +%H:%M:%S)] full probe" | tee -a "$LOG"
LOG="$LOG" bash "$ROOT/scripts/robotwin_final_probe.sh" \
  || echo "[$(date -u +%H:%M:%S)] full probe failed" | tee -a "$LOG"

echo "[$(date -u +%H:%M:%S)] C/D and probe pipeline done" | tee -a "$LOG"
