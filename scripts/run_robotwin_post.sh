#!/usr/bin/env bash
# Run the third RoboTwin arm and then the three-way A/B/C probe.
#
# Arm C only makes sense once arms A and B are done, because it is a
# single-variable change against arm A (action conditioning of the delta head)
# and the probe has to compare all of them on the same batches.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
STEPS="${STEPS:-15000}"
LOG="${LOG:-/workspace/robotwin_post.log}"

echo "[$(date -u +%H:%M:%S)] waiting for the A/B continuation to finish" | tee -a "$LOG"
while pgrep -f "scripts/train.py" >/dev/null 2>&1 \
   || pgrep -f "run_robotwin_arms_continue.sh" >/dev/null 2>&1; do
  sleep 60
done
sleep 30
if pgrep -f "scripts/train.py" >/dev/null 2>&1; then
  echo "[$(date -u +%H:%M:%S)] training restarted while waiting; giving up" | tee -a "$LOG"
  exit 1
fi
echo "[$(date -u +%H:%M:%S)] A/B finished; starting arm C" | tee -a "$LOG"

# The probe is still worth running for A and B even if arm C cannot start, so
# failures here are reported rather than fatal.
STEPS="$STEPS" bash "$ROOT/scripts/run_robotwin_arm_c.sh" \
  || echo "[$(date -u +%H:%M:%S)] arm C failed; probing A and B only" | tee -a "$LOG"

LOG="$LOG" bash "$ROOT/scripts/robotwin_final_probe.sh" \
  || echo "[$(date -u +%H:%M:%S)] final probe failed" | tee -a "$LOG"

echo "[$(date -u +%H:%M:%S)] post-training pipeline done" | tee -a "$LOG"
