#!/usr/bin/env bash
# Run the third RoboTwin arm and then the three-way A/B/C probe.
#
# Arm C only makes sense once arms A and B are done, because it is a
# single-variable change against arm A (action conditioning of the delta head)
# and the probe has to compare all of them on the same batches.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
# Arms C and D read their own defaults (9,000); this only forwards an explicit
# override. It used to default to 15,000, which overrode the arm scripts' 9,000
# and silently made C and D twice as long as planned.
STEPS="${STEPS:-9000}"
LOG="${LOG:-/workspace/robotwin_post.log}"

# Wait for the previous trainer to actually release the GPUs. It exits its
# process before its XLA allocator is gone, and the first early probe died with
# a 143 MB allocation failure against a still-occupied device.
wait_for_gpu() {
  local tries=0 busy
  while (( tries < 20 )); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 > 2000' | wc -l)
    if (( busy == 0 )); then
      return 0
    fi
    echo "[$(date -u +%H:%M:%S)] $busy GPU(s) still above 2 GB; waiting" | tee -a "$LOG"
    sleep 30
    tries=$((tries + 1))
  done
  echo "[$(date -u +%H:%M:%S)] GPUs still busy after 10 minutes; proceeding anyway" | tee -a "$LOG"
  return 0
}

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
echo "[$(date -u +%H:%M:%S)] A/B finished" | tee -a "$LOG"
wait_for_gpu

# Report A and B first instead of waiting for C and D to train. Only A and B
# have checkpoints at this point, so the shared step is theirs (15k) and the
# rows for C and D are skipped automatically. Written to its own directory so
# the canonical 12k/9k probe later does not overwrite these files.
echo "[$(date -u +%H:%M:%S)] early probe: A and B at their shared step" | tee -a "$LOG"
OUT_DIR="${OUT_DIR:-/workspace/artifacts/con2}/step15k" \
  LOG="$LOG" bash "$ROOT/scripts/robotwin_final_probe.sh" \
  || echo "[$(date -u +%H:%M:%S)] early A/B probe failed; continuing" | tee -a "$LOG"

echo "[$(date -u +%H:%M:%S)] starting arms C and D" | tee -a "$LOG"

# The probe is still worth running for A and B even if arm C cannot start, so
# failures here are reported rather than fatal.
STEPS="$STEPS" bash "$ROOT/scripts/run_robotwin_arm_c.sh" \
  || echo "[$(date -u +%H:%M:%S)] arm C failed; probing A and B only" | tee -a "$LOG"

STEPS="$STEPS" bash "$ROOT/scripts/run_robotwin_arm_d.sh" \
  || echo "[$(date -u +%H:%M:%S)] arm D failed; probing the arms that exist" | tee -a "$LOG"

echo "[$(date -u +%H:%M:%S)] full probe for all arms" | tee -a "$LOG"
LOG="$LOG" bash "$ROOT/scripts/robotwin_final_probe.sh" \
  || echo "[$(date -u +%H:%M:%S)] final probe failed" | tee -a "$LOG"

echo "[$(date -u +%H:%M:%S)] post-training pipeline done" | tee -a "$LOG"
