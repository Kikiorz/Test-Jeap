#!/usr/bin/env bash
# Unattended chain: wait for the RoboTwin randomized finetune to finish, then run
# the 20-task Clean/Random closed-loop evaluation on the final checkpoint.
#
# Env: RUN_NAME, CKPT_ROOT, TRIALS, WORKERS, POLL_SECONDS, LOG.
set -uo pipefail

ROOT=${ROOT:-/workspace/robotwin_ws}
RUN_NAME=${RUN_NAME:-robotwin_random20_ft}
CKPT_ROOT=${CKPT_ROOT:-/workspace/artifacts/checkpoints_robotwin_random}
CKPT_DIR="${CKPT_ROOT}/${RUN_NAME}"
TRIALS=${TRIALS:-20}
WORKERS=${WORKERS:-8}
POLL_SECONDS=${POLL_SECONDS:-120}
LOG=${LOG:-/workspace/robotwin_ft_then_eval.log}

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "waiting for the trainer (${RUN_NAME}) to exit"
while pgrep -f "${RUN_NAME}" >/dev/null 2>&1; do
  sleep "$POLL_SECONDS"
done
log "trainer is gone; last log lines:"
tail -3 /workspace/robotwin_random_ft.log | tr '\r' '\n' | tail -3 | tee -a "$LOG"

STEP=$(ls -1 "$CKPT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
if [[ -z "$STEP" ]]; then
  log "no checkpoint under $CKPT_DIR; aborting"
  exit 1
fi
log "final checkpoint step=${STEP}; starting the evaluation suite (${TRIALS} trials/task, ${WORKERS} parallel sims)"

TRIALS="$TRIALS" WORKERS="$WORKERS" STEP="$STEP" CKPT_DIR="$CKPT_DIR" \
  bash "$ROOT/scripts/robotwin_eval_suite.sh" >>"$LOG" 2>&1

log "evaluation finished; summary:"
cat "/workspace/robotwin_eval_results/summary_${RUN_NAME}_${STEP}.tsv" 2>/dev/null | tee -a "$LOG"
