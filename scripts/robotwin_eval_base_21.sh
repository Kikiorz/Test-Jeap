#!/usr/bin/env bash
# Control run for the RoboTwin closed-loop numbers: evaluate the *un-finetuned*
# pi0.5 base on the same 20 tasks, Clean and Random, through the same policy
# server and the same simulator.
#
# Why: the finetune's headline number only means something next to what the base
# already scores on this exact harness. The paper's pi0.5 column is not that
# control - it is a pi0.5 trained on RoboTwin *clean* demonstrations, so it
# differs from our base in both data and recipe.
#
# The base checkpoint is already laid out like a checkpoint step directory
# (`params/` + `assets/`), so it only needs a numeric parent entry to satisfy the
# server's `model_path`/`checkpoint_num` convention.
#
# Usage: bash scripts/robotwin_eval_base_21.sh
# Env: TRIALS WORKERS POLICY_GPU ENV_GPU BASE_DIR RUN_DIR
set -uo pipefail

BASE_DIR=${BASE_DIR:-/dev/shm/rt_ft/models/pi05_base}
RUN_DIR=${RUN_DIR:-/dev/shm/rt_ft/ckpt/base_eval_run}
ROOT=${ROOT:-/workspace/robotwin_ws}
BENCH=${BENCH:-/dev/shm/robotwin/code}
EVAL_PY=${EVAL_PY:-/opt/rt-eval/bin/python}
POLICY_PY=${POLICY_PY:-/opt/rt-eval/bin/python}
TRIALS=${TRIALS:-25}
WORKERS=${WORKERS:-8}
POLICY_GPU=${POLICY_GPU:-0}
ENV_GPU=${ENV_GPU:-1}
LOG=${LOG:-/dev/shm/rt_base_eval_chain.log}
LOG_DIR=${LOG_DIR:-/dev/shm/rt_base_eval_logs}
RESULTS_DIR=${RESULTS_DIR:-/dev/shm/rt_base_eval_results}
REPORT=${REPORT:-robotwin_eval_report.py}

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

if [[ ! -d "$BASE_DIR/params" ]]; then
  log "abort: $BASE_DIR/params not found"
  exit 1
fi
if [[ ! -d "$BENCH/assets/objects" ]]; then
  log "abort: simulator not staged at $BENCH (run the eval chain first)"
  exit 1
fi

mkdir -p "$RUN_DIR"
ln -sfn "$BASE_DIR" "$RUN_DIR/0"
log "base checkpoint: $BASE_DIR -> $RUN_DIR/0"

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
export ROOT BENCH EVAL_PY POLICY_PY TRIALS WORKERS POLICY_GPU ENV_GPU LOG_DIR RESULTS_DIR

CKPT_DIR="$RUN_DIR" STEP=0 \
  bash "$ROOT/scripts/robotwin_eval_suite.sh" >>"$LOG" 2>&1
rc=$?
log "base suite exit=${rc}; summary:"
cat "$RESULTS_DIR/summary_base_eval_run_0.tsv" 2>/dev/null | tee -a "$LOG"

for cand in "$ROOT/scripts/$REPORT" "/root/$REPORT" "$(dirname "$0")/$REPORT"; do
  if [[ -f "$cand" ]]; then
    log "report from $cand:"
    "$EVAL_PY" "$cand" "$RESULTS_DIR/summary_base_eval_run_0.tsv" >>"$LOG" 2>&1
    tail -35 "$LOG"
    break
  fi
done
log "base chain finished"
