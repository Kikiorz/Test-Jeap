#!/usr/bin/env bash
# Control run for the RoboTwin closed-loop numbers, meant to be run on the
# simulator box: evaluate the *un-finetuned* pi0.5 base on the same 20 tasks,
# Clean and Random, through the same policy server and simulator.
#
# Why: the finetune's number only means something next to what the base already
# scores on this exact harness. The paper's pi0.5 column is not that control - it
# is a pi0.5 trained on RoboTwin *clean* demonstrations, so it differs from our
# base in both data and recipe.
#
# The base checkpoint is laid out like a checkpoint step directory (`params/` +
# `assets/`), so it only needs a numeric parent entry to satisfy the server's
# `model_path`/`checkpoint_num` convention.
#
# Env: BASE_DIR RUN_DIR TRIALS WORKERS POLICY_GPU ENV_GPU
set -uo pipefail

BASE_DIR=${BASE_DIR:-/dev/shm/pi05_base}
RUN_DIR=${RUN_DIR:-/dev/shm/ckpt_robotwin/base_eval_run}
ROOT=${ROOT:-/workspace/robotwin_ws}
TRIALS=${TRIALS:-25}
WORKERS=${WORKERS:-8}
POLICY_GPU=${POLICY_GPU:-1}
ENV_GPU=${ENV_GPU:-2}
LOG_DIR=${LOG_DIR:-/dev/shm/rt_base_logs}
RESULTS_DIR=${RESULTS_DIR:-/dev/shm/rt_base_results}

[[ -d "$BASE_DIR/params" ]] || { echo "abort: $BASE_DIR/params not found"; exit 1; }

mkdir -p "$RUN_DIR" "$LOG_DIR" "$RESULTS_DIR"
ln -sfn "$BASE_DIR" "$RUN_DIR/0"
echo "[base] $BASE_DIR -> $RUN_DIR/0"

cd "$ROOT"
ROOT="$ROOT" CKPT_DIR="$RUN_DIR" STEP=0 \
  TRIALS="$TRIALS" WORKERS="$WORKERS" POLICY_GPU="$POLICY_GPU" ENV_GPU="$ENV_GPU" \
  LOG_DIR="$LOG_DIR" RESULTS_DIR="$RESULTS_DIR" \
  bash scripts/robotwin_eval_suite.sh
rc=$?
echo "[base] suite exit=$rc"
cat "$RESULTS_DIR/summary_base_eval_run_0.tsv" 2>/dev/null
