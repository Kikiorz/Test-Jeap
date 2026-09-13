#!/usr/bin/env bash
# Full LIBERO-Plus evaluation for one served checkpoint.
#
# Protocol requested: 8 rollout environments per GPU (4 GPUs -> 32 shards) and
# every LIBERO-Plus task in all four suites (10,030 tasks total: spatial 2,402;
# object 2,518; goal 2,591; libero_10 2,519). No difficulty filter is applied,
# so the run covers all seven perturbation categories at every level.
#
# Usage: RUN_ID=pi05-plus-30k PORT=8000 bash scripts/run_libero_plus_full_sweep.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:?Set RUN_ID to a stable identifier for the served checkpoint}"
PORT="${PORT:-8000}"
NUM_TASK_SHARDS="${NUM_TASK_SHARDS:-32}"
GPUS="${GPUS:-4}"
LOG_ROOT="${LOG_ROOT:-$ROOT/data/libero-eval/$RUN_ID}"
mkdir -p "$LOG_ROOT"

declare -A SUITE_TASKS=(
  [libero_spatial]=2402
  [libero_object]=2518
  [libero_goal]=2591
  [libero_10]=2519
)

for suite in libero_spatial libero_object libero_goal libero_10; do
  printf '[%s] launching %s (%s tasks, %s shards)\n' \
    "$(date -u +%H:%M:%S)" "$suite" "${SUITE_TASKS[$suite]}" "$NUM_TASK_SHARDS"
  pids=()
  for ((shard = 0; shard < NUM_TASK_SHARDS; shard++)); do
    gpu=$((shard % GPUS))
    RUN_ID="$RUN_ID" \
    PORT="$PORT" \
    TASK_SUITE="$suite" \
    TASK_START=0 \
    TASK_END="${SUITE_TASKS[$suite]}" \
    NUM_TASK_SHARDS="$NUM_TASK_SHARDS" \
    TASK_SHARD_ID="$shard" \
    EVAL_GPU="$gpu" \
    MUJOCO_EGL_DEVICE_ID="$gpu" \
    nohup bash "$ROOT/scripts/run_libero_evaluation.sh" plus \
      >"$LOG_ROOT/${suite}.shard-$(printf '%02d' "$shard").log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  printf '[%s] finished %s\n' "$(date -u +%H:%M:%S)" "$suite"
done

printf '\nAll suites finished. Summary:\n'
EVAL_PYTHON="${EVAL_PYTHON:-$ROOT/examples/libero/.venv-plus/bin/python}" \
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/data/libero-plus/config}" \
  bash "$ROOT/scripts/run_libero_evaluation.sh" summary \
  "$LOG_ROOT/plus-libero_spatial.shard-"*"-of-$(printf '%05d' "$NUM_TASK_SHARDS").jsonl" \
  "$LOG_ROOT/plus-libero_object.shard-"*"-of-$(printf '%05d' "$NUM_TASK_SHARDS").jsonl" \
  "$LOG_ROOT/plus-libero_goal.shard-"*"-of-$(printf '%05d' "$NUM_TASK_SHARDS").jsonl" \
  "$LOG_ROOT/plus-libero_10.shard-"*"-of-$(printf '%05d' "$NUM_TASK_SHARDS").jsonl"
