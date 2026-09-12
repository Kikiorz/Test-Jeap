#!/usr/bin/env bash
# Evaluate one LIBERO-Plus checkpoint across every difficulty-5 task.
#
# libero_10 at level 5 covers all seven perturbation categories (705 tasks), so
# a single sweep gives the per-category table plus the official task-micro and
# category-macro aggregates. Shards run in parallel; the policy server must
# already be up (see scripts/run_libero_policy_server.sh).
#
# Usage:
#   RUN_ID=pi05-plus-30k PORT=8000 bash scripts/run_libero_plus_l5_sweep.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:?Set RUN_ID to a stable identifier for the served checkpoint}"
PORT="${PORT:-8000}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-2519}"
NUM_TASK_SHARDS="${NUM_TASK_SHARDS:-12}"
DIFFICULTY="${DIFFICULTY:-5}"
LOG_ROOT="${LOG_ROOT:-$ROOT/data/libero-eval/$RUN_ID}"
mkdir -p "$LOG_ROOT"

pids=()
for ((shard = 0; shard < NUM_TASK_SHARDS; shard++)); do
  RUN_ID="$RUN_ID" \
  PORT="$PORT" \
  TASK_SUITE="$TASK_SUITE" \
  TASK_START="$TASK_START" \
  TASK_END="$TASK_END" \
  NUM_TASK_SHARDS="$NUM_TASK_SHARDS" \
  TASK_SHARD_ID="$shard" \
  ONLY_DIFFICULTY_LEVEL="$DIFFICULTY" \
  nohup bash "$ROOT/scripts/run_libero_evaluation.sh" plus \
    >"$LOG_ROOT/shard-$(printf '%02d' "$shard").log" 2>&1 &
  pids+=("$!")
done

printf 'launched %d shards for %s difficulty %s; waiting...\n' "$NUM_TASK_SHARDS" "$TASK_SUITE" "$DIFFICULTY"
for pid in "${pids[@]}"; do wait "$pid"; done

printf '\nAll shards finished. Summary:\n'
EVAL_PYTHON="${EVAL_PYTHON:-$ROOT/examples/libero/.venv-plus/bin/python}" \
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$ROOT/data/libero-plus/config}" \
  bash "$ROOT/scripts/run_libero_evaluation.sh" summary \
  "$LOG_ROOT/plus-$TASK_SUITE.shard-"*"-of-$(printf '%05d' "$NUM_TASK_SHARDS").jsonl"
