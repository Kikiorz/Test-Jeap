#!/usr/bin/env bash
# Full LIBERO-Plus sweep for one policy server: the whole libero_10 suite
# (every perturbation category, every difficulty level), sharded 16 ways so one
# GPU drives ~16 MuJoCo environments in parallel.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_jepa/ts_JEPA_libero}"
cd "$ROOT"

SIDE="${SIDE:?Set SIDE to the run-id prefix, e.g. con1con2}"
PORT="${PORT:?Set PORT to the policy server port}"
SHARDS="${NUM_TASK_SHARDS:-16}"
TASK_SUITE="${TASK_SUITE:-libero_10}"
LOG_ROOT="${LOG_ROOT:-/workspace/ts_jepa/logs/plus_eval}"
mkdir -p "$LOG_ROOT"

RUN_ID="${SIDE}_plus_full"
for ((shard = 0; shard < SHARDS; shard++)); do
  RUN_ID="$RUN_ID" \
  PORT="$PORT" \
  TASK_SUITE="$TASK_SUITE" \
  NUM_TASK_SHARDS="$SHARDS" \
  TASK_SHARD_ID="$shard" \
  nohup bash scripts/run_libero_evaluation.sh plus \
    >"$LOG_ROOT/${RUN_ID}.shard-$(printf '%02d' "$shard").log" 2>&1 &
done
printf 'launched %s shards for %s (port %s)\n' "$SHARDS" "$RUN_ID" "$PORT"
