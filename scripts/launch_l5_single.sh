#!/usr/bin/env bash
# Run the seven LIBERO-Plus L5 categories for one policy server, using the same
# sharding as the baseline/adapter sweeps so the journals pair up by
# (task_id, episode_idx). Baseline journals already exist on disk.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_con1_clean}"
cd "$ROOT"

SIDE="${SIDE:?Set SIDE to the run-id prefix, e.g. livecross}"
PORT="${PORT:?Set PORT to the policy server port}"
SHARDS="${NUM_TASK_SHARDS:-12}"
LOG_ROOT="${LOG_ROOT:-/workspace/artifacts/libero_eval_logs}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STARTUP_GRACE_SECONDS="${STARTUP_GRACE_SECONDS:-120}"
ONLY_SLUG="${ONLY_SLUG:-}"
mkdir -p "$LOG_ROOT"

SPECS=(
  "Background Textures|0|289|Background_Textures"
  "Language Instructions|1101|1484|Language_Instructions"
  "Sensor Noise|1484|1933|Sensor_Noise"
  "Objects Layout|1933|2245|Objects_Layout"
  "Light Conditions|2245|2519|Light_Conditions"
  "Camera Viewpoints|682|1101|Camera_Viewpoints"
  "Robot Initial States|289|682|Robot_Initial_States"
)

wait_for_category() {
  sleep "$STARTUP_GRACE_SECONDS"
  while pgrep -f "examples/libero/main[.]py" >/dev/null; do
    sleep "$POLL_SECONDS"
  done
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r category start end slug <<<"$spec"
  if [[ -n "$ONLY_SLUG" && "$slug" != "$ONLY_SLUG" ]]; then
    continue
  fi
  run_id="${SIDE}_L5_${slug}"
  printf '[%s] launching %s (%s)\n' "$(date -u +%H:%M:%S)" "$category" "$run_id"
  for ((shard = 0; shard < SHARDS; shard++)); do
    RUN_ID="$run_id" \
    PORT="$PORT" \
    TASK_SUITE="libero_10" \
    TASK_START="$start" \
    TASK_END="$end" \
    NUM_TASK_SHARDS="$SHARDS" \
    TASK_SHARD_ID="$shard" \
    ONLY_CATEGORY="$category" \
    ONLY_DIFFICULTY_LEVEL=5 \
    nohup bash scripts/run_libero_evaluation.sh plus \
      >"$LOG_ROOT/${run_id}.shard-$(printf '%02d' "$shard").log" 2>&1 &
  done
  wait_for_category
  printf '[%s] finished %s\n' "$(date -u +%H:%M:%S)" "$category"
done
printf '[%s] all requested categories complete\n' "$(date -u +%H:%M:%S)"
