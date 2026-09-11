#!/usr/bin/env bash
# Run the remaining LIBERO-Plus L5 categories one after another, 12 shards per
# side. Category-level sequencing keeps the two policy servers from being
# over-subscribed (120 concurrent clients starved them earlier); every shard
# journals as it goes, so a re-run resumes instead of repeating work.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_con1_clean}"
cd "$ROOT"

LOG_ROOT="${LOG_ROOT:-/workspace/artifacts/libero_eval_logs}"
BASELINE_PORT="${BASELINE_PORT:-8001}"
ADAPTER_PORT="${ADAPTER_PORT:-8000}"
SHARDS="${NUM_TASK_SHARDS:-12}"
POLL_SECONDS="${POLL_SECONDS:-30}"
STARTUP_GRACE_SECONDS="${STARTUP_GRACE_SECONDS:-120}"
mkdir -p "$LOG_ROOT"

SPECS=(
  "Background Textures|0|289|Background_Textures"
  "Language Instructions|1101|1484|Language_Instructions"
  "Sensor Noise|1484|1933|Sensor_Noise"
  "Objects Layout|1933|2245|Objects_Layout"
  "Light Conditions|2245|2519|Light_Conditions"
)

launch_category() {
  local category="$1" start="$2" end="$3" slug="$4"
  for side in baseline adapter; do
    if [[ "$side" == "baseline" ]]; then port="$BASELINE_PORT"; else port="$ADAPTER_PORT"; fi
    local run_id="${side}_L5_${slug}"
    for ((shard = 0; shard < SHARDS; shard++)); do
      RUN_ID="$run_id" \
      PORT="$port" \
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
  done
}

wait_for_category() {
  # The shards spend their first seconds in the wrapper shell, so give them time
  # to reach the evaluation entrypoint before trusting an empty pgrep.
  sleep "$STARTUP_GRACE_SECONDS"
  while pgrep -f "examples/libero/main[.]py" >/dev/null; do
    sleep "$POLL_SECONDS"
  done
}

for spec in "${SPECS[@]}"; do
  IFS='|' read -r category start end slug <<<"$spec"
  printf '[%s] launching %s (tasks %s-%s)\n' "$(date -u +%H:%M:%S)" "$category" "$start" "$end"
  launch_category "$category" "$start" "$end" "$slug"
  wait_for_category
  printf '[%s] finished %s\n' "$(date -u +%H:%M:%S)" "$category"
done
printf '[%s] all categories complete\n' "$(date -u +%H:%M:%S)"
