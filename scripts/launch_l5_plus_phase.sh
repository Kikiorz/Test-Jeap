#!/usr/bin/env bash
# Launch the paired baseline-vs-adapter LIBERO-Plus L5 sweep across the five
# perturbation categories that have not been evaluated yet.
#
# Both sides run concurrently (one policy server per side) and every category is
# split over NUM_TASK_SHARDS shards of the pinned L5 task range, so the run
# headers stay byte-comparable with the completed Camera/Robot L5 runs.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_con1_clean}"
cd "$ROOT"

LOG_ROOT="${LOG_ROOT:-/workspace/artifacts/libero_eval_logs}"
BASELINE_PORT="${BASELINE_PORT:-8001}"
ADAPTER_PORT="${ADAPTER_PORT:-8000}"
NUM_TASK_SHARDS="${NUM_TASK_SHARDS:-12}"
mkdir -p "$LOG_ROOT"

# category label | task block start | task block end | slug
SPECS=(
  "Background Textures|0|289|Background_Textures"
  "Language Instructions|1101|1484|Language_Instructions"
  "Sensor Noise|1484|1933|Sensor_Noise"
  "Objects Layout|1933|2245|Objects_Layout"
  "Light Conditions|2245|2519|Light_Conditions"
)

launch_side() {
  local side="$1" port="$2"
  for spec in "${SPECS[@]}"; do
    IFS='|' read -r category start end slug <<<"$spec"
    local run_id="${side}_L5_${slug}"
    for ((shard = 0; shard < NUM_TASK_SHARDS; shard++)); do
      RUN_ID="$run_id" \
      PORT="$port" \
      TASK_SUITE="libero_10" \
      TASK_START="$start" \
      TASK_END="$end" \
      NUM_TASK_SHARDS="$NUM_TASK_SHARDS" \
      TASK_SHARD_ID="$shard" \
      ONLY_CATEGORY="$category" \
      ONLY_DIFFICULTY_LEVEL=5 \
      nohup bash scripts/run_libero_evaluation.sh plus \
        >"$LOG_ROOT/${run_id}.shard-$(printf '%02d' "$shard").log" 2>&1 &
    done
  done
}

launch_side baseline "$BASELINE_PORT"
sleep 2
launch_side adapter "$ADAPTER_PORT"

sleep 3
printf 'launched %d shard processes\n' "$((2 * ${#SPECS[@]} * NUM_TASK_SHARDS))"
pgrep -fa 'examples/libero/main.py' | wc -l
