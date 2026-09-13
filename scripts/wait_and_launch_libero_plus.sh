#!/usr/bin/env bash
# Launch the LIBERO-Plus fine-tune as soon as the shared box can host it.
#
# The instance at 93.91.156.102 is shared with a RoboTwin run that occupies all
# four GPUs (66 GB each) and keeps the 150 GB overlay at ~100%. Rather than
# fighting for memory - the first attempt died with RESOURCE_EXHAUSTED and the
# restart hit ENOSPC mid-download - this watcher waits for both resources to
# clear and then runs the standard chain.
#
# Usage: nohup bash scripts/wait_and_launch_libero_plus.sh > /workspace/wait_launch.log 2>&1 &
set -uo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_simpenv}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/libero_plus_lerobot}"
EXP_NAME="${EXP_NAME:-libero_plus_30k}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-64}"
MIN_FREE_GB="${MIN_FREE_GB:-45}"
MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-60000}"
DEADLINE_HOURS="${DEADLINE_HOURS:-72}"

cd "$ROOT"
deadline=$(( $(date +%s) + DEADLINE_HOURS * 3600 ))

while :; do
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  min_gpu=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sort -n | head -1)
  printf '[%s] free=%sG min_gpu_free=%sMiB\n' "$(date -u +%H:%M:%S)" "$free_gb" "$min_gpu"
  if [ "${free_gb:-0}" -ge "$MIN_FREE_GB" ] && [ "${min_gpu:-0}" -ge "$MIN_FREE_GPU_MIB" ]; then
    printf '[%s] resources are available, launching\n' "$(date -u +%H:%M:%S)"
    break
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    printf '[%s] gave up waiting for resources\n' "$(date -u +%H:%M:%S)"
    exit 1
  fi
  sleep 300
done

printf '[%s] restoring the pi0.5 base checkpoint\n' "$(date -u +%H:%M:%S)"
.venv/bin/python -u scripts/fetch_pi05_base.py || exit 1

# The waiting eval chain gives up after 36 h, so make sure it is alive again
# before the trainer starts producing checkpoints.
if ! pgrep -f 'auto_libero_plus_eva[l]' >/dev/null; then
  printf '[%s] restarting the waiting evaluation chain\n' "$(date -u +%H:%M:%S)"
  RUN_ID=pi05-plus-30k nohup bash scripts/auto_libero_plus_eval.sh \
    >>/workspace/auto_eval.log 2>&1 &
fi

HF_LEROBOT_HOME=/workspace/lerobot_home HF_HOME=/workspace/.hf_home \
VIDEOS_TOTAL=28694 EXP_NAME="$EXP_NAME" STEPS="$STEPS" BATCH="$BATCH" \
DATASET_ROOT="$DATASET_ROOT" \
  bash scripts/auto_libero_plus_pipeline.sh >"$ROOT/../auto_pipeline_gated.log" 2>&1

printf '[%s] training chain finished\n' "$(date -u +%H:%M:%S)"
