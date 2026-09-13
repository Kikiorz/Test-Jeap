#!/usr/bin/env bash
# Unattended chain: wait for the LIBERO-Plus LeRobot download, compute the
# normalisation statistics, then run the 30k-step pi0.5 fine-tune.
#
# Everything is resumable: the download loop and the trainer both skip work that
# is already done, and each stage writes its own log under /workspace.
set -u

ROOT="${ROOT:-/workspace/ts_JEPA_simpenv}"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export HF_HOME=/workspace/.hf_home
export HF_LEROBOT_HOME=/workspace/lerobot_home
export HF_HUB_OFFLINE=1

VIDEOS_TOTAL="${VIDEOS_TOTAL:-28694}"
EXP_NAME="${EXP_NAME:-libero_plus_30k}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-64}"
DATASET_ROOT="${DATASET_ROOT:-/workspace/libero_plus_lerobot}"

printf '[%s] syncing dataset at %s (%s videos expected)\n' \
  "$(date -u +%H:%M:%S)" "$DATASET_ROOT" "$VIDEOS_TOTAL"
while :; do
  .venv/bin/python -u scripts/fetch_libero_plus_dataset.py \
    --root "$DATASET_ROOT" --workers 16 >/workspace/dl_pipeline_sync.log 2>&1
  # `find` rather than a shell glob: once the dataset holds ~23k videos the
  # expanded argument list exceeds ARG_MAX and `ls` silently returns nothing.
  v=$(find "$DATASET_ROOT/videos" -name '*.mp4' 2>/dev/null | wc -l)
  p=$(find "$DATASET_ROOT/data" -name '*.parquet' 2>/dev/null | wc -l)
  printf '[%s] videos %s/%s parquet %s/14347\n' "$(date -u +%H:%M:%S)" "$v" "$VIDEOS_TOTAL" "$p"
  [ "$v" -ge "$VIDEOS_TOTAL" ] && [ "$p" -ge 14347 ] && break
  sleep 120
done

printf '[%s] computing norm stats\n' "$(date -u +%H:%M:%S)"
norm_ok=0
for frames in 100000 50000 20000; do
  printf '[%s] norm stats attempt max-frames=%s\n' "$(date -u +%H:%M:%S)" "$frames"
  if CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compute_norm_stats.py \
      --config-name pi05_libero_plus --max-frames "$frames" \
      >/workspace/norm_libero_plus.log 2>&1; then
    norm_ok=1
    break
  fi
  printf '[%s] attempt failed, retrying\n' "$(date -u +%H:%M:%S)"
  tail -5 /workspace/norm_libero_plus.log
  sleep 60
done
if [ "$norm_ok" != "1" ]; then
  printf '[%s] norm stats failed after all attempts\n' "$(date -u +%H:%M:%S)"
  exit 1
fi

printf '[%s] starting %s-step fine-tune\n' "$(date -u +%H:%M:%S)" "$STEPS"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
OPENPI_SAVE_OPT_STATE=0 \
  .venv/bin/python -u scripts/train.py pi05_libero_plus \
    --exp-name="$EXP_NAME" \
    --checkpoint-base-dir=/workspace/checkpoints \
    --overwrite --no-wandb-enabled \
    --num-train-steps="$STEPS" --batch-size="$BATCH" --fsdp-devices=4 \
    --save-interval=5000 --keep-period=30000 \
    >/workspace/train_libero_plus.log 2>&1

printf '[%s] TRAIN_DONE\n' "$(date -u +%H:%M:%S)"
