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

VIDEOS_TOTAL="${VIDEOS_TOTAL:-28684}"
EXP_NAME="${EXP_NAME:-libero_plus_30k}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-64}"

printf '[%s] waiting for videos (%s needed)\n' "$(date -u +%H:%M:%S)" "$VIDEOS_TOTAL"
while :; do
  v=$(ls /workspace/data/libero_plus_lerobot/videos/*/observation.images.*/*.mp4 2>/dev/null | wc -l)
  printf '[%s] videos %s/%s\n' "$(date -u +%H:%M:%S)" "$v" "$VIDEOS_TOTAL"
  [ "$v" -ge "$VIDEOS_TOTAL" ] && break
  sleep 300
done

printf '[%s] computing norm stats\n' "$(date -u +%H:%M:%S)"
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/compute_norm_stats.py \
  --config-name pi05_libero_plus --max-frames 200000 >/workspace/norm_libero_plus.log 2>&1 \
  || { printf 'norm stats failed\n'; tail -20 /workspace/norm_libero_plus.log; exit 1; }

printf '[%s] starting %s-step fine-tune\n' "$(date -u +%H:%M:%S)" "$STEPS"
CUDA_VISIBLE_DEVICES=0,1,2,3 \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  .venv/bin/python -u scripts/train.py pi05_libero_plus \
    --exp-name="$EXP_NAME" \
    --checkpoint-base-dir=/workspace/checkpoints \
    --overwrite --no-wandb-enabled \
    --num-train-steps="$STEPS" --batch-size="$BATCH" --fsdp-devices=4 \
    --save-interval=5000 --keep-period=30000 \
    >/workspace/train_libero_plus.log 2>&1

printf '[%s] TRAIN_DONE\n' "$(date -u +%H:%M:%S)"
