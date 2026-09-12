#!/usr/bin/env bash
# Build the Con1 latent cache for RoboTwin: V-JEPA frame states first, then the
# frozen VLM prefix tokens, four workers (one per GPU) at each stage.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
cd "$ROOT"

# Both stages read the inline-image copy. The video dataset needs a torchcodec
# AV1 decode per camera per episode, which measured ~8 episodes/min against ~29
# for the inline JPEGs, and RoboTwin v2.1 stores one video per episode rather
# than the per-chunk layout the loader expects. Set DATASET to
# /workspace/robotwin2/RoboTwin_v21 only if the inline copy is unavailable; the
# states cache identity follows DATASET, so a rebuild against a different copy
# must go to a fresh STATES directory.
DATASET="${DATASET:-/workspace/robotwin2/RoboTwin_v21_inline}"
POLICY_CKPT="${POLICY_CKPT:-/workspace/artifacts/models/jepa_wam_pi05_robotwin_publish/pi05_robotwin_clean_20_vjepa_aux/19999}"
VJEPA_ROOT="${VJEPA_ROOT:-/workspace/vjepa2}"
VJEPA_CKPT="${VJEPA_CKPT:-$VJEPA_ROOT/vjepa2_1_vitg_384.pt}"
STATES="${STATES:-/workspace/artifacts/con1/robotwin_states_v1}"
FEATURES="${FEATURES:-/workspace/artifacts/con1/robotwin_clean20_19999_features_v1}"
LOG="${LOG:-/workspace/robotwin_cache_build.log}"
IMAGES=(observation.images.cam_high observation.images.cam_left_wrist observation.images.cam_right_wrist)

WORKERS="${WORKERS:-8}"

echo "[$(date -u +%H:%M:%S)] stage 1: V-JEPA frame states ($WORKERS workers)" | tee -a "$LOG"
for ((rank = 0; rank < WORKERS; rank++)); do
  gpu=$((rank % 4))
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT/src" \
    nohup .venv/bin/python -u scripts/cache_con1_frame_states.py \
      --dataset-root "$DATASET" \
      --checkpoint "$VJEPA_CKPT" \
      --vjepa-source-root "$VJEPA_ROOT" \
      --output-root "$STATES" \
      --image-keys "${IMAGES[@]}" \
      --device cuda:0 --batch-size 16 \
      --worker-rank "$rank" --world-size "$WORKERS" \
      >>"$LOG.rank$rank" 2>&1 &
done
wait
echo "[$(date -u +%H:%M:%S)] stage 1 done" | tee -a "$LOG"

echo "[$(date -u +%H:%M:%S)] stage 2: frozen VLM prefix tokens" | tee -a "$LOG"
PYTHONPATH="$ROOT/src" \
  .venv/bin/python -u scripts/cache_con1_features.py \
    --dataset "${DATASET_INLINE:-$DATASET}" \
    --states "$STATES" \
    --checkpoint "$POLICY_CKPT" \
    --output "$FEATURES" \
    --gpus 0,1,2,3 \
    --base-config pi05_robotwin_con1con2_ctx_20k \
    --num-queries 64 --latent-dim 4224 --horizon 50 \
    --image-keys "${IMAGES[@]}" \
    >>"$LOG" 2>&1
echo "[$(date -u +%H:%M:%S)] stage 2 done" | tee -a "$LOG"
