#!/usr/bin/env bash
# Stage 2 of the Con1 cache: frozen VLM prefix tokens. Two workers per GPU
# (`--gpus 0,0,1,1,2,2,3,3`) because a single worker leaves the GPUs idle --
# one worker per GPU measured 7 episodes/min, and the script only spawns as many
# workers as there are entries in --gpus while the device mapping is rank % list.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
cd "$ROOT"

DATASET="${DATASET:-/workspace/robotwin2/RoboTwin_v21_inline}"
STATES="${STATES:-/workspace/artifacts/con1/robotwin_states_v2}"
POLICY_CKPT="${POLICY_CKPT:-/workspace/artifacts/models/jepa_wam_pi05_robotwin_publish/pi05_robotwin_clean_20_vjepa_aux/19999}"
OUTPUT="${OUTPUT:-/workspace/artifacts/con1/robotwin_clean20_19999_features_v1}"
LOG="${LOG:-/workspace/features_stage2.log}"

PYTHONPATH="$ROOT/src" HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 \
  .venv/bin/python -u scripts/cache_con1_features.py \
    --dataset "$DATASET" \
    --states "$STATES" \
    --checkpoint "$POLICY_CKPT" \
    --output "$OUTPUT" \
    --gpus 0,0,1,1,2,2,3,3 \
    --base-config pi05_robotwin_con1con2_ctx_20k \
    --batch-size "${BATCH_SIZE:-32}" \
    --num-queries 64 --latent-dim 4224 --horizon 16 \
    --image-keys observation.images.cam_high observation.images.cam_left_wrist observation.images.cam_right_wrist \
    >>"$LOG" 2>&1
