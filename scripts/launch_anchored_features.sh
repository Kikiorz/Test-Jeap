#!/usr/bin/env bash
# Stage 2 of the Con1 cache rebuild: combine the frozen V-JEPA frame states with
# the PI0.5 predictive tokens (R) into the anchored cache the head and the
# policy fine-tune consume.
set -euo pipefail

ROOT="${ROOT:-/dev/shm/ts_jepa}"
REPO="${REPO:-$ROOT/ts_JEPA_libero}"
PY="${PY:-/opt/venv-jepa/bin/python}"
DATASET="${DATASET:-$ROOT/data/libero_lerobot}"
STATES="${STATES:-$ROOT/con1/frame_states_v1}"
CHECKPOINT="${CHECKPOINT:-$ROOT/models/jepa_wam/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/40000}"
OUTPUT="${OUTPUT:-$ROOT/con1/anchored_40k_features_v1}"
GPUS="${GPUS:-0,1,2,3}"
BATCH="${BATCH:-8}"

cd "$REPO"
PYTHONPATH="$REPO/src:$REPO/packages/openpi-client/src" \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  nohup "$PY" -u scripts/cache_con1_features.py \
    --dataset "$DATASET" \
    --states "$STATES" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --gpus "$GPUS" \
    --batch-size "$BATCH" \
    >"$ROOT/logs/anchored_features.log" 2>&1 &
printf 'launched stage 2 -> %s\n' "$OUTPUT"
