#!/usr/bin/env bash
# Train the Con1 anchored delta head on the rebuilt cache. The head predicts the
# future latent delta from the 64 predictive tokens; Con1's cross-attention
# consumes it and Con2 refines it.
set -euo pipefail

# XLA spawns ptxas/nvcc subprocesses and the image ships a 1024 fd soft limit,
# which makes the GEMM autotuner fail with "Too many open files".
ulimit -n 65535 2>/dev/null || true

ROOT="${ROOT:-/dev/shm/ts_jepa}"
REPO="${REPO:-$ROOT/ts_JEPA_libero}"
PY="${PY:-/opt/venv-jepa/bin/python}"
CACHE="${CACHE:-$ROOT/con1/anchored_40k_features_v1}"
OUTPUT="${OUTPUT:-$ROOT/checkpoints/con1_anchored_head_40k}"
STEPS="${STEPS:-12000}"
BATCH="${BATCH:-256}"
LR="${LR:-1e-5}"
WIDTH="${WIDTH:-512}"
LATENT="${LATENT:-2816}"
SEED="${SEED:-42}"

mkdir -p "$OUTPUT"
cd "$REPO"
PYTHONPATH="$REPO/src" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
  nohup "$PY" -u -m openpi.con1.train_head \
    --cache "$CACHE" \
    --output "$OUTPUT" \
    --steps "$STEPS" \
    --batch-size "$BATCH" \
    --horizon 10 \
    --latent-dim "$LATENT" \
    --width "$WIDTH" \
    --learning-rate "$LR" \
    --seed "$SEED" \
    >"$ROOT/logs/head_training.log" 2>&1 &
printf 'launched head training -> %s\n' "$OUTPUT"
