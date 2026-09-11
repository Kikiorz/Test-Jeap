#!/usr/bin/env bash
# Head-to-head: anchored delta head on the VLM pooled latent vs the JEPA pooled
# latent, identical budget/recipe, two GPUs each.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_con1_clean}"
cd "$ROOT"
STEPS="${STEPS:-10000}"
BATCH="${BATCH:-256}"
WIDTH="${WIDTH:-512}"
LR="${LR:-1e-5}"
LOG_ROOT="${LOG_ROOT:-/workspace/artifacts/checkpoints}"
mkdir -p "$LOG_ROOT"

common_env=(
  "PYTHONPATH=$ROOT/src"
  "XLA_PYTHON_CLIENT_PREALLOCATE=false"
  "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2"
  "NCCL_P2P_DISABLE=1"
  "NCCL_IB_DISABLE=1"
  "HF_HUB_OFFLINE=1"
)

run_job() {
  local name="$1" cache="$2" latent_dim="$3" gpus="$4"
  local out="$LOG_ROOT/$name"
  mkdir -p "$out"
  nohup env "${common_env[@]}" "CUDA_VISIBLE_DEVICES=$gpus" \
    "$ROOT/.venv/bin/python" -u -m openpi.con1.train_head \
      --cache "$cache" \
      --output "$out" \
      --steps "$STEPS" \
      --batch-size "$BATCH" \
      --horizon 10 \
      --latent-dim "$latent_dim" \
      --width "$WIDTH" \
      --learning-rate "$LR" \
      >"$out/train.log" 2>&1 &
  printf 'started %s (latent_dim=%s, gpus=%s) pid=%s\n' "$name" "$latent_dim" "$gpus" "$!"
}

run_job con1_vlm_head_40k_10k /workspace/artifacts/con1/anchored_40k_features_vlm_pooled 2048 0,1
sleep 3
run_job con1_jepa_head_40k_10k_rerun /workspace/artifacts/con1/anchored_40k_features_v1 2816 2,3
