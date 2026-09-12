#!/usr/bin/env bash
# Serve one SimpENV checkpoint and score it on the four WidowX tasks.
#
# Usage: CONFIG=pi05_bridge_con1 CKPT=/workspace/checkpoints/.../6000 TAG=armA \
#          OUT=/workspace/simpler_eval bash examples/simpler_env/run_eval.sh [n_trajs]
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_simpenv}"
SIMPLER_ENV_ROOT="${SIMPLER_ENV_ROOT:-/workspace/simpler/SimplerEnv}"
SIMPLER_VENV="${SIMPLER_VENV:-/workspace/simpler/.venv}"
CONFIG="${CONFIG:?Set CONFIG to the training config name}"
CKPT="${CKPT:?Set CKPT to the checkpoint directory}"
TAG="${TAG:-eval}"
N_TRAJS="${1:-24}"
OUT="${OUT:-/workspace/simpler_eval}"
PORT="${PORT:-8000}"

export HF_HOME=/workspace/.hf_home
export HF_LEROBOT_HOME=/workspace/lerobot_home
export HF_HUB_OFFLINE=1

mkdir -p "$OUT"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT/src" nohup .venv/bin/python scripts/serve_policy.py \
  --env SIMPLER_ENV --port "$PORT" policy:checkpoint \
  --policy.config "$CONFIG" --policy.dir "$CKPT" \
  >"$OUT/policy_server_${TAG}.log" 2>&1 &
server_pid=$!

cleanup() { kill "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT

for _ in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null; then break; fi
  sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null || { echo "policy server failed to start"; exit 1; }

cd "$SIMPLER_ENV_ROOT"
CUDA_VISIBLE_DEVICES=0 \
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
VK_DRIVER_FILES=/etc/vulkan/icd.d/nvidia_icd.json \
  "$SIMPLER_VENV/bin/python" "$ROOT/examples/simpler_env/main.py" \
  --task all --n-trajs "$N_TRAJS" --host 127.0.0.1 --port "$PORT" \
  --replan-steps 8 --log-dir "$OUT"
