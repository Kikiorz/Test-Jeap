#!/usr/bin/env bash
# One-command rebuild of the LIBERO Con1/Con2 pipeline on a fresh Blackwell box.
#
# Every step encodes a failure that cost time during the 2026-09-12 rebuild:
#   * the image ships a vLLM server holding ~89 GB per GPU -> stop it first
#   * the repo pins jax[cuda12]==0.5.3, whose jaxlib has no sm_120 kernels
#   * the bundled torch 2.7.1+cu126 only ships up to sm_90 (conv3d -> no kernel)
#   * orbax 0.11.13 needs jax.experimental.layout.DeviceLocalLayout (absent in
#     jax 0.7.2) while orbax 0.12.x changes the metadata API the repo uses
#   * XLA's GEMM autotuner spawns ptxas and dies under the default 1024 fd limit
#   * container disks can be as small as 32 GB and /dev/shm is noexec, so data
#     lives on the RAM disk while virtualenvs must live on a real disk
#   * mujoco needs libEGL.so.1, which the image does not ship
#
# Usage:  HF_TOKEN=hf_xxx bash libero_rebuild.sh [env|data|cache|head|arms|eval|all]
set -euo pipefail

STAGE="${1:-all}"
WORK="${WORK:-/dev/shm/ts_jepa}"
VENV="${VENV:-/opt/venv-jepa}"
EVAL_VENV="${EVAL_VENV:-/opt/venv-libero-plus}"
REPO="${REPO:-$WORK/ts_JEPA_libero}"
REPO_URL="${REPO_URL:-https://github.com/Kikiorz/Test-Jeap.git}"
BRANCH="${BRANCH:-feat/LIBERO}"
HF_TOKEN="${HF_TOKEN:-$(cat "$WORK/.hf_token" 2>/dev/null || true)}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/dev/shm/uv-cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/dev/shm/pip-cache}"
export TMPDIR="${TMPDIR:-/dev/shm/tmp}"
mkdir -p "$WORK" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR" "$WORK/logs"

# XLA spawns ptxas/nvcc per kernel; the default soft limit is far too small.
ulimit -n 65535 2>/dev/null || true

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

stage_env() {
  log "freeing the GPUs (the image runs a vLLM server on all four)"
  supervisorctl stop vllm model-ui 2>/dev/null || true
  pkill -f 'VLLM::Worker' 2>/dev/null || true
  sleep 10

  log "repo"
  if [[ ! -d "$REPO/.git" ]]; then
    git clone -q -b "$BRANCH" "$REPO_URL" "$REPO"
  fi
  git -C "$REPO" fetch -q origin "$BRANCH" && git -C "$REPO" checkout -q "$BRANCH" && git -C "$REPO" pull -q --ff-only

  log "python env: jax 0.7.2 + orbax 0.11.20 + torch cu128 (sm_120)"
  uv venv --clear --python 3.11 "$VENV"
  uv pip install --python "$VENV/bin/python" -e "$REPO" 2>&1 | tail -2
  uv pip install --python "$VENV/bin/python" "jax[cuda12]==0.7.2" "jaxlib==0.7.2" "jax-cuda12-plugin==0.7.2" "jax-cuda12-pjrt==0.7.2" "orbax-checkpoint==0.11.20" "ml_dtypes>=0.5" tensorstore pyarrow tyro wandb tqdm-loggable pytest huggingface_hub 2>&1 | tail -3
  uv pip install --python "$VENV/bin/python" --index-url https://download.pytorch.org/whl/cu128 --upgrade "torch==2.9.1" "torchvision==0.24.1" 2>&1 | tail -2
  "$VENV/bin/python" -c "import jax, torch; print('jax', jax.__version__, jax.default_backend(), len(jax.devices())); print('torch', torch.__version__, torch.cuda.get_arch_list()[-2:])"
}

stage_data() {
  log "base checkpoint, LIBERO dataset, V-JEPA 2.1"
  mkdir -p "$WORK/models" "$WORK/data" "$WORK/vjepa2"
  "$VENV/bin/hf" download CokeAnd1ce/JEPA_WAM --token "$HF_TOKEN" \
    --include "checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/40000/params/*" \
              "checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/40000/assets/*" \
    --local-dir "$WORK/models/jepa_wam" >"$WORK/logs/dl_base.log" 2>&1
  "$VENV/bin/hf" download physical-intelligence/libero --repo-type dataset --token "$HF_TOKEN" \
    --local-dir "$WORK/data/libero_lerobot" >"$WORK/logs/dl_libero.log" 2>&1
  [[ -f "$WORK/vjepa2/vjepa2_1_vitg_384.pt" ]] || curl -L --retry 3 -o "$WORK/vjepa2/vjepa2_1_vitg_384.pt" https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitg_384.pt
  [[ -d "$WORK/vjepa2_src" ]] || git clone -q --depth 1 https://github.com/facebookresearch/vjepa2.git "$WORK/vjepa2_src"
  log "base $(du -sh "$WORK/models/jepa_wam" | cut -f1); libero $(du -sh "$WORK/data/libero_lerobot" | cut -f1)"
}

stage_cache() {
  log "stage 1: frozen V-JEPA per-frame states (4 workers)"
  ROOT="$WORK" REPO="$REPO" PY="$VENV/bin/python" bash "$WORK/launch_frame_states.sh"
  while pgrep -f 'cache_con1_frame_states.py' >/dev/null; do sleep 60; done
  log "stage 2: anchored cache (r + z)"
  ROOT="$WORK" REPO="$REPO" PY="$VENV/bin/python" bash "$WORK/launch_anchored_features.sh"
  while pgrep -f 'cache_con1_features.py' >/dev/null; do sleep 60; done
  "$VENV/bin/python" -c "import json;d=json.load(open('$WORK/con1/anchored_40k_features_v1/manifest.json'));print('anchored complete:',d.get('complete'),'episodes:',len(d.get('episodes',[])))"
}

stage_head() {
  log "Con1 anchored delta head (12k steps)"
  ROOT="$WORK" REPO="$REPO" PY="$VENV/bin/python" bash "$WORK/launch_head_training.sh"
  while pgrep -f 'openpi.con1.train_head' >/dev/null; do sleep 60; done
}

stage_arms() {
  log "P2 arms: Con1 livecross vs complete algorithm"
  ROOT="$WORK" REPO="$REPO" PY="$VENV/bin/python" bash "$WORK/run_libero_arms.sh"
}

stage_eval() {
  log "LIBERO-Plus benchmark + eval venv"
  apt-get update -qq && apt-get install -y -qq libegl1 libgles2 libglvnd0
  mkdir -p "$REPO/data/libero-plus"
  [[ -d "$REPO/data/libero-plus/LIBERO-plus" ]] || git clone -q https://github.com/sylvestf/LIBERO-plus.git "$REPO/data/libero-plus/LIBERO-plus"
  git -C "$REPO/data/libero-plus/LIBERO-plus" checkout -q 4976dc30028e805ff8094b55501d532c48fec182
  uv venv --clear --python 3.8 "$EVAL_VENV"
  uv pip install --python "$EVAL_VENV/bin/python" -r "$REPO/examples/libero/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu113 2>&1 | tail -2
  MUJOCO_GL=egl PYTHONPATH="$REPO/data/libero-plus/LIBERO-plus" "$EVAL_VENV/bin/python" -c "import robosuite, mujoco; print('robosuite', robosuite.__version__, 'mujoco', mujoco.__version__)"
}

case "$STAGE" in
  env) stage_env ;;
  data) stage_data ;;
  cache) stage_cache ;;
  head) stage_head ;;
  arms) stage_arms ;;
  eval) stage_eval ;;
  all) stage_env; stage_data; stage_cache; stage_head; stage_arms; stage_eval ;;
  *) echo "unknown stage: $STAGE" >&2; exit 2 ;;
esac
log "stage '$STAGE' done"
