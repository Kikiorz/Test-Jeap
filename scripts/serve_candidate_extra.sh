#!/usr/bin/env bash
# Two extra candidate policy servers, because a single server is CPU-bound on
# the online V-JEPA latent path (one thread at 100%, GPU at 33%).
set -euo pipefail
ulimit -n 65535 2>/dev/null || true

WORK="${WORK:-/workspace/ts_jepa}"
REPO="${REPO:-$WORK/ts_JEPA_libero}"
PY="${PY:-/opt/venv-jepa/bin/python}"
CAND_CFG="${CAND_CFG:-pi05_libero_con1con2_ctx_40k}"
CAND_DIR="${CAND_DIR:-$WORK/checkpoints/pi05_libero_con1con2_ctx_40k/libero_b_full/4499}"

start() {
  local port="$1" gpu="$2"
  printf '[%s] candidate server :%s on GPU %s\n' "$(date -u +%H:%M:%S)" "$port" "$gpu"
  PYTHONPATH="$REPO/src" HF_HUB_OFFLINE=1 \
  OPENPI_CON1_ONLINE_LATENT=1 \
  OPENPI_CON1_VJEPA_CHECKPOINT="$WORK/vjepa2/vjepa2_1_vitg_384.pt" \
  OPENPI_CON1_VJEPA_ROOT="$WORK/vjepa2_src" \
  OPENPI_CON1_VJEPA_DEVICE="cuda:0" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 \
    nohup "$PY" "$REPO/scripts/serve_policy.py" --env LIBERO --port "$port" \
      policy:checkpoint --policy.config "$CAND_CFG" --policy.dir "$CAND_DIR" \
      >"$WORK/logs/policy_candidate_${port}.log" 2>&1 &
}

cd "$REPO"
start 8004 0
start 8005 1
for port in 8004 8005; do
  tries=90
  until curl -sf -o /dev/null "http://127.0.0.1:$port/healthz"; do
    tries=$((tries - 1)); [[ $tries -gt 0 ]] || { echo ":$port never healthy" >&2; exit 3; }
    sleep 10
  done
done
printf '[%s] extra candidate servers healthy\n' "$(date -u +%H:%M:%S)"
