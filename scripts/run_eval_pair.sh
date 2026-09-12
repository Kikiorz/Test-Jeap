#!/usr/bin/env bash
# Paired LIBERO-Plus evaluation: the official base policy and the Con1/Con2
# candidate, each driving 16 MuJoCo environments over the whole libero_10
# suite, then a paired summary.
set -euo pipefail
ulimit -n 65535 2>/dev/null || true

WORK="${WORK:-/workspace/ts_jepa}"
REPO="${REPO:-$WORK/ts_JEPA_libero}"
PY="${PY:-/opt/venv-jepa/bin/python}"
BASE_CKPT="${BASE_CKPT:-$WORK/models/jepa_wam/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/40000}"
CAND_CFG="${CAND_CFG:-pi05_libero_con1con2_ctx_40k}"
CAND_EXP="${CAND_EXP:-libero_b_full}"
BASE_PORT="${BASE_PORT:-8001}"
CAND_PORT="${CAND_PORT:-8000}"
SHARDS="${SHARDS:-16}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

CAND_DIR="$WORK/checkpoints/$CAND_CFG/$CAND_EXP"
CAND_STEP="${CAND_STEP:-$(ls "$CAND_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)}"
[[ -n "${CAND_STEP:-}" ]] || { echo "no candidate checkpoint under $CAND_DIR" >&2; exit 2; }
log "candidate $CAND_CFG/$CAND_EXP step $CAND_STEP"

start_server() {
  local name="$1" port="$2" config="$3" directory="$4" online_latent="$5" gpu="$6"
  log "starting $name policy server on :$port"
  PYTHONPATH="$REPO/src" HF_HUB_OFFLINE=1 OPENPI_CON1_ONLINE_LATENT="$online_latent" \
  CUDA_VISIBLE_DEVICES="$gpu" \
  OPENPI_CON1_VJEPA_CHECKPOINT="$WORK/vjepa2/vjepa2_1_vitg_384.pt" \
  OPENPI_CON1_VJEPA_ROOT="$WORK/vjepa2_src" \
  OPENPI_CON1_VJEPA_DEVICE="cuda:0" \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
    nohup "$PY" scripts/serve_policy.py --env LIBERO --port "$port" \
      policy:checkpoint --policy.config "$config" --policy.dir "$directory" \
      >"$WORK/logs/policy_${name}.log" 2>&1 &
}

wait_health() {
  local port="$1" tries=90
  until curl -sf -o /dev/null "http://127.0.0.1:$port/healthz"; do
    tries=$((tries - 1))
    [[ $tries -gt 0 ]] || { echo "policy server on :$port never became healthy" >&2; exit 3; }
    sleep 10
  done
}

cd "$REPO"
start_server baseline "$BASE_PORT" pi05_libero_vjepa_aux "$BASE_CKPT" 0 "${BASE_GPU:-2}"
start_server candidate "$CAND_PORT" "$CAND_CFG" "$CAND_DIR/$CAND_STEP" 1 "${CAND_GPU:-3}"
wait_health "$BASE_PORT"
wait_health "$CAND_PORT"
log "both policy servers healthy"

SIDE=baseline PORT="$BASE_PORT" NUM_TASK_SHARDS="$SHARDS" \
  LOG_ROOT="$WORK/logs/plus_eval" ROOT="$REPO" bash "$REPO/scripts/launch_plus_full.sh"
SIDE=con1con2 PORT="$CAND_PORT" NUM_TASK_SHARDS="$SHARDS" \
  LOG_ROOT="$WORK/logs/plus_eval" ROOT="$REPO" bash "$REPO/scripts/launch_plus_full.sh"
log "16+16 shards launched"
