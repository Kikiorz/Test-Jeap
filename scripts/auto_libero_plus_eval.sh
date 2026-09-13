#!/usr/bin/env bash
# Wait for the LIBERO-Plus fine-tune to finish, then serve the final checkpoint
# and run the full 32-shard LIBERO-Plus sweep (8 rollout envs per GPU, all four
# suites, every difficulty level).
set -u

ROOT="${ROOT:-/workspace/ts_JEPA_simpenv}"
cd "$ROOT"

CKPT_ROOT=/workspace/checkpoints/pi05_libero_plus
EXP_NAME="${EXP_NAME:-libero_plus_30k}"
STEPS="${STEPS:-30000}"
RUN_ID="${RUN_ID:-pi05-plus-30k}"
PORT="${PORT:-8000}"

printf '[%s] waiting for the final checkpoint %s/%s/%s\n' "$(date -u +%H:%M:%S)" "$CKPT_ROOT" "$EXP_NAME" "$((STEPS - 1))"
while [ ! -d "$CKPT_ROOT/$EXP_NAME/$((STEPS - 1))" ]; do
  if ! pgrep -f "scripts/train[.]py" >/dev/null; then
    printf '[%s] no trainer running and no final checkpoint; aborting\n' "$(date -u +%H:%M:%S)"
    exit 1
  fi
  sleep 300
done
printf '[%s] final checkpoint present\n' "$(date -u +%H:%M:%S)"

CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python scripts/serve_policy.py \
  --env LIBERO --port "$PORT" policy:checkpoint \
  --policy.config pi05_libero_plus \
  --policy.dir "$CKPT_ROOT/$EXP_NAME/$((STEPS - 1))" \
  >/workspace/policy_server_plus.log 2>&1 &

for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null && break
  sleep 10
done
curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null \
  || { printf 'policy server failed to start\n'; tail -20 /workspace/policy_server_plus.log; exit 1; }
printf '[%s] policy server up on %s\n' "$(date -u +%H:%M:%S)" "$PORT"

RUN_ID="$RUN_ID" PORT="$PORT" NUM_TASK_SHARDS=32 \
  bash scripts/run_libero_plus_full_sweep.sh >/workspace/libero_plus_sweep.log 2>&1
printf '[%s] SWEEP_DONE\n' "$(date -u +%H:%M:%S)"
