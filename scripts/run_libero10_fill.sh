#!/usr/bin/env bash
# Fill the libero_10 coverage hole: the suite was restarted several times while
# the parallel layout was tuned, so ~42% of its tasks were evaluated. This waits
# for the current suite to drain, then re-runs libero_10 with the settled
# 64-shard / 8-server / 4-GPU-EGL layout; already-recorded episodes are skipped
# by the per-shard resume, so only the missing ones execute.
set -euo pipefail
ulimit -n 65535 2>/dev/null || true

REPO="${REPO:-/workspace/ts_jepa/ts_JEPA_libero}"
LOG_ROOT="${LOG_ROOT:-/workspace/ts_jepa/logs/plus_eval}"
EVAL_PYTHON="${EVAL_PYTHON:-/opt/venv-libero-plus/bin/python}"
SHARDS="${SHARDS:-64}"
PORTS=(8002 8004 8005 8006 8007 8008 8009 8010)

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

cd "$REPO"
while pgrep -f 'libero/main[.]py' >/dev/null; do sleep 60; done

log "filling libero_10"
i=0
for port in "${PORTS[@]}"; do
  for k in 0 1 2 3 4 5 6 7; do
    RUN_ID=con1con2_plus_full \
    PORT="$port" \
    TASK_SUITE="libero_10" \
    NUM_TASK_SHARDS="$SHARDS" \
    TASK_SHARD_ID="$i" \
    EVAL_PYTHON="$EVAL_PYTHON" \
    EVAL_GPU="$((k % 4))" \
    MUJOCO_EGL_DEVICE_ID="$((k % 4))" \
    nohup bash scripts/run_libero_evaluation.sh plus \
      >"$LOG_ROOT/con1con2_libero_10_fill.shard-$(printf '%02d' "$i").log" 2>&1 &
    i=$((i + 1))
  done
done
sleep 30
log "libero_10 fill launched ($(pgrep -f 'libero/main[.]py' | wc -l) evaluators)"
