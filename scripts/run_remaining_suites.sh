#!/usr/bin/env bash
# Wait for the current suite to drain, then sweep the remaining LIBERO-Plus
# suites with the same 64-shard / 8-server / 4-GPU-EGL layout.
set -euo pipefail
ulimit -n 65535 2>/dev/null || true

REPO="${REPO:-/workspace/ts_jepa/ts_JEPA_libero}"
LOG_ROOT="${LOG_ROOT:-/workspace/ts_jepa/logs/plus_eval}"
EVAL_PYTHON="${EVAL_PYTHON:-/opt/venv-libero-plus/bin/python}"
SHARDS="${SHARDS:-32}"
PORTS=(8002 8004 8005 8006 8007 8008 8009 8010 8011 8012 8013 8014)
SUITES=(libero_spatial libero_object libero_goal)

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

cd "$REPO"
for suite in "${SUITES[@]}"; do
  while pgrep -f 'libero/main[.]py' >/dev/null; do sleep 60; done
  log "launching $suite"
  i=0
  for port in "${PORTS[@]}"; do
    for k in 0 1 2; do
      RUN_ID=con1con2_plus_full \
      PORT="$port" \
      TASK_SUITE="$suite" \
      NUM_TASK_SHARDS="$SHARDS" \
      TASK_SHARD_ID="$i" \
      EVAL_PYTHON="$EVAL_PYTHON" \
      EVAL_GPU="$((k % 4))" \
      MUJOCO_EGL_DEVICE_ID="$((k % 4))" \
      nohup bash scripts/run_libero_evaluation.sh plus \
        >"$LOG_ROOT/con1con2_${suite}_s${SHARDS}.shard-$(printf '%02d' "$i").log" 2>&1 &
      i=$((i + 1))
    done
  done
  sleep 30
  log "$suite launched ($(pgrep -f 'libero/main[.]py' | wc -l) evaluators)"
done
log "all remaining suites launched"
