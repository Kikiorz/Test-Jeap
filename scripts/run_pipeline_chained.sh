#!/usr/bin/env bash
# Chained rebuild so the GPUs never idle between stages:
#   wait for V-JEPA weights -> frame states -> anchored cache -> head -> arms
# Budgets are trimmed for the current deadline; the head still sees 8k steps
# and each arm 3k steps (which crosses the 2k stage-1 boundary, so the action
# expert's last four blocks do get trained).
set -euo pipefail

WORK="${WORK:-/workspace/ts_jepa}"
REPO="${REPO:-$WORK/ts_JEPA_libero}"
HEAD_STEPS="${HEAD_STEPS:-8000}"
ARM_STEPS="${ARM_STEPS:-3000}"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# 1. wait for the V-JEPA checkpoint to be fully written
target=16878318788
while true; do
  size=$(stat -c%s "$WORK/vjepa2/vjepa2_1_vitg_384.pt" 2>/dev/null || echo 0)
  [[ "$size" -ge "$target" ]] && break
  log "waiting for V-JEPA weights: $size / $target"
  sleep 120
done
log "V-JEPA weights ready"

# 2. cache (frame states then anchored features)
WORK="$WORK" REPO="$REPO" bash "$WORK/libero_rebuild.sh" cache

# 3. head, then the two arms, with the trimmed budgets
WORK="$WORK" REPO="$REPO" STEPS="$HEAD_STEPS" bash "$WORK/libero_rebuild.sh" head
WORK="$WORK" REPO="$REPO" STEPS="$ARM_STEPS" bash "$WORK/libero_rebuild.sh" arms

log "pipeline done"
