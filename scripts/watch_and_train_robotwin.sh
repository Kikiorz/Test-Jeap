#!/usr/bin/env bash
# Wait for the Con1 feature cache to be marked complete, then launch the two-arm
# RoboTwin ablation. Keeps the GPU box busy without polling by hand.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
FEATURES="${FEATURES:-/workspace/artifacts/con1/robotwin_clean20_19999_features_v1}"
STEPS="${STEPS:-4500}"
LOG="${LOG:-/workspace/robotwin_watch.log}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"
POLL="${POLL:-60}"

cd "$ROOT"
deadline=$(( $(date +%s) + MAX_WAIT_SECONDS ))
while (( $(date +%s) < deadline )); do
  if [[ -f "$FEATURES/manifest.json" ]] && grep -q '"complete": true' "$FEATURES/manifest.json"; then
    episodes=$(python3 -c "import json;print(len(json.load(open('$FEATURES/manifest.json'))['episodes']))")
    echo "[$(date -u +%H:%M:%S)] cache complete with $episodes episodes; starting both arms" | tee -a "$LOG"
    STEPS="$STEPS" bash scripts/run_robotwin_arms.sh
    echo "[$(date -u +%H:%M:%S)] arms finished" | tee -a "$LOG"
    exit 0
  fi
  sleep "$POLL"
done
echo "[$(date -u +%H:%M:%S)] gave up waiting for the cache" | tee -a "$LOG"
exit 1
