#!/bin/bash
set -euo pipefail

seen_running=0

while true; do
  status="$(supervisorctl status con1_stage1_nojl || true)"
  if printf '%s\n' "${status}" | grep -Eq 'FATAL|BACKOFF|UNKNOWN'; then
    printf '%s\n' "${status}"
    exit 1
  fi
  if printf '%s\n' "${status}" | grep -q RUNNING; then
    seen_running=1
  fi
  if [[ "${seen_running}" == "1" ]] && printf '%s\n' "${status}" | grep -Eq 'STOPPED|EXITED'; then
    if [[ -f /workspace/artifacts/con1/stage1_h10_t16_d128_nojl/summary.json ]] && \
       [[ -f /workspace/artifacts/con1/stage1_h10_t16_d128_nojl/change_targets_raw/manifest.json ]]; then
      supervisorctl start con1_stage2_warmup
      exit 0
    fi
  fi
  sleep 60
done
