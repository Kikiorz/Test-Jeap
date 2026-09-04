#!/bin/bash
set -euo pipefail
cd /workspace/ts_JEPA_con

while true; do
  status="$(supervisorctl status con1_precompute_displacement:* || true)"
  if printf '%s\n' "${status}" | grep -Eq 'FATAL|BACKOFF|UNKNOWN'; then
    printf '%s\n' "${status}"
    exit 1
  fi
  running="$(printf '%s\n' "${status}" | grep -c RUNNING || true)"
  if [[ "${running}" == "0" ]]; then
    complete="$(find /workspace/artifacts/vjepa_targets/libero_vjepa2_1_vitg_384_offset10_displacement1408/targets -name 'episode_*.npy' | wc -l)"
    if [[ "${complete}" != "1693" ]]; then
      printf 'precompute stopped with %s/1693 episodes\n' "${complete}"
      exit 1
    fi
    supervisorctl start con1_stage1_nojl
    exit 0
  fi
  sleep 60
done
