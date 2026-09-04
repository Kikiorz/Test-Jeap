#!/bin/bash
set -euo pipefail

program_name=con1_stage2_full
checkpoint_dir=/workspace/artifacts/checkpoints/pi05_libero_vjepa_con1/h10_t16_d128_nojl/1000
log_path=/workspace/artifacts/logs/con1_stage2_full.log
seen_running=0

while true; do
  status="$(supervisorctl status "${program_name}" || true)"
  if printf '%s\n' "${status}" | grep -Eq 'FATAL|BACKOFF|UNKNOWN'; then
    printf '%s\n' "${status}"
    exit 1
  fi
  if printf '%s\n' "${status}" | grep -q RUNNING; then
    seen_running=1
  fi
  if [[ "${seen_running}" == "1" ]] && \
     [[ -f "${checkpoint_dir}/_CHECKPOINT_METADATA" ]] && \
     [[ -d "${checkpoint_dir}/params" ]]; then
    /workspace/openpi_jepawam/.venv/bin/python - "${log_path}" <<'PY'
import math
import re
import sys

pattern = re.compile(
    r"^Step (?P<step>\d+): .*change_flow_loss=(?P<change>[0-9.eE+-]+).*"
    r"flow_loss=(?P<action>[0-9.eE+-]+).*grad_norm=(?P<grad>[0-9.eE+-]+)"
)
records = []
with open(sys.argv[1]) as handle:
    for line in handle:
        if match := pattern.search(line):
            records.append({key: float(match[key]) for key in ("step", "change", "action", "grad")})
if not records or records[0]["step"] != 0 or records[-1]["step"] < 1000:
    raise SystemExit("Full-phase log does not contain step 0 through step 1000")
if not all(math.isfinite(value) for record in records for value in record.values()):
    raise SystemExit("Full phase contains non-finite metrics")
print({"event": "first_full_checkpoint_ready", "first": records[0], "last": records[-1]}, flush=True)
PY
    supervisorctl stop "${program_name}"
    echo "Paused Con1 full training at finalized checkpoint 1000 for paired LIBERO rollout."
    supervisorctl start con1_paired_smoke
    exit 0
  fi
  if [[ "${seen_running}" == "1" ]] && printf '%s\n' "${status}" | grep -Eq 'STOPPED|EXITED'; then
    echo "Con1 full phase ended before finalized checkpoint 1000" >&2
    exit 1
  fi
  sleep 30
done
