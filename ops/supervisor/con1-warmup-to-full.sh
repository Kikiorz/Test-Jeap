#!/bin/bash
set -euo pipefail

warmup_name=con1_stage2_warmup
full_name=con1_stage2_full
log_path=/workspace/artifacts/logs/con1_stage2_warmup.log
checkpoint_root=/workspace/artifacts/checkpoints/pi05_libero_vjepa_con1_warmup/h10_t16_d128_nojl
seen_running=0

while true; do
  status="$(supervisorctl status "${warmup_name}" || true)"
  if printf '%s\n' "${status}" | grep -Eq 'FATAL|BACKOFF|UNKNOWN'; then
    printf '%s\n' "${status}"
    exit 1
  fi
  if printf '%s\n' "${status}" | grep -q RUNNING; then
    seen_running=1
  fi
  if [[ "${seen_running}" == "1" ]] && printf '%s\n' "${status}" | grep -Eq 'STOPPED|EXITED'; then
    final_checkpoint="$(find "${checkpoint_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
      | grep -E '^[0-9]+$' | sort -n | tail -1)"
    if [[ "${final_checkpoint}" != "4999" ]] || [[ ! -d "${checkpoint_root}/4999/params" ]]; then
      echo "Warm-up ended without the required step-4999 params checkpoint" >&2
      exit 1
    fi
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
if not records or records[0]["step"] != 0 or records[-1]["step"] < 4900:
    raise SystemExit("Warm-up log does not contain the required step-0 through step-4900 records")
if not all(math.isfinite(value) for record in records for value in record.values()):
    raise SystemExit("Warm-up contains non-finite metrics")
first, last = records[0], records[-1]
if not last["change"] < first["change"]:
    raise SystemExit(f"Change flow did not improve: {first['change']} -> {last['change']}")
if not last["action"] < first["action"]:
    raise SystemExit(f"Action flow did not improve: {first['action']} -> {last['action']}")
print({"event": "warmup_health_pass", "first": first, "last": last}, flush=True)
PY
    supervisorctl start "${full_name}"
    exit 0
  fi
  sleep 60
done
