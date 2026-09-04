#!/bin/bash
set -euo pipefail

program_name=con1_stage2_full
checkpoint_dir=/workspace/artifacts/checkpoints/pi05_libero_vjepa_con1/h10_t16_d128_nojl/10000
log_path=/workspace/artifacts/logs/con1_stage2_full.log
health_path=/workspace/artifacts/eval/con1_10k_plus_l45_paired40/train_health.json
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
    /workspace/openpi_jepawam/.venv/bin/python - "${log_path}" "${health_path}" <<'PY'
import json
import math
from pathlib import Path
import re
import statistics
import sys

pattern = re.compile(
    r"^Step (?P<step>\d+): .*change_flow_loss=(?P<change>[0-9.eE+-]+).*"
    r"flow_loss=(?P<action>[0-9.eE+-]+).*grad_norm=(?P<grad>[0-9.eE+-]+).*loss=(?P<loss>[0-9.eE+-]+)"
)
by_step = {}
with open(sys.argv[1]) as handle:
    for line in handle:
        if match := pattern.search(line):
            record = {key: float(match[key]) for key in ("step", "change", "action", "grad", "loss")}
            if 1000 <= record["step"] <= 10000:
                by_step[int(record["step"])] = record
records = [by_step[key] for key in sorted(by_step)]
if not records or records[0]["step"] > 1100 or records[-1]["step"] < 9900:
    raise SystemExit("Full-phase log does not cover checkpoint 1000 through the 10k gate")
if not all(math.isfinite(value) for record in records for value in record.values()):
    raise SystemExit("Full phase contains non-finite metrics")
early = [record for record in records if 1000 <= record["step"] < 2000]
prior = [record for record in records if 8000 <= record["step"] < 9000]
late = [record for record in records if 9000 <= record["step"] <= 10000]
if len(early) < 5 or len(prior) < 5 or len(late) < 5:
    raise SystemExit("Insufficient rolling metrics for the 10k health gate")
means = lambda values, key: statistics.fmean(record[key] for record in values)
summary = {
    "event": "con1_10k_health_pass",
    "early": {key: means(early, key) for key in ("change", "action", "loss", "grad")},
    "prior": {key: means(prior, key) for key in ("change", "action", "loss", "grad")},
    "late": {key: means(late, key) for key in ("change", "action", "loss", "grad")},
}
if summary["late"]["change"] >= summary["early"]["change"]:
    raise SystemExit(f"Change loss did not improve: {summary}")
if summary["late"]["action"] > 1.25 * summary["early"]["action"]:
    raise SystemExit(f"Action loss regressed by more than 25%: {summary}")
summary["has_improvement_room"] = (
    summary["late"]["loss"] <= 0.99 * summary["prior"]["loss"]
    or summary["late"]["action"] <= 0.99 * summary["prior"]["action"]
)
destination = Path(sys.argv[2])
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(summary, indent=2) + "\n")
print(summary, flush=True)
PY
    supervisorctl stop "${program_name}"
    echo "Paused Con1 at finalized checkpoint 10000 for paired LIBERO-Plus L4/L5 gate."
    supervisorctl start con1_plus_10k_gate
    exit 0
  fi
  if [[ "${seen_running}" == "1" ]] && printf '%s\n' "${status}" | grep -Eq 'STOPPED|EXITED'; then
    echo "Con1 ended before finalized checkpoint 10000" >&2
    exit 1
  fi
  sleep 30
done
