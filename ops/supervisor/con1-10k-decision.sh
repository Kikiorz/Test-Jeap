#!/bin/bash
set -euo pipefail

summary=/workspace/artifacts/eval/con1_10k_plus_l45_paired40/paired_summary.json
health=/workspace/artifacts/eval/con1_10k_plus_l45_paired40/train_health.json
test -f "${summary}"
test -f "${health}"

decision="$(/workspace/openpi_jepawam/.venv/bin/python - "${summary}" "${health}" <<'PY'
import json
import sys

value = json.load(open(sys.argv[1]))
health = json.load(open(sys.argv[2]))
if value.get("episodes_per_model") != 40:
    raise SystemExit(f"Unexpected gate size: {value.get('episodes_per_model')}")
if value.get("baseline_successes") is None or value.get("con1_successes") is None:
    raise SystemExit("Gate summary is missing success counts")
if not isinstance(health.get("has_improvement_room"), bool):
    raise SystemExit("Training health summary is missing has_improvement_room")
continue_con1 = value["success_rate_delta"] > 0 or (
    value["success_rate_delta"] == 0 and health["has_improvement_room"]
)
print("continue_con1" if continue_con1 else "enter_con2")
PY
)"

echo "Con1 10k decision: ${decision}"
if [[ "${decision}" == "continue_con1" ]]; then
  echo "Con1 improved over baseline; resume formal Con1 training."
  supervisorctl start con1_stage2_full
else
  echo "Con1 did not improve over baseline; start Con2 offline adapter training."
  supervisorctl start con2_offline
fi
