#!/bin/bash
set -euo pipefail

# Paired 40-task LIBERO-Plus development gate for the Con1 step-10k
# checkpoint.  Four suites run in parallel, one per GPU.  Within every suite
# we evaluate five L4 and five L5 tasks selected deterministically while
# balancing the seven official perturbation categories across strata.

repo_root=/workspace/ts_JEPA_con
policy_python=/workspace/openpi_jepawam/.venv/bin/python
eval_python=/workspace/openpi_jepawam/examples/libero/.venv-plus/bin/python
plus_root=/workspace/Test-Jeap/third_party/LIBERO-plus
classification_path=${plus_root}/libero/libero/benchmark/task_classification.json
base_checkpoint=/workspace/artifacts/models/jepa_wam_pi05_60k/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/59999
con1_checkpoint=/workspace/artifacts/checkpoints/pi05_libero_vjepa_con1/h10_t16_d128_nojl/10000
result_root=/workspace/artifacts/eval/con1_10k_plus_l45_paired40
manifest=${result_root}/manifest.json
seed=431

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

test -x "${policy_python}"
test -x "${eval_python}"
test -f "${classification_path}"
test -d "${base_checkpoint}/params"
test -f "${con1_checkpoint}/_CHECKPOINT_METADATA"
test -d "${con1_checkpoint}/params"
test "$(git -C "${plus_root}" rev-parse HEAD)" = 4976dc30028e805ff8094b55501d532c48fec182
mkdir -p "${result_root}/logs"

"${policy_python}" - "${classification_path}" "${manifest}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

source = Path(sys.argv[1])
destination = Path(sys.argv[2])
raw = source.read_bytes()
payload = json.loads(raw)
categories = sorted({row["category"] for rows in payload.values() for row in rows})
manifest = {
    "schema_version": 1,
    "benchmark_revision": "4976dc30028e805ff8094b55501d532c48fec182",
    "classification_sha256": hashlib.sha256(raw).hexdigest(),
    "seed": 431,
    "selection": (
        "five cyclically balanced categories per suite/difficulty stratum; "
        "minimum sha256(seed|suite|difficulty|category|task_name)"
    ),
    "suites": {},
}
stratum = 0
for suite, rows in payload.items():
    selected = []
    for difficulty in (4, 5):
        offset = (stratum * 2) % len(categories)
        selected_categories = [categories[(offset + index) % len(categories)] for index in range(5)]
        for category in selected_categories:
            candidates = [
                row
                for row in rows
                if row.get("difficulty_level") == difficulty and row["category"] == category
            ]
            if not candidates:
                raise SystemExit(f"No candidate for {suite=} {difficulty=} {category=}")

            def selection_key(row):
                value = f"431|{suite}|{difficulty}|{category}|{row['name']}"
                return hashlib.sha256(value.encode()).hexdigest()

            row = min(candidates, key=selection_key)
            selected.append(
                {
                    "task_id": row["id"] - 1,
                    "task_name": row["name"],
                    "category": category,
                    "difficulty_level": difficulty,
                }
            )
        stratum += 1
    if len(selected) != 10 or sum(row["difficulty_level"] == 4 for row in selected) != 5:
        raise SystemExit(f"Invalid selected stratum for {suite}")
    manifest["suites"][suite] = selected
destination.write_text(json.dumps(manifest, indent=2) + "\n")
PY

suites=(libero_spatial libero_object libero_goal libero_10)
server_pids=()

cleanup_servers() {
  if ((${#server_pids[@]})); then
    kill "${server_pids[@]}" 2>/dev/null || true
    for pid in "${server_pids[@]}"; do
      wait "${pid}" 2>/dev/null || true
    done
    server_pids=()
  fi
}
trap cleanup_servers EXIT INT TERM

wait_for_port() {
  local port="$1"
  local pid="$2"
  for _ in $(seq 1 180); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return 1
    fi
    if "${policy_python}" - "${port}" <<'PY'
import socket
import sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
    pass
PY
    then
      return 0
    fi
    sleep 2
  done
  return 1
}

run_suite() {
  local label="$1"
  local index="$2"
  local suite="$3"
  local port=$((8200 + index))
  local task_id task_end results_path
  mapfile -t task_ids < <("${policy_python}" - "${manifest}" "${suite}" <<'PY'
import json
import sys
value = json.load(open(sys.argv[1]))
for row in value["suites"][sys.argv[2]]:
    print(row["task_id"])
PY
  )
  for task_id in "${task_ids[@]}"; do
    task_end=$((task_id + 1))
    results_path=$(printf '%s/%s/plus-%s-task-%05d.jsonl' "${result_root}" "${label}" "${suite}" "${task_id}")
    RUN_ID="${label}_seed${seed}" \
    HOST=127.0.0.1 \
    PORT="${port}" \
    TASK_SUITE="${suite}" \
    TASK_START="${task_id}" \
    TASK_END="${task_end}" \
    NUM_TRIALS=1 \
    REPLAN_STEPS=10 \
    SAVE_VIDEO=0 \
    EVAL_GPU="${index}" \
    MUJOCO_EGL_DEVICE_ID="${index}" \
    EVAL_PYTHON="${eval_python}" \
    LIBERO_PLUS_ROOT="${plus_root}" \
    RESULTS_ROOT="${result_root}/${label}" \
    RESULTS_PATH="${results_path}" \
    LIBERO_CONFIG_PATH="${result_root}/${label}/config_${suite}" \
    BENCHMARK_REVISION="$(git -C "${plus_root}" rev-parse HEAD)" \
    RESUME=1 \
      bash "${repo_root}/scripts/run_libero_evaluation.sh" plus \
      >>"${result_root}/logs/${label}_eval_${suite}.log" 2>&1
  done
}

run_model() {
  local label="$1"
  local config_name="$2"
  local checkpoint="$3"
  local index suite
  local client_pids=()
  local failed=0

  cleanup_servers
  for index in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${index}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.82 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    PYTHONUNBUFFERED=1 \
      "${policy_python}" "${repo_root}/scripts/serve_policy.py" \
        --env LIBERO \
        --port "$((8200 + index))" \
        policy:checkpoint \
        --policy.config "${config_name}" \
        --policy.dir "${checkpoint}" \
        >"${result_root}/logs/${label}_server_${index}.log" 2>&1 &
    server_pids+=("$!")
  done
  for index in 0 1 2 3; do
    wait_for_port "$((8200 + index))" "${server_pids[$index]}" || {
      echo "${label} policy server ${index} failed to become ready" >&2
      return 1
    }
  done
  for index in 0 1 2 3; do
    suite="${suites[$index]}"
    run_suite "${label}" "${index}" "${suite}" &
    client_pids+=("$!")
  done
  for pid in "${client_pids[@]}"; do
    wait "${pid}" || failed=1
  done
  cleanup_servers
  return "${failed}"
}

run_model baseline pi05_libero_vjepa_aux "${base_checkpoint}"
run_model con1_10k pi05_libero_vjepa_con1 "${con1_checkpoint}"

"${policy_python}" - "${result_root}" "${manifest}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
manifest = json.load(open(sys.argv[2]))
expected = {
    (suite, row["task_id"]): row
    for suite, rows in manifest["suites"].items()
    for row in rows
}

def load(label):
    records = {}
    for path in sorted((root / label).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            value = json.loads(line)
            if value.get("record_type") != "episode":
                continue
            key = (value["task_suite_name"], value["task_id"])
            if key in records:
                raise SystemExit(f"Duplicate episode for {label}: {key}")
            records[key] = value
    return records

baseline = load("baseline")
con1 = load("con1_10k")
if set(baseline) != set(expected) or set(con1) != set(expected):
    raise SystemExit(
        f"Expected {len(expected)} paired tasks, got baseline={len(baseline)} con1={len(con1)}"
    )
for label, records in (("baseline", baseline), ("con1_10k", con1)):
    for key, record in records.items():
        target = expected[key]
        if record["status"] == "error":
            raise SystemExit(f"Infrastructure/policy error in {label}: {key}")
        for field in ("task_name", "category", "difficulty_level"):
            if record[field] != target[field]:
                raise SystemExit(f"Manifest mismatch in {label} {key} {field}")

def success(record):
    return record["status"] == "success"

def grouped(records, field):
    output = {}
    for key, record in records.items():
        group = expected[key][field]
        item = output.setdefault(str(group), {"episodes": 0, "successes": 0})
        item["episodes"] += 1
        item["successes"] += int(success(record))
    for item in output.values():
        item["success_rate"] = item["successes"] / item["episodes"]
    return output

b_success = sum(map(success, baseline.values()))
c_success = sum(map(success, con1.values()))
con1_only = [key for key in expected if not success(baseline[key]) and success(con1[key])]
baseline_only = [key for key in expected if success(baseline[key]) and not success(con1[key])]
summary = {
    "protocol": "paired-development-gate-not-leaderboard",
    "seed": manifest["seed"],
    "episodes_per_model": len(expected),
    "baseline_successes": b_success,
    "con1_successes": c_success,
    "success_rate_delta": (c_success - b_success) / len(expected),
    "con1_only": con1_only,
    "baseline_only": baseline_only,
    "baseline_by_difficulty": grouped(baseline, "difficulty_level"),
    "con1_by_difficulty": grouped(con1, "difficulty_level"),
    "baseline_by_category": grouped(baseline, "category"),
    "con1_by_category": grouped(con1, "category"),
}
(root / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
PY

supervisorctl start con1_10k_decision
