#!/bin/bash
set -euo pipefail

# Minimal policy-level check after the first full Con1 checkpoint.  This is a
# paired development run, not a leaderboard estimate: five task IDs from each
# standard LIBERO suite, one initial state each, with identical seeds.

repo_root=/workspace/ts_JEPA_con
policy_python=/workspace/openpi_jepawam/.venv/bin/python
eval_python=/workspace/openpi_jepawam/examples/libero/.venv-plus/bin/python
standard_libero_root=/workspace/openpi_jepawam/third_party/libero
base_checkpoint=/workspace/artifacts/models/jepa_wam_pi05_60k/checkpoints/openpi/pi05_libero_vjepa_aux/pi05_vjepa_pair32_q64_w01_seed42_fsdp2_b128_continue60k_exact/59999
con1_checkpoint=/workspace/artifacts/checkpoints/pi05_libero_vjepa_con1/h10_t16_d128_nojl/1000
result_root=/workspace/artifacts/eval/con1_first_1k_paired20
seed=431

test -x "${policy_python}"
test -x "${eval_python}"
test -d "${standard_libero_root}/libero/libero"
test -d "${base_checkpoint}/params"
test -f "${con1_checkpoint}/_CHECKPOINT_METADATA"
test -d "${con1_checkpoint}/params"
mkdir -p "${result_root}/logs"

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

run_model() {
  local label="$1"
  local config_name="$2"
  local checkpoint="$3"
  local index port suite
  local client_pids=()
  local failed=0

  cleanup_servers
  for index in 0 1 2 3; do
    port=$((8100 + index))
    CUDA_VISIBLE_DEVICES="${index}" \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.82 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    PYTHONUNBUFFERED=1 \
      "${policy_python}" "${repo_root}/scripts/serve_policy.py" \
        --env LIBERO \
        --port "${port}" \
        policy:checkpoint \
        --policy.config "${config_name}" \
        --policy.dir "${checkpoint}" \
        >"${result_root}/logs/${label}_server_${index}.log" 2>&1 &
    server_pids+=("$!")
  done

  for index in 0 1 2 3; do
    wait_for_port "$((8100 + index))" "${server_pids[$index]}" || {
      echo "${label} policy server ${index} failed to become ready" >&2
      return 1
    }
  done

  for index in 0 1 2 3; do
    suite="${suites[$index]}"
    RUN_ID="${label}_seed${seed}" \
    HOST=127.0.0.1 \
    PORT="$((8100 + index))" \
    TASK_SUITE="${suite}" \
    TASK_START=0 \
    TASK_END=5 \
    NUM_TRIALS=1 \
    REPLAN_STEPS=10 \
    SAVE_VIDEO=0 \
    EVAL_GPU="${index}" \
    EVAL_PYTHON="${eval_python}" \
    STANDARD_LIBERO_ROOT="${standard_libero_root}" \
    RESULTS_ROOT="${result_root}/${label}" \
    RESULTS_PATH="${result_root}/${label}/standard-${suite}.jsonl" \
    LIBERO_CONFIG_PATH="${result_root}/${label}/config_${suite}" \
    BENCHMARK_REVISION="$(git -C "${standard_libero_root}" rev-parse HEAD)" \
    RESUME=1 \
      bash "${repo_root}/scripts/run_libero_evaluation.sh" standard \
      >"${result_root}/logs/${label}_eval_${suite}.log" 2>&1 &
    client_pids+=("$!")
  done

  for pid in "${client_pids[@]}"; do
    wait "${pid}" || failed=1
  done
  cleanup_servers
  return "${failed}"
}

run_model baseline pi05_libero_vjepa_aux "${base_checkpoint}"
run_model con1_1k pi05_libero_vjepa_con1 "${con1_checkpoint}"

"${policy_python}" - "${result_root}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])

def load(label):
    records = {}
    for path in sorted((root / label).glob("standard-*.jsonl")):
        with path.open() as handle:
            for line in handle:
                value = json.loads(line)
                if value.get("record_type") != "episode":
                    continue
                key = (value["task_suite_name"], value["task_id"], value["episode_idx"])
                records[key] = value
    return records

baseline = load("baseline")
con1 = load("con1_1k")
if set(baseline) != set(con1) or len(baseline) != 20:
    raise SystemExit(f"Expected 20 paired episodes, got baseline={len(baseline)} con1={len(con1)}")
if any(record["status"] == "error" for record in [*baseline.values(), *con1.values()]):
    raise SystemExit("At least one paired rollout ended in an infrastructure/policy error")

b_success = sum(record["status"] == "success" for record in baseline.values())
c_success = sum(record["status"] == "success" for record in con1.values())
con1_only = sum(
    baseline[key]["status"] != "success" and con1[key]["status"] == "success" for key in baseline
)
baseline_only = sum(
    baseline[key]["status"] == "success" and con1[key]["status"] != "success" for key in baseline
)
summary = {
    "protocol": "paired-development-smoke-not-leaderboard",
    "seed": 431,
    "episodes_per_model": 20,
    "baseline_successes": b_success,
    "con1_successes": c_success,
    "success_rate_delta": (c_success - b_success) / 20,
    "con1_only_successes": con1_only,
    "baseline_only_successes": baseline_only,
}
(root / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2), flush=True)
PY
