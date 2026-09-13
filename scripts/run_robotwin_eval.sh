#!/usr/bin/env bash
# Closed-loop RoboTwin 2.0 evaluation of an openpi checkpoint, no conda.
#
# The released XPolicyLab harness drives the simulator with a conda-activated
# client and launches the policy server through a `conda info --base` shim. This
# box has neither, so the two halves are started directly:
#   * policy server: XPolicyLab/setup_policy_server.py with our openpi config
#   * environment client: RoboTwin/scripts/eval_policy_xpolicylab.py
#
# Usage:
#   bash scripts/run_robotwin_eval.sh <task> <task_config> <ckpt_dir> <step> <trials> \
#        [policy_gpu] [env_gpu] [exp_name]
# Example:
#   bash scripts/run_robotwin_eval.sh adjust_bottle demo_randomized \
#        /workspace/artifacts/checkpoints_robotwin_random/robotwin_random20_ft 20000 20
set -uo pipefail

TASK=${1:?task}
TASK_CONFIG=${2:?task_config (demo_clean|demo_randomized)}
CKPT_DIR=${3:?checkpoint dir (the run dir, not the step dir)}
STEP=${4:?checkpoint step}
TRIALS=${5:-20}
POLICY_GPU=${6:-0}
ENV_GPU=${7:-0}
EXP=${8:-$(basename "$CKPT_DIR")}

ROOT=${ROOT:-/workspace/robotwin_ws}
BENCH=${BENCH:-/workspace/robotwin/code}
EVAL_PY=${EVAL_PY:-/workspace/robotwin/eval_venv/bin/python}
POLICY_PY=${POLICY_PY:-$ROOT/.venv/bin/python}
TRAIN_CONFIG=${TRAIN_CONFIG:-pi05_robotwin_random20_ft}
ASSET_ID=${ASSET_ID:-robotwin_random20}
PORT=${PORT:-$(( 20000 + RANDOM % 20000 ))}
LOG_DIR=${LOG_DIR:-/workspace/robotwin_eval_logs}

mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/${TASK}_${TASK_CONFIG}_${STEP}_server.log"
CLIENT_LOG="$LOG_DIR/${TASK}_${TASK_CONFIG}_${STEP}_client.log"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill -TERM -- -"${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[eval] task=$TASK config=$TASK_CONFIG ckpt=$CKPT_DIR step=$STEP trials=$TRIALS port=$PORT"

setsid env \
  PYTHONPATH="$BENCH:$ROOT/src:$ROOT/packages/openpi-client/src" \
  CUDA_VISIBLE_DEVICES="$POLICY_GPU" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}" \
  HF_HOME=/workspace/.hf_home \
  "$POLICY_PY" "$BENCH/XPolicyLab/setup_policy_server.py" \
    --config_path "$BENCH/XPolicyLab/policy/Pi_05/deploy.yml" \
    --overrides \
      port="$PORT" \
      host=127.0.0.1 \
      bench_name=RoboTwin \
      task_name="$TASK" \
      ckpt_name="$EXP" \
      env_cfg_type=aloha_agilex \
      action_type=joint \
      seed=0 \
      policy_name=Pi_05 \
      train_config_name="$TRAIN_CONFIG" \
      repo_id="$ASSET_ID" \
      model_path="$CKPT_DIR" \
      checkpoint_num="$STEP" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "[eval] server pid=${SERVER_PID}; waiting for port ${PORT}"
for _ in $(seq 1 180); do
  if grep -q "Listening\|listening\|serve_forever\|Server started" "$SERVER_LOG" 2>/dev/null; then break; fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[eval] server died; tail of $SERVER_LOG:"; tail -25 "$SERVER_LOG"; exit 1
  fi
  sleep 5
done

cd "$BENCH"
PYTHONPATH="$BENCH" \
  CUDA_VISIBLE_DEVICES="$ENV_GPU" \
  "$EVAL_PY" scripts/eval_policy_xpolicylab.py \
    --bench_name RoboTwin \
    --task_name "$TASK" \
    --env_cfg_type aloha_agilex \
    --policy_name Pi_05 \
    --host 127.0.0.1 \
    --port "$PORT" \
    --protocol ws \
    --eval_batch true \
    --root_dir "$BENCH" \
    --device_id "$ENV_GPU" \
    --seed 0 \
    --task_config "$TASK_CONFIG" \
    --test_num "$TRIALS" \
    2>&1 | tee "$CLIENT_LOG"

echo "[eval] done: $TASK ($TASK_CONFIG); client log $CLIENT_LOG"
