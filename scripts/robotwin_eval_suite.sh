#!/usr/bin/env bash
# Closed-loop RoboTwin 2.0 evaluation of one checkpoint over all 20 benchmark
# tasks, Clean and Random, with one policy server shared by every client run.
#
# Env: TRIALS (episodes per task/config), WORKERS (parallel sims per task),
#      CKPT_DIR, STEP, POLICY_GPU, ENV_GPU, LOG_DIR, RESULTS_DIR, ONLY (task list).
set -uo pipefail

TASKS=(
  adjust_bottle
  beat_block_hammer
  click_alarmclock
  click_bell
  dump_bin_bigbin
  grab_roller
  handover_mic
  lift_pot
  place_bread_basket
  place_bread_skillet
  place_burger_fries
  place_cans_plasticbox
  place_empty_cup
  place_object_basket
  place_shoe
  press_stapler
  shake_bottle_horizontally
  shake_bottle
  stack_bowls_three
  stack_bowls_two
)

ROOT=${ROOT:-/workspace/robotwin_ws}
BENCH=${BENCH:-/workspace/robotwin/code}
EVAL_PY=${EVAL_PY:-/workspace/robotwin/eval_venv/bin/python}
POLICY_PY=${POLICY_PY:-$ROOT/.venv/bin/python}
CKPT_DIR=${CKPT_DIR:-/workspace/artifacts/checkpoints_robotwin_random/robotwin_random20_ft}
if [[ -z "${STEP:-}" ]]; then
  STEP=$(ls -1 "$CKPT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
fi
if [[ -z "${STEP:-}" ]]; then
  echo "[suite] no checkpoint found under $CKPT_DIR" >&2
  exit 1
fi
TRIALS=${TRIALS:-25}
WORKERS=${WORKERS:-4}
# A client run can livelock: observed on `adjust_bottle / demo_randomized`, where
# the sim workers spun at ~30% CPU with every GPU at 0% and the client log froze
# mid-episode. Without a bound, one hung config stalls the whole 40-config sweep.
# A healthy random config takes ~15-25 min at 8-12 workers, so 45 min is a
# generous ceiling; a killed config is recorded as FAILED and retried.
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-2700}
POLICY_GPU=${POLICY_GPU:-0}
ENV_GPU=${ENV_GPU:-1}
TRAIN_CONFIG=${TRAIN_CONFIG:-pi05_robotwin_random20_ft}
ASSET_ID=${ASSET_ID:-robotwin_random20}
EXP=${EXP:-$(basename "$CKPT_DIR")}
LOG_DIR=${LOG_DIR:-/workspace/robotwin_eval_logs}
RESULTS_DIR=${RESULTS_DIR:-/workspace/robotwin_eval_results}
PORT=${PORT:-$(( 20000 + RANDOM % 20000 ))}
ONLY=${ONLY:-}

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
SUMMARY="$RESULTS_DIR/summary_${EXP}_${STEP}.tsv"

SERVER_LOG="$LOG_DIR/suite_server_${STEP}.log"
if [[ -z "${SERVER_PORT:-}" ]]; then
  echo "[suite] starting policy server on GPU ${POLICY_GPU}, port ${PORT}"
  setsid env \
    PYTHONPATH="$BENCH:$ROOT/src:$ROOT/packages/openpi-client/src" \
    CUDA_VISIBLE_DEVICES="$POLICY_GPU" \
    XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.35}" \
    HF_HOME=/workspace/.hf_home \
    "$POLICY_PY" "$BENCH/XPolicyLab/setup_policy_server.py" \
      --config_path "$BENCH/XPolicyLab/policy/Pi_05/deploy.yml" \
      --overrides \
        port="$PORT" host=127.0.0.1 bench_name=RoboTwin \
        task_name=adjust_bottle ckpt_name="$EXP" env_cfg_type=aloha_agilex \
        action_type=joint seed=0 policy_name=Pi_05 \
        train_config_name="$TRAIN_CONFIG" repo_id="$ASSET_ID" \
        model_path="$CKPT_DIR" checkpoint_num="$STEP" \
    >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 180); do
    if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
      exec 3>&- 3<&- 2>/dev/null || true
      echo "[suite] server listening"
      break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[suite] server died; tail of $SERVER_LOG:"; tail -25 "$SERVER_LOG"; exit 1
    fi
    sleep 5
  done
  trap 'kill -TERM -- -"${SERVER_PID}" 2>/dev/null || true' EXIT
else
  echo "[suite] reusing server on port ${PORT}"
fi

for task in "${TASKS[@]}"; do
  if [[ -n "$ONLY" && " $ONLY " != *" $task "* ]]; then
    continue
  fi
  for spec in "demo_clean:seen" "demo_randomized:unseen"; do
    cfg="${spec%%:*}"
    instruction="${spec##*:}"
    client_log="$LOG_DIR/${task}_${cfg}_${STEP}_client.log"
    if [[ -f "$RESULTS_DIR/${task}_${cfg}_${STEP}.txt" ]]; then
      echo "[suite] ${task}/${cfg} already evaluated; skipping"
      continue
    fi
    echo "[suite] ${task} / ${cfg} (instruction=${instruction}) trials=${TRIALS} workers=${WORKERS}"
    ( cd "$BENCH" && PYTHONPATH="$BENCH" CUDA_VISIBLE_DEVICES="$ENV_GPU" \
      timeout --signal=TERM --kill-after=120 "$CLIENT_TIMEOUT" \
      "$EVAL_PY" scripts/eval_policy_xpolicylab.py \
        --bench_name RoboTwin \
        --task_name "$task" \
        --env_cfg_type aloha_agilex \
        --policy_name Pi_05 \
        --host 127.0.0.1 \
        --port "$PORT" \
        --protocol ws \
        --eval_batch true \
        --num_workers "$WORKERS" \
        --root_dir "$BENCH" \
        --device_id "$ENV_GPU" \
        --seed 0 \
        --instruction_type "$instruction" \
        --task_config "$cfg" \
        --test_num "$TRIALS" ) >"$client_log" 2>&1
    rate=$(grep -o "Final batch success rate: [0-9]*/[0-9]* = [0-9.]*%" "$client_log" | tail -1)
    if [[ -z "$rate" ]]; then
      rate="FAILED (see $client_log)"
    fi
    printf "%s\t%s\t%s\n" "$task" "$cfg" "$rate" >>"$SUMMARY"
    cp "$client_log" "$RESULTS_DIR/${task}_${cfg}_${STEP}.txt" 2>/dev/null || true
    echo "[suite] ${task}/${cfg}: ${rate}"
  done
done

echo "[suite] summary written to $SUMMARY"
