#!/usr/bin/env bash
# Unattended chain across the two boxes: wait for the RoboTwin randomized
# finetune on this box, ship the final params to the simulator box, and run the
# 20-task Clean/Random closed-loop evaluation there.
#
# Why the simulator box and not this one: RoboTwin validates every seed with an
# expert demonstration before the policy runs (`expert_check`, default on), and
# that demonstration is planned by curobo. This box's simulator environment has
# no curobo (building it needs the CUDA toolkit that matches torch, which the
# training venv's cu126 torch does not), so a run here could only be done with
# `--expert_check false` - a protocol deviation from the numbers we compare to.
# The other box's eval environment has curobo working, 192 cores at load ~9, and 76 GB
# free per GPU while it trains, so the official protocol is affordable there.
#
# Env: RUN_NAME TRAIN_MATCH LOCAL_CKPT_ROOT REMOTE REMOTE_PORT REMOTE_ROOT
#      REMOTE_CKPT_ROOT TRIALS WORKERS POLICY_GPU ENV_GPU LOG
set -uo pipefail

RUN_NAME=${RUN_NAME:-robotwin_random20_ft}
TRAIN_MATCH=${TRAIN_MATCH:-train\.py pi05_robotwin_random20_ft}
LOCAL_CKPT_ROOT=${LOCAL_CKPT_ROOT:-/dev/shm/rt_ft/ckpt/pi05_robotwin_random20_ft}
LOCAL_CKPT_DIR="${LOCAL_CKPT_ROOT}/${RUN_NAME}"

REMOTE=${REMOTE:-root@93.91.156.102}
REMOTE_PORT=${REMOTE_PORT:-43728}
REMOTE_ROOT=${REMOTE_ROOT:-/workspace/robotwin_ws}
REMOTE_CKPT_ROOT=${REMOTE_CKPT_ROOT:-/dev/shm/ckpt_robotwin/pi05_robotwin_random20_ft}
REMOTE_CKPT_DIR="${REMOTE_CKPT_ROOT}/${RUN_NAME}"
REMOTE_LOG_DIR=${REMOTE_LOG_DIR:-/dev/shm/rt_eval_logs}
REMOTE_RESULTS_DIR=${REMOTE_RESULTS_DIR:-/dev/shm/rt_eval_results}

TRIALS=${TRIALS:-25}
# Measured on the preview run: a clean episode is ~1.4 min, a randomized one
# ~6 min (the randomized scenes run the 400-step budget much more often), so 40
# configs x 25 episodes is ~8.5 h at 8 sims. The box has 192 cores and only the
# LIBERO sweep (32 shards) is co-resident, so 12 is a safe step up.
WORKERS=${WORKERS:-12}
POLICY_GPU=${POLICY_GPU:-1}
ENV_GPU=${ENV_GPU:-2}
EXPECT_MIN_STEP=${EXPECT_MIN_STEP:-9999}
POLL=${POLL:-120}
LOG=${LOG:-/dev/shm/rt_split_chain.log}
RESULTS_BACK=${RESULTS_BACK:-/dev/shm/rt_eval_results_remote}
REPORT=${REPORT:-robotwin_eval_report.py}

SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "$REMOTE_PORT" "$REMOTE")
RSYNC_SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p $REMOTE_PORT"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "chain armed: waiting for the trainer (${TRAIN_MATCH}) to exit"
while pgrep -f "${TRAIN_MATCH}" >/dev/null 2>&1; do sleep "$POLL"; done
log "trainer is gone"

# An in-flight save leaves an `*.orbax-checkpoint-tmp-*` directory; wait it out so
# the step we pick is a real one.
for _ in $(seq 1 30); do
  if ! ls -d "$LOCAL_CKPT_DIR"/*.orbax-checkpoint-tmp-* >/dev/null 2>&1; then break; fi
  log "waiting for an in-flight checkpoint save to settle"; sleep 60
done

STEP=$(ls -1 "$LOCAL_CKPT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
if [[ -z "$STEP" ]]; then
  log "abort: no checkpoint under $LOCAL_CKPT_DIR"; exit 1
fi
if [[ ! -f "$LOCAL_CKPT_DIR/$STEP/params/manifest.ocdbt" ]]; then
  log "abort: $LOCAL_CKPT_DIR/$STEP/params has no manifest.ocdbt"; exit 1
fi
if (( STEP < EXPECT_MIN_STEP )); then
  log "abort: trainer exited at step ${STEP} < ${EXPECT_MIN_STEP}; it probably crashed"
  exit 3
fi
log "final checkpoint step=${STEP}"

# The optimizer state stays on disk. It is 19 GB the evaluation never reads, but
# it is also the only way to continue this run: deleting it would turn any later
# "train it longer" into a restart from the base weights.

log "copying params to ${REMOTE}:${REMOTE_CKPT_DIR}/${STEP}"
"${SSH[@]}" "mkdir -p '${REMOTE_CKPT_DIR}/${STEP}'"
rsync -a --info=progress2 -e "$RSYNC_SSH" \
  "$LOCAL_CKPT_DIR/$STEP/params" "$LOCAL_CKPT_DIR/$STEP/assets" "$LOCAL_CKPT_DIR/$STEP/_CHECKPOINT_METADATA" \
  "${REMOTE}:${REMOTE_CKPT_DIR}/${STEP}/" >>"$LOG" 2>&1
log "copy done; remote size: $("${SSH[@]}" "du -sh '${REMOTE_CKPT_DIR}/${STEP}'" 2>/dev/null)"

"${SSH[@]}" "ls '${REMOTE_CKPT_DIR}/${STEP}/params/manifest.ocdbt' >/dev/null 2>&1" \
  || { log "abort: remote copy is incomplete"; exit 4; }

# One episode of one task first, under the official protocol.
log "remote smoke: adjust_bottle / demo_clean, 1 episode"
"${SSH[@]}" "cd '${REMOTE_ROOT}' && ROOT='${REMOTE_ROOT}' CKPT_DIR='${REMOTE_CKPT_DIR}' STEP='${STEP}' \
  TRIALS=1 WORKERS=2 POLICY_GPU='${POLICY_GPU}' ENV_GPU='${ENV_GPU}' \
  LOG_DIR='${REMOTE_LOG_DIR}/smoke' RESULTS_DIR='${REMOTE_RESULTS_DIR}/smoke' ONLY=adjust_bottle \
  timeout 1800 bash scripts/robotwin_eval_suite.sh" >>"$LOG" 2>&1
smoke_rc=$?
"${SSH[@]}" "cat '${REMOTE_RESULTS_DIR}/smoke/summary_${RUN_NAME}_${STEP}.tsv'" 2>/dev/null | tee -a "$LOG"
if [[ "$smoke_rc" -ne 0 ]]; then
  log "smoke failed (rc=${smoke_rc}); not starting the full suite"
  exit 2
fi

log "full suite on ${REMOTE}: 20 tasks x {clean, random} x ${TRIALS} episodes, ${WORKERS} sims"
"${SSH[@]}" "cd '${REMOTE_ROOT}' && ROOT='${REMOTE_ROOT}' CKPT_DIR='${REMOTE_CKPT_DIR}' STEP='${STEP}' \
  TRIALS='${TRIALS}' WORKERS='${WORKERS}' POLICY_GPU='${POLICY_GPU}' ENV_GPU='${ENV_GPU}' \
  LOG_DIR='${REMOTE_LOG_DIR}' RESULTS_DIR='${REMOTE_RESULTS_DIR}' \
  setsid nohup bash scripts/robotwin_eval_suite.sh >'${REMOTE_RESULTS_DIR}/suite_${STEP}.log' 2>&1 </dev/null & \
  echo started" | tee -a "$LOG"

# Wait for the remote suite to finish, then pull the results back.
for _ in $(seq 1 480); do
  sleep "$POLL"
  running=$("${SSH[@]}" "pgrep -f 'robotwin_eval_suite[.]sh' | head -1" 2>/dev/null || true)
  nrows=$("${SSH[@]}" "wc -l < '${REMOTE_RESULTS_DIR}/summary_${RUN_NAME}_${STEP}.tsv' 2>/dev/null" 2>/dev/null || echo 0)
  log "remote suite running=${running:-no} rows=${nrows}/40"
  if [[ -z "$running" ]] && (( nrows >= 40 )); then break; fi
done

log "pulling results back"
mkdir -p "$RESULTS_BACK"
rsync -a -e "$RSYNC_SSH" "${REMOTE}:${REMOTE_RESULTS_DIR}/" "$RESULTS_BACK/" >>"$LOG" 2>&1
SUMMARY="$RESULTS_BACK/summary_${RUN_NAME}_${STEP}.tsv"
log "summary:"; cat "$SUMMARY" 2>/dev/null | tee -a "$LOG"

for cand in "/root/$REPORT" "$(dirname "$0")/$REPORT" "$REMOTE_ROOT/scripts/$REPORT"; do
  if [[ -f "$cand" ]]; then
    log "report from $cand:"
    python3 "$cand" "$SUMMARY" >>"$LOG" 2>&1
    tail -40 "$LOG"
    break
  fi
done
log "chain finished"
