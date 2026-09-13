#!/usr/bin/env bash
# Unattended chain for the noexec-scratch box: wait for the RoboTwin randomized
# finetune to exit, stage the simulator, then evaluate the final checkpoint on
# the 20 benchmark tasks, Clean and Random, in place.
#
# Why in place: /dev/shm is mounted noexec on this box, so anything that has to
# load a .so (the simulator venv) must live on the root disk. /opt/rt-eval is a
# venv whose base is the training venv's Python and which picks up the training
# venv's packages through a .pth, so the same interpreter serves the policy
# server and the simulator client. The simulator assets are data and go to
# /dev/shm, which is why the asset staging has to wait for the trainer to exit
# (the checkpoint working set is what fills the scratch mount while it runs).
#
# Env knobs:
#   RUN_NAME     exp name under CKPT_ROOT                 (robotwin_random20_ft)
#   CKPT_ROOT    checkpoint root                          (/dev/shm/rt_ft/ckpt/pi05_robotwin_random20_ft)
#   ROOT         openpi checkout                          (/workspace/robotwin_ws)
#   BENCH        RoboTwin checkout (sim + XPolicyLab)     (/dev/shm/robotwin/code)
#   ASSET_SRC    ssh source for a staged RoboTwin checkout (root@93.91.156.102:/workspace/robotwin/code)
#   EVAL_PY      simulator interpreter                    (/opt/rt-eval/bin/python)
#   POLICY_PY    policy-server interpreter                (/opt/rt-eval/bin/python)
#   STAGE_ASSETS 1 to rsync the simulator from ASSET_SRC when assets are missing
#   TRIALS WORKERS POLICY_GPU ENV_GPU
set -uo pipefail

RUN_NAME=${RUN_NAME:-robotwin_random20_ft}
TRAIN_MATCH=${TRAIN_MATCH:-train\.py pi05_robotwin_random20_ft}
CKPT_ROOT=${CKPT_ROOT:-/dev/shm/rt_ft/ckpt/pi05_robotwin_random20_ft}
CKPT_DIR="${CKPT_ROOT}/${RUN_NAME}"
ROOT=${ROOT:-/workspace/robotwin_ws}
BENCH=${BENCH:-/dev/shm/robotwin/code}
ASSET_SRC=${ASSET_SRC:-root@93.91.156.102:/workspace/robotwin/code}
EVAL_PY=${EVAL_PY:-/opt/rt-eval/bin/python}
POLICY_PY=${POLICY_PY:-/opt/rt-eval/bin/python}
STAGE_ASSETS=${STAGE_ASSETS:-1}
TRIALS=${TRIALS:-25}
WORKERS=${WORKERS:-8}
POLICY_GPU=${POLICY_GPU:-0}
ENV_GPU=${ENV_GPU:-1}
POLL=${POLL:-120}
LOG=${LOG:-/dev/shm/rt_ft_eval_chain.log}
LOG_DIR=${LOG_DIR:-/dev/shm/rt_eval_logs}
RESULTS_DIR=${RESULTS_DIR:-/dev/shm/rt_eval_results}

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "chain armed: waiting for the trainer (${TRAIN_MATCH}) to exit"
while pgrep -f "${TRAIN_MATCH}" >/dev/null 2>&1; do
  sleep "$POLL"
done
log "trainer is gone"

# A save that runs out of scratch space leaves an `*.orbax-checkpoint-tmp-*`
# directory behind and no usable step directory (that is what happened at step
# 4000 on 2026-09-13). Wait for any in-flight save to settle so the step we pick
# is the real one, then require its params to be present.
for _ in $(seq 1 30); do
  if ! ls -d "$CKPT_DIR"/*.orbax-checkpoint-tmp-* >/dev/null 2>&1; then break; fi
  log "waiting for an in-flight checkpoint save to settle"
  sleep 60
done

STEP=$(ls -1 "$CKPT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
if [[ -z "$STEP" ]]; then
  log "abort: no checkpoint under $CKPT_DIR"
  ls -la "$CKPT_DIR" | tee -a "$LOG"
  exit 1
fi
if [[ ! -f "$CKPT_DIR/$STEP/params/manifest.ocdbt" ]]; then
  log "abort: $CKPT_DIR/$STEP/params has no manifest.ocdbt (incomplete save)"
  ls -la "$CKPT_DIR/$STEP" | tee -a "$LOG"
  exit 1
fi
# The trainer exiting is not on its own a reason to evaluate: on 2026-09-13 it
# died at step 6000 when the step-4000 async save failed, and the chain happily
# started evaluating the stale step-2000 checkpoint. Only a run that reached the
# end of the schedule should be evaluated.
EXPECT_MIN_STEP=${EXPECT_MIN_STEP:-9999}
if (( STEP < EXPECT_MIN_STEP )); then
  log "abort: trainer exited at step ${STEP}, expected >= ${EXPECT_MIN_STEP}; it probably crashed"
  tail -5 /dev/shm/rt_ft/train.log 2>/dev/null | tr '\r' '\n' | tail -5 | tee -a "$LOG"
  exit 3
fi
log "final checkpoint step=${STEP}"
df -h /dev/shm | tail -1 | tee -a "$LOG"

# Free the scratch mount: the evaluation reads params only, and every other
# checkpoint plus the optimizer state is dead weight at this point.
for d in "$CKPT_DIR"/*; do
  [[ -d "$d" ]] || continue
  base=$(basename "$d")
  if [[ "$base" =~ ^[0-9]+$ ]]; then
    if [[ "$base" == "$STEP" ]]; then
      rm -rf "$d/train_state" && log "removed train_state from $base (keep params)"
    else
      rm -rf "$d" && log "removed superseded checkpoint $base"
    fi
  else
    rm -rf "$d" && log "removed partial checkpoint dir $base"
  fi
done
df -h /dev/shm | tail -1 | tee -a "$LOG"

# The simulator checkout on this box was downloaded but never staged with its
# assets (the conda bootstrap that would have unpacked them cannot run from a
# noexec mount). Pull the validated checkout from the other box.
asset_bytes=$(du -sb "$BENCH/assets" 2>/dev/null | cut -f1 || echo 0)
if [[ "$STAGE_ASSETS" == "1" && "${asset_bytes:-0}" -lt 1000000000 ]]; then
  log "staging the simulator from ${ASSET_SRC} (assets are ${asset_bytes:-0} bytes)"
  mkdir -p "$BENCH"
  rsync -a --info=progress2 "$ASSET_SRC/" "$BENCH/" >>"$LOG" 2>&1
  log "rsync done; assets now $(du -sh "$BENCH/assets" 2>/dev/null | cut -f1)"
fi

if [[ -f "$BENCH/scripts/update_embodiment_config_path.py" ]]; then
  ( cd "$BENCH" && "$EVAL_PY" scripts/update_embodiment_config_path.py >>"$LOG" 2>&1 ) \
    && log "embodiment config paths expanded"
fi

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
export ROOT BENCH EVAL_PY POLICY_PY CKPT_DIR STEP LOG_DIR RESULTS_DIR
export TRIALS WORKERS POLICY_GPU ENV_GPU

# One task, one episode first: the full suite is 40 client runs, so a failure
# here is worth catching before it.
log "smoke: adjust_bottle / demo_clean, 1 episode"
SMOKE_RESULTS="$RESULTS_DIR/smoke"
mkdir -p "$SMOKE_RESULTS"
ONLY=adjust_bottle TRIALS=1 WORKERS=2 RESULTS_DIR="$SMOKE_RESULTS" \
  bash "$ROOT/scripts/robotwin_eval_suite.sh" >>"$LOG" 2>&1
smoke=$?
tail -3 "$SMOKE_RESULTS/summary_${RUN_NAME}_${STEP}.tsv" 2>/dev/null | tee -a "$LOG"
if [[ "$smoke" -ne 0 ]]; then
  log "smoke failed (exit ${smoke}); see ${LOG_DIR} - not starting the full suite"
  exit 2
fi

log "full suite: 20 tasks x {clean, random} x ${TRIALS} episodes, ${WORKERS} sims"
bash "$ROOT/scripts/robotwin_eval_suite.sh" >>"$LOG" 2>&1
rc=$?
log "suite exit=${rc}; summary:"
cat "$RESULTS_DIR/summary_${RUN_NAME}_${STEP}.tsv" 2>/dev/null | tee -a "$LOG"

# PACE-comparable table (ours vs the paper's pi0.5 column). The reporter ships
# next to this script; it is looked up in a few places so the chain works whether
# it was deployed from the repo or copied to /root.
REPORT=${REPORT:-robotwin_eval_report.py}
for cand in "$ROOT/scripts/$REPORT" "/root/$REPORT" "$(dirname "$0")/$REPORT"; do
  if [[ -f "$cand" ]]; then
    log "report from $cand:"
    "$EVAL_PY" "$cand" "$RESULTS_DIR/summary_${RUN_NAME}_${STEP}.tsv" >>"$LOG" 2>&1
    tail -35 "$LOG" 2>/dev/null
    break
  fi
done
log "chain finished"
