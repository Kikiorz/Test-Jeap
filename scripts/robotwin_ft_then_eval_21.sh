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

STEP=$(ls -1 "$CKPT_DIR" 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1)
if [[ -z "$STEP" ]]; then
  log "abort: no checkpoint under $CKPT_DIR"
  exit 1
fi
log "final checkpoint step=${STEP}"

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
log "chain finished"
