#!/usr/bin/env bash
# Paired held-out flow probe for the RoboTwin ablation:
#   base      = arm A's checkpoint with the Con1 correction silenced
#   arm A     = Con1 livecross
#   arm B     = Con1 livecross + Con2 refinement + whole-prefix VLM context
#   arm C     = arm A + action conditioning of the anchored-delta head
#   arm D     = arm A + a 5x learning rate on the anchored-delta head
#               (C and D are probed only when they have a checkpoint at $STEP)
# All three use the same data loader (same dataset, same seed, shuffled) so the
# batches are identical and the comparison is paired.
set -euo pipefail

ROOT="${ROOT:-/workspace/ts_JEPA_robotwin}"
cd "$ROOT"
STEP="${STEP:-4499}"
BATCHES="${BATCHES:-48}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUT_DIR="${OUT_DIR:-/workspace/artifacts/con2}"
CKPT_BASE="${CKPT_BASE:-/workspace/artifacts/checkpoints}"

probe() {
  local config="$1" exp="$2" out="$3"; shift 3
  echo "[$(date -u +%H:%M:%S)] probing $exp -> $out"
  PYTHONPATH="$ROOT/src" CUDA_VISIBLE_DEVICES=0,1,2,3 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnccl.so.2 \
  NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 HF_HOME=/workspace/.hf_home HF_HUB_OFFLINE=1 \
    .venv/bin/python -u scripts/probe_con1_budget_and_conditioning.py \
      --config "$config" --exp-name "$exp" --checkpoint-step "$STEP" \
      --batch-size "$BATCH_SIZE" --batches "$BATCHES" \
      --budgets ${PROBE_BUDGETS:-0.05} \
      --out "$out" "$@"
}

A_CFG=pi05_robotwin_con1_livecross_20k
B_CFG=pi05_robotwin_con1con2_ctx_20k
C_CFG=pi05_robotwin_con1_actcond_20k
D_CFG=pi05_robotwin_con1_headlr_20k

# The probe self-checks that a silenced correction measures exactly zero and
# raises if it does not. That row is guarded so a broken control shows up as a
# missing row instead of silently becoming a wrong baseline, without costing the
# arms that follow.
probe "$A_CFG" robotwin_a_con1 "$OUT_DIR/robotwin_ab_base.json" --zero-correction \
  || echo "[$(date -u +%H:%M:%S)] ZERO-CORRECTION CONTROL FAILED its self-check; no base row"
# True released base: the config's own weight loader restores the released
# checkpoint into every non-Con1/Con2 parameter and the correction is silenced,
# so this row is the released policy rather than "arm A minus its correction".
# Guarded because it is the newest code path and the A/B/C rows matter more.
probe "$A_CFG" robotwin_a_con1 "$OUT_DIR/robotwin_ab_basetrue.json" --base-weights \
  || echo "[$(date -u +%H:%M:%S)] released-base probe failed; continuing without it"
probe "$A_CFG" robotwin_a_con1 "$OUT_DIR/robotwin_ab_a.json"
probe "$B_CFG" robotwin_b_full "$OUT_DIR/robotwin_ab_b.json"

C_DIR="$CKPT_BASE/pi05_robotwin_con1_actcond_20k/robotwin_c_actcond"
if [[ -d "$C_DIR/$STEP" ]]; then
  probe "$C_CFG" robotwin_c_actcond "$OUT_DIR/robotwin_ab_c.json"
else
  echo "[$(date -u +%H:%M:%S)] no arm C checkpoint at step $STEP in $C_DIR; skipping arm C"
fi

D_DIR="$CKPT_BASE/pi05_robotwin_con1_headlr_20k/robotwin_d_headlr"
if [[ -d "$D_DIR/$STEP" ]]; then
  probe "$D_CFG" robotwin_d_headlr "$OUT_DIR/robotwin_ab_d.json"
else
  echo "[$(date -u +%H:%M:%S)] no arm D checkpoint at step $STEP in $D_DIR; skipping arm D"
fi

# Diagnostic row: arm A's checkpoint with the offline-trained head grafted in.
# It answers "would a better latent predictor buy action accuracy?" without a
# second joint training run. Guarded, and clearly not a trained arm.
HEAD_CKPT="${HEAD_CKPT:-/workspace/artifacts/con2/robotwin_head_warmstart.msgpack}"
if [[ -f "$HEAD_CKPT" ]]; then
  probe "$A_CFG" robotwin_a_con1 "$OUT_DIR/robotwin_ab_a_graft.json" --head-weights "$HEAD_CKPT" \
    || echo "[$(date -u +%H:%M:%S)] grafted-head probe failed; continuing"
else
  echo "[$(date -u +%H:%M:%S)] no offline head at $HEAD_CKPT; skipping the grafted-head row"
fi

# Last, and in a subshell so PROBE_BUDGETS stays local: does relaxing Con1's 5%
# residual cap buy accuracy? The probe's own notes say the cap - not the gate -
# is what limits the correction, which is a different explanation for Con2's
# weakness than "the predicted delta is uninformative". Running it at the very
# end means an OOM here costs only this row.
(
  export PROBE_BUDGETS="0.05 0.2"
  probe "$A_CFG" robotwin_a_con1 "$OUT_DIR/robotwin_ab_a_budgets.json"
) || echo "[$(date -u +%H:%M:%S)] multi-budget probe failed (likely OOM); continuing"
echo "[$(date -u +%H:%M:%S)] probes done"
