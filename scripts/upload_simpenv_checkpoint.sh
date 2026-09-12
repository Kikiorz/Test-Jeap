#!/usr/bin/env bash
# Upload one SimpENV checkpoint (params + assets + eval notes) to the private
# delivery repo so it can be pulled and evaluated locally.
#
# Usage: CKPT=/workspace/checkpoints/<config>/<exp>/<step> NAME=armA \
#          bash scripts/upload_simpenv_checkpoint.sh
set -euo pipefail

CKPT="${CKPT:?Set CKPT to the step directory}"
NAME="${NAME:?Set NAME to the destination folder, e.g. armA}"
REPO="${REPO:-QRP123/simpenv-pi05-con1}"
STAGE="${STAGE:-/workspace/hf_stage/$NAME}"

export HF_HOME=/workspace/.hf_home
mkdir -p "$STAGE"
rm -rf "${STAGE:?}/"*
cp -r "$CKPT/params" "$STAGE/params"
if [ -d "$CKPT/assets" ]; then cp -r "$CKPT/assets" "$STAGE/assets"; fi
if [ -f "$CKPT/_METADATA" ]; then cp "$CKPT/_METADATA" "$STAGE/_METADATA"; fi
du -sh "$STAGE"

HF_CLI=/workspace/ts_JEPA_simpenv/.venv/bin/huggingface-cli
[ -x "$HF_CLI" ] || HF_CLI=/workspace/ts_JEPA_simpenv/.venv/bin/hf
"$HF_CLI" upload "$REPO" "$STAGE" "$NAME" --repo-type model
echo "UPLOADED $REPO/$NAME"
