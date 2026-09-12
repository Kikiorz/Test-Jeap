#!/usr/bin/env bash
# Bootstrap the LIBERO branch on a Blackwell box.
#
# Two environment facts drive this script:
#   1. the repository pins jax[cuda12]==0.5.3, whose jaxlib predates Blackwell
#      (sm_120) kernels, so the GPU is invisible to it;
#   2. this image has no JAX at all.
# We therefore create a Python 3.11 venv, install the project, then upgrade JAX
# to the 0.7.2 build that was verified on sm_120 previously.
set -euo pipefail

WORK="${WORK:-/dev/shm/ts_jepa}"
REPO="${REPO:-$WORK/ts_JEPA_libero}"
# /dev/shm is mounted noexec, so the virtualenv (shared objects) has to live
# on the container disk while the data lives on the RAM disk.
VENV="${VENV:-/opt/venv-jepa}"
JAX_VERSION="${JAX_VERSION:-0.7.2}"

cd "$REPO"
# Keep package caches on the RAM disk: the container disk is only 32 GB and the
# image already ships ~15 GB of unrelated content.
export UV_CACHE_DIR=/dev/shm/uv-cache
export PIP_CACHE_DIR=/dev/shm/pip-cache
export TMPDIR=/dev/shm/tmp
mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$TMPDIR"
printf '[%s] creating venv\n' "$(date -u +%H:%M:%S)"
uv venv --clear --python 3.11 "$VENV"
uv pip install --python "$VENV/bin/python" -U pip 2>&1 | tail -2
printf '[%s] installing project (this pulls the pinned jax first)\n' "$(date -u +%H:%M:%S)"
uv pip install --python "$VENV/bin/python" -e . 2>&1 | tail -5
printf '[%s] upgrading jax to %s for Blackwell\n' "$(date -u +%H:%M:%S)" "$JAX_VERSION"
uv pip install --python "$VENV/bin/python" \
  "jax[cuda12]==$JAX_VERSION" "jaxlib==$JAX_VERSION" 2>&1 | tail -5
printf '[%s] extra runtime deps\n' "$(date -u +%H:%M:%S)"
uv pip install --python "$VENV/bin/python" \
  tensorstore orbax-checkpoint pyarrow tyro wandb tqdm-loggable pytest 2>&1 | tail -3
printf '[%s] jax device check\n' "$(date -u +%H:%M:%S)"
"$VENV/bin/python" - <<'PY'
import jax
print("jax", jax.__version__)
print("backend", jax.default_backend())
print("devices", jax.devices())
PY
printf '[%s] bootstrap done\n' "$(date -u +%H:%M:%S)"
