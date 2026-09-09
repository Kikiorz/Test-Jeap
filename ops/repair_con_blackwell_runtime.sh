#!/usr/bin/env bash
set -euo pipefail
cd /workspace/ts_JEPA_con

# The instance exports UV_NO_CACHE=1 globally. Keep completed downloads across
# bounded retries without changing its global configuration.
export UV_NO_CACHE=0
export UV_CACHE_DIR=/workspace/.uv-cache
export UV_HTTP_TIMEOUT=600
export UV_CONCURRENT_DOWNLOADS=3

# Run under supervisor, with all .venv GPU training/inference jobs stopped.
# This changes only the project virtualenv; never the injected host driver.
.venv/bin/python -c 'import importlib.metadata as m,json; print(json.dumps({"event":"runtime_before","packages":{d.metadata["Name"]:d.version for d in m.distributions() if d.metadata["Name"].lower().replace("_","-").startswith(("jax","nvidia-","torch","numpy","ml-dtypes","orbax","flax"))}},sort_keys=True),flush=True)'
# Upstream [tool.uv].override-dependencies forces ml-dtypes==0.4.1 even over
# an explicit requirements pin. This runtime override must ignore that config.
uv pip install --no-config --python .venv/bin/python \
  -r ops/requirements-con-blackwell.txt \
  --overrides ops/requirements-con-blackwell-downloads.txt \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
.venv/bin/python -c 'import importlib.metadata as m,json; print(json.dumps({"event":"runtime_after","packages":{d.metadata["Name"]:d.version for d in m.distributions() if d.metadata["Name"].lower().replace("_","-").startswith(("jax","nvidia-","torch","numpy","ml-dtypes","orbax","flax"))}},sort_keys=True),flush=True); assert m.version("jax")=="0.7.2"; assert m.version("ml-dtypes")=="0.5.4"; assert tuple(map(int,m.version("nvidia-cublas-cu12").split(".")[:2])) >= (12,8)'
.venv/bin/python -c 'import torch,torchvision,torchcodec,jax; print({"torch":torch.__version__,"torch_cuda":torch.version.cuda,"torch_architectures":torch.cuda.get_arch_list(),"torchvision":torchvision.__version__,"torchcodec":torchcodec.__version__,"jax":jax.__version__},flush=True); assert "sm_120" in torch.cuda.get_arch_list()'
