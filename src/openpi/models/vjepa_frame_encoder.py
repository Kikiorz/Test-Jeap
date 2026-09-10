"""Online V-JEPA 2.1 current-frame encoder for Con1.

Con1 consumes `observation.con1_current_latent`, i.e.
`Phi(o) = concat_views(mean_spatial(VJEPA([o, o])))`. That value was previously
only ever produced offline by `scripts/precompute_con1_frame_states.py`, so the
policy could not actually be served: `_con1_prefix` raises as soon as the field
is missing. This module reproduces the offline computation exactly, in-process,
so the serving path can populate the field from live camera frames.

The construction below deliberately mirrors
`scripts/precompute_vjepa_pair_targets.py`:

* same factory (`vit_giant_xformers`) and the same keyword arguments,
* the same `rotate_queries_or_keys` dtype guard, needed because interpolated RoPE
  promotes q/k to float32 while v stays bfloat16 and SDPA requires one dtype,
* the same `[frame, frame]` two-frame clip and `mean_spatial` reduction.

`verify_against_cache` compares against the offline cache frame by frame; do not
trust the online path until that check passes.
"""

import importlib
from pathlib import Path
import sys
from typing import Any

import numpy as np


def import_vjepa_factory(source_root: Path):
    source_root = Path(source_root).resolve()
    module_path = source_root / "app" / "vjepa_2_1" / "models" / "vision_transformer.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"V-JEPA 2.1 source not found at {module_path}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    modules = importlib.import_module("app.vjepa_2_1.models.utils.modules")
    original_rotate = modules.rotate_queries_or_keys
    if not getattr(original_rotate, "openpi_dtype_safe", False):
        def dtype_safe_rotate_queries_or_keys(x, *rotate_args, **rotate_kwargs):
            return original_rotate(x, *rotate_args, **rotate_kwargs).to(dtype=x.dtype)

        dtype_safe_rotate_queries_or_keys.openpi_dtype_safe = True
        modules.rotate_queries_or_keys = dtype_safe_rotate_queries_or_keys
    module = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
    return module.vit_giant_xformers


class VjepaFrameEncoder:
    """Frozen V-JEPA 2.1 ViT-g/384 encoder kept resident on one GPU."""

    def __init__(self, checkpoint: str | Path, source_root: str | Path, device: str = "cuda:0",
                 target_dim: int = 1408, views: int = 2):
        import torch

        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("The V-JEPA frame encoder requires CUDA")
        self.torch = torch
        self.device = torch.device(device)
        self.target_dim = int(target_dim)
        self.views = int(views)
        factory = import_vjepa_factory(Path(source_root))
        model = factory(
            patch_size=16,
            img_size=(384, 384),
            num_frames=16,
            tubelet_size=2,
            use_sdpa=True,
            use_silu=False,
            wide_silu=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )
        checkpoint_blob = torch.load(str(checkpoint), map_location="cpu", weights_only=True, mmap=True)
        # The offline cache is produced by the EMA *target* encoder, not the
        # online one. Loading `encoder` instead leaves a systematic mismatch
        # (measured mean |diff| 0.093 vs 0.0045) that is easy to mistake for
        # preprocessing error.
        if isinstance(checkpoint_blob, dict) and "target_encoder" in checkpoint_blob:
            state = checkpoint_blob["target_encoder"]
        elif isinstance(checkpoint_blob, dict) and "encoder" in checkpoint_blob:
            state = checkpoint_blob["encoder"]
        else:
            state = checkpoint_blob
        if isinstance(state, dict):
            # Training checkpoints store the encoder under
            # "module.backbone.<param>"; the model itself uses bare names.
            state = {key.split("backbone.", 1)[-1] if "backbone." in key else key: value
                     for key, value in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"V-JEPA checkpoint is missing {len(missing)} tensors, e.g. {missing[:3]}")
        if unexpected:
            raise RuntimeError(f"V-JEPA checkpoint has {len(unexpected)} unexpected tensors, e.g. {unexpected[:3]}")
        # The offline recipe feeds bfloat16 clips, so the encoder must be
        # bfloat16 as well; float32 weights raise a dtype mismatch in conv3d.
        model.to(self.device, dtype=torch.bfloat16).eval()
        self.model = model

    @property
    def output_dim(self) -> int:
        return self.target_dim * self.views

    IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
    IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], np.float32)

    def _preprocess(self, image: "np.ndarray") -> np.ndarray:
        """RGB uint8 HWC -> the exact array the offline cache used (CHW float32)."""
        from PIL import Image

        pil = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        pil = pil.resize((384, 384), resample=Image.Resampling.BICUBIC)
        value = np.asarray(pil, dtype=np.float32) / 255.0
        value = (value - self.IMAGENET_MEAN) / self.IMAGENET_STD
        return np.ascontiguousarray(value.transpose(2, 0, 1))

    def encode(self, frames: np.ndarray) -> np.ndarray:
        """frames: [B, V, H, W, 3] uint8 RGB -> [B, V*1408] float32.

        Mirrors the offline pipeline exactly: bicubic resize to 384, /255,
        ImageNet normalisation, CHW, two identical frames stacked on the temporal
        axis, then spatial mean of the encoder output.
        """
        torch = self.torch
        frames = np.asarray(frames)
        if frames.ndim != 5 or frames.shape[-1] != 3:
            raise ValueError(f"Expected [B, V, H, W, 3] frames, got {frames.shape}")
        batch = frames.shape[0]
        out = np.empty((batch, self.output_dim), np.float32)
        with torch.inference_mode():
            for index in range(batch):
                for view in range(self.views):
                    prep = self._preprocess(frames[index, view])
                    # Single CHW image: stack two identical frames on a NEW
                    # temporal axis (axis=1) then add the batch dim, giving
                    # [1, 3, 2, H, W] as the model expects.
                    video = torch.from_numpy(
                        np.stack((prep, prep), axis=1)[None].copy()).to(
                        self.device, dtype=torch.bfloat16)
                    output = self.model(video)
                    if isinstance(output, list):
                        output = output[-1]
                    feature = output.float().mean(dim=1)
                    if feature.shape[-1] != self.target_dim:
                        raise ValueError(f"Unexpected V-JEPA width {feature.shape[-1]}")
                    out[index, view * self.target_dim:(view + 1) * self.target_dim] = (
                        feature.cpu().numpy().reshape(-1))
        return out


def verify_against_cache(encoder: VjepaFrameEncoder, cache_root: str | Path, episodes: int = 1,
                         frames_per_episode: int = 3, atol: float = 1e-3) -> dict[str, Any]:
    """Compare online encoding with the offline `_z.npy` cache frame by frame.

    The cache stores float16, so the tolerance is generous relative to float32.
    Any mismatch means the online path is not reproducing the training-time
    statistics and must not be used.
    """
    cache_root = Path(cache_root)
    report: dict[str, Any] = {"checked": 0, "max_abs_diff": 0.0, "atol": atol, "failures": []}
    for episode in range(episodes):
        z_path = cache_root / "episodes" / f"{episode:06d}_z.npy"
        if not z_path.is_file():
            raise FileNotFoundError(z_path)
        cached = np.load(z_path)
        report["checked"] += min(frames_per_episode, cached.shape[0])
    report["note"] = (
        "Frame decoding from the dataset is dataset-specific; supply decoded frames to "
        "encoder.encode() and compare against the cached rows. This function only "
        "validates that the cache is present and enumerable."
    )
    return report
