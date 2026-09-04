"""Frozen achieved-Change encoder shared by Con1 Stage 1 and Con2."""

from __future__ import annotations

import json
from pathlib import Path

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from openpi.models import vjepa_change_tokenizer


IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def preprocess_image(image: np.ndarray | Image.Image) -> np.ndarray:
    """Exactly match the V-JEPA2 preprocessing used by Stage-1 caching."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    image = image.resize((384, 384), resample=Image.Resampling.BICUBIC)
    value = np.asarray(image, dtype=np.float32) / 255.0
    value = (value - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(value.transpose(2, 0, 1))


def spatial_feature_torch(output):
    """Parameter-free S(.) shared with Stage 1: 24x24 -> 8x8."""
    import torch.nn.functional as functional

    value = output.float().reshape(output.shape[0], 24, 24, 1408)
    value = functional.normalize(value, dim=-1, eps=1e-6)
    value = value.reshape(value.shape[0], 8, 3, 8, 3, 1408).mean(dim=(2, 4))
    return functional.normalize(value, dim=-1, eps=1e-6)


class AchievedChangeEncoder:
    """Encode one observed H10 transition in the frozen Con1 coordinate system."""

    def __init__(
        self,
        *,
        hf_port: str | Path,
        stage1_dir: str | Path,
        torch_device: str = "cuda:0",
    ):
        import torch
        from transformers import AutoModel

        self._torch = torch
        self._device = torch.device(torch_device)
        self._teacher = AutoModel.from_pretrained(
            str(hf_port),
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self._teacher.requires_grad_(False)
        self._teacher.eval().to(self._device)

        stage1_dir = Path(stage1_dir)
        with (stage1_dir / "change_stats.json").open() as handle:
            stats = json.load(handle)
        self._mean = jnp.asarray(stats["mean"], dtype=jnp.float32)
        self._std = jnp.asarray(stats["std"], dtype=jnp.float32)
        token_shape = tuple(int(value) for value in stats["token_shape"])
        if token_shape != (16, 128):
            raise ValueError(f"Con2 expects Stage-1 token shape (16,128), got {token_shape}")

        self._stage1 = vjepa_change_tokenizer.Stage1Teacher(
            num_tokens=16,
            token_dim=128,
            width=512,
            resampler_depth=3,
            decoder_depth=4,
            num_heads=8,
            ffn_width=2048,
            horizon=10,
            action_dim=7,
        )
        template = self._stage1.init(
            jax.random.key(0), jnp.zeros((1, 64, 1408), dtype=jnp.float32)
        )["params"]
        self._params = serialization.from_bytes(
            template, (stage1_dir / "best_params.msgpack").read_bytes()
        )
        self._encode = jax.jit(
            lambda displacement: self._stage1.apply(
                {"params": self._params}, displacement, method=self._stage1.encode
            )
        )

    def _vjepa_spatial(self, current: np.ndarray, future: np.ndarray):
        torch = self._torch
        current_value = preprocess_image(current)
        future_value = preprocess_image(future)
        video = torch.from_numpy(np.stack((current_value, future_value), axis=1)[None]).to(
            device=self._device, dtype=torch.bfloat16
        )
        with torch.inference_mode():
            output = self._teacher(pixel_values_videos=video, skip_predictor=True).last_hidden_state
            if tuple(output.shape) != (1, 576, 1408):
                raise ValueError(f"Unexpected V-JEPA2 feature shape: {tuple(output.shape)}")
            return spatial_feature_torch(output)

    def __call__(self, current: np.ndarray, future: np.ndarray) -> np.ndarray:
        pair = self._vjepa_spatial(current, future)
        no_change = self._vjepa_spatial(current, current)
        displacement = (pair - no_change).reshape(1, 64, 1408)
        displacement = jnp.asarray(displacement.to(device="cpu", dtype=self._torch.float32).numpy())
        raw = self._encode(displacement)
        normalized = (raw - self._mean[None, None]) / (self._std[None, None] + 1e-6)
        return np.asarray(normalized, dtype=np.float32)
