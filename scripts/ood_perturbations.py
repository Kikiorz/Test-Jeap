"""Approximate LIBERO-Plus perturbation categories in image space.

LIBERO-Plus has seven categories: Background Textures, Camera Viewpoints, Light
Conditions, Objects Layout, Robot Initial States, Sensor Noise, Language. Four of
them change the *scene* (objects, robot pose, background, prompt) and cannot be
reproduced from recorded frames; three are image-space effects and are
approximated here:

  * Light Conditions   -> per-channel gain + gamma
  * Sensor Noise       -> gaussian / salt-pepper noise, blur
  * Camera Viewpoints  -> random perspective warp + crop (a view change, not a
    re-render, so it is an approximation of the real camera perturbation)

Language, Background Textures, Objects Layout and Robot Initial States are left
to the real LIBERO-Plus environment.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

CONDITIONS = ("clean", "lighting", "noise", "blur", "camera")


def _to_image(image: np.ndarray) -> Image.Image:
    return Image.fromarray(np.asarray(image, dtype=np.uint8))


def perturb(image: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    if condition == "clean":
        return np.asarray(image, dtype=np.uint8)
    if condition == "lighting":
        value = _to_image(image).convert("RGB")
        array = np.asarray(value, dtype=np.float32) / 255.0
        gain = rng.uniform(0.55, 1.45, size=3).astype(np.float32)
        gamma = float(rng.uniform(0.7, 1.4))
        array = np.clip(array * gain[None, None, :], 0, 1) ** gamma
        return (array * 255).astype(np.uint8)
    if condition == "noise":
        array = np.asarray(image, dtype=np.float32)
        sigma = float(rng.uniform(6.0, 26.0))
        array = array + rng.normal(0, sigma, size=array.shape)
        if rng.random() < 0.5:
            mask = rng.random(array.shape[:2]) < 0.01
            array[mask] = rng.choice([0.0, 255.0], size=(mask.sum(), 1))
        return np.clip(array, 0, 255).astype(np.uint8)
    if condition == "blur":
        radius = float(rng.uniform(0.8, 2.2))
        return np.asarray(_to_image(image).filter(ImageFilter.GaussianBlur(radius)), dtype=np.uint8)
    if condition == "camera":
        # Perspective warp of the recorded view: a stand-in for a moved camera.
        height, width = image.shape[:2]
        margin = 0.12
        src = np.asarray([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
        offsets = rng.uniform(-margin, margin, size=(4, 2)).astype(np.float32)
        dst = src + offsets * np.asarray([width, height], dtype=np.float32)[None, :]
        coefficients = _homography(src, dst)
        warped = _to_image(image).transform((width, height), Image.PERSPECTIVE, coefficients,
                                            resample=Image.BICUBIC)
        scale = float(rng.uniform(0.82, 0.95))
        box = (int(width * (1 - scale) / 2), int(height * (1 - scale) / 2),
               int(width * (1 + scale) / 2), int(height * (1 + scale) / 2))
        return np.asarray(warped.crop(box).resize((width, height), Image.BICUBIC), dtype=np.uint8)
    raise ValueError(f"Unknown condition {condition!r}")


def _homography(src: np.ndarray, dst: np.ndarray) -> tuple:
    """Least-squares homography mapping dst -> src, for PIL's PERSPECTIVE transform."""
    matrix = []
    for (x, y), (u, v) in zip(dst, src, strict=True):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    a = np.asarray(matrix, dtype=np.float64)
    b = src.reshape(-1).astype(np.float64)
    solution = np.linalg.lstsq(a, b, rcond=None)[0]
    return tuple(float(v) for v in np.append(solution, 1.0))
