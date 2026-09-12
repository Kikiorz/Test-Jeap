import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SimplerEnvInputs(transforms.DataTransformFn):
    """SimplerEnv (WidowX / Bridge) inputs.

    The benchmark's own reference policies (RT-1, Octo) consume a single
    third-person image plus the language instruction, so by default the wrist
    image is zero-filled and masked out. ``use_wrist_image`` enables the wrist
    camera for checkpoints that were trained with it.
    """

    model_type: _model.ModelType
    use_wrist_image: bool = False

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": (
                    _parse_image(data["observation/wrist_image"])
                    if self.use_wrist_image
                    else np.zeros_like(base_image)
                ),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_ if self.use_wrist_image else np.False_,
                "right_wrist_0_rgb": (
                    np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_
                ),
            },
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SimplerEnvOutputs(transforms.DataTransformFn):
    """WidowX action: 7 dims (xyz delta, rpy delta, gripper)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :7])}
