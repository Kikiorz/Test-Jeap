import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robotwin_example() -> dict:
    """Creates a random input example for the RoboTwin policy."""
    return {
        "observation/state": np.random.rand(14),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_right": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboTwinInputs(transforms.DataTransformFn):
    """Maps RoboTwin LeRobot v3 keys to the openpi policy input schema.

    RoboTwin provides one head camera and two wrist cameras. All three are real
    observations, so unlike LIBERO we do not pad any view with zeros.
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        left_wrist = _parse_image(data["observation/wrist_image"])
        right_wrist = _parse_image(data["observation/wrist_image_right"])

        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "vjepa_target" in data:
            inputs["vjepa_target"] = data["vjepa_target"]
        if "con1_current_latent" in data:
            inputs["con1_current_latent"] = data["con1_current_latent"]
        if "con1_future_latents" in data:
            inputs["con1_future_latents"] = data["con1_future_latents"]
        if "con1_future_valid" in data:
            inputs["con1_future_valid"] = data["con1_future_valid"]

        return inputs


@dataclasses.dataclass(frozen=True)
class RoboTwinOutputs(transforms.DataTransformFn):
    """Returns the physical 14-dimensional bimanual action (drops padding)."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :14])}
