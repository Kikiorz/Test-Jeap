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

    Two spellings occur in practice and both are accepted, because the training
    pipeline and the RoboTwin evaluation client disagree:

    * training / LeRobot v2.1 datasets: flat ``observation/image`` keys, produced
      by ``LeRobotRoboTwinDataConfig``'s repack transform;
    * the RoboTwin simulator client (``XPolicyLab/policy/Pi_05/model.py``):
      ``{"state": ..., "images": {"cam_high": ...}, "prompt": ...}``, the shape
      the released RoboTwin pi0.5 config is written against.
    """

    model_type: _model.ModelType

    @staticmethod
    def _image(data: dict, flat_key: str, nested_key: str) -> np.ndarray:
        if flat_key in data:
            return _parse_image(data[flat_key])
        images = data.get("images")
        if isinstance(images, dict) and nested_key in images:
            return _parse_image(images[nested_key])
        raise KeyError(f"missing image: neither '{flat_key}' nor 'images.{nested_key}' is present")

    def __call__(self, data: dict) -> dict:
        base_image = self._image(data, "observation/image", "cam_high")
        left_wrist = self._image(data, "observation/wrist_image", "cam_left_wrist")
        right_wrist = self._image(data, "observation/wrist_image_right", "cam_right_wrist")
        state = data["observation/state"] if "observation/state" in data else data["state"]

        inputs = {
            "state": state,
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
