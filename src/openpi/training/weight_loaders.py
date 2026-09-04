import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # Existing loaders only initialize missing LoRA weights. Specialized configs may explicitly allow other new params.
    missing_regex: str = ".*lora.*"

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


@dataclasses.dataclass(frozen=True)
class ActionChangeCheckpointWeightLoader(WeightLoader):
    """Warm-start Con1 and initialize its Change expert from the Action expert."""

    params_path: str
    change_kv_init_scale: float = 1e-3

    def load(self, params: at.Params) -> at.Params:
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded, sep="/")
        result = {}
        new_pattern = re.compile(
            r".*(future_context_proj|change_in_proj|change_out_proj|change_spatial_embedding).*"
        )
        for key, reference in flat_ref.items():
            source_key = key
            if source_key not in flat_loaded and "_2" in source_key:
                source_key = source_key.replace("_2", "_1")
            if source_key in flat_loaded:
                value = flat_loaded[source_key]
                if source_key != key and (
                    "kv_einsum_2" in key or "attn_vec_einsum_2" in key
                ):
                    # A full Action->Change clone makes the new Change keys and
                    # values perturb the pretrained joint softmax too strongly
                    # before any Con1 update. Keep the useful Q/norm/FFN
                    # initialization, but introduce Change K/V and its readout
                    # in the local neighborhood of the released policy.
                    value = value * self.change_kv_init_scale
                if value.shape != reference.shape:
                    raise ValueError(
                        f"Checkpoint shape mismatch for {key} from {source_key}: {value.shape} != {reference.shape}"
                    )
                result[key] = value.astype(reference.dtype) if value.dtype != reference.dtype else value
            elif new_pattern.fullmatch(key):
                result[key] = reference
            else:
                raise KeyError(f"Base checkpoint is missing non-Con1 parameter: {key}")
        return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
