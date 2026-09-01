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
    # ACTR introduces an intervention point into the scanned Gemma stack.
    # Released JEPA-WAM checkpoints store all layers under one leading scan
    # axis; split that axis losslessly when warm-starting the split module.
    split_scanned_layers_at: int | None = None

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        if self.split_scanned_layers_at is not None:
            loaded_params = _split_scanned_layers(loaded_params, self.split_scanned_layers_at)
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


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
            reference_value = flat_ref[k]
            if v is None or reference_value is None:
                if v is not None or reference_value is not None:
                    raise ValueError(f"Optional parameter mismatch at {k}: loaded={v}, reference={reference_value}")
                result[k] = None
            else:
                result[k] = v.astype(reference_value.dtype) if v.dtype != reference_value.dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _split_scanned_layers(params: at.Params, split_layer: int) -> at.Params:
    """Converts ``llm/layers`` into exact early/late scan parameter axes."""
    if split_layer < 1:
        raise ValueError("split_layer must be positive")

    flat = flax.traverse_util.flatten_dict(params, sep="/")
    result = {}
    matched = 0
    marker = "PaliGemma/llm/layers/"
    for key, value in flat.items():
        if marker not in key:
            result[key] = value
            continue
        if value.ndim < 1 or value.shape[0] <= split_layer:
            raise ValueError(f"Cannot split scanned layer parameter {key} with shape {value.shape} at {split_layer}")
        prefix, suffix = key.split(marker, maxsplit=1)
        result[f"{prefix}PaliGemma/llm/early_layers/{suffix}"] = value[:split_layer]
        result[f"{prefix}PaliGemma/llm/late_layers/{suffix}"] = value[split_layer:]
        matched += 1

    if matched == 0:
        raise ValueError("No PaliGemma/llm/layers parameters found to split")
    logger.info("Split %d scanned Gemma parameter leaves at layer %d", matched, split_layer)
    return flax.traverse_util.unflatten_dict(result, sep="/")
