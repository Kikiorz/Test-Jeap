"""Slice-aware freezing without changing the resumed optimizer tree or dtype."""

import math

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models import gemma
from openpi.shared import nnx_utils


ACTION = nnx_utils.PathRegex(
    ".*(llm.*_1|action_in_proj|action_out_proj|time_mlp_in|time_mlp_out|"
    "state_proj|action_time_mlp_in|action_time_mlp_out).*"
)
SCANNED_ACTION = nnx_utils.PathRegex(".*llm/layers/.*_1.*")


def transform_frozen_action(config, state, *, stop_gradient=False):
    """Stop differentiation, or zero gradients/updates, outside the last N blocks.

    Retaining the full original optimizer tree is intentional: it permits exact
    8k AdamW restoration without recasting newly frozen fp32 values to bf16.
    The final update mask is required even with zero gradients because restored
    momentum and decoupled weight decay otherwise move the frozen parameters.
    """
    count = config.rapr_action_train_last_n
    if not count:
        return state
    depth = gemma.get_config(config.model.action_expert_variant).depth
    if not 0 < count < depth:
        raise ValueError("Late Action training requires 0 < last_n < expert depth")
    freeze = jax.lax.stop_gradient if stop_gradient else jnp.zeros_like

    def transform(path, variable):
        if not ACTION(path, variable):
            return variable
        value = variable.value
        if value is None:
            return variable
        if SCANNED_ACTION(path, variable):
            if value.ndim < 1 or value.shape[0] != depth:
                raise ValueError(f"Unexpected scanned Action shape at {path}: {value.shape}")
            return variable.replace(jnp.concatenate([freeze(value[:-count]), value[-count:]], axis=0))
        return variable.replace(freeze(value))

    return state.map(transform)


def validate_start(config, *, resuming, completed_updates):
    """Separate intentional 5k phase starts from exact legacy 8k continuations."""
    if not config.rapr_action_train_last_n:
        return
    if config.rapr_action_freeze_from_start:
        if not resuming and (
            completed_updates != 0
            or not getattr(config.weight_loader, "require_complete", False)
            or getattr(config.weight_loader, "params_path", None) != config.rapr_action_freeze_anchor
        ):
            raise ValueError("Fresh late-two phase must load the complete freeze-anchor checkpoint")
    elif not resuming or completed_updates < 3001:
        raise ValueError("Last-two continuation must restore the saved 8k-or-later full training state")


def scope_summary(config, state):
    count = config.rapr_action_train_last_n
    depth = gemma.get_config(config.model.action_expert_variant).depth
    original_action = active_action = con1 = 0
    for path, variable in state.flat_state().items():
        if variable.value is None:
            continue
        size = math.prod(variable.value.shape)
        if ACTION(path, variable):
            original_action += size
            if not count:
                active_action += size
            elif SCANNED_ACTION(path, variable):
                if variable.value.shape[0] != depth:
                    raise ValueError("Action depth and scanned parameter shape differ")
                active_action += size // depth * count
        elif isinstance(variable, nnx.VariableState) and variable.type is nnx.Param:
            if str(path[0]) in ("rapr_delta_head", "rapr_router"):
                con1 += size
    return {"expert_depth": depth, "train_last_n": count,
            "frozen_action_blocks": list(range(depth - count)) if count else [],
            "active_action_blocks": list(range(depth - count, depth)) if count else list(range(depth)),
            "original_action_parameters": original_action,
            "effective_trainable_action_parameters": active_action,
            "effective_frozen_action_parameters": original_action - active_action,
            "con1_and_alpha_parameters": con1,
            "effective_trainable_parameters": active_action + con1,
            "freeze_anchor_params": config.rapr_action_freeze_anchor,
            "optimizer_tree_preserved": True, "parameter_dtypes_preserved": True}
