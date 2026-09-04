import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # Optional future-representation objective. Disabled by default so existing Pi0/Pi0.5 models are unchanged.
    use_vjepa_aux: bool = False
    vjepa_num_queries: int = 64
    vjepa_query_grid_size: int = 8
    vjepa_target_grid_size: int = 24
    vjepa_target_dim: int = 1408
    vjepa_aux_weight: float = 0.1
    vjepa_aux_warmup_steps: int = 1000
    vjepa_action_attends_queries: bool = False
    vjepa_disable_geometric_augmentation: bool = True

    # Con1: co-generate action and a frozen Stage-1 change endpoint. Disabled by
    # default so all upstream Pi0/Pi0.5 parameter trees remain unchanged.
    use_action_change_mmdit: bool = False
    change_num_tokens: int = 16
    change_token_dim: int = 128
    change_joint_start_layer: int = 12
    change_loss_weight: float = 0.3
    change_train_action_late: bool = True
    # Con2 adds a zero-function, strictly directional Change->Action adapter
    # only after the complete Con1 model has been trained and frozen.
    use_achieved_change_adapter: bool = False
    achieved_change_adapter_rank: int = 4
    achieved_change_inverse_probability: float = 0.3

    # Low-rank residual shared by the JEPA future-query and action paths.
    # The first control-aligned TTT experiment updates only this adapter.
    use_jepa_ttt_adapter: bool = False
    jepa_ttt_adapter_rank: int = 8

    # Observable point-flow interface.  Stage 1 decodes the already learned
    # JEPA future-query representation into short, image-plane trajectories.
    use_point_flow: bool = False
    point_flow_num_points: int = 32
    point_flow_horizon: int = 10
    point_flow_hidden_dim: int = 256
    point_flow_num_layers: int = 3
    point_flow_num_heads: int = 4
    point_flow_action_loss_weight: float = 1.0
    point_flow_loss_weight: float = 1.0
    point_flow_visibility_weight: float = 0.1
    point_flow_smoothness_weight: float = 0.05

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.use_vjepa_aux:
            if not self.pi05:
                raise ValueError("V-JEPA auxiliary training is only supported for Pi0.5")
            if self.vjepa_num_queries != self.vjepa_query_grid_size**2:
                raise ValueError("vjepa_num_queries must equal vjepa_query_grid_size squared")
            if self.vjepa_target_grid_size < 1 or self.vjepa_target_dim < 1:
                raise ValueError("V-JEPA target grid size and dimension must be positive")
            if self.vjepa_aux_weight < 0 or self.vjepa_aux_warmup_steps < 0:
                raise ValueError("V-JEPA auxiliary weight and warmup steps must be non-negative")
        if self.use_action_change_mmdit:
            if not self.use_vjepa_aux or not self.pi05:
                raise ValueError("Action–Change MMDiT requires the Pi0.5 JEPA-WAM future-query branch")
            if self.change_num_tokens != 16:
                raise ValueError("The first Con1 implementation requires a 4x4 (16-token) change grid")
            if self.change_token_dim < 1 or self.change_loss_weight < 0:
                raise ValueError("Change token dimension must be positive and loss weight non-negative")
            action_depth = _gemma.get_config(self.action_expert_variant).depth
            if not 0 < self.change_joint_start_layer < action_depth:
                raise ValueError("change_joint_start_layer must split the Action Expert depth")
        if self.use_achieved_change_adapter:
            if not self.use_action_change_mmdit:
                raise ValueError("Achieved-Change adaptation requires Action–Change MMDiT")
            if self.achieved_change_adapter_rank < 1:
                raise ValueError("Achieved-Change adapter rank must be positive")
            if not 0.0 < self.achieved_change_inverse_probability < 1.0:
                raise ValueError("Achieved-Change inverse probability must lie strictly between zero and one")
        if self.use_point_flow:
            if not self.use_vjepa_aux:
                raise ValueError("Point-flow prediction requires the JEPA-WAM future-query branch")
            if self.point_flow_num_points < 1 or self.point_flow_horizon < 1:
                raise ValueError("Point-flow point count and horizon must be positive")
            if self.point_flow_horizon != self.action_horizon:
                raise ValueError("The first point-flow experiment requires point and action horizons to match")
            if self.point_flow_hidden_dim % self.point_flow_num_heads:
                raise ValueError("point_flow_hidden_dim must be divisible by point_flow_num_heads")
            if (
                self.point_flow_num_layers < 1
                or self.point_flow_action_loss_weight < 0
                or self.point_flow_loss_weight < 0
            ):
                raise ValueError("Point-flow layer count must be positive and loss weight non-negative")
        if self.use_jepa_ttt_adapter:
            if not self.use_vjepa_aux:
                raise ValueError("JEPA TTT adapter requires the JEPA-WAM future-query branch")
            if self.jepa_ttt_adapter_rank < 1:
                raise ValueError("jepa_ttt_adapter_rank must be positive")

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                vjepa_target=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.vjepa_target_grid_size**2, self.vjepa_target_dim], jnp.float16
                    )
                    if self.use_vjepa_aux and not self.use_action_change_mmdit
                    else None
                ),
                change_target=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.change_num_tokens, self.change_token_dim], jnp.float32
                    )
                    if self.use_action_change_mmdit
                    else None
                ),
                point_flow_queries=(
                    jax.ShapeDtypeStruct([batch_size, self.point_flow_num_points, 2], jnp.float32)
                    if self.use_point_flow
                    else None
                ),
                point_flow_target=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.point_flow_num_points, self.point_flow_horizon, 2], jnp.float32
                    )
                    if self.use_point_flow
                    else None
                ),
                point_flow_visibility=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.point_flow_num_points, self.point_flow_horizon], jnp.float32
                    )
                    if self.use_point_flow
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        if self.use_achieved_change_adapter:
            trainable = nnx_utils.PathRegex(".*change_to_action_(k|v)_(down|up).*")
            return nnx.Not(trainable)
        if self.use_action_change_mmdit:
            trainable_parts = [
                "future_context_proj",
                "change_in_proj",
                "change_out_proj",
                "change_spatial_embedding",
                "llm.*_2",
            ]
            if self.change_train_action_late:
                # The scanned layer arrays contain all 18 blocks. train.py
                # masks parameter gradients on rows 0:change_joint_start_layer;
                # do not include the final norm or other Action parameters.
                trainable_parts.append("llm/layers/.*_1")
            trainable = nnx_utils.PathRegex(".*(" + "|".join(trainable_parts) + ").*")
            return nnx.Not(trainable)
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
