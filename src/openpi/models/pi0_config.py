import dataclasses
from typing import TYPE_CHECKING, Literal

import flax.nnx as nnx
import flax.traverse_util as traverse_util
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

    # final_1 Reciprocal Action--Predictive Routing (RAPR).  This is an
    # incremental module on top of a pretrained JEPA-WAM Pi0.5 checkpoint.
    # Continuous residual coefficient.  The pretrained stream is always kept
    # in the computation; this scalar only scales the learned Con1 residual.
    use_rapr: bool = False
    rapr_delta_dim: int = 128
    rapr_width: int = 256
    rapr_loss_weight: float = 0.1
    # Historical runs sum feature errors; the new branch averages only D,
    # preserving the detached q weighting over valid future steps.
    rapr_prediction_reduction: Literal["sum", "mean"] = "sum"
    rapr_beta: float = 0.5
    rapr_temperature: float = 1.0
    rapr_gate: float = 0.05
    rapr_learnable_alpha: bool = False
    rapr_inference_gate: float = 1.0
    rapr_nonregression_weight: float = 1.0
    rapr_gate_max_rms: float = 0.05
    rapr_gate_max_abs: float = 0.2
    # New manuscript variant, kept separate from historical RAPR checkpoints.
    rapr_paper_orthogonal: bool = False
    rapr_train_action_expert: bool = False
    # Do not silently equate a training-time final layer with the last step
    # of iterative denoising. The run protocol must choose explicitly.
    rapr_control_stage: Literal["unresolved", "final_expert_layer", "final_denoise"] = "unresolved"
    rapr_control_num_steps: int = 10
    # Two shared retrievals after the last two Action blocks; r uses the
    # second A from the same sampled-time training forward, never extra sampling.
    rapr_late_layer_count: int = 0

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
        if self.rapr_paper_orthogonal and not self.use_rapr:
            raise ValueError("Paper orthogonal Con1 requires use_rapr=True")
        if self.rapr_prediction_reduction not in ("sum", "mean"):
            raise ValueError("Con1 prediction reduction must be sum or mean")
        if self.rapr_prediction_reduction == "mean" and not self.rapr_paper_orthogonal:
            raise ValueError("Mean-feature prediction loss requires paper Con1")
        if self.rapr_control_num_steps < 1:
            raise ValueError("Control retrieval needs at least one denoising step")
        if self.rapr_late_layer_count not in (0, 2):
            raise ValueError("Supported retrieval placement is legacy output-only or late-two")
        if self.rapr_late_layer_count and (
            not self.rapr_paper_orthogonal or self.rapr_control_stage != "final_expert_layer"
            or _gemma.get_config(self.action_expert_variant).depth < 2
        ):
            raise ValueError("Late-two paper retrieval requires final_expert_layer supervision")
        if self.rapr_train_action_expert and not self.rapr_paper_orthogonal:
            raise ValueError("Action expert joint training is only configured for paper Con1")
        if self.use_rapr:
            if not self.pi05 or not self.use_vjepa_aux:
                raise ValueError("RAPR requires a Pi0.5 JEPA-WAM future-query branch")
            if self.use_action_change_mmdit or self.use_jepa_ttt_adapter:
                raise ValueError("RAPR cannot be combined with the legacy Change or image-TTT branches")
            if min(self.rapr_delta_dim, self.rapr_width) < 1:
                raise ValueError("RAPR Delta-Z dimensions must be positive")
            if self.rapr_loss_weight < 0 or self.rapr_nonregression_weight < 0 or not 0 <= self.rapr_beta < 1:
                raise ValueError("RAPR loss weight must be nonnegative and beta must lie in [0, 1)")
            if self.rapr_temperature <= 0:
                raise ValueError("RAPR sensitivity temperature must be positive")
            if not 0 < self.rapr_gate < 1:
                raise ValueError("rapr_gate must lie strictly inside (0, 1)")
            if not 0 <= self.rapr_inference_gate <= 1:
                raise ValueError("rapr_inference_gate must lie in [0, 1]")
            if self.rapr_gate_max_rms < 0 or self.rapr_gate_max_abs < 0:
                raise ValueError("RAPR gate drift limits must be nonnegative")

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

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "Pi0":
        """Load a base checkpoint while retaining freshly initialized RAPR leaves.

        A pretrained JEPA-WAM checkpoint predates ``use_rapr`` and therefore
        has no Delta-Z/router leaves.  The generic base loader requires an
        exact tree and would reject that valid warm start.  Existing leaves
        are loaded verbatim; only missing RAPR leaves keep their deterministic
        initialization (including the zero residual readout and closed runtime
        gate).
        """
        if not self.use_rapr:
            return super().load(params, remove_extra_params=remove_extra_params)
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        reference = traverse_util.flatten_dict(state.to_pure_dict(), sep="/")
        loaded = traverse_util.flatten_dict(params, sep="/")
        if remove_extra_params:
            merged = {key: loaded.get(key, value) for key, value in reference.items()}
        else:
            merged = {**reference, **loaded}
        at.check_pytree_equality(
            expected=reference,
            got=merged,
            check_shapes=True,
            check_dtypes=False,
        )
        state.replace_by_pure_dict(traverse_util.unflatten_dict(merged, sep="/"))
        return nnx.merge(graphdef, state)

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
                transition_target=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.rapr_delta_dim], jnp.float32)
                    if self.use_rapr
                    else None
                ),
                transition_valid=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon], jnp.bool_)
                    if self.rapr_paper_orthogonal else None
                ),
                task_index=(
                    jax.ShapeDtypeStruct([batch_size], jnp.int32)
                    if self.rapr_paper_orthogonal else None
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
        if self.use_rapr:
            # Incremental learning starts from a pretrained base. Alpha is an
            # ordinary non-Param variable when fixed, so the optimizer sees
            # only Delta-Z/router residual parameters.
            if self.rapr_train_action_expert:
                return nnx.Not(nnx_utils.PathRegex(
                    ".*(rapr_delta_head|rapr_router|llm.*_1|action_in_proj|action_out_proj|"
                    "time_mlp_in|time_mlp_out|state_proj|action_time_mlp_in|action_time_mlp_out).*"
                ))
            return nnx.Not(nnx_utils.PathRegex(".*(rapr_delta_head|rapr_router).*"))
        if self.use_jepa_ttt_adapter:
            # Only the newly added residual and its scalar gate are trainable;
            # every released JEPA-WAM/π0.5 parameter remains frozen.
            return nnx.Not(nnx_utils.PathRegex(".*jepa_ttt_adapter.*"))
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
