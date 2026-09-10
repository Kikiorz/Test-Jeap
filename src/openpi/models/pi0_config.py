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

    # Con1 reciprocal action-latent adapter.  Kept disabled by default so an
    # unmodified JEPA-WAM checkpoint remains bit-compatible until the explicit
    # Con1 training configuration enables it.
    use_con1: bool = False
    con1_latent_dim: int = 2816
    con1_width: int = 512
    con1_alpha_initial: float = 0.05
    con1_train_action_layers_from: int = 14
    con1_delta_weight: float = 0.2
    con1_sgr_beta: float = 0.5
    con1_residual_weight: float = 1e-3
    # Flow loss starts boosted and cosinely decays toward `final`, keeping the
    # action objective strong for most of training while easing coupling late.
    con1_flow_weight_initial: float = 2.0
    con1_flow_weight_final: float = 1.0
    con1_flow_weight_decay_steps: int = 15_000
    # Feed the action chunk into the delta head so it predicts the consequence
    # of the planned actions instead of a marginal future.
    con1_action_conditioning: bool = False
    # Add the direct pooled linear readout to the delta head. Off by default so
    # checkpoints from the earlier head structure keep restoring.
    con1_direct_readout: bool = False
    # Feed the pooled full VLM prefix (image patches, language, state) into the
    # delta head alongside the predictive queries.
    con1_vlm_context: bool = False
    # Zero-initialised, bounded action-side adapter (Con1's LoRA analogue).
    con1_action_adapter: bool = False
    con1_adapter_scale: float = 1.0
    con1_action_dims: int = 7
    # Three-stage coupling schedule: 2k adapter warm-up, 5k joint flow,
    # then 5k joint flow with the action-sensitivity weighting enabled.
    con1_stage1_steps: int = 2000
    con1_stage2_steps: int = 5000
    con1_stage3_steps: int = 5000

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
        if self.use_con1:
            if not self.pi05 or not self.use_vjepa_aux:
                raise ValueError("Con1 requires Pi0.5 with predictive R tokens")
            if self.con1_latent_dim < 1 or self.con1_width < 8:
                raise ValueError("Con1 dimensions must be positive")
            if not 0 < self.con1_alpha_initial < 1:
                raise ValueError("Con1 alpha initial must be inside (0,1)")
            depth = _gemma.get_config(self.action_expert_variant).depth
            if not 0 <= self.con1_train_action_layers_from < depth:
                raise ValueError("Con1 action-layer split is outside expert depth")
            if not 0 <= self.con1_sgr_beta < 1 or self.con1_delta_weight < 0:
                raise ValueError("Invalid Con1 loss weights")
            if min(self.con1_stage1_steps, self.con1_stage2_steps, self.con1_stage3_steps) < 1:
                raise ValueError("Con1 stage lengths must be positive")
            if self.con1_residual_weight < 0 or not 1 <= self.con1_action_dims <= self.action_dim:
                raise ValueError("Invalid residual weight or physical action dimension")
            if not self.con1_flow_weight_initial > 0 or not self.con1_flow_weight_final > 0:
                raise ValueError("Con1 flow weights must be positive")
            if self.con1_flow_weight_decay_steps < 1:
                raise ValueError("Con1 flow weight decay steps must be positive")

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
                    if self.use_vjepa_aux
                    else None
                ),
                con1_current_latent=(
                    jax.ShapeDtypeStruct([batch_size, self.con1_latent_dim], jnp.float32)
                    if self.use_con1 else None
                ),
                con1_future_latents=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.con1_latent_dim], jnp.float32)
                    if self.use_con1 else None
                ),
                con1_future_valid=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon], jnp.bool_)
                    if self.use_con1 else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
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
