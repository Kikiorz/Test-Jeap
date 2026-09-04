import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models import point_flow as _point_flow
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

# NNX stores initializer callables in graph metadata.  Construct them once so
# repeated model creation (shape tracing, then real initialization) produces
# identical GraphDefs.
_ZERO_KERNEL_INIT = nnx.initializers.zeros_init()
_SMALL_CHANGE_KERNEL_INIT = nnx.initializers.normal(stddev=1e-3)


class JepaTTTAdapter(nnx.Module):
    """Zero-initialized low-rank residual shared by JEPA and action paths."""

    def __init__(self, width: int, rank: int, *, rngs: nnx.Rngs):
        self.down = nnx.Linear(width, rank, use_bias=False, rngs=rngs)
        self.up = nnx.Linear(
            rank,
            width,
            use_bias=False,
            kernel_init=_ZERO_KERNEL_INIT,
            rngs=rngs,
        )

    def __call__(self, tokens: jax.Array) -> jax.Array:
        dtype = tokens.dtype
        value = tokens.astype(jnp.float32)
        value = value * jax.lax.rsqrt(jnp.mean(jnp.square(value), axis=-1, keepdims=True) + 1e-6)
        residual = self.up(nnx.silu(self.down(value)))
        return tokens + residual.astype(dtype)


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_vjepa_aux = config.use_vjepa_aux
        self.vjepa_num_queries = config.vjepa_num_queries
        self.vjepa_query_grid_size = config.vjepa_query_grid_size
        self.vjepa_target_grid_size = config.vjepa_target_grid_size
        self.vjepa_target_dim = config.vjepa_target_dim
        self.vjepa_aux_weight = config.vjepa_aux_weight
        self.vjepa_action_attends_queries = config.vjepa_action_attends_queries
        self.vjepa_disable_geometric_augmentation = config.vjepa_disable_geometric_augmentation
        self.use_action_change_mmdit = config.use_action_change_mmdit
        self.change_num_tokens = config.change_num_tokens
        self.change_token_dim = config.change_token_dim
        self.change_joint_start_layer = config.change_joint_start_layer
        self.change_loss_weight = config.change_loss_weight
        self.change_train_action_late = config.change_train_action_late
        self.use_achieved_change_adapter = config.use_achieved_change_adapter
        self.achieved_change_inverse_probability = config.achieved_change_inverse_probability
        self.use_jepa_ttt_adapter = config.use_jepa_ttt_adapter
        self.use_point_flow = config.use_point_flow
        self.point_flow_loss_weight = config.point_flow_loss_weight
        self.point_flow_visibility_weight = config.point_flow_visibility_weight
        self.point_flow_smoothness_weight = config.point_flow_smoothness_weight
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.action_expert_depth = action_expert_config.depth
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm_configs = [paligemma_config, action_expert_config]
        if self.use_action_change_mmdit:
            change_expert_config = dataclasses.replace(
                action_expert_config,
                directional_adapter_rank=(
                    config.achieved_change_adapter_rank if config.use_achieved_change_adapter else 0
                ),
            )
            llm_configs.append(change_expert_config)
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=llm_configs,
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        use_adarms = [False, True] if config.pi05 else [False, False]
        if self.use_action_change_mmdit:
            use_adarms.append(True)
        llm.lazy_init(rngs=rngs, method="init", use_adarms=use_adarms)
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        if self.use_jepa_ttt_adapter:
            self.jepa_ttt_adapter = JepaTTTAdapter(
                paligemma_config.width, config.jepa_ttt_adapter_rank, rngs=rngs
            )
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)
        if self.use_action_change_mmdit:
            self.change_in_proj = nnx.Linear(config.change_token_dim, action_expert_config.width, rngs=rngs)
            self.change_out_proj = nnx.Linear(
                action_expert_config.width,
                config.change_token_dim,
                kernel_init=_SMALL_CHANGE_KERNEL_INIT,
                rngs=rngs,
            )
            self.future_context_proj = nnx.Linear(config.vjepa_target_dim, action_expert_config.width, rngs=rngs)
            spatial_init = jax.random.normal(
                rngs.params(), (config.change_num_tokens, action_expert_config.width), dtype=jnp.float32
            )
            self.change_spatial_embedding = nnx.Param(spatial_init * 0.02)

        if self.use_vjepa_aux:
            query_init = jax.random.normal(
                rngs.params(), (self.vjepa_num_queries, paligemma_config.width), dtype=jnp.float32
            )
            self.vjepa_query_tokens = nnx.Param(query_init * 0.02)
            self.vjepa_alignment_norm = nnx.LayerNorm(paligemma_config.width, rngs=rngs)
            self.vjepa_alignment_in = nnx.Linear(paligemma_config.width, paligemma_config.width, rngs=rngs)
            self.vjepa_alignment_out = nnx.Linear(paligemma_config.width, self.vjepa_target_dim, rngs=rngs)
            if self.use_point_flow:
                self.point_flow_planner = _point_flow.PointFlowPlanner(
                    transition_width=paligemma_config.width,
                    query_grid_size=self.vjepa_query_grid_size,
                    horizon=config.point_flow_horizon,
                    hidden_width=config.point_flow_hidden_dim,
                    num_layers=config.point_flow_num_layers,
                    num_heads=config.point_flow_num_heads,
                    ffn_width=4 * config.point_flow_hidden_dim,
                    rngs=rngs,
                )

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
            if self.use_jepa_ttt_adapter:
                image_tokens = self.jepa_ttt_adapter(image_tokens)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        if self.use_vjepa_aux:
            query_tokens = jnp.broadcast_to(
                self.vjepa_query_tokens.value.astype(tokens[0].dtype),
                (tokens[0].shape[0], self.vjepa_num_queries, self.vjepa_query_tokens.value.shape[-1]),
            )
            tokens.append(query_tokens)
            input_mask.append(jnp.ones(query_tokens.shape[:2], dtype=jnp.bool_))
            # Queries read image/language and one another. Earlier prefix tokens cannot read the queries.
            ar_mask += [True] + ([False] * (self.vjepa_num_queries - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        time_emb = self.embed_flow_time(timestep)
        if self.pi05:
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def embed_flow_time(self, timestep: at.Float[at.Array, " b"]) -> jax.Array:
        """Embed a modality-specific rectified-flow time with the Pi0.5 time MLP."""
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
        return time_emb

    def _predict_action_velocity_from_preprocessed(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> tuple[_model.Actions, at.Float[at.Array, "b q emb"] | None]:
        """Run the training-style joint forward without constructing a loss."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, noisy_actions, timestep
        )
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        if self.use_vjepa_aux and not self.vjepa_action_attends_queries:
            prefix_length = prefix_tokens.shape[1]
            query_start = prefix_length - self.vjepa_num_queries
            attn_mask = attn_mask.at[:, prefix_length:, query_start:prefix_length].set(False)
            positions = positions.at[:, prefix_length:].add(-self.vjepa_num_queries)
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        velocity = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        query_out = None
        if self.use_vjepa_aux:
            query_out = prefix_out[:, -self.vjepa_num_queries :]
        return velocity, query_out

    def predict_action_velocity(
        self,
        observation: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"],
    ) -> _model.Actions:
        """Return the frozen policy velocity for an explicit flow state."""
        observation = _model.preprocess_observation(None, observation, train=False)
        velocity, _ = self._predict_action_velocity_from_preprocessed(
            observation, noisy_actions, timestep
        )
        return velocity

    def predict_vjepa_from_observation(
        self, observation: _model.Observation
    ) -> at.Float[at.Array, "b p d"]:
        """Predict a JEPA target from deploy-time prefix inputs only."""
        if not self.use_vjepa_aux:
            raise ValueError("V-JEPA prediction requires use_vjepa_aux=True")
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        embedded = [prefix_tokens, None]
        if self.use_action_change_mmdit:
            embedded.append(None)
        (prefix_out, *_), _ = self.PaliGemma.llm(embedded, mask=prefix_attn_mask, positions=positions)
        assert prefix_out is not None
        query_out = prefix_out[:, -self.vjepa_num_queries :]
        return self.predict_vjepa_target(query_out)

    def _project_future_context(self, query_out: jax.Array) -> jax.Array:
        """Map frozen JEPA-WAM future queries to the 4x4 Change grid."""
        value = self.vjepa_alignment_norm(query_out)
        value = nnx.gelu(self.vjepa_alignment_in(value))
        value = self.vjepa_alignment_out(value).astype(jnp.float32)
        value = value.reshape(value.shape[0], 8, 8, self.vjepa_target_dim)
        value *= jax.lax.rsqrt(jnp.sum(jnp.square(value), axis=-1, keepdims=True) + 1e-6)
        value = value.reshape(value.shape[0], 4, 2, 4, 2, self.vjepa_target_dim).mean(axis=(2, 4))
        value *= jax.lax.rsqrt(jnp.sum(jnp.square(value), axis=-1, keepdims=True) + 1e-6)
        value = value.reshape(value.shape[0], self.change_num_tokens, self.vjepa_target_dim)
        return self.future_context_proj(value)

    def _prepare_action_change_prefix(self, observation: _model.Observation):
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None, None], mask=prefix_attn_mask, positions=positions
        )
        assert prefix_out is not None
        query_out = prefix_out[:, -self.vjepa_num_queries :]
        return prefix_mask, kv_cache, self._project_future_context(query_out)

    def _predict_action_change_suffix(
        self,
        observation: _model.Observation,
        noisy_actions: jax.Array,
        noisy_change: jax.Array,
        action_timestep: jax.Array,
        change_timestep: jax.Array,
        prefix_mask: jax.Array,
        kv_cache,
        future_context: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        action_tokens, action_mask, _, action_adarms_cond = self.embed_suffix(
            observation, noisy_actions, action_timestep
        )
        change_adarms_cond = self.embed_flow_time(change_timestep)
        batch_size = action_tokens.shape[0]
        action_length = action_tokens.shape[1]
        change_input = self.change_in_proj(noisy_change)
        change_input = change_input + future_context
        change_input = change_input + self.change_spatial_embedding.value[None].astype(change_input.dtype)
        change_tokens = jnp.zeros_like(change_input)

        depth = self.action_expert_depth
        injections = [
            None,
            jnp.zeros((depth, *action_tokens.shape), dtype=action_tokens.dtype),
            jnp.zeros((depth, *change_tokens.shape), dtype=change_tokens.dtype).at[
                self.change_joint_start_layer
            ].set(change_input),
        ]
        active = jnp.zeros((depth, 3), dtype=jnp.bool_)
        active = active.at[:, 1].set(True)
        active = active.at[self.change_joint_start_layer :, 2].set(True)
        total_suffix = action_length + self.change_num_tokens
        prefix_visible = prefix_mask.at[:, -self.vjepa_num_queries :].set(False)
        prefix_part = einops.repeat(prefix_visible, "b p -> b s p", s=total_suffix)
        early_suffix = jnp.zeros((batch_size, total_suffix, total_suffix), dtype=jnp.bool_)
        early_suffix = early_suffix.at[:, :action_length, :action_length].set(True)
        early_suffix = early_suffix.at[:, action_length:, action_length:].set(True)
        late_suffix = jnp.ones_like(early_suffix)
        early_mask = jnp.concatenate([prefix_part, early_suffix], axis=-1)
        late_mask = jnp.concatenate([prefix_part, late_suffix], axis=-1)
        layer_masks = jnp.broadcast_to(early_mask[None], (depth, *early_mask.shape))
        layer_masks = layer_masks.at[self.change_joint_start_layer :].set(late_mask)

        position_start = jnp.sum(prefix_mask, axis=-1) - self.vjepa_num_queries
        action_positions = position_start[:, None] + jnp.arange(action_length)[None]
        change_positions = position_start[:, None] + action_length + jnp.arange(self.change_num_tokens)[None]
        positions = jnp.concatenate([action_positions, change_positions], axis=1)
        (prefix_out, action_out, change_out), _ = self.PaliGemma.llm(
            [None, action_tokens, change_tokens],
            mask=late_mask,
            positions=positions,
            kv_cache=kv_cache,
            adarms_cond=[None, action_adarms_cond, change_adarms_cond],
            layer_masks=layer_masks,
            injections=injections,
            active_experts=active,
        )
        assert prefix_out is None and action_out is not None and change_out is not None
        return self.action_out_proj(action_out), self.change_out_proj(change_out)

    def _predict_action_change_velocity_from_preprocessed(
        self,
        observation: _model.Observation,
        noisy_actions: jax.Array,
        noisy_change: jax.Array,
        action_timestep: jax.Array,
        change_timestep: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        prefix_mask, kv_cache, future_context = self._prepare_action_change_prefix(observation)
        return self._predict_action_change_suffix(
            observation,
            noisy_actions,
            noisy_change,
            action_timestep,
            change_timestep,
            prefix_mask,
            kv_cache,
            future_context,
        )

    def predict_action_change_velocity(
        self,
        observation: _model.Observation,
        noisy_actions: jax.Array,
        noisy_change: jax.Array,
        action_timestep: jax.Array,
        change_timestep: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Return Con1 velocities for explicit Action/Change flow states."""
        if not self.use_action_change_mmdit:
            raise ValueError("Action–Change velocity requires use_action_change_mmdit=True")
        observation = _model.preprocess_observation(None, observation, train=False)
        return self._predict_action_change_velocity_from_preprocessed(
            observation,
            noisy_actions,
            noisy_change,
            action_timestep,
            change_timestep,
        )

    def compute_action_change_loss_components(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        if observation.change_target is None:
            raise ValueError("Action–Change training requires a frozen Stage-1 change_target")
        preprocess_rng, action_noise_rng, change_noise_rng, time_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, geometric_augmentation=False
        )
        change_target = jax.lax.stop_gradient(observation.change_target.astype(jnp.float32))
        action_noise = jax.random.normal(action_noise_rng, actions.shape)
        change_noise = jax.random.normal(change_noise_rng, change_target.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        expanded = time[..., None, None]
        action_t = expanded * action_noise + (1 - expanded) * actions
        change_t = expanded * change_noise + (1 - expanded) * change_target
        action_velocity, change_velocity = self._predict_action_change_velocity_from_preprocessed(
            observation, action_t, change_t, time, time
        )
        action_target = action_noise - actions
        change_velocity_target = change_noise - change_target
        action_loss = jnp.mean(jnp.square(action_velocity[..., :7] - action_target[..., :7]), axis=-1)
        change_loss = jnp.mean(jnp.square(change_velocity - change_velocity_target), axis=(1, 2))
        return action_loss, change_loss

    def compute_achieved_change_adapter_loss_components(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Train only Con2's directional adapter in joint or clean-Change mode.

        The complete Con1 model is frozen by the config's parameter filter.
        A scalar mode is sampled per optimizer step to avoid running two large
        MMDiT forwards for every batch.
        """
        if observation.change_target is None:
            raise ValueError("Achieved-Change adapter training requires frozen Stage-1 change targets")
        preprocess_rng, action_noise_rng, change_noise_rng, time_rng, mode_rng, inverse_rng = jax.random.split(rng, 6)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, geometric_augmentation=False
        )
        change_target = jax.lax.stop_gradient(observation.change_target.astype(jnp.float32))
        action_noise = jax.random.normal(action_noise_rng, actions.shape)
        change_noise = jax.random.normal(change_noise_rng, change_target.shape)
        inverse_mode = jax.random.bernoulli(
            mode_rng, p=self.achieved_change_inverse_probability
        )

        def joint_mode(_):
            time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
            expanded = time[..., None, None]
            action_t = expanded * action_noise + (1 - expanded) * actions
            change_t = expanded * change_noise + (1 - expanded) * change_target
            action_velocity, change_velocity = self._predict_action_change_velocity_from_preprocessed(
                observation, action_t, change_t, time, time
            )
            action_target = action_noise - actions
            change_target_velocity = change_noise - change_target
            action_loss = jnp.mean(
                jnp.square(action_velocity[..., :7] - action_target[..., :7]), axis=-1
            )
            change_loss = jnp.mean(jnp.square(change_velocity - change_target_velocity), axis=(1, 2))
            return action_loss, change_loss

        def inverse_mode_fn(_):
            batch_shape = actions.shape[:-2]
            high_noise = jax.random.uniform(inverse_rng, batch_shape, minval=0.8, maxval=1.0)
            pure_noise = jax.random.bernoulli(jax.random.fold_in(inverse_rng, 1), 0.5, batch_shape)
            action_time = jnp.where(pure_noise, 1.0, high_noise)
            expanded = action_time[..., None, None]
            action_t = expanded * action_noise + (1 - expanded) * actions
            action_velocity, _ = self._predict_action_change_velocity_from_preprocessed(
                observation,
                action_t,
                change_target,
                action_time,
                jnp.zeros_like(action_time),
            )
            action_target = action_noise - actions
            action_loss = jnp.mean(
                jnp.square(action_velocity[..., :7] - action_target[..., :7]), axis=-1
            )
            return action_loss, jnp.zeros(actions.shape[:-2], dtype=action_loss.dtype)

        action_loss, change_loss = jax.lax.cond(inverse_mode, inverse_mode_fn, joint_mode, operand=None)
        return action_loss, change_loss, inverse_mode.astype(jnp.float32)

    def compute_online_achieved_change_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        executed_actions: jax.Array,
        achieved_change: jax.Array,
        *,
        num_noise_samples: int = 4,
    ) -> jax.Array:
        """Reward-free Con2 loss for one completed deployment transition.

        `executed_actions` must already be mapped back to the normalized Pi0.5
        10x32 action space. `achieved_change` must use the frozen Stage-1
        normalization. The action input is pure Gaussian noise (tau_A=1), so
        the executed command appears only in the velocity target.
        """
        if not self.use_achieved_change_adapter:
            raise ValueError("Online achieved-Change loss requires the directional adapter")
        if num_noise_samples < 1:
            raise ValueError("num_noise_samples must be positive")
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_mask, kv_cache, future_context = self._prepare_action_change_prefix(observation)
        batch_size = executed_actions.shape[0]
        repeats = num_noise_samples
        repeated_observation = jax.tree.map(
            lambda value: None if value is None else jnp.repeat(value, repeats, axis=0),
            observation,
            is_leaf=lambda value: value is None,
        )
        repeated_actions = jnp.repeat(executed_actions, repeats, axis=0)
        repeated_change = jnp.repeat(achieved_change, repeats, axis=0)
        repeated_prefix_mask = jnp.repeat(prefix_mask, repeats, axis=0)
        repeated_future_context = jnp.repeat(future_context, repeats, axis=0)
        repeated_cache = jax.tree.map(lambda value: jnp.repeat(value, repeats, axis=1), kv_cache)
        noise = jax.random.normal(rng, repeated_actions.shape)
        action_time = jnp.ones((batch_size * repeats,), dtype=jnp.float32)
        change_time = jnp.zeros_like(action_time)
        action_velocity, _ = self._predict_action_change_suffix(
            repeated_observation,
            noise,
            repeated_change,
            action_time,
            change_time,
            repeated_prefix_mask,
            repeated_cache,
            repeated_future_context,
        )
        target = noise - repeated_actions
        return jnp.mean(jnp.square(action_velocity[..., :7] - target[..., :7]))

    def compute_all_loss_components(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[
        at.Float[at.Array, "*b ah"],
        at.Float[at.Array, "*b"] | None,
        at.Float[at.Array, "*b"] | None,
        dict[str, at.Float[at.Array, "*b"]],
    ]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            geometric_augmentation=not (self.use_vjepa_aux and self.vjepa_disable_geometric_augmentation),
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        v_t, query_out = self._predict_action_velocity_from_preprocessed(observation, x_t, time)

        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if not self.use_vjepa_aux:
            return flow_loss, None, None, {}
        if observation.vjepa_target is None:
            raise ValueError("V-JEPA auxiliary training requires observation.vjepa_target")

        assert query_out is not None
        predicted_target = self.predict_vjepa_target(query_out)
        target = jax.lax.stop_gradient(observation.vjepa_target.astype(jnp.float32))
        if predicted_target.shape != target.shape:
            raise ValueError(f"V-JEPA prediction/target shape mismatch: {predicted_target.shape} != {target.shape}")
        predicted_target = predicted_target.astype(jnp.float32)
        predicted_target /= jnp.maximum(jnp.linalg.norm(predicted_target, axis=-1, keepdims=True), 1e-6)
        target /= jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
        aux_loss = jnp.mean(1.0 - jnp.sum(predicted_target * target, axis=-1), axis=-1)

        if not self.use_point_flow:
            return flow_loss, aux_loss, None, {}
        if (
            observation.point_flow_queries is None
            or observation.point_flow_target is None
            or observation.point_flow_visibility is None
        ):
            raise ValueError("Point-flow training requires queries, target tracks, and visibility labels")
        predicted_tracks, predicted_visibility_logits = self.point_flow_planner(
            jax.lax.stop_gradient(query_out), observation.point_flow_queries.astype(jnp.float32)
        )
        point_loss, point_metrics = _point_flow.point_flow_loss(
            predicted_tracks,
            predicted_visibility_logits,
            jax.lax.stop_gradient(observation.point_flow_target.astype(jnp.float32)),
            jax.lax.stop_gradient(observation.point_flow_visibility),
            visibility_weight=self.point_flow_visibility_weight,
            smoothness_weight=self.point_flow_smoothness_weight,
        )
        return flow_loss, aux_loss, point_loss, point_metrics

    def compute_loss_components(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b"] | None]:
        """Backward-compatible action and JEPA losses used by existing configs."""
        flow_loss, aux_loss, _, _ = self.compute_all_loss_components(rng, observation, actions, train=train)
        return flow_loss, aux_loss

    def predict_vjepa_target(self, query_out: at.Float[at.Array, "b q emb"]) -> at.Float[at.Array, "b p d"]:
        value = self.vjepa_alignment_norm(query_out)
        value = nnx.gelu(self.vjepa_alignment_in(value))
        value = self.vjepa_alignment_out(value)
        value = value.reshape(
            value.shape[0], self.vjepa_query_grid_size, self.vjepa_query_grid_size, self.vjepa_target_dim
        )
        value = jax.image.resize(
            value,
            (value.shape[0], self.vjepa_target_grid_size, self.vjepa_target_grid_size, self.vjepa_target_dim),
            method="linear",
        )
        return value.reshape(value.shape[0], self.vjepa_target_grid_size**2, self.vjepa_target_dim)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.use_action_change_mmdit:
            action_loss, change_loss = self.compute_action_change_loss_components(
                rng, observation, actions, train=train
            )
            return action_loss + self.change_loss_weight * change_loss[..., None]
        flow_loss, aux_loss, point_loss, _ = self.compute_all_loss_components(rng, observation, actions, train=train)
        if aux_loss is None:
            return flow_loss
        total_loss = flow_loss + self.vjepa_aux_weight * aux_loss[..., None]
        if point_loss is not None:
            total_loss = total_loss + self.point_flow_loss_weight * point_loss[..., None]
        return total_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        if self.use_action_change_mmdit:
            return self._sample_action_change(rng, observation, num_steps=num_steps, noise=noise)[0]
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            if self.use_vjepa_aux and not self.vjepa_action_attends_queries:
                prefix_attn_mask = prefix_attn_mask.at[:, :, -self.vjepa_num_queries :].set(False)
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            if self.use_vjepa_aux and not self.vjepa_action_attends_queries:
                positions = positions - self.vjepa_num_queries

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def sample_action_change(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Return both generated endpoints for Con1/Con2 diagnostics."""
        if not self.use_action_change_mmdit:
            raise ValueError("Action–Change sampling requires use_action_change_mmdit=True")
        observation = _model.preprocess_observation(None, observation, train=False)
        return self._sample_action_change(rng, observation, num_steps=num_steps, noise=noise)

    def _sample_action_change(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Jointly integrate action and predicted change using current inputs only."""
        action_rng, change_rng = jax.random.split(rng)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(action_rng, (batch_size, self.action_horizon, self.action_dim))
        change_noise = jax.random.normal(
            change_rng, (batch_size, self.change_num_tokens, self.change_token_dim)
        )
        prefix_mask, kv_cache, future_context = self._prepare_action_change_prefix(observation)
        dt = -1.0 / num_steps

        def step(carry):
            action_t, change_t, time = carry
            action_velocity, change_velocity = self._predict_action_change_suffix(
                observation,
                action_t,
                change_t,
                jnp.broadcast_to(time, batch_size),
                jnp.broadcast_to(time, batch_size),
                prefix_mask,
                kv_cache,
                future_context,
            )
            return action_t + dt * action_velocity, change_t + dt * change_velocity, time + dt

        def cond(carry):
            return carry[2] >= -dt / 2

        action_0, change_0, _ = jax.lax.while_loop(cond, step, (noise, change_noise, 1.0))
        return action_0, change_0
