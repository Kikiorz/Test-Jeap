import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.con1.modules import (
    AnchoredDeltaHead,
    ActionDeltaCrossAttention,
    DeltaMetric,
    DeltaRefinement,
    balanced_latent_weight,
    control_weighted_delta_loss,
    direction_alignment_loss,
    metric_delta_loss,
    sensitivity_horizon_weights,
)

logger = logging.getLogger("openpi")


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
        self.use_con1 = config.use_con1
        self.con1_delta_weight = config.con1_delta_weight
        self.con1_sgr_beta = config.con1_sgr_beta
        self.con1_residual_weight = config.con1_residual_weight
        self.con1_action_dims = config.con1_action_dims
        self.con1_action_conditioning = config.con1_action_conditioning
        self.con1_action_conditioning_source = config.con1_action_conditioning_source
        if self.con1_action_conditioning_source not in ("demonstration", "estimate"):
            raise ValueError("con1_action_conditioning_source must be 'demonstration' or 'estimate'")
        self.con1_train_action_layers_from = config.con1_train_action_layers_from
        self.con1_metric = config.con1_metric
        self.use_con2 = config.use_con2
        self.con1_metric_align_weight = config.con1_metric_align_weight
        self.con1_balance_strength = config.con1_balance_strength
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.action_depth = action_expert_config.depth
        self.con1_insert_from = max(0, self.action_depth - 4)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
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
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        if config.use_con1:
            # Linen modules are bridged into NNX so their parameters are part of
            # the normal checkpoint/optimizer tree.  The zero-initialized output
            # projection preserves the pretrained action function at step zero.
            self.con1_delta_head = nnx_bridge.ToNNX(
                AnchoredDeltaHead(horizon=config.action_horizon,
                                  latent_dim=config.con1_latent_dim,
                                  width=config.con1_width,
                                  action_dim=config.con1_action_dims,
                                  use_action_conditioning=config.con1_action_conditioning,
                                  use_direct_readout=config.con1_direct_readout,
                                  vlm_context_dim=paligemma_config.width,
                                  use_vlm_context=config.con1_vlm_context,
                                  use_vlm_context_tokens=config.con1_vlm_context_tokens)
            )
            self.con1_delta_head.lazy_init(
                jnp.zeros((1, self.vjepa_num_queries, paligemma_config.width), dtype=jnp.float32),
                jnp.zeros((1, config.con1_latent_dim), dtype=jnp.float32),
                (jnp.zeros((1, config.action_horizon, config.con1_action_dims), dtype=jnp.float32)
                 if config.con1_action_conditioning else None),
                (jnp.zeros((1, paligemma_config.width), dtype=jnp.float32)
                 if config.con1_vlm_context else None),
                (jnp.zeros((1, 256, paligemma_config.width), dtype=jnp.float32)
                 if config.con1_vlm_context_tokens else None),
                (jnp.zeros((1, 256), dtype=jnp.bool_)
                 if config.con1_vlm_context_tokens else None),
                rngs=rngs,
            )
            self.con1_cross_attention = nnx_bridge.ToNNX(ActionDeltaCrossAttention(
                action_expert_config.width, width=config.con1_width,
                alpha_initial=config.con1_alpha_initial,
                use_action_adapter=config.con1_action_adapter,
                adapter_scale=config.con1_adapter_scale,
                out_init_std=config.con1_cross_attention_out_init,
                residual_budget=config.con1_residual_budget))
            self.con1_cross_attention.lazy_init(
                jnp.zeros((1, config.action_horizon, action_expert_config.width), dtype=jnp.float32),
                jnp.zeros((1, config.action_horizon, config.con1_latent_dim), dtype=jnp.float32), rngs=rngs)
            if config.con1_metric:
                # Bounded diagonal metric on the latent residual; zero-initialised
                # so it is the identity (the Euclidean loss) at step 0.
                self.con1_delta_metric = nnx_bridge.ToNNX(
                    DeltaMetric(latent_dim=config.con1_latent_dim,
                                max_scale=config.con1_metric_max_scale))
                self.con1_delta_metric.lazy_init(
                    jnp.zeros((1, config.action_horizon, config.con1_latent_dim), dtype=jnp.float32),
                    rngs=rngs)
            if config.use_con2:
                # Con2: learned refinement of the predicted latent delta, trained
                # through the action objective as well as the latent objective.
                self.con2_refine = nnx_bridge.ToNNX(
                    DeltaRefinement(latent_dim=config.con1_latent_dim,
                                    width=config.con2_width))
                self.con2_refine.lazy_init(
                    jnp.zeros((1, config.action_horizon, config.con1_latent_dim), dtype=jnp.float32),
                    jnp.zeros((1, config.con1_latent_dim), dtype=jnp.float32),
                    rngs=rngs)

        if self.use_vjepa_aux:
            query_init = jax.random.normal(
                rngs.params(), (self.vjepa_num_queries, paligemma_config.width), dtype=jnp.float32
            )
            self.vjepa_query_tokens = nnx.Param(query_init * 0.02)
            self.vjepa_alignment_norm = nnx.LayerNorm(paligemma_config.width, rngs=rngs)
            self.vjepa_alignment_in = nnx.Linear(paligemma_config.width, paligemma_config.width, rngs=rngs)
            self.vjepa_alignment_out = nnx.Linear(paligemma_config.width, self.vjepa_target_dim, rngs=rngs)

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
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
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

    def compute_loss_components(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> tuple[at.Float[at.Array, "*b ah"], at.Float[at.Array, "*b"] | None]:
        if self.use_con1:
            _, info = self.compute_con1_loss(rng, observation, actions, beta=self.con1_sgr_beta)
            return info["flow_loss"], info["con1_delta_loss"]
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

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
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
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        action_hidden = suffix_out[:, -self.action_horizon :]
        v_t = self.action_out_proj(action_hidden)

        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if not self.use_vjepa_aux or observation.vjepa_target is None:
            return flow_loss, None

        query_out = prefix_out[:, -self.vjepa_num_queries :]
        predicted_target = self.predict_vjepa_target(query_out)
        target = jax.lax.stop_gradient(observation.vjepa_target.astype(jnp.float32))
        if predicted_target.shape != target.shape:
            raise ValueError(f"V-JEPA prediction/target shape mismatch: {predicted_target.shape} != {target.shape}")
        predicted_target = predicted_target.astype(jnp.float32)
        predicted_target /= jnp.maximum(jnp.linalg.norm(predicted_target, axis=-1, keepdims=True), 1e-6)
        target /= jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
        aux_loss = jnp.mean(1.0 - jnp.sum(predicted_target * target, axis=-1), axis=-1)
        return flow_loss, aux_loss

    def _con1_prefix(self, observation):
        """Frozen prefix pass; independent of the action chunk, so callers can
        run it once and re-derive the delta as the action estimate changes."""
        if observation.con1_current_latent is None and self.use_vjepa_aux:
            # The JEPA-WAM path needs the offline teacher latent; the SimpENV path
            # computes its latent from the prefix it is already running (the
            # pooled VLM prefix is exactly what the head was trained against).
            raise ValueError("Con1 requires a CURRENT-only teacher latent at both training and inference")
        tokens, mask, ar_mask = self.embed_prefix(observation)
        (prefix, _), cache = self.PaliGemma.llm(
            [tokens, None], mask=make_attn_mask(mask, ar_mask),
            positions=jnp.cumsum(mask, axis=1) - 1)
        prefix = jax.lax.stop_gradient(prefix)
        cache = jax.tree.map(jax.lax.stop_gradient, cache)
        if self.use_vjepa_aux:
            queries = jax.lax.stop_gradient(prefix[:, -self.vjepa_num_queries:])
            context_tokens = jax.lax.stop_gradient(prefix[:, : -self.vjepa_num_queries])
            context_mask = mask[:, : -self.vjepa_num_queries]
        else:
            # SimpENV branch: the checkpoint has no JEPA predictive-query tokens
            # (pi0.5 was never trained with R_t on this data), so the delta head
            # attends over the whole VLM prefix - language, every image patch and
            # the state - instead of a dedicated R_t.
            queries = jax.lax.stop_gradient(prefix)
            context_tokens = jax.lax.stop_gradient(prefix)
            context_mask = mask
        context_mask_f = context_mask.astype(jnp.float32)
        pooled = (context_tokens * context_mask_f[..., None]).sum(1)
        pooled = pooled / jnp.maximum(context_mask_f.sum(1, keepdims=True), 1.0)
        # `pooled` is the collapsed variant; `tokens`/`mask` let the delta head
        # attend over the whole VLM prefix (language, every image patch, state)
        # instead of seeing a single averaged vector.
        return (mask, cache), queries, {
            "pooled": jax.lax.stop_gradient(pooled),
            "tokens": context_tokens,
            "mask": context_mask,
        }

    def _con1_delta(self, r_tokens, current_latent, action_chunk=None, vlm_context=None):
        pooled = tokens = context_mask = None
        if vlm_context is not None:
            pooled = vlm_context.get("pooled")
            tokens = vlm_context.get("tokens")
            context_mask = vlm_context.get("mask")
        delta = self.con1_delta_head(
            r_tokens, current_latent, action_chunk, pooled, tokens, context_mask)["delta"]
        if self.use_con2:
            # Con2 refinement: the correction is zero at init, so this is a
            # strict extension of Con1. Both the latent term and the action flow
            # term are computed on the refined delta, which is what makes the
            # refinement action-supervised rather than latent-supervised only.
            delta = self.con2_refine(delta, current_latent)["delta"]
        return delta

    def _con1_context(self, observation, action_chunk=None):
        context, r_tokens, vlm_context = self._con1_prefix(observation)
        current_latent = observation.con1_current_latent
        if current_latent is None:
            current_latent = vlm_context["pooled"]
        delta = self._con1_delta(r_tokens, current_latent, action_chunk, vlm_context)
        return context, delta

    def _con1_velocity(self, observation, x_t, time, context, delta):
        """One residual retrieval before blocks 14--17, shared by train/inference."""
        prefix_mask, cache = context
        hidden, mask, ar_mask, cond = self.embed_suffix(observation, x_t, time)
        prefix_attention = einops.repeat(prefix_mask, "b p -> b s p", s=hidden.shape[1])
        if not self.vjepa_action_attends_queries:
            prefix_attention = prefix_attention.at[:, :, -self.vjepa_num_queries:].set(False)
        full_mask = jnp.concatenate([prefix_attention, make_attn_mask(mask, ar_mask)], axis=-1)
        positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
        if not self.vjepa_action_attends_queries:
            positions -= self.vjepa_num_queries
        # Run the frozen lower stack once. No extra denoising loop for training.
        if self.con1_insert_from:
            hidden = self.PaliGemma.llm(
                hidden, positions, full_mask, cond, cache, start=0, stop=self.con1_insert_from,
                frozen=True, method="suffix_segment")
        fused = self.con1_cross_attention(hidden, delta)
        hidden = self.PaliGemma.llm(
            fused["hidden"].astype(hidden.dtype), positions, full_mask, cond, cache,
            start=self.con1_insert_from, stop=self.action_depth, frozen=False, method="suffix_segment")
        hidden = self.PaliGemma.llm(hidden, cond, method="normalize_suffix")
        return self.action_out_proj(hidden[:, -self.action_horizon:]), {
            "attention": fused["attention"],
            "con1_residual_energy": jnp.mean(jnp.square(fused["correction"])),
            "con1_alpha": fused["alpha"],
        }

    def compute_con1_loss(self, rng, observation, actions, *, beta=0., flow_weight=1.):
        # Offline z labels describe unaugmented observations. Do not silently
        # photometrically/geometrically augment only the policy side.
        observation = _model.preprocess_observation(None, observation, train=False)
        noise_rng, time_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * .999 + .001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions
        # Action conditioning: predict the consequence of the action chunk rather
        # than the marginal future.
        if not self.con1_action_conditioning:
            context, delta = self._con1_context(observation, None)
        elif self.con1_action_conditioning_source == "estimate":
            # Sampling feeds the head the *previous denoising step's* estimate,
            # not the demonstrated chunk. Reproduce that distribution: a
            # stop-gradient probe pass with the demonstrated chunk gives the
            # one-step-lagged estimate `x_t - t * v`, and the real (differentiable)
            # pass is then conditioned on it. Without this the head only ever sees
            # the training-time chunk and cannot learn to use the action.
            context, delta_probe = self._con1_context(observation, actions[..., : self.con1_action_dims])
            velocity_probe, _ = self._con1_velocity(
                observation, x_t, time, context, jax.lax.stop_gradient(delta_probe))
            estimate = jax.lax.stop_gradient(
                x_t - time[..., None, None] * velocity_probe)[..., : self.con1_action_dims]
            context, delta = self._con1_context(observation, estimate)
        else:
            action_chunk = actions[..., : self.con1_action_dims]
            context, delta = self._con1_context(observation, action_chunk)
        # One main forward. Its VJP supplies action sensitivity; stop_gradient
        # on q avoids second-order optimization of importance weights.
        v_t, pullback, aux = jax.vjp(
            lambda d: self._con1_velocity(observation, x_t, time, context, d), delta, has_aux=True)
        if observation.con1_future_latents is None or observation.con1_future_valid is None:
            raise ValueError("Con1 training requires future labels AND episode-local validity mask")
        # action[t:t+H] versus latent[t+1:t+H+1]: the masks differ by one.
        action_valid = jnp.concatenate([jnp.ones_like(observation.con1_future_valid[:, :1]),
                                       observation.con1_future_valid[:, :-1]], axis=1)
        action_mask = action_valid[..., None] & (jnp.arange(self.action_dim) < self.con1_action_dims)
        error = jnp.where(action_mask, v_t.astype(jnp.float32) - u_t.astype(jnp.float32), 0.)
        action_count = jnp.maximum(action_mask.sum(), 1)
        # The action-direction VJP is needed by SGR, by the metric's alignment
        # objective, and by the gradient balancer, so it is computed whenever any
        # of them is active - and only then.
        need_sensitivity = (jnp.asarray(beta) > 0) | (
            (self.con1_balance_strength > 0)
            or (self.con1_metric and self.con1_metric_align_weight > 0))
        sensitivity = jax.lax.cond(
            jnp.asarray(need_sensitivity),
            # The pullback is taken with respect to the action chunk, which the
            # action expert produces in bf16, so the cotangent must match that
            # dtype even though the loss itself is computed in fp32.
            lambda _: jax.lax.stop_gradient(
                pullback(jnp.asarray(2 * error / action_count, v_t.dtype))[0]),
            lambda _: jnp.zeros_like(delta), operand=None)
        if self.con1_metric:
            # Learned bounded metric on the same residual/target: identity at
            # step 0, positive definite for every parameter value, so the only
            # zero stays at delta = delta*.
            _, scale = self.con1_delta_metric(delta)
            weights = sensitivity_horizon_weights(
                aux["attention"], sensitivity, observation.con1_future_valid, beta=beta,
                action_valid=action_valid)
            delta_loss, delta_metrics = metric_delta_loss(
                delta, observation.con1_current_latent, observation.con1_future_latents,
                observation.con1_future_valid, scale, weights=weights)
            delta_metrics["con1_metric_scale_mean"] = jnp.mean(scale)
            align_loss = jnp.zeros(())
            if self.con1_metric_align_weight > 0:
                align_loss, align_metrics = direction_alignment_loss(
                    delta, observation.con1_current_latent, observation.con1_future_latents,
                    observation.con1_future_valid, sensitivity, scale)
                delta_metrics.update(align_metrics)
        else:
            delta_loss, delta_metrics = control_weighted_delta_loss(
                delta, observation.con1_current_latent, observation.con1_future_latents,
                observation.con1_future_valid, aux["attention"], sensitivity, beta=beta,
                action_valid=action_valid)
            align_loss = jnp.zeros(())
        # The configured 0.2-vs-1.0 weights are not the effective balance: the
        # latent term's gradient on the head is measured to be ~700x larger. The
        # factor below is detached, so it cannot be gamed by shrinking the loss.
        if self.con1_balance_strength > 0:
            if self.con1_metric:
                delta_grad = jax.grad(
                    lambda d: metric_delta_loss(
                        d, observation.con1_current_latent, observation.con1_future_latents,
                        observation.con1_future_valid, scale, weights=weights)[0])(delta)
            else:
                delta_grad = jax.grad(
                    lambda d: control_weighted_delta_loss(
                        d, observation.con1_current_latent, observation.con1_future_latents,
                        observation.con1_future_valid, aux["attention"], sensitivity, beta=beta,
                        action_valid=action_valid)[0])(delta)
            latent_weight = self.con1_delta_weight * balanced_latent_weight(
                sensitivity, delta_grad, self.con1_balance_strength)
        else:
            latent_weight = self.con1_delta_weight
        flow = jnp.square(error).sum() / action_count
        weighted_flow = flow_weight * flow
        weighted_delta = latent_weight * delta_loss
        total = (weighted_flow + weighted_delta
                 + self.con1_metric_align_weight * align_loss
                 + self.con1_residual_weight * aux["con1_residual_energy"])
        return total, dict(delta_metrics, flow_loss=flow, weighted_flow_loss=weighted_flow,
                          flow_weight=flow_weight, con1_delta_loss=delta_loss,
                          weighted_con1_delta_loss=weighted_delta,
                          con1_latent_weight=latent_weight,
                          con1_metric_align_loss=align_loss,
                          con1_alpha=aux["con1_alpha"], con1_residual_energy=aux["con1_residual_energy"],
                          sgr_beta=jnp.asarray(beta))

    def extract_pooled_prefix(self, observation: _model.Observation):
        """Frozen pooled VLM prefix - the Con1 latent for the SimpENV branch.

        pi0.5 has no JEPA predictive-query tokens on this data, so the Con1
        latent target is simply the mask-weighted mean of the whole VLM prefix
        (language, every image patch, state). No suffix, action labels or future
        images enter, and the result is stop-gradiented like the JEPA teacher.
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        tokens, mask, ar_mask = self.embed_prefix(observation)
        (prefix, _), _ = self.PaliGemma.llm(
            [tokens, None], mask=make_attn_mask(mask, ar_mask),
            positions=jnp.cumsum(mask, axis=1) - 1)
        weights = mask.astype(jnp.float32)[..., None]
        pooled = (prefix * weights).sum(1) / jnp.maximum(weights.sum(1), 1.0)
        return jax.lax.stop_gradient(pooled.astype(jnp.float32))

    def extract_predictive_tokens(self, observation: _model.Observation):
        """Frozen current-only R, exactly the prefix used by action sampling.

        No suffix, action labels, future images, or auxiliary targets enter.
        This does not change the author's policy/action computation.
        """
        if not self.use_vjepa_aux:
            raise ValueError("Predictive tokens require a JEPA-WAM checkpoint")
        observation = _model.preprocess_observation(None, observation, train=False)
        tokens, mask, ar_mask = self.embed_prefix(observation)
        (prefix, _), _cache = self.PaliGemma.llm(
            [tokens, None], mask=make_attn_mask(mask, ar_mask),
            positions=jnp.cumsum(mask, axis=1) - 1)
        return jax.lax.stop_gradient(prefix[:, -self.vjepa_num_queries:].astype(jnp.float32))

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
        if self.use_con1:
            loss, _ = self.compute_con1_loss(rng, observation, actions, beta=self.con1_sgr_beta)
            return jnp.broadcast_to(loss, actions.shape[:-1])
        flow_loss, aux_loss = self.compute_loss_components(rng, observation, actions, train=train)
        if aux_loss is None:
            return flow_loss

        return flow_loss + self.vjepa_aux_weight * aux_loss[..., None]

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
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        if self.use_con1:
            # The prefix does not depend on the action chunk, so run it once.
            # With action conditioning the delta must be re-derived each step from
            # the current clean-action estimate a_hat = x_t - time * velocity.
            context, r_tokens, vlm_context = self._con1_prefix(observation)
            current_latent = observation.con1_current_latent
            if current_latent is None:
                current_latent = vlm_context["pooled"]
            initial_estimate = jnp.zeros(
                (batch_size, self.action_horizon, self.con1_action_dims), dtype=noise.dtype
            )

            def con1_step(carry):
                x_t, time, action_estimate = carry
                condition = action_estimate if self.con1_action_conditioning else None
                delta = self._con1_delta(r_tokens, current_latent, condition, vlm_context)
                velocity, _ = self._con1_velocity(
                    observation, x_t, jnp.broadcast_to(time, (batch_size,)), context, delta)
                next_estimate = (x_t - time * velocity)[..., : self.con1_action_dims]
                return x_t + dt * velocity, time + dt, next_estimate

            return jax.lax.while_loop(lambda carry: carry[1] >= -dt / 2,
                                      con1_step, (noise, 1.0, initial_estimate))[0]

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
