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


class _SwiGLUUpdate(nnx.Module):
    """A compact, modality-specific nonlinear update used inside ACTR."""

    def __init__(self, width: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.gate = nnx.Linear(width, hidden_dim, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(width, hidden_dim, use_bias=False, rngs=rngs)
        self.output = nnx.Linear(hidden_dim, width, use_bias=False, rngs=rngs)

    def __call__(self, x: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b n d"]:
        return self.output(nnx.swish(self.gate(x)) * self.value(x))


class ActionContingentTransitionRefiner(nnx.Module):
    """Ordered A->R->A co-refinement over pretrained JEPA-WAM tokens.

    The two streams retain their native widths and receive independent norms,
    projections and FFNs.  They only meet in a low-dimensional QK-normalized
    attention space.  Zero-initialized scalar gates make the module an exact
    identity at warm start.
    """

    def __init__(
        self,
        transition_width: int,
        action_width: int,
        interaction_dim: int,
        num_heads: int,
        ffn_dim: int,
        flow_onset: float,
        stage: int,
        *,
        rngs: nnx.Rngs,
    ):
        if interaction_dim % num_heads:
            raise ValueError("interaction_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = interaction_dim // num_heads
        self.flow_onset = flow_onset
        self.stage = stage

        self.transition_norm_ar = nnx.RMSNorm(transition_width, rngs=rngs)
        self.action_norm_ar = nnx.RMSNorm(action_width, rngs=rngs)
        self.transition_q_ar = nnx.Linear(transition_width, interaction_dim, use_bias=False, rngs=rngs)
        self.action_k_ar = nnx.Linear(action_width, interaction_dim, use_bias=False, rngs=rngs)
        self.action_v_ar = nnx.Linear(action_width, interaction_dim, use_bias=False, rngs=rngs)
        self.transition_out_ar = nnx.Linear(interaction_dim, transition_width, use_bias=False, rngs=rngs)
        self.transition_ffn_norm = nnx.RMSNorm(transition_width, rngs=rngs)
        self.transition_ffn = _SwiGLUUpdate(transition_width, ffn_dim, rngs=rngs)
        self.gate_ar = nnx.Param(jnp.zeros((), dtype=jnp.float32))

        self.action_norm_ra = nnx.RMSNorm(action_width, rngs=rngs)
        self.transition_norm_ra = nnx.RMSNorm(transition_width, rngs=rngs)
        self.action_q_ra = nnx.Linear(action_width, interaction_dim, use_bias=False, rngs=rngs)
        self.transition_k_ra = nnx.Linear(transition_width, interaction_dim, use_bias=False, rngs=rngs)
        self.transition_v_ra = nnx.Linear(transition_width, interaction_dim, use_bias=False, rngs=rngs)
        self.action_out_ra = nnx.Linear(interaction_dim, action_width, use_bias=False, rngs=rngs)
        self.action_ffn_norm = nnx.RMSNorm(action_width, rngs=rngs)
        self.action_ffn = _SwiGLUUpdate(action_width, ffn_dim, rngs=rngs)
        self.gate_ra = nnx.Param(jnp.zeros((), dtype=jnp.float32))

    def _heads(self, x: at.Float[at.Array, "b n d"]) -> at.Float[at.Array, "b n h dh"]:
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim)

    @staticmethod
    def _qk_norm(x: at.Float[at.Array, "b n h d"]) -> at.Float[at.Array, "b n h d"]:
        variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        return (x * jax.lax.rsqrt(variance + 1e-6)).astype(x.dtype)

    def _attention(
        self,
        query: at.Float[at.Array, "b q h d"],
        key: at.Float[at.Array, "b k h d"],
        value: at.Float[at.Array, "b k h d"],
    ) -> at.Float[at.Array, "b q i"]:
        query = self._qk_norm(query)
        key = self._qk_norm(key)
        logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, preferred_element_type=jnp.float32)
        logits *= self.head_dim**-0.5
        weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
        output = jnp.einsum("bhqk,bkhd->bqhd", weights, value)
        return output.reshape(output.shape[0], output.shape[1], self.num_heads * self.head_dim)

    def __call__(
        self,
        transition_prior: at.Float[at.Array, "b r dr"],
        action_hidden: at.Float[at.Array, "b a da"],
        time: at.Float[at.Array, " b"],
    ) -> tuple[at.Float[at.Array, "b r dr"], at.Float[at.Array, "b a da"]]:
        late_weight = (time <= self.flow_onset).astype(transition_prior.dtype)[:, None, None]

        transition_norm = self.transition_norm_ar(transition_prior)
        # The action values condition the transition working copy, but the
        # transition objective must not reshape the pretrained action stream.
        action_norm = jax.lax.stop_gradient(self.action_norm_ar(action_hidden))
        transition_query = self._heads(self.transition_q_ar(transition_norm))
        action_key = self._heads(self.action_k_ar(action_norm))
        action_value = self._heads(self.action_v_ar(action_norm))
        transition_message = self.transition_out_ar(
            self._attention(transition_query, action_key, action_value)
        )
        transition_update = transition_message + self.transition_ffn(
            self.transition_ffn_norm(transition_prior + transition_message)
        )
        transition_work = transition_prior + late_weight * jnp.tanh(self.gate_ar.value) * transition_update

        if self.stage == 1:
            return transition_work, action_hidden

        action_norm = self.action_norm_ra(action_hidden)
        transition_norm = self.transition_norm_ra(transition_work)
        action_query = self._heads(self.action_q_ra(action_norm))
        transition_key = self._heads(self.transition_k_ra(transition_norm))
        transition_value = self._heads(self.transition_v_ra(transition_norm))
        action_message = self.action_out_ra(self._attention(action_query, transition_key, transition_value))
        action_update = action_message + self.action_ffn(
            self.action_ffn_norm(action_hidden + action_message)
        )
        action_work = action_hidden + late_weight.astype(action_hidden.dtype) * jnp.tanh(
            self.gate_ra.value
        ) * action_update
        return transition_work, action_work


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.use_vjepa_aux = config.use_vjepa_aux
        self.vjepa_num_queries = config.vjepa_num_queries
        self.vjepa_query_grid_size = config.vjepa_query_grid_size
        self.vjepa_target_grid_size = config.vjepa_target_grid_size
        self.vjepa_target_dim = config.vjepa_target_dim
        self.vjepa_compact_target_dim = config.vjepa_compact_target_dim
        self.vjepa_compact_projection_seed = config.vjepa_compact_projection_seed
        self.vjepa_aux_weight = config.vjepa_aux_weight
        self.vjepa_action_attends_queries = config.vjepa_action_attends_queries
        self.vjepa_disable_geometric_augmentation = config.vjepa_disable_geometric_augmentation
        self.use_actr = config.use_actr
        self.actr_stage = config.actr_stage
        self.actr_action_loss_weight = config.actr_action_loss_weight
        self.actr_injection_layer = config.actr_injection_layer
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        if self.use_actr and not 0 < self.actr_injection_layer < action_expert_config.depth:
            raise ValueError(
                f"ACTR injection layer must be in (0, {action_expert_config.depth}), "
                f"got {self.actr_injection_layer}"
            )
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm_cls = _gemma.SplitModule if self.use_actr else _gemma.Module
        llm_kwargs = {"split_layer": self.actr_injection_layer} if self.use_actr else {}
        llm = nnx_bridge.ToNNX(
            llm_cls(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                **llm_kwargs,
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

        if self.use_vjepa_aux:
            query_init = jax.random.normal(
                rngs.params(), (self.vjepa_num_queries, paligemma_config.width), dtype=jnp.float32
            )
            self.vjepa_query_tokens = nnx.Param(query_init * 0.02)
            self.vjepa_alignment_norm = nnx.LayerNorm(paligemma_config.width, rngs=rngs)
            self.vjepa_alignment_in = nnx.Linear(paligemma_config.width, paligemma_config.width, rngs=rngs)
            self.vjepa_alignment_out = nnx.Linear(paligemma_config.width, self.vjepa_target_dim, rngs=rngs)
        if self.use_actr:
            self.actr = ActionContingentTransitionRefiner(
                transition_width=paligemma_config.width,
                action_width=action_expert_config.width,
                interaction_dim=config.actr_interaction_dim,
                num_heads=config.actr_num_heads,
                ffn_dim=config.actr_ffn_dim,
                flow_onset=config.actr_flow_onset,
                stage=config.actr_stage,
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
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng,
            observation,
            train=train,
            geometric_augmentation=not (self.use_vjepa_aux and self.vjepa_disable_geometric_augmentation),
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        if self.use_actr and self.actr_stage == 1:
            # Stage 1 explicitly studies whether a meaningful partial action
            # disambiguates the frozen transition prior.
            time = jax.random.uniform(time_rng, batch_shape, minval=0.001, maxval=self.actr.flow_onset)
        else:
            time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        if self.use_actr:
            # ACTR needs a genuine intermediate Action Expert state.  Build
            # the frozen transition prior once from the complete prefix, then
            # run suffix blocks 1:injection, co-refine A->R->A, and finish the
            # remaining frozen blocks.
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
            (prefix_out, _), kv_cache = self.PaliGemma.llm(
                [prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions
            )
            transition_prior = prefix_out[:, -self.vjepa_num_queries :]

            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            suffix_to_prefix_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            if not self.vjepa_action_attends_queries:
                suffix_to_prefix_mask = suffix_to_prefix_mask.at[:, :, -self.vjepa_num_queries :].set(False)
            full_attn_mask = jnp.concatenate([suffix_to_prefix_mask, suffix_attn_mask], axis=-1)
            suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            if not self.vjepa_action_attends_queries:
                suffix_positions = suffix_positions - self.vjepa_num_queries

            early_cache, late_cache = self._split_actr_cache(kv_cache)
            (_, suffix_draft), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=early_cache,
                adarms_cond=[None, adarms_cond],
                method="forward_early",
            )
            transition_work, action_grounded = self.actr(
                transition_prior, suffix_draft[:, -self.action_horizon :], time
            )
            # Pi0.5 has only action tokens in the suffix, so the grounded
            # action sequence is exactly the late-block input sequence.
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, action_grounded],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=late_cache,
                adarms_cond=[None, adarms_cond],
                method="forward_late",
            )
            query_out = transition_work
            action_hidden = suffix_out[:, -self.action_horizon :]
        else:
            # Keep the released model's joint training forward unchanged.
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
            query_out = prefix_out[:, -self.vjepa_num_queries :] if self.use_vjepa_aux else None
            action_hidden = suffix_out[:, -self.action_horizon :]
        if not self.use_vjepa_aux:
            v_t = self.action_out_proj(action_hidden)
            flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
            return flow_loss, None
        if observation.vjepa_target is None:
            raise ValueError("V-JEPA auxiliary training requires observation.vjepa_target")

        assert query_out is not None
        v_t = self.action_out_proj(action_hidden)
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        predicted_target = self.predict_vjepa_supervision(query_out)
        target = jax.lax.stop_gradient(observation.vjepa_target.astype(jnp.float32))
        if predicted_target.shape != target.shape:
            raise ValueError(f"V-JEPA prediction/target shape mismatch: {predicted_target.shape} != {target.shape}")
        predicted_target = predicted_target.astype(jnp.float32)
        predicted_target /= jnp.maximum(jnp.linalg.norm(predicted_target, axis=-1, keepdims=True), 1e-6)
        target /= jnp.maximum(jnp.linalg.norm(target, axis=-1, keepdims=True), 1e-6)
        aux_loss = jnp.mean(1.0 - jnp.sum(predicted_target * target, axis=-1), axis=-1)
        return flow_loss, aux_loss

    def predict_vjepa_target(self, query_out: at.Float[at.Array, "b q emb"]) -> at.Float[at.Array, "b p d"]:
        value = self.predict_vjepa_query_grid(query_out)
        value = jax.image.resize(
            value,
            (value.shape[0], self.vjepa_target_grid_size, self.vjepa_target_grid_size, self.vjepa_target_dim),
            method="linear",
        )
        return value.reshape(value.shape[0], self.vjepa_target_grid_size**2, self.vjepa_target_dim)

    def predict_vjepa_query_grid(self, query_out: at.Float[at.Array, "b q emb"]) -> at.Float[at.Array, "b h w d"]:
        value = self.vjepa_alignment_norm(query_out)
        value = nnx.gelu(self.vjepa_alignment_in(value))
        value = self.vjepa_alignment_out(value)
        return value.reshape(
            value.shape[0], self.vjepa_query_grid_size, self.vjepa_query_grid_size, self.vjepa_target_dim
        )

    def predict_vjepa_supervision(self, query_out: at.Float[at.Array, "b q emb"]) -> at.Float[at.Array, "b p d"]:
        if not self.vjepa_compact_target_dim:
            return self.predict_vjepa_target(query_out)

        value = self.predict_vjepa_query_grid(query_out).astype(jnp.float32)
        value /= jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-6)
        projection = jax.random.rademacher(
            jax.random.key(self.vjepa_compact_projection_seed),
            (self.vjepa_target_dim, self.vjepa_compact_target_dim),
            dtype=jnp.float32,
        ) * (self.vjepa_compact_target_dim**-0.5)
        value = jnp.einsum("bhwd,dc->bhwc", value, projection)
        value /= jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-6)
        return value.reshape(value.shape[0], self.vjepa_query_grid_size**2, self.vjepa_compact_target_dim)

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        flow_loss, aux_loss = self.compute_loss_components(rng, observation, actions, train=train)
        if aux_loss is None:
            return flow_loss

        action_weight = self.actr_action_loss_weight if self.use_actr else 1.0
        return action_weight * flow_loss + self.vjepa_aux_weight * aux_loss[..., None]

    def _split_actr_cache(self, kv_cache: _gemma.KVCache) -> tuple[_gemma.KVCache, _gemma.KVCache]:
        keys, values = kv_cache
        split = self.actr_injection_layer
        return (keys[:split], values[:split]), (keys[split:], values[split:])

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

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions
        )
        transition_prior = (
            prefix_out[:, -self.vjepa_num_queries :] if self.use_actr else None
        )

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

            if self.use_actr:
                early_cache, late_cache = self._split_actr_cache(kv_cache)
                (prefix_out, suffix_draft), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=early_cache,
                    adarms_cond=[None, adarms_cond],
                    method="forward_early",
                )
                assert prefix_out is None and transition_prior is not None
                _, action_grounded = self.actr(
                    transition_prior,
                    suffix_draft[:, -self.action_horizon :],
                    jnp.broadcast_to(time, (batch_size,)),
                )
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, action_grounded],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=late_cache,
                    adarms_cond=[None, adarms_cond],
                    method="forward_late",
                )
            else:
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
                )
            assert prefix_out is None
            action_hidden = suffix_out[:, -self.action_horizon :]
            v_t = self.action_out_proj(action_hidden)

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
