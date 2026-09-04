"""Action-conditioned prediction in a frozen JEPA transition space."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def normalize(value: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


class ActionConditionedConsequence(nn.Module):
    """Embed a realized semantic change and predict it from state/action.

    A learned query first reads the current state.  That current-anchored query
    then reads the realized transition, and only its query update is retained as
    the target.  The predictor receives the current anchor and an action chunk,
    but never the future target.
    """

    horizon: int
    action_dim: int
    transition_dim: int
    width: int = 128
    num_heads: int = 4

    @nn.compact
    def __call__(
        self,
        current: jax.Array,
        action: jax.Array,
        target: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if current.ndim != 3 or current.shape[1] != 64:
            raise ValueError(f"Expected current [B,64,D], got {current.shape}")
        if target.shape != current.shape:
            raise ValueError(f"Expected target shape {current.shape}, got {target.shape}")
        if action.shape[-2:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"Expected action [B,{self.horizon},{self.action_dim}], got {action.shape}"
            )
        if self.width % self.num_heads:
            raise ValueError("width must be divisible by num_heads")

        batch = current.shape[0]
        projection = nn.Dense(self.width, use_bias=False, name="transition_projection")
        spatial_pool = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.width,
            out_features=self.width,
            dropout_rate=0.0,
            name="current_pool",
        )
        temporal_readout = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.width,
            out_features=self.width,
            dropout_rate=0.0,
            name="temporal_readout",
        )
        position = self.param(
            "transition_position", nn.initializers.normal(0.02), (64, self.width)
        )
        query = self.param("transition_query", nn.initializers.normal(0.02), (1, self.width))

        current_tokens = projection(normalize(current)) + position[None]
        target_tokens = projection(normalize(target)) + position[None]
        queries = jnp.broadcast_to(query[None], (batch, 1, self.width))
        current_embedding = spatial_pool(queries, current_tokens, deterministic=True)[:, 0]
        target_embedding = temporal_readout(
            current_embedding[:, None], target_tokens, deterministic=True
        )[:, 0]
        target_update = target_embedding - current_embedding

        action_flat = action.reshape(batch, self.horizon * self.action_dim)
        action_hidden = nn.Dense(4 * self.width, name="action_hidden")(action_flat)
        action_embedding = nn.Dense(self.width, name="action_output")(nn.gelu(action_hidden))

        joint = jnp.concatenate(
            [current_embedding, action_embedding, current_embedding * action_embedding], axis=-1
        )
        consequence_hidden = nn.Dense(4 * self.width, name="consequence_hidden")(joint)
        consequence_embedding = nn.Dense(self.width, name="consequence_output")(
            nn.gelu(consequence_hidden)
        )
        target_update = nn.Dense(self.width, name="target_output")(
            nn.gelu(nn.LayerNorm(name="target_norm")(target_update))
        )

        logit_scale = self.param("logit_scale", lambda key: jnp.asarray(jnp.log(10.0)))
        return (
            normalize(consequence_embedding),
            normalize(target_update),
            jnp.exp(jnp.clip(logit_scale, jnp.log(1.0), jnp.log(100.0))),
        )
