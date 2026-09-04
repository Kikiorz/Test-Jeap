"""Action-conditioned prediction in a frozen JEPA transition space."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def normalize(value: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


class ActionConditionedConsequence(nn.Module):
    """Embed a realized transition and predict its embedding from state/action.

    Both current state and realized outcome are represented in the same frozen
    V-JEPA feature space.  A shared transition encoder prevents two arbitrary
    coordinate systems, while the predictor receives only current-state tokens
    and an action chunk.
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
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.width,
            out_features=self.width,
            dropout_rate=0.0,
            name="transition_pool",
        )
        position = self.param(
            "transition_position", nn.initializers.normal(0.02), (64, self.width)
        )
        query = self.param("transition_query", nn.initializers.normal(0.02), (1, self.width))

        def pool_transition(value: jax.Array) -> jax.Array:
            tokens = projection(normalize(value)) + position[None]
            queries = jnp.broadcast_to(query[None], (batch, 1, self.width))
            return attention(queries, tokens, deterministic=True)[:, 0]

        current_embedding = pool_transition(current)
        target_embedding = pool_transition(target)

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
        target_embedding = nn.Dense(self.width, name="target_output")(
            nn.gelu(nn.LayerNorm(name="target_norm")(target_embedding))
        )

        logit_scale = self.param("logit_scale", lambda key: jnp.asarray(jnp.log(10.0)))
        return (
            normalize(consequence_embedding),
            normalize(target_embedding),
            jnp.exp(jnp.clip(logit_scale, jnp.log(1.0), jnp.log(100.0))),
        )
