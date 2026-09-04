"""Contrastive compatibility energy between a JEPA transition and an action chunk."""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def normalize(value: jax.Array) -> jax.Array:
    value = value.astype(jnp.float32)
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


class TransitionActionEnergy(nn.Module):
    """A small CLIP-style transition/action compatibility model.

    A learned query pools the 8x8 JEPA transition grid.  The action chunk is
    encoded independently.  Their normalized dot product is an energy score;
    no transition feature is injected into the policy hidden state.
    """

    horizon: int
    action_dim: int
    transition_dim: int
    width: int = 128
    num_heads: int = 4

    @nn.compact
    def __call__(self, transition: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        if transition.ndim != 3 or transition.shape[1] != 64:
            raise ValueError(f"Expected transition [B,64,D], got {transition.shape}")
        if action.shape[-2:] != (self.horizon, self.action_dim):
            raise ValueError(
                f"Expected action [B,{self.horizon},{self.action_dim}], got {action.shape}"
            )
        if self.width % self.num_heads:
            raise ValueError("width must be divisible by num_heads")

        batch = transition.shape[0]
        transition = nn.Dense(self.width, use_bias=False, name="transition_projection")(
            normalize(transition)
        )
        position = self.param(
            "transition_position", nn.initializers.normal(0.02), (64, self.width)
        )
        transition = transition + position[None]

        query = self.param("transition_query", nn.initializers.normal(0.02), (1, self.width))
        query = jnp.broadcast_to(query[None], (batch, 1, self.width))
        query = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.width,
            out_features=self.width,
            dropout_rate=0.0,
            name="transition_pool",
        )(query, transition, deterministic=True)[:, 0]
        transition_embedding = nn.Dense(self.width, name="transition_output")(
            nn.gelu(nn.LayerNorm(name="transition_norm")(query))
        )

        action_flat = action[..., : self.action_dim].reshape(batch, self.horizon * self.action_dim)
        action_hidden = nn.Dense(4 * self.width, name="action_hidden")(action_flat)
        action_embedding = nn.Dense(self.width, name="action_output")(nn.gelu(action_hidden))

        logit_scale = self.param("logit_scale", lambda key: jnp.asarray(jnp.log(10.0)))
        return normalize(transition_embedding), normalize(action_embedding), jnp.exp(
            jnp.clip(logit_scale, jnp.log(1.0), jnp.log(100.0))
        )

