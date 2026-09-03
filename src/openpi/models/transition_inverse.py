"""Inverse readout from a JEPA transition grid to an action chunk."""

from __future__ import annotations

from typing import Literal

from flax import linen as nn
import jax
import jax.numpy as jnp


InverseMode = Literal["state", "transition"]


class TransitionInverseDecoder(nn.Module):
    horizon: int
    action_dim: int
    width: int
    num_heads: int
    mode: InverseMode

    @nn.compact
    def __call__(self, transition: jax.Array, state: jax.Array) -> jax.Array:
        if self.width % self.num_heads:
            raise ValueError("width must be divisible by num_heads")
        batch = transition.shape[0]
        state_token = nn.Dense(self.width, name="state_projection")(state)
        queries = self.param(
            "action_queries", nn.initializers.normal(0.02), (self.horizon, self.width)
        )
        queries = jnp.broadcast_to(queries[None], (batch, self.horizon, self.width))
        queries = queries + state_token[:, None, :]

        if self.mode == "state":
            context = jnp.zeros_like(queries)
        elif self.mode == "transition":
            tokens = nn.Dense(self.width, use_bias=False, name="transition_projection")(transition)
            position = self.param(
                "transition_position", nn.initializers.normal(0.02), (64, self.width)
            )
            tokens = tokens + position[None]
            query = nn.Dense(self.width, use_bias=False, name="query")(nn.RMSNorm()(queries))
            key = nn.Dense(self.width, use_bias=False, name="key")(nn.RMSNorm()(tokens))
            value = nn.Dense(self.width, use_bias=False, name="value")(tokens)
            head_width = self.width // self.num_heads
            query = query.reshape(batch, self.horizon, self.num_heads, head_width)
            key = key.reshape(batch, 64, self.num_heads, head_width)
            value = value.reshape(batch, 64, self.num_heads, head_width)
            logits = jnp.einsum("bqhd,bkhd->bhqk", query, key) * head_width**-0.5
            weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(value.dtype)
            context = jnp.einsum("bhqk,bkhd->bqhd", weights, value).reshape(
                batch, self.horizon, self.width
            )
            context = nn.Dense(self.width, use_bias=False, name="context_output")(context)
        else:
            raise ValueError(f"Unknown inverse mode {self.mode}")

        hidden = nn.RMSNorm(name="output_norm")(queries + context)
        gate = nn.Dense(4 * self.width, name="ffn_gate")(hidden)
        content = nn.Dense(4 * self.width, name="ffn_content")(hidden)
        hidden = nn.Dense(self.width, name="ffn_output")(nn.silu(gate) * content)
        return nn.Dense(self.action_dim, name="action_output")(hidden)
