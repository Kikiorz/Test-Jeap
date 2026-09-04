"""Compact action-identifiable change bottlenecks for Con1 falsification tests."""

from __future__ import annotations

from flax import linen as nn
import jax
import jax.numpy as jnp


def l2_normalize_tokens(tokens: jax.Array, eps: float = 1e-6) -> jax.Array:
    """Normalize each token without changing its token ordering."""
    value = tokens.astype(jnp.float32)
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), eps)


def latent_displacement(realized: jax.Array, nochange: jax.Array) -> jax.Array:
    """Return the no-change-referenced displacement in JEPA cosine geometry."""
    if realized.shape != nochange.shape:
        raise ValueError(f"JEPA target shapes differ: {realized.shape} != {nochange.shape}")
    return l2_normalize_tokens(realized) - l2_normalize_tokens(nochange)


class _SwiGLU(nn.Module):
    width: int
    expansion: int = 4

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        hidden_width = self.width * self.expansion
        gate = nn.Dense(hidden_width, use_bias=False, name="gate")(value)
        content = nn.Dense(hidden_width, use_bias=False, name="content")(value)
        return nn.Dense(self.width, use_bias=False, name="output")(nn.silu(gate) * content)


class ChangeEncoder(nn.Module):
    """Read JEPA displacement and optional current context into compact tokens."""

    num_tokens: int = 16
    token_dim: int = 16
    width: int = 256
    depth: int = 2
    num_heads: int = 4

    @nn.compact
    def __call__(
        self,
        displacement: jax.Array,
        current_tokens: jax.Array,
        state: jax.Array,
    ) -> jax.Array:
        if displacement.ndim != 3:
            raise ValueError(f"Expected [batch, patches, channels], got {displacement.shape}")
        if current_tokens.shape != displacement.shape:
            raise ValueError(f"Current tokens must match displacement: {current_tokens.shape} != {displacement.shape}")
        if state.ndim != 2 or state.shape[0] != displacement.shape[0]:
            raise ValueError(f"Expected state [batch, channels] aligned with displacement, got {state.shape}")
        if self.width % self.num_heads:
            raise ValueError("width must be divisible by num_heads")

        batch_size = displacement.shape[0]
        delta_hidden = nn.Dense(self.width, use_bias=False, name="delta_projection")(displacement.astype(jnp.float32))
        current_hidden = nn.Dense(self.width, use_bias=False, name="current_projection")(
            current_tokens.astype(jnp.float32)
        )
        state_hidden = nn.Dense(self.width, use_bias=False, name="state_projection")(state.astype(jnp.float32))
        dense_tokens = delta_hidden + current_hidden + state_hidden[:, None, :]
        queries = self.param(
            "change_queries",
            nn.initializers.normal(0.02),
            (self.num_tokens, self.width),
        )
        queries = jnp.broadcast_to(queries[None], (batch_size, self.num_tokens, self.width))

        for layer in range(self.depth):
            query_input = nn.RMSNorm(name=f"query_norm_{layer}")(queries)
            dense_input = nn.RMSNorm(name=f"dense_norm_{layer}")(dense_tokens)
            message = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.width,
                out_features=self.width,
                use_bias=False,
                dropout_rate=0.0,
                name=f"cross_attention_{layer}",
            )(query_input, dense_input, deterministic=True)
            queries = queries + message
            queries = queries + _SwiGLU(self.width, name=f"ffn_{layer}")(nn.RMSNorm(name=f"ffn_norm_{layer}")(queries))

        return nn.Dense(self.token_dim, name="change_output")(queries)


class InverseActionDecoder(nn.Module):
    """Decode physical actions only from the compact bottleneck."""

    horizon: int = 10
    action_dim: int = 7
    width: int = 256

    @nn.compact
    def __call__(self, change_tokens: jax.Array) -> jax.Array:
        if change_tokens.ndim != 3:
            raise ValueError(f"Expected [batch, tokens, channels], got {change_tokens.shape}")
        hidden = change_tokens.reshape(change_tokens.shape[0], -1)
        hidden = nn.LayerNorm(name="input_norm")(hidden)
        hidden = nn.silu(nn.Dense(self.width, name="hidden")(hidden))
        actions = nn.Dense(self.horizon * self.action_dim, name="action_output")(hidden)
        return actions.reshape(change_tokens.shape[0], self.horizon, self.action_dim)


class PhaseAModel(nn.Module):
    """Phase A change encoder and its training-only inverse decoder."""

    num_tokens: int = 16
    token_dim: int = 16
    width: int = 256
    depth: int = 2
    num_heads: int = 4
    horizon: int = 10
    action_dim: int = 7

    @nn.compact
    def __call__(
        self,
        displacement: jax.Array,
        current_tokens: jax.Array,
        state: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        change = ChangeEncoder(
            num_tokens=self.num_tokens,
            token_dim=self.token_dim,
            width=self.width,
            depth=self.depth,
            num_heads=self.num_heads,
            name="change_encoder",
        )(displacement, current_tokens, state)
        actions = InverseActionDecoder(
            horizon=self.horizon,
            action_dim=self.action_dim,
            width=self.width,
            name="inverse_decoder",
        )(change)
        return change, actions
