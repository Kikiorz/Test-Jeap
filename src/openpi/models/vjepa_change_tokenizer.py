"""Action-identifiable spatial change tokenizer for Con1.

The deployable tokenizer compresses a frozen V-JEPA2 no-change-referenced
displacement into a 4x4 grid of change tokens.  A training-only inverse
Transformer is included to make those tokens retain information about the
corresponding physical action chunk.
"""

from __future__ import annotations

import math

from flax import linen as nn
import jax
import jax.numpy as jnp


def l2_normalize_tokens(tokens: jax.Array, eps: float = 1e-6) -> jax.Array:
    value = tokens.astype(jnp.float32)
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), eps)


def latent_displacement(realized: jax.Array, nochange: jax.Array) -> jax.Array:
    """No-change-referenced displacement in the frozen JEPA cosine geometry."""
    if realized.shape != nochange.shape:
        raise ValueError(f"JEPA target shapes differ: {realized.shape} != {nochange.shape}")
    return l2_normalize_tokens(realized) - l2_normalize_tokens(nochange)


def square_grid_positions(token_count: int) -> jax.Array:
    """Return row-major token centers in the shared [-1, 1]^2 coordinate frame."""
    side = int(round(math.sqrt(token_count)))
    if side * side != token_count:
        raise ValueError(f"Token count must form a square grid, got {token_count}")
    axis = ((jnp.arange(side, dtype=jnp.float32) + 0.5) / side) * 2.0 - 1.0
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    return jnp.stack((xx, yy), axis=-1).reshape(token_count, 2)


def _rotate_pairs(value: jax.Array, angle: jax.Array) -> jax.Array:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    cosine = jnp.cos(angle).astype(value.dtype)
    sine = jnp.sin(angle).astype(value.dtype)
    rotated_even = even * cosine - odd * sine
    rotated_odd = even * sine + odd * cosine
    return jnp.stack((rotated_even, rotated_odd), axis=-1).reshape(value.shape)


def apply_2d_rope(value: jax.Array, positions: jax.Array) -> jax.Array:
    """Apply separable 2-D rotary position encoding to [B,N,H,D] tensors."""
    if value.ndim != 4 or positions.shape != (value.shape[1], 2):
        raise ValueError(f"Invalid RoPE inputs: value={value.shape}, positions={positions.shape}")
    head_dim = value.shape[-1]
    if head_dim % 4:
        raise ValueError(f"2-D RoPE requires head_dim divisible by four, got {head_dim}")
    axis_dim = head_dim // 2
    pair_count = axis_dim // 2
    frequencies = 2.0 ** jnp.linspace(0.0, 4.0, pair_count, dtype=jnp.float32)
    x_angle = jnp.pi * positions[:, 0, None] * frequencies[None]
    y_angle = jnp.pi * positions[:, 1, None] * frequencies[None]
    x_angle = x_angle[None, :, None, :]
    y_angle = y_angle[None, :, None, :]
    x_value, y_value = jnp.split(value, 2, axis=-1)
    return jnp.concatenate((_rotate_pairs(x_value, x_angle), _rotate_pairs(y_value, y_angle)), axis=-1)


class SwiGLU(nn.Module):
    width: int
    hidden_width: int

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        gate = nn.Dense(self.hidden_width, use_bias=False, name="gate")(value)
        content = nn.Dense(self.hidden_width, use_bias=False, name="content")(value)
        return nn.Dense(self.width, use_bias=False, name="output")(nn.silu(gate) * content)


class SpatialMultiHeadAttention(nn.Module):
    """Multi-head attention whose queries and keys use a shared 2-D RoPE frame."""

    width: int
    num_heads: int

    @nn.compact
    def __call__(
        self,
        query: jax.Array,
        key_value: jax.Array,
        query_positions: jax.Array,
        key_positions: jax.Array,
    ) -> jax.Array:
        if self.width % self.num_heads:
            raise ValueError("width must be divisible by num_heads")
        head_dim = self.width // self.num_heads
        if head_dim % 4:
            raise ValueError("2-D RoPE requires per-head width divisible by four")

        def split_heads(value: jax.Array) -> jax.Array:
            return value.reshape(value.shape[0], value.shape[1], self.num_heads, head_dim)

        q = split_heads(nn.Dense(self.width, use_bias=False, name="query")(query))
        k = split_heads(nn.Dense(self.width, use_bias=False, name="key")(key_value))
        v = split_heads(nn.Dense(self.width, use_bias=False, name="value")(key_value))
        q = apply_2d_rope(q, query_positions)
        k = apply_2d_rope(k, key_positions)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32)
        logits = logits * (head_dim**-0.5)
        weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
        attended = attended.reshape(attended.shape[0], attended.shape[1], self.width)
        return nn.Dense(self.width, use_bias=False, name="output")(attended)


class SpatialChangeResampler(nn.Module):
    """Compress a square JEPA displacement grid into spatially indexed tokens."""

    num_tokens: int = 16
    token_dim: int = 128
    width: int = 512
    depth: int = 3
    num_heads: int = 8
    ffn_width: int = 2048

    @nn.compact
    def __call__(self, displacement: jax.Array) -> jax.Array:
        if displacement.ndim != 3:
            raise ValueError(f"Expected [batch, patches, channels], got {displacement.shape}")
        dense_positions = square_grid_positions(displacement.shape[1])
        latent_positions = square_grid_positions(self.num_tokens)
        dense = nn.Dense(self.width, use_bias=False, name="input_projection")(
            displacement.astype(jnp.float32)
        )
        latent = self.param(
            "change_queries",
            nn.initializers.normal(0.02),
            (self.num_tokens, self.width),
        )
        latent = jnp.broadcast_to(latent[None], (displacement.shape[0], self.num_tokens, self.width))

        for layer in range(self.depth):
            cross_message = SpatialMultiHeadAttention(
                width=self.width,
                num_heads=self.num_heads,
                name=f"cross_attention_{layer}",
            )(
                nn.RMSNorm(name=f"cross_query_norm_{layer}")(latent),
                nn.RMSNorm(name=f"cross_key_norm_{layer}")(dense),
                latent_positions,
                dense_positions,
            )
            latent = latent + cross_message
            self_message = SpatialMultiHeadAttention(
                width=self.width,
                num_heads=self.num_heads,
                name=f"self_attention_{layer}",
            )(
                nn.RMSNorm(name=f"self_norm_{layer}")(latent),
                nn.RMSNorm(name=f"self_key_norm_{layer}")(latent),
                latent_positions,
                latent_positions,
            )
            latent = latent + self_message
            latent = latent + SwiGLU(
                width=self.width,
                hidden_width=self.ffn_width,
                name=f"ffn_{layer}",
            )(nn.RMSNorm(name=f"ffn_norm_{layer}")(latent))

        return nn.Dense(self.token_dim, name="change_output")(
            nn.RMSNorm(name="output_norm")(latent)
        )


class DirectInverseDecoder(nn.Module):
    """Training-only temporal Transformer that reads change tokens and nothing else."""

    horizon: int = 10
    action_dim: int = 7
    width: int = 512
    depth: int = 4
    num_heads: int = 8
    ffn_width: int = 2048

    @nn.compact
    def __call__(self, change_tokens: jax.Array) -> jax.Array:
        if change_tokens.ndim != 3:
            raise ValueError(f"Expected [batch,tokens,channels], got {change_tokens.shape}")
        memory = nn.Dense(self.width, use_bias=False, name="memory_projection")(
            change_tokens.astype(jnp.float32)
        )
        queries = self.param(
            "action_queries",
            nn.initializers.normal(0.02),
            (self.horizon, self.width),
        )
        temporal_positions = self.param(
            "temporal_positions",
            nn.initializers.normal(0.02),
            (self.horizon, self.width),
        )
        action = jnp.broadcast_to(
            (queries + temporal_positions)[None],
            (change_tokens.shape[0], self.horizon, self.width),
        )

        for layer in range(self.depth):
            normalized = nn.RMSNorm(name=f"self_norm_{layer}")(action)
            action = action + nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.width,
                out_features=self.width,
                use_bias=False,
                dropout_rate=0.0,
                name=f"self_attention_{layer}",
            )(normalized, normalized, deterministic=True)
            action = action + nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.width,
                out_features=self.width,
                use_bias=False,
                dropout_rate=0.0,
                name=f"cross_attention_{layer}",
            )(
                nn.RMSNorm(name=f"cross_query_norm_{layer}")(action),
                nn.RMSNorm(name=f"cross_memory_norm_{layer}")(memory),
                deterministic=True,
            )
            action = action + SwiGLU(
                width=self.width,
                hidden_width=self.ffn_width,
                name=f"ffn_{layer}",
            )(nn.RMSNorm(name=f"ffn_norm_{layer}")(action))

        return nn.Dense(self.action_dim, name="action_output")(
            nn.RMSNorm(name="output_norm")(action)
        )


class Stage1Teacher(nn.Module):
    """Final Stage-1 model: deployable tokenizer plus training-only decoder."""

    num_tokens: int = 16
    token_dim: int = 128
    width: int = 512
    resampler_depth: int = 3
    decoder_depth: int = 4
    num_heads: int = 8
    ffn_width: int = 2048
    horizon: int = 10
    action_dim: int = 7

    def setup(self) -> None:
        self.change_tokenizer = SpatialChangeResampler(
            num_tokens=self.num_tokens,
            token_dim=self.token_dim,
            width=self.width,
            depth=self.resampler_depth,
            num_heads=self.num_heads,
            ffn_width=self.ffn_width,
            name="change_tokenizer",
        )
        self.inverse_decoder = DirectInverseDecoder(
            horizon=self.horizon,
            action_dim=self.action_dim,
            width=self.width,
            depth=self.decoder_depth,
            num_heads=self.num_heads,
            ffn_width=self.ffn_width,
            name="inverse_decoder",
        )

    def encode(self, displacement: jax.Array) -> jax.Array:
        """Encode a JEPA displacement into the deployable change endpoint."""
        return self.change_tokenizer(displacement)

    def __call__(self, displacement: jax.Array) -> tuple[jax.Array, jax.Array]:
        change = self.encode(displacement)
        return change, self.inverse_decoder(change)
