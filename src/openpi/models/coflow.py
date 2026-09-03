"""Minimal action-transition coupled rectified flow.

This module isolates the scientific core of CoFlow-JEPA from the large Pi0
implementation.  It is used for the small falsification experiment before the
same four blocks are inserted into the final Action Expert layers.
"""

from __future__ import annotations

from typing import Literal

import flax.linen as nn
import jax
import jax.numpy as jnp


CoFlowMode = Literal["coflow", "fixed", "independent"]


def normalize_transition(tokens: jax.Array) -> jax.Array:
    """Tokenwise float32 normalization used by both transition endpoints."""
    value = tokens.astype(jnp.float32)
    return value / jnp.maximum(jnp.linalg.norm(value, axis=-1, keepdims=True), 1e-6)


def transition_interpolant(
    prior: jax.Array, observed: jax.Array, time: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return U_tau and its source-minus-target rectified-flow velocity."""
    if prior.shape != observed.shape:
        raise ValueError(f"Transition endpoint shapes differ: {prior.shape} != {observed.shape}")
    prior = normalize_transition(prior)
    observed = normalize_transition(observed)
    expanded_time = time.astype(jnp.float32)
    while expanded_time.ndim < prior.ndim:
        expanded_time = expanded_time[..., None]
    return expanded_time * prior + (1.0 - expanded_time) * observed, prior - observed


def _time_features(time: jax.Array, width: int) -> jax.Array:
    if width % 2:
        raise ValueError("Time embedding width must be even")
    frequencies = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(1000.0), width // 2))
    angles = time.astype(jnp.float32)[:, None] * frequencies[None]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


def _grid_features(token_count: int, width: int) -> jax.Array:
    grid_size = int(round(token_count**0.5))
    if grid_size * grid_size != token_count:
        raise ValueError(f"Transition token count {token_count} is not a square grid")
    axis = (jnp.arange(grid_size, dtype=jnp.float32) + 0.5) / grid_size
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    coordinates = jnp.stack([xx, yy], axis=-1).reshape(token_count, 2)
    frequency_count = max(width // 4, 1)
    frequencies = 2.0 ** jnp.arange(frequency_count, dtype=jnp.float32)
    angles = 2.0 * jnp.pi * coordinates[..., None] * frequencies
    encoded = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1).reshape(token_count, -1)
    if encoded.shape[-1] < width:
        encoded = jnp.pad(encoded, ((0, 0), (0, width - encoded.shape[-1])))
    return encoded[:, :width]


class _SwiGLU(nn.Module):
    width: int
    expansion: int = 4

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        hidden = self.expansion * self.width
        gate = nn.Dense(hidden, use_bias=False, name="gate")(value)
        content = nn.Dense(hidden, use_bias=False, name="content")(value)
        return nn.Dense(self.width, use_bias=False, name="output")(nn.silu(gate) * content)


class CoFlowBlock(nn.Module):
    """MMDiT-style two-stream block with asymmetric action reliability."""

    width: int
    num_heads: int
    mode: CoFlowMode = "coflow"

    def setup(self):
        if self.width % self.num_heads:
            raise ValueError("width must be divisible by num_heads")
        if self.mode not in ("coflow", "fixed", "independent"):
            raise ValueError(f"Unknown CoFlow mode: {self.mode}")

    def _heads(self, value: jax.Array) -> jax.Array:
        return value.reshape(*value.shape[:-1], self.num_heads, self.width // self.num_heads)

    @staticmethod
    def _attention(query: jax.Array, key: jax.Array, value: jax.Array, bias: jax.Array | None = None):
        head_width = query.shape[-1]
        logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, preferred_element_type=jnp.float32)
        logits *= head_width**-0.5
        if bias is not None:
            logits = logits + bias
        weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", weights, value)
        return attended.reshape(*attended.shape[:-2], -1)

    @nn.compact
    def __call__(
        self,
        action: jax.Array,
        transition: jax.Array,
        transition_prior: jax.Array,
        time: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        action_norm = nn.RMSNorm(name="action_attention_norm")(action)
        transition_norm = nn.RMSNorm(name="transition_attention_norm")(transition)
        prior_norm = nn.RMSNorm(name="prior_attention_norm")(transition_prior)

        q_action, k_action, v_action = jnp.split(
            nn.Dense(3 * self.width, use_bias=False, name="action_qkv")(action_norm), 3, axis=-1
        )
        q_transition, k_transition, v_transition = jnp.split(
            nn.Dense(3 * self.width, use_bias=False, name="transition_qkv")(transition_norm), 3, axis=-1
        )
        _, k_prior, v_prior = jnp.split(
            nn.Dense(3 * self.width, use_bias=False, name="prior_qkv")(prior_norm), 3, axis=-1
        )
        q_action, k_action, v_action = map(self._heads, (q_action, k_action, v_action))
        q_transition, k_transition, v_transition = map(
            self._heads, (q_transition, k_transition, v_transition)
        )
        k_prior, v_prior = map(self._heads, (k_prior, v_prior))

        if self.mode == "independent":
            action_keys, action_values = k_action, v_action
        elif self.mode == "fixed":
            action_keys = jnp.concatenate([k_action, k_prior], axis=1)
            action_values = jnp.concatenate([v_action, v_prior], axis=1)
        else:
            action_keys = jnp.concatenate([k_action, k_transition], axis=1)
            action_values = jnp.concatenate([v_action, v_transition], axis=1)
        action_message = self._attention(q_action, action_keys, action_values)
        action = action + nn.Dense(self.width, use_bias=False, name="action_attention_output")(action_message)

        if self.mode == "coflow":
            stopped_k_action = jax.lax.stop_gradient(k_action)
            stopped_v_action = jax.lax.stop_gradient(v_action)
            transition_keys = jnp.concatenate([k_transition, stopped_k_action], axis=1)
            transition_values = jnp.concatenate([v_transition, stopped_v_action], axis=1)
            transition_count = transition.shape[1]
            action_count = action.shape[1]
            rho = jnp.clip(1.0 - time.astype(jnp.float32), 0.0, 1.0)
            action_log_weight = jnp.where(rho > 0, jnp.log(jnp.maximum(rho, 1e-12)), -1e30)
            bias = jnp.concatenate(
                [
                    jnp.zeros((time.shape[0], transition_count), dtype=jnp.float32),
                    jnp.broadcast_to(action_log_weight[:, None], (time.shape[0], action_count)),
                ],
                axis=1,
            )[:, None, None, :]
            transition_message = self._attention(
                q_transition, transition_keys, transition_values, bias=bias
            )
        else:
            transition_message = self._attention(q_transition, k_transition, v_transition)
        transition = transition + nn.Dense(
            self.width, use_bias=False, name="transition_attention_output"
        )(transition_message)

        action = action + _SwiGLU(self.width, name="action_ffn")(
            nn.RMSNorm(name="action_ffn_norm")(action)
        )
        transition = transition + _SwiGLU(self.width, name="transition_ffn")(
            nn.RMSNorm(name="transition_ffn_norm")(transition)
        )
        return action, transition


class CoFlowCore(nn.Module):
    """Compact four-block core used to falsify coupled flow on frozen data."""

    action_dim: int
    transition_dim: int
    width: int = 256
    depth: int = 4
    num_heads: int = 4
    mode: CoFlowMode = "coflow"

    @nn.compact
    def __call__(
        self,
        action_tau: jax.Array,
        transition_tau: jax.Array,
        transition_prior: jax.Array,
        time: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        action = nn.Dense(self.width, name="action_input")(action_tau)
        transition = nn.Dense(self.width, name="transition_input")(transition_tau)
        prior = nn.Dense(self.width, name="prior_input")(transition_prior)

        time_embedding = nn.Dense(self.width, name="time_input")(_time_features(time, self.width))
        action = action + time_embedding[:, None, :]
        transition = transition + time_embedding[:, None, :]
        prior = prior + time_embedding[:, None, :]
        transition_position = _grid_features(transition.shape[1], self.width)
        transition = transition + transition_position[None]
        prior = prior + transition_position[None]

        for layer in range(self.depth):
            action, transition = CoFlowBlock(
                self.width, self.num_heads, mode=self.mode, name=f"block_{layer}"
            )(action, transition, prior, time)
        action = nn.RMSNorm(name="action_output_norm")(action)
        transition = nn.RMSNorm(name="transition_output_norm")(transition)
        action_velocity = nn.Dense(self.action_dim, name="action_output")(action)
        transition_velocity = nn.Dense(self.transition_dim, name="transition_output")(transition)
        return action_velocity, transition_velocity

