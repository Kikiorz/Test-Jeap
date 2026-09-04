"""Current-conditioned joint rectified flow for actions and JEPA change tokens."""

from __future__ import annotations

from typing import Literal

from flax import linen as nn
import jax
import jax.numpy as jnp

FlowMode = Literal["joint", "independent"]


def rectified_interpolant(target: jax.Array, noise: jax.Array, time: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Interpolate from noise at tau=1 to data at tau=0."""
    if target.shape != noise.shape:
        raise ValueError(f"Target/noise shapes differ: {target.shape} != {noise.shape}")
    expanded_time = time.astype(jnp.float32)
    while expanded_time.ndim < target.ndim:
        expanded_time = expanded_time[..., None]
    return expanded_time * noise + (1.0 - expanded_time) * target, noise - target


def _time_features(time: jax.Array, width: int) -> jax.Array:
    if width % 2:
        raise ValueError("Time embedding width must be even")
    frequencies = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(1000.0), width // 2))
    angles = time.astype(jnp.float32)[:, None] * frequencies[None]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class _SwiGLU(nn.Module):
    width: int
    expansion: int = 4

    @nn.compact
    def __call__(self, value: jax.Array) -> jax.Array:
        hidden = self.width * self.expansion
        gate = nn.Dense(hidden, use_bias=False, name="gate")(value)
        content = nn.Dense(hidden, use_bias=False, name="content")(value)
        return nn.Dense(self.width, use_bias=False, name="output")(nn.silu(gate) * content)


class ActionChangeBlock(nn.Module):
    """MMDiT-style block with matched joint and independent attention modes."""

    width: int = 256
    num_heads: int = 4
    mode: FlowMode = "joint"

    def setup(self) -> None:
        if self.width % self.num_heads:
            raise ValueError("width must be divisible by num_heads")
        if self.mode not in ("joint", "independent"):
            raise ValueError(f"Unknown flow mode: {self.mode}")

    def _heads(self, value: jax.Array) -> jax.Array:
        return value.reshape(*value.shape[:-1], self.num_heads, self.width // self.num_heads)

    @staticmethod
    def _attention(query: jax.Array, key: jax.Array, value: jax.Array) -> jax.Array:
        scale = query.shape[-1] ** -0.5
        logits = jnp.einsum("bqhd,bkhd->bhqk", query, key, preferred_element_type=jnp.float32) * scale
        weights = jax.nn.softmax(logits, axis=-1).astype(value.dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", weights, value)
        return attended.reshape(*attended.shape[:-2], -1)

    @nn.compact
    def __call__(
        self,
        action: jax.Array,
        change: jax.Array,
        condition: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        action_norm = nn.RMSNorm(name="action_attention_norm")(action)
        change_norm = nn.RMSNorm(name="change_attention_norm")(change)
        condition_norm = nn.RMSNorm(name="condition_attention_norm")(condition)

        qa, ka, va = jnp.split(nn.Dense(3 * self.width, use_bias=False, name="action_qkv")(action_norm), 3, axis=-1)
        qb, kb, vb = jnp.split(nn.Dense(3 * self.width, use_bias=False, name="change_qkv")(change_norm), 3, axis=-1)
        ka_condition, va_condition = jnp.split(
            nn.Dense(2 * self.width, use_bias=False, name="action_condition_kv")(condition_norm), 2, axis=-1
        )
        kb_condition, vb_condition = jnp.split(
            nn.Dense(2 * self.width, use_bias=False, name="change_condition_kv")(condition_norm), 2, axis=-1
        )
        qa, ka, va, qb, kb, vb, ka_condition, va_condition, kb_condition, vb_condition = map(
            self._heads,
            (qa, ka, va, qb, kb, vb, ka_condition, va_condition, kb_condition, vb_condition),
        )

        action_keys = [ka, ka_condition]
        action_values = [va, va_condition]
        change_keys = [kb, kb_condition]
        change_values = [vb, vb_condition]
        if self.mode == "joint":
            action_keys.insert(1, kb)
            action_values.insert(1, vb)
            change_keys.insert(1, ka)
            change_values.insert(1, va)

        action_message = self._attention(
            qa, jnp.concatenate(action_keys, axis=1), jnp.concatenate(action_values, axis=1)
        )
        change_message = self._attention(
            qb, jnp.concatenate(change_keys, axis=1), jnp.concatenate(change_values, axis=1)
        )
        action = action + nn.Dense(self.width, use_bias=False, name="action_attention_output")(action_message)
        change = change + nn.Dense(self.width, use_bias=False, name="change_attention_output")(change_message)

        action = action + _SwiGLU(self.width, name="action_ffn")(nn.RMSNorm(name="action_ffn_norm")(action))
        change = change + _SwiGLU(self.width, name="change_ffn")(nn.RMSNorm(name="change_ffn_norm")(change))
        return action, change


class ActionChangeCoFlow(nn.Module):
    """Jointly predict action and latent-change rectified-flow velocities."""

    action_dim: int = 7
    change_dim: int = 16
    width: int = 256
    depth: int = 2
    num_heads: int = 4
    mode: FlowMode = "joint"

    @nn.compact
    def __call__(
        self,
        action_tau: jax.Array,
        change_tau: jax.Array,
        current_hidden: jax.Array,
        state: jax.Array,
        time: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        if action_tau.ndim != 3 or action_tau.shape[-1] != self.action_dim:
            raise ValueError(f"Unexpected action shape: {action_tau.shape}")
        if change_tau.ndim != 3 or change_tau.shape[-1] != self.change_dim:
            raise ValueError(f"Unexpected change shape: {change_tau.shape}")
        if current_hidden.ndim != 3 or current_hidden.shape[0] != action_tau.shape[0]:
            raise ValueError(f"Unexpected current-hidden shape: {current_hidden.shape}")
        if state.ndim != 2 or state.shape[0] != action_tau.shape[0]:
            raise ValueError(f"Unexpected state shape: {state.shape}")
        if time.shape != (action_tau.shape[0],):
            raise ValueError(f"Unexpected time shape: {time.shape}")

        action = nn.Dense(self.width, name="action_input")(action_tau.astype(jnp.float32))
        change = nn.Dense(self.width, name="change_input")(change_tau.astype(jnp.float32))
        condition = nn.Dense(self.width, use_bias=False, name="current_condition")(
            nn.RMSNorm(name="current_condition_norm")(current_hidden.astype(jnp.float32))
        )
        state_token = nn.Dense(self.width, name="state_condition")(state.astype(jnp.float32))[:, None, :]
        condition = jnp.concatenate([condition, state_token], axis=1)

        action_position = self.param("action_position", nn.initializers.normal(0.02), (action.shape[1], self.width))
        change_position = self.param("change_position", nn.initializers.normal(0.02), (change.shape[1], self.width))
        time_embedding = nn.Dense(self.width, name="time_projection")(_time_features(time, self.width))
        action = action + action_position[None] + time_embedding[:, None]
        change = change + change_position[None] + time_embedding[:, None]
        condition = condition + time_embedding[:, None]

        for layer in range(self.depth):
            action, change = ActionChangeBlock(
                width=self.width,
                num_heads=self.num_heads,
                mode=self.mode,
                name=f"block_{layer}",
            )(action, change, condition)

        action = nn.RMSNorm(name="action_output_norm")(action)
        change = nn.RMSNorm(name="change_output_norm")(change)
        return (
            nn.Dense(self.action_dim, name="action_output")(action),
            nn.Dense(self.change_dim, name="change_output")(change),
        )
