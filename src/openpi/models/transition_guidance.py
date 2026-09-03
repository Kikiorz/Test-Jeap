"""A compact JEPA-transition guidance field for frozen action flow."""

from __future__ import annotations

from flax import linen as nn
import jax
import jax.numpy as jnp

from openpi.models import coflow


def _time_features(time: jax.Array, width: int) -> jax.Array:
    if width % 2:
        raise ValueError("width must be even")
    frequency = jnp.exp(jnp.linspace(jnp.log(1.0), jnp.log(1000.0), width // 2))
    angle = time.astype(jnp.float32)[:, None] * frequency[None]
    return jnp.concatenate([jnp.sin(angle), jnp.cos(angle)], axis=-1)


class TransitionGuidance(nn.Module):
    """Predict an additive velocity field from JEPA transition tokens."""

    action_dim: int
    state_dim: int
    width: int = 256
    depth: int = 2
    num_heads: int = 4

    @nn.compact
    def __call__(
        self,
        action_tau: jax.Array,
        state: jax.Array,
        transition: jax.Array,
        time: jax.Array,
        *,
        use_transition: bool = True,
    ) -> jax.Array:
        if self.width % self.num_heads:
            raise ValueError("width must be divisible by num_heads")
        batch, horizon, _ = action_tau.shape
        action = nn.Dense(self.width, name="action_input")(action_tau)
        action += nn.Dense(self.width, name="state_input")(state)[:, None, :]
        action += nn.Dense(self.width, name="time_input")(_time_features(time, self.width))[:, None, :]

        transition = coflow.normalize_transition(transition)
        transition = nn.Dense(self.width, use_bias=False, name="transition_input")(transition)
        position = self.param(
            "transition_position", nn.initializers.normal(0.02), (transition.shape[1], self.width)
        )
        transition = transition + position[None]
        if not use_transition:
            transition = jax.lax.stop_gradient(jnp.zeros_like(transition))

        head_width = self.width // self.num_heads
        for layer in range(self.depth):
            action_norm = nn.RMSNorm(name=f"action_norm_{layer}")(action)
            transition_norm = nn.RMSNorm(name=f"transition_norm_{layer}")(transition)
            query = nn.Dense(self.width, use_bias=False, name=f"query_{layer}")(action_norm)
            key = nn.Dense(self.width, use_bias=False, name=f"key_{layer}")(transition_norm)
            value = nn.Dense(self.width, use_bias=False, name=f"value_{layer}")(transition_norm)
            query = query.reshape(batch, horizon, self.num_heads, head_width)
            key = key.reshape(batch, transition.shape[1], self.num_heads, head_width)
            value = value.reshape(batch, transition.shape[1], self.num_heads, head_width)
            logits = jnp.einsum("bqhd,bkhd->bhqk", query, key) * head_width**-0.5
            weights = jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(value.dtype)
            context = jnp.einsum("bhqk,bkhd->bqhd", weights, value).reshape(
                batch, horizon, self.width
            )
            action += nn.Dense(self.width, use_bias=False, name=f"context_output_{layer}")(
                context
            )
            hidden = nn.RMSNorm(name=f"ffn_norm_{layer}")(action)
            gate = nn.Dense(4 * self.width, use_bias=False, name=f"ffn_gate_{layer}")(hidden)
            content = nn.Dense(4 * self.width, use_bias=False, name=f"ffn_content_{layer}")(
                hidden
            )
            action += nn.Dense(self.width, use_bias=False, name=f"ffn_output_{layer}")(
                nn.silu(gate) * content
            )

        action = nn.RMSNorm(name="output_norm")(action)
        return nn.Dense(
            self.action_dim,
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init(),
            name="guidance_output",
        )(action)
