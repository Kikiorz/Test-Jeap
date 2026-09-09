"""RAPR (final_1) and the predictive gradient adapter (final_2).

No inverse-dynamics objective or action labels are used by the inner update.
Teacher targets must be adjacent observed transitions, not a repeated chunk
endpoint. Delta-Z is a temporal interface, not a separate novelty claim.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp


def _matrix(key, rows, cols):
    return jax.random.normal(key, (rows, cols), dtype=jnp.float32) / math.sqrt(rows)


def init_delta_head(key, *, input_dim, horizon, delta_dim=128, width=256, adapter_rank=8):
    if min(input_dim, horizon, delta_dim, width, adapter_rank) < 1:
        raise ValueError("All Delta-Z dimensions must be positive")
    keys = jax.random.split(key, 6)
    # Learned temporal queries read all frozen JEPA future tokens. No future
    # observation can enter this current-only path.
    head = {
        "key": _matrix(keys[0], input_dim, width),
        "value": _matrix(keys[1], input_dim, width),
        "time_queries": jax.random.normal(keys[2], (horizon, width)) * 0.02,
        "out": _matrix(keys[3], width, delta_dim),
        "bias": jnp.zeros((delta_dim,), dtype=jnp.float32),
    }
    # Only these tiny parameters are plastic at deployment (Con2).
    adapter = {
        "down": _matrix(keys[4], delta_dim, adapter_rank),
        "up": jnp.zeros((adapter_rank, delta_dim), dtype=jnp.float32),
    }
    return head, adapter


def delta_head(head, adapter, future_tokens):
    tokens = future_tokens.astype(jnp.float32)
    tokens = (tokens - tokens.mean(-1, keepdims=True)) * jax.lax.rsqrt(
        tokens.var(-1, keepdims=True) + 1e-6
    )
    keys, values = tokens @ head["key"], tokens @ head["value"]
    weights = jax.nn.softmax(
        jnp.einsum("hd,bqd->bhq", head["time_queries"], keys) / math.sqrt(keys.shape[-1]), axis=-1
    )
    delta = jax.nn.gelu(weights @ values) @ head["out"] + head["bias"]
    return delta + jax.nn.gelu(delta @ adapter["down"]) @ adapter["up"]


def init_routing(key, *, action_dim, delta_dim, horizon, width=128, gate=0.0):
    if min(action_dim, delta_dim, horizon, width) < 1:
        raise ValueError("All routing dimensions must be positive")
    keys = jax.random.split(key, 4)
    return {
        "query": _matrix(keys[0], action_dim, width),
        "key": _matrix(keys[1], delta_dim, width),
        "value": _matrix(keys[2], delta_dim, width),
        "out": _matrix(keys[3], width, action_dim),
        "relative_bias": jnp.zeros((2 * horizon - 1,), dtype=jnp.float32),
        "gate": jnp.asarray(gate, dtype=jnp.float32),
    }


def route_action(parameters, action_hidden, delta, *, gate_override=None):
    """Query a predicted trajectory using *intermediate* action tokens."""
    horizon = delta.shape[-2]
    if action_hidden.shape[-2] != horizon:
        raise ValueError("Action and predicted transition horizons must match")
    q = action_hidden.astype(jnp.float32) @ parameters["query"]
    k = delta.astype(jnp.float32) @ parameters["key"]
    v = delta.astype(jnp.float32) @ parameters["value"]
    offsets = jnp.arange(horizon)[None, :] - jnp.arange(horizon)[:, None] + horizon - 1
    logits = q @ jnp.swapaxes(k, -1, -2) / math.sqrt(q.shape[-1])
    routes = jax.nn.softmax(logits + parameters["relative_bias"][offsets], axis=-1)
    # The route is a residual on the pretrained action stream. A runtime gate
    # can force this whole newly added module off without changing any base
    # checkpoint parameters. Clipping also prevents an unconstrained learned
    # scalar from turning a safe residual into an action amplifier.
    gate = parameters["gate"] if gate_override is None else gate_override
    gate = jnp.clip(gate, 0.0, 1.0)
    residual = gate * (routes @ v @ parameters["out"])
    return action_hidden + residual.astype(action_hidden.dtype), routes


def module_output_gate(
    baseline_output,
    candidate_output,
    *,
    baseline_score,
    candidate_score,
    max_rms=0.05,
    max_abs=0.2,
):
    """Select a newly added module output transactionally.

    ``score`` is a *predeclared* self-supervised proxy evaluated on the same
    state/noise (for this paper: predictive residual or held-out action loss).
    Without a score, the candidate is rejected. This is a negative-update
    guard, not a claim that proxy improvement guarantees task success.
    """
    base = jnp.asarray(baseline_output)
    candidate = jnp.asarray(candidate_output)
    if base.shape != candidate.shape:
        raise ValueError("Baseline and candidate outputs must have identical shapes")
    if not 0 <= max_rms or not 0 <= max_abs:
        raise ValueError("Gate drift limits must be nonnegative")
    finite = (
        bool(jnp.isfinite(base).all())
        and bool(jnp.isfinite(candidate).all())
        and bool(jnp.isfinite(jnp.asarray(baseline_score)))
        and bool(jnp.isfinite(jnp.asarray(candidate_score)))
    )
    difference = (candidate.astype(jnp.float32) - base.astype(jnp.float32))
    rms = float(jnp.sqrt(jnp.mean(jnp.square(difference))))
    maximum = float(jnp.max(jnp.abs(difference)))
    accepted = (
        finite
        and float(candidate_score) <= float(baseline_score)
        and rms <= max_rms
        and maximum <= max_abs
    )
    return (candidate if accepted else base), {
        "enabled": accepted,
        "reason": "proxy_improved" if accepted else "negative_update_or_drift",
        "baseline_score": float(baseline_score),
        "candidate_score": float(candidate_score),
        "output_rms": rms,
        "output_max_abs": maximum,
    }


def supervision_weights(routes, delta, action_gradient, *, beta=0.5, temperature=1.0):
    """Detached usage x gradient-activation sensitivity, with a uniform anchor.

    Follow the softmax formula in final_1 literally: a zero raw sensitivity
    does NOT imply zero weight after softmax (the prose example overclaims).
    """
    if not 0 <= beta < 1 or temperature <= 0:
        raise ValueError("Require 0 <= beta < 1 and temperature > 0")
    usage = routes.mean(axis=-2)
    sensitivity = jnp.abs(jnp.sum(action_gradient * delta, axis=-1))
    importance = usage * jax.nn.softmax(sensitivity / temperature, axis=-1)
    relevance = importance / jnp.maximum(importance.sum(-1, keepdims=True), 1e-12)
    weights = (1 - beta) / delta.shape[-2] + beta * relevance
    return jax.lax.stop_gradient(weights)


def prediction_loss(delta, target, weights=None, mask=None):
    """Squared distance, with absent/unexecuted transitions excluded."""
    if delta.shape != target.shape:
        raise ValueError("Prediction and adjacent-transition target shapes must match")
    error = jnp.mean(jnp.square(delta - jax.lax.stop_gradient(target)), axis=-1)
    if weights is None:
        weights = jnp.ones_like(error) / error.shape[-1]
    if mask is not None:
        weights = weights * mask
    weights = jax.lax.stop_gradient(weights)
    return (weights * error).sum() / jnp.maximum(weights.sum(), 1e-12)


def init_gradient_adapter(key, plastic, *, rank=4):
    """Per plastic tensor: a*g + U(V^T*g), initialized to identity.

    U starts at zero, V is random: initializing BOTH factors to zero would
    prevent learning the low-rank transform.
    """
    if rank < 1:
        raise ValueError("Gradient adapter rank must be positive")
    leaves, structure = jax.tree_util.tree_flatten(plastic)
    keys = jax.random.split(key, len(leaves))
    transformed = [
        {"scale": jnp.asarray(1.0), "u": jnp.zeros((leaf.size, rank)),
         "v": _matrix(k, leaf.size, rank)}
        for k, leaf in zip(keys, leaves, strict=True)
    ]
    return jax.tree_util.tree_unflatten(structure, transformed)


def transform_gradient(transform, gradients):
    def apply(gradient, block):
        flat = gradient.reshape(-1)
        return (block["scale"] * flat + block["u"] @ (block["v"].T @ flat)).reshape(gradient.shape)
    return jax.tree.map(apply, gradients, transform)


def predictive_update(plastic, transform, gradient, *, learning_rate):
    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")
    direction = transform_gradient(transform, gradient)
    return jax.tree.map(lambda theta, update: theta - learning_rate * update, plastic, direction)


def meta_objective(transform, plastic, support_loss: Callable, query_loss: Callable, *, learning_rate):
    """Differentiate through the update to omega; no action-gradient target.

    Frozen theta, head, routing and backbone are not outer optimizer inputs.
    S must finish before Q starts; the dataset layer enforces this contract.
    """
    gradient = jax.grad(support_loss)(plastic)
    adapted = predictive_update(plastic, transform, gradient, learning_rate=learning_rate)
    return query_loss(adapted)
