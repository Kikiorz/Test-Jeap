"""Observable point-flow interface for predictive robot control.

The modules in this file deliberately stay independent of the Pi0 transformer
implementation.  ``semantic_transport_flow`` derives self-supervised motion
from the policy's own spatial visual features, without an external tracker.
``PointFlowPlanner`` turns JEPA-WAM future-query states into short-horizon 2-D
tracks.  ``PointFlowConditioner`` turns either predicted or observed tracks
into a residual that can be inserted inside an action expert.

Coordinates are normalized to ``[0, 1]``.  A zero-initialized conditioner
output makes the untrained interface an exact identity without adding a
learned gate.
"""

from __future__ import annotations

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.shared import array_typing as at


def _grid_coordinates(grid_size: int, dtype: jnp.dtype = jnp.float32) -> jax.Array:
    axis = (jnp.arange(grid_size, dtype=jnp.float32) + 0.5) / grid_size
    yy, xx = jnp.meshgrid(axis, axis, indexing="ij")
    return jnp.stack([xx, yy], axis=-1).reshape(-1, 2).astype(dtype)


def semantic_transport_flow(
    current_tokens: at.Float[at.Array, "b q d"],
    future_tokens: at.Float[at.Array, "b q d"],
    *,
    grid_size: int,
    temperature: float = 0.07,
    spatial_sigma: float = 0.35,
) -> tuple[at.Float[at.Array, "b q xy"], at.Float[at.Array, "b q"]]:
    """Derive a semantic 2-D transport field from two feature grids.

    The same frozen visual encoder produces both grids.  A dual-softmax
    correspondence converts semantic similarity into displacement while a
    weak spatial prior resolves repeated textures.  This deterministic
    observer is used both for offline self-supervision and for measuring
    achieved motion after deployment; it is not a separately trained tracker.

    Returns future endpoint coordinates and a soft mutual-match confidence.
    """
    if current_tokens.shape != future_tokens.shape:
        raise ValueError(f"Current/future feature shapes differ: {current_tokens.shape}, {future_tokens.shape}")
    expected_tokens = grid_size**2
    if current_tokens.shape[1] != expected_tokens:
        raise ValueError(f"Expected {expected_tokens} tokens for a {grid_size}x{grid_size} grid")
    if temperature <= 0 or spatial_sigma <= 0:
        raise ValueError("temperature and spatial_sigma must be positive")

    current = current_tokens.astype(jnp.float32)
    future = future_tokens.astype(jnp.float32)
    current /= jnp.maximum(jnp.linalg.norm(current, axis=-1, keepdims=True), 1e-6)
    future /= jnp.maximum(jnp.linalg.norm(future, axis=-1, keepdims=True), 1e-6)
    similarity = jnp.einsum("bqd,brd->bqr", current, future)

    coordinates = _grid_coordinates(grid_size)
    squared_distance = jnp.sum(
        jnp.square(coordinates[:, None, :] - coordinates[None, :, :]), axis=-1
    )
    logits = similarity / temperature - squared_distance[None] / (2.0 * spatial_sigma**2)

    # Reciprocal (dual-softmax) matching suppresses one-to-many assignments
    # without introducing a learned observer or an iterative tracker.
    row_probability = jax.nn.softmax(logits, axis=-1)
    column_probability = jax.nn.softmax(logits, axis=-2)
    mutual = row_probability * column_probability
    transport = mutual / jnp.maximum(jnp.sum(mutual, axis=-1, keepdims=True), 1e-8)
    endpoints = jnp.einsum("bqr,rd->bqd", transport, coordinates)
    confidence = jnp.sqrt(jnp.max(mutual, axis=-1))
    return endpoints, confidence


def _fourier_features(x: jax.Array, num_frequencies: int = 4) -> jax.Array:
    """Encode normalized geometric values without learning a coordinate table."""
    frequencies = 2.0 ** jnp.arange(num_frequencies, dtype=jnp.float32)
    angles = 2.0 * jnp.pi * x.astype(jnp.float32)[..., None] * frequencies
    encoded = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    return jnp.concatenate([x.astype(jnp.float32), encoded.reshape(*x.shape[:-1], -1)], axis=-1)


class _SwiGLU(nnx.Module):
    def __init__(self, width: int, hidden_width: int, *, rngs: nnx.Rngs):
        self.gate = nnx.Linear(width, hidden_width, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(width, hidden_width, use_bias=False, rngs=rngs)
        self.output = nnx.Linear(hidden_width, width, use_bias=False, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.output(nnx.swish(self.gate(x)) * self.value(x))


class _MultiHeadAttention(nnx.Module):
    """Small attention implementation used only by the point-flow interface."""

    def __init__(
        self,
        query_width: int,
        key_value_width: int,
        output_width: int,
        hidden_width: int,
        num_heads: int,
        *,
        zero_output: bool = False,
        rngs: nnx.Rngs,
    ):
        if hidden_width % num_heads:
            raise ValueError("hidden_width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_width = hidden_width // num_heads
        self.query = nnx.Linear(query_width, hidden_width, use_bias=False, rngs=rngs)
        self.key = nnx.Linear(key_value_width, hidden_width, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(key_value_width, hidden_width, use_bias=False, rngs=rngs)
        self.output = nnx.Linear(hidden_width, output_width, use_bias=False, rngs=rngs)
        if zero_output:
            self.output.kernel.value = jnp.zeros_like(self.output.kernel.value)

    def _split_heads(self, x: jax.Array) -> jax.Array:
        return x.reshape(*x.shape[:-1], self.num_heads, self.head_width)

    def __call__(self, query: jax.Array, key_value: jax.Array, key_mask: jax.Array | None = None) -> jax.Array:
        q = self._split_heads(self.query(query))
        k = self._split_heads(self.key(key_value))
        v = self._split_heads(self.value(key_value))
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32)
        logits *= self.head_width**-0.5
        if key_mask is not None:
            logits = jnp.where(key_mask[:, None, None, :], logits, jnp.finfo(logits.dtype).min)
        weights = jax.nn.softmax(logits, axis=-1).astype(v.dtype)
        attended = jnp.einsum("bhqk,bkhd->bqhd", weights, v)
        attended = attended.reshape(*attended.shape[:-2], self.num_heads * self.head_width)
        return self.output(attended)


class _PointFlowBlock(nnx.Module):
    def __init__(self, width: int, num_heads: int, ffn_width: int, *, rngs: nnx.Rngs):
        self.cross_norm = nnx.RMSNorm(width, rngs=rngs)
        self.memory_norm = nnx.RMSNorm(width, rngs=rngs)
        self.cross_attention = _MultiHeadAttention(width, width, width, width, num_heads, rngs=rngs)
        self.self_norm = nnx.RMSNorm(width, rngs=rngs)
        self.self_attention = _MultiHeadAttention(width, width, width, width, num_heads, rngs=rngs)
        self.ffn_norm = nnx.RMSNorm(width, rngs=rngs)
        self.ffn = _SwiGLU(width, ffn_width, rngs=rngs)

    def __call__(self, points: jax.Array, transition: jax.Array) -> jax.Array:
        points = points + self.cross_attention(self.cross_norm(points), self.memory_norm(transition))
        points = points + self.self_attention(self.self_norm(points), self.self_norm(points))
        return points + self.ffn(self.ffn_norm(points))


class PointFlowPlanner(nnx.Module):
    """Decode JEPA future-query states into observable point trajectories."""

    def __init__(
        self,
        transition_width: int,
        query_grid_size: int,
        horizon: int,
        hidden_width: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        ffn_width: int = 1024,
        *,
        rngs: nnx.Rngs,
    ):
        if horizon < 1 or query_grid_size < 1 or num_layers < 1:
            raise ValueError("horizon, query_grid_size, and num_layers must be positive")
        self.query_grid_size = query_grid_size
        self.horizon = horizon
        coordinate_width = 2 * (1 + 2 * 4)
        self.transition_norm = nnx.RMSNorm(transition_width, rngs=rngs)
        self.transition_projection = nnx.Linear(transition_width, hidden_width, rngs=rngs)
        self.transition_position = nnx.Linear(coordinate_width, hidden_width, use_bias=False, rngs=rngs)
        self.point_projection = nnx.Linear(coordinate_width, hidden_width, rngs=rngs)
        # Plain Python containers are traversed as graph nodes by the NNX
        # version pinned in openpi.  ``nnx.List`` only exists in newer Flax.
        self.blocks = [_PointFlowBlock(hidden_width, num_heads, ffn_width, rngs=rngs) for _ in range(num_layers)]
        self.output_norm = nnx.RMSNorm(hidden_width, rngs=rngs)
        self.track_head = nnx.Linear(hidden_width, horizon * 2, rngs=rngs)
        self.visibility_head = nnx.Linear(hidden_width, horizon, rngs=rngs)
        # The initial planner is the persistence predictor.  This gives a
        # meaningful, deterministic baseline before any optimization step.
        self.track_head.kernel.value = jnp.zeros_like(self.track_head.kernel.value)
        self.track_head.bias.value = jnp.zeros_like(self.track_head.bias.value)
        self.visibility_head.kernel.value = jnp.zeros_like(self.visibility_head.kernel.value)
        self.visibility_head.bias.value = jnp.full_like(self.visibility_head.bias.value, 4.0)

    def _transition_coordinates(self, dtype: jnp.dtype) -> jax.Array:
        return _grid_coordinates(self.query_grid_size, dtype)

    def __call__(
        self,
        transition_tokens: at.Float[at.Array, "b q dt"],
        query_points: at.Float[at.Array, "b k xy"],
    ) -> tuple[at.Float[at.Array, "b k h xy"], at.Float[at.Array, "b k h"]]:
        expected_queries = self.query_grid_size**2
        if transition_tokens.shape[1] != expected_queries:
            raise ValueError(
                f"Expected {expected_queries} JEPA queries for a {self.query_grid_size}x{self.query_grid_size} grid, "
                f"got {transition_tokens.shape[1]}"
            )
        if query_points.shape[-1] != 2:
            raise ValueError(f"Point coordinates must have width 2, got {query_points.shape}")

        transition = self.transition_projection(self.transition_norm(transition_tokens))
        transition_coordinates = self._transition_coordinates(query_points.dtype)
        transition_coordinates = jnp.broadcast_to(
            transition_coordinates[None], (transition_tokens.shape[0], expected_queries, 2)
        )
        transition = transition + self.transition_position(_fourier_features(transition_coordinates)).astype(
            transition.dtype
        )

        points = self.point_projection(_fourier_features(query_points)).astype(transition.dtype)
        for block in self.blocks:
            points = block(points, transition)
        points = self.output_norm(points)

        # Predict a displacement in logit-coordinate space.  Zero displacement
        # maps every future step to the current query point exactly.
        epsilon = 1e-4
        current = jnp.clip(query_points.astype(jnp.float32), epsilon, 1.0 - epsilon)
        current_logits = jnp.log(current) - jnp.log1p(-current)
        displacement_logits = self.track_head(points).astype(jnp.float32)
        displacement_logits = displacement_logits.reshape(
            query_points.shape[0], query_points.shape[1], self.horizon, 2
        )
        tracks = jax.nn.sigmoid(current_logits[:, :, None, :] + displacement_logits)
        visibility_logits = self.visibility_head(points).astype(jnp.float32)
        return tracks, visibility_logits


class PointFlowConditioner(nnx.Module):
    """Encode geometric tracks and inject them into action hidden states."""

    def __init__(
        self,
        action_width: int,
        plan_width: int = 256,
        num_heads: int = 4,
        *,
        rngs: nnx.Rngs,
    ):
        # start xy, displacement xy, visibility, normalized time, followed by
        # four-frequency Fourier features of the four geometric coordinates.
        geometric_width = 4
        fourier_width = geometric_width * (1 + 2 * 4)
        feature_width = fourier_width + 2
        self.plan_projection = nnx.Linear(feature_width, plan_width, rngs=rngs)
        self.plan_norm = nnx.RMSNorm(plan_width, rngs=rngs)
        self.action_norm = nnx.RMSNorm(action_width, rngs=rngs)
        self.cross_attention = _MultiHeadAttention(
            action_width,
            plan_width,
            action_width,
            plan_width,
            num_heads,
            zero_output=True,
            rngs=rngs,
        )

    def encode_plan(
        self,
        query_points: at.Float[at.Array, "b k xy"],
        tracks: at.Float[at.Array, "b k h xy"],
        visibility: at.Float[at.Array, "b k h"],
    ) -> at.Float[at.Array, "b n d"]:
        if tracks.shape[:2] != query_points.shape[:2] or tracks.shape[-1] != 2:
            raise ValueError(f"Incompatible point/track shapes: {query_points.shape}, {tracks.shape}")
        if visibility.shape != tracks.shape[:-1]:
            raise ValueError(f"Visibility shape {visibility.shape} does not match tracks {tracks.shape}")
        batch, points, horizon, _ = tracks.shape
        starts = jnp.broadcast_to(query_points[:, :, None, :], tracks.shape)
        geometry = jnp.concatenate([starts, tracks - starts], axis=-1)
        geometry = _fourier_features(geometry)
        time = (jnp.arange(horizon, dtype=jnp.float32) + 1.0) / horizon
        time = jnp.broadcast_to(time[None, None, :, None], (batch, points, horizon, 1))
        features = jnp.concatenate([geometry, visibility.astype(jnp.float32)[..., None], time], axis=-1)
        features = features.reshape(batch, points * horizon, -1)
        return self.plan_norm(self.plan_projection(features))

    def __call__(
        self,
        action_hidden: at.Float[at.Array, "b a da"],
        query_points: at.Float[at.Array, "b k xy"],
        tracks: at.Float[at.Array, "b k h xy"],
        visibility: at.Float[at.Array, "b k h"],
    ) -> at.Float[at.Array, "b a da"]:
        plan = self.encode_plan(query_points, tracks, visibility)
        return action_hidden + self.cross_attention(self.action_norm(action_hidden), plan)


def point_flow_loss(
    predicted_tracks: jax.Array,
    predicted_visibility_logits: jax.Array,
    target_tracks: jax.Array,
    target_visibility: jax.Array,
    *,
    visibility_weight: float = 0.1,
    smoothness_weight: float = 0.05,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Per-example point-flow loss and diagnostics."""
    if predicted_tracks.shape != target_tracks.shape:
        raise ValueError(f"Track shape mismatch: {predicted_tracks.shape} != {target_tracks.shape}")
    if predicted_visibility_logits.shape != target_visibility.shape:
        raise ValueError(
            f"Visibility shape mismatch: {predicted_visibility_logits.shape} != {target_visibility.shape}"
        )
    visible = target_visibility.astype(jnp.float32)
    error = predicted_tracks.astype(jnp.float32) - target_tracks.astype(jnp.float32)
    absolute = jnp.abs(error)
    huber = jnp.where(absolute <= 0.01, 0.5 * jnp.square(error) / 0.01, absolute - 0.005)
    visible_coordinates = visible[..., None]
    denominator = jnp.maximum(jnp.sum(visible_coordinates, axis=(1, 2, 3)), 1.0)
    track_loss = jnp.sum(huber * visible_coordinates, axis=(1, 2, 3)) / denominator

    visibility_loss = jax.nn.softplus(predicted_visibility_logits) - visible * predicted_visibility_logits
    visibility_loss = jnp.mean(visibility_loss, axis=(1, 2))

    velocity = jnp.diff(predicted_tracks.astype(jnp.float32), axis=2)
    acceleration = jnp.diff(velocity, axis=2)
    smoothness_loss = (
        jnp.mean(jnp.abs(acceleration), axis=(1, 2, 3))
        if predicted_tracks.shape[2] > 2
        else jnp.zeros(predicted_tracks.shape[0], dtype=jnp.float32)
    )
    loss = track_loss + visibility_weight * visibility_loss + smoothness_weight * smoothness_loss

    euclidean = jnp.linalg.norm(error, axis=-1)
    visible_count = jnp.maximum(jnp.sum(visible, axis=(1, 2)), 1.0)
    ade = jnp.sum(euclidean * visible, axis=(1, 2)) / visible_count
    final_visible = visible[:, :, -1]
    final_count = jnp.maximum(jnp.sum(final_visible, axis=1), 1.0)
    fde = jnp.sum(euclidean[:, :, -1] * final_visible, axis=1) / final_count
    visibility_accuracy = jnp.mean(
        (predicted_visibility_logits >= 0) == (target_visibility >= 0.5), axis=(1, 2)
    )
    metrics = {
        "track_loss": track_loss,
        "visibility_loss": visibility_loss,
        "smoothness_loss": smoothness_loss,
        "ade": ade,
        "fde": fde,
        "visibility_accuracy": visibility_accuracy,
    }
    return loss, metrics
