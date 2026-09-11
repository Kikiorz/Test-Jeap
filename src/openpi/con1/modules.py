"""No Q0/Gamma branch, no historical Con1 implementations.

The head's fixed horizon encoding is not a learned action query Q0.
Only CURRENT-observation features enter prediction. Labels enter loss only.
"""
import math

import flax.linen as nn
import jax
import jax.numpy as jnp


def horizon_encoding(horizon, width):
    if horizon < 1 or width < 2 or width % 2:
        raise ValueError("Positive horizon and even width >=2 required")
    positions = jnp.arange(1, horizon + 1, dtype=jnp.float32)[:, None]
    frequencies = jnp.exp(-math.log(10000.) * jnp.arange(width // 2) / (width // 2))
    angles = positions * frequencies[None]
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)


class AnchoredDeltaHead(nn.Module):
    horizon: int = 10
    latent_dim: int = 2816
    width: int = 512
    action_dim: int = 0
    use_action_conditioning: bool = False
    use_direct_readout: bool = False
    vlm_context_dim: int = 0
    use_vlm_context: bool = False

    @nn.compact
    def __call__(self, r_tokens, current_latent, action_chunk=None, vlm_context=None):
        if r_tokens.ndim != 3 or current_latent.ndim != 2:
            raise ValueError("Expected R[B,N,E] and current latent[B,D]")
        if r_tokens.shape[0] != current_latent.shape[0] or current_latent.shape[-1] != self.latent_dim:
            raise ValueError("Anchor/batch dimensions disagree")
        if self.use_action_conditioning:
            if action_chunk is None:
                raise ValueError("Action conditioning enabled but no action chunk was supplied")
            if action_chunk.ndim != 3 or action_chunk.shape[1] != self.horizon:
                raise ValueError("Action chunk must be [B, horizon, action_dim]")
            if action_chunk.shape[-1] != self.action_dim:
                raise ValueError(f"Action chunk width {action_chunk.shape[-1]} != {self.action_dim}")
            action_chunk = jax.lax.stop_gradient(action_chunk.astype(jnp.float32))
        if self.use_vlm_context:
            if vlm_context is None:
                raise ValueError("VLM context conditioning enabled but no context was supplied")
            if vlm_context.shape[-1] != self.vlm_context_dim:
                raise ValueError(
                    f"VLM context width {vlm_context.shape[-1]} != {self.vlm_context_dim}")
            vlm_context = jax.lax.stop_gradient(vlm_context.astype(jnp.float32))
        r = jax.lax.stop_gradient(r_tokens.astype(jnp.float32))
        anchor = jax.lax.stop_gradient(current_latent.astype(jnp.float32))
        r = nn.LayerNorm(name="r_norm")(r)
        # Raw anchor projection retains magnitude information; normalize hidden,
        # not the teacher latent defining the delta/reconstruction target.
        anchor_hidden = nn.Dense(self.width, name="anchor_in")(anchor)
        slots = nn.Dense(self.horizon * self.width, name="chunk_expand")(nn.gelu(anchor_hidden))
        slots = slots.reshape(anchor.shape[0], self.horizon, self.width)
        q = nn.Dense(self.width, use_bias=False, name="temporal_query")(
            nn.LayerNorm(name="slot_norm")(slots))
        k = nn.Dense(self.width, use_bias=False, name="r_key")(r)
        v = nn.Dense(self.width, use_bias=False, name="r_value")(r)
        if self.width % 4:
            raise ValueError("Width must be divisible by four attention heads")
        q = q.reshape(q.shape[0], q.shape[1], 4, self.width // 4)
        k = k.reshape(k.shape[0], k.shape[1], 4, self.width // 4)
        v = v.reshape(v.shape[0], v.shape[1], 4, self.width // 4)
        attention = jax.nn.softmax(jnp.einsum("bmhd,bnhd->bhmn", q, k) / math.sqrt(self.width // 4), -1)
        context = jnp.einsum("bhmn,bnhd->bmhd", attention, v).reshape(slots.shape)
        hidden = slots + context

        if self.use_vlm_context:
            # Zero-initialise the *output* projection, not the input: zeroing the
            # input makes the output projection's gradient identically zero, so
            # it would freeze at a random draw. This keeps the branch an exact
            # no-op at step zero while letting both projections train.
            merged = nn.Dense(self.width, name="vlm_context_in")(vlm_context)
            merged = nn.Dense(self.width, name="vlm_context_out",
                              kernel_init=nn.initializers.zeros_init())(nn.gelu(merged))
            hidden = hidden + merged[:, None, :]

        # Optional direct per-horizon linear readout of the pooled features. The
        # attention path alone was measured to underfit badly (0.483 held-out
        # NMSE against a 0.414 linear floor), because every output had to be
        # learnt through attention + FFN. Gated by a flag so checkpoints written
        # with the older head structure still restore exactly.
        direct = None
        if self.use_direct_readout:
            # Raw token mean, so the exact input to this layer can be rebuilt
            # outside the model for a closed-form warm start.
            pooled = r_tokens.astype(jnp.float32).mean(1)
            readout = jnp.concatenate([pooled, anchor], axis=-1)
            direct = nn.Dense(
                self.horizon * self.latent_dim, name="direct_readout",
                kernel_init=nn.initializers.normal(1e-4),
            )(readout)
            direct = direct.reshape(anchor.shape[0], self.horizon, self.latent_dim)

        if self.use_action_conditioning:
            # Actions enter as horizon-indexed tokens. A causal mask stops
            # horizon j from reading actions that happen after step j.
            action_hidden = nn.Dense(self.width, name="action_in")(action_chunk)
            action_hidden = action_hidden + horizon_encoding(self.horizon, self.width)[None]
            action_hidden = nn.LayerNorm(name="action_norm")(action_hidden)
            aq = nn.Dense(self.width, use_bias=False, name="action_query")(hidden)
            ak = nn.Dense(self.width, use_bias=False, name="action_key")(action_hidden)
            # Zero-initialised value projection: at step zero the action branch
            # contributes exactly nothing, so the head still reproduces the
            # pretrained head and only gains action conditioning as it trains.
            av = nn.Dense(self.width, use_bias=False, name="action_value",
                          kernel_init=nn.initializers.zeros_init())(action_hidden)
            heads = 4
            aq = aq.reshape(*aq.shape[:2], heads, self.width // heads)
            ak = ak.reshape(*ak.shape[:2], heads, self.width // heads)
            av = av.reshape(*av.shape[:2], heads, self.width // heads)
            logits = jnp.einsum("bqhd,bkhd->bhqk", aq, ak) / math.sqrt(self.width // heads)
            step = jnp.arange(self.horizon)
            causal = (step[None, :] <= step[:, None])[None, None]
            logits = jnp.where(causal, logits, -1e30)
            action_attn = jax.nn.softmax(logits, -1)
            action_context = jnp.einsum("bhqk,bkhd->bqhd", action_attn, av).reshape(*hidden.shape)
            hidden = hidden + action_context

        hidden = hidden + nn.Dense(self.width, name="ff_out")(
            nn.gelu(nn.Dense(2 * self.width, name="ff_in")(nn.LayerNorm(name="ff_norm")(hidden))))
        delta = nn.Dense(self.latent_dim, name="delta_out",
                         kernel_init=nn.initializers.normal(1e-4))(hidden)
        if direct is not None:
            delta = delta + direct
        return {"delta": delta, "future": anchor[:, None] + delta}


class ActionDeltaCrossAttention(nn.Module):
    action_width: int
    width: int = 512
    alpha_initial: float = .05
    # Bounded, zero-initialised action-side adapter. It is the Con1 analogue of
    # a LoRA bottleneck on the action expert: it starts as an exact no-op and
    # adds capacity that does not depend on the latent prediction at all.
    use_action_adapter: bool = False
    adapter_scale: float = 1.0
    # Zero keeps the original behaviour: the correction is exactly zero at step
    # zero, but the key/value/query projections then receive no gradient until
    # the output projection has moved on its own. A small non-zero standard
    # deviation lets the latent-conditioned path train from the first step while
    # the gate keeps the perturbation negligible. The `alpha=0` identity with
    # the base policy is preserved by the gate regardless of this value.
    out_init_std: float = 0.0
    # Hard relative budget on the correction: per token, its RMS is capped at
    # `residual_budget` times the RMS of the incoming action hidden state. 0
    # disables the cap. The rescaling factor is stop-gradiented, so it bounds
    # the forward perturbation without changing the gradient direction.
    residual_budget: float = 0.0

    @nn.compact
    def __call__(self, action_hidden, predicted_delta):
        if not 0 < self.alpha_initial < 1:
            raise ValueError("alpha_initial must be inside (0,1)")
        if action_hidden.ndim != 3 or predicted_delta.ndim != 3:
            raise ValueError("Expected action[B,H,A] and delta[B,J,D]")
        if action_hidden.shape[0] != predicted_delta.shape[0] or action_hidden.shape[-1] != self.action_width:
            raise ValueError("Action/delta dimensions disagree")
        # Q = W_Q LN(H_A): no learnable query tokens, no Q0 + Gamma(H).
        q = nn.Dense(self.width, use_bias=False, name="query")(
            nn.LayerNorm(name="action_norm")(action_hidden.astype(jnp.float32)))
        k = nn.Dense(self.width, use_bias=False, name="key")(predicted_delta.astype(jnp.float32))
        v = nn.Dense(self.width, use_bias=False, name="value")(predicted_delta.astype(jnp.float32))
        attention = jax.nn.softmax(jnp.einsum("bhd,bjd->bhj", q, k) / math.sqrt(self.width), -1)
        out_init = (nn.initializers.zeros_init() if self.out_init_std <= 0
                    else nn.initializers.normal(stddev=self.out_init_std))
        residual = nn.Dense(self.action_width, use_bias=False, name="out",
                            kernel_init=out_init)(attention @ v)
        logit = self.param("alpha_logit", lambda _: jnp.asarray(
            math.log(self.alpha_initial / (1 - self.alpha_initial)), dtype=jnp.float32))
        correction = jax.nn.sigmoid(logit) * residual
        if self.use_action_adapter:
            normed = nn.LayerNorm(name="adapter_norm")(action_hidden.astype(jnp.float32))
            # See the vlm_context note: the output projection carries the zero
            # initialisation so both projections remain trainable.
            hidden = nn.Dense(self.width, name="adapter_in")(normed)
            hidden = nn.gelu(hidden)
            adapter = nn.Dense(self.action_width, name="adapter_out",
                               kernel_init=nn.initializers.zeros_init())(hidden)
            correction = correction + self.adapter_scale * adapter
        if self.residual_budget > 0:
            correction_rms = jnp.sqrt(jnp.mean(jnp.square(correction), axis=-1, keepdims=True))
            base_rms = jnp.sqrt(
                jnp.mean(jnp.square(action_hidden.astype(jnp.float32)), axis=-1, keepdims=True))
            limit = self.residual_budget * jnp.maximum(base_rms, 1e-6)
            shrink = jnp.minimum(1.0, limit / jnp.maximum(correction_rms, 1e-12))
            correction = correction * jax.lax.stop_gradient(shrink)
        return {"hidden": action_hidden.astype(jnp.float32) + correction,
                "correction": correction, "attention": attention,
                "alpha": jax.nn.sigmoid(logit),
                "budget_shrink": (jnp.ones_like(correction[..., :1]) if self.residual_budget <= 0
                                  else jax.lax.stop_gradient(shrink))}


def anchored_loss(delta, anchor, future_target, valid, *, delta_weight=1., feature_reduction="mean"):
    """Exact requested two-term objective, detached CURRENT anchor and teacher.

    With anchor=z_t*=Phi(o_t), the terms are algebraically identical:
    L_recon + lambda L_delta = (1+lambda) L_delta. No extra constraint.
    Mean reduction avoids inflating scale by latent_dim; 'sum' is explicit.
    Average over valid positions; all-invalid batches yield zero loss/gradient.
    """
    if delta_weight < 0 or not math.isfinite(delta_weight):
        raise ValueError("delta_weight must be finite and nonnegative")
    if feature_reduction not in ("mean", "sum"):
        raise ValueError("feature_reduction must be mean or sum")
    if delta.shape != future_target.shape or valid.shape != delta.shape[:-1] or anchor.shape != (delta.shape[0], delta.shape[-1]):
        raise ValueError("Delta/anchor/future/mask shapes disagree")
    anchor = jax.lax.stop_gradient(anchor.astype(jnp.float32))
    target = jax.lax.stop_gradient(future_target.astype(jnp.float32))
    mask = valid.astype(bool)
    # Sanitize padded entries BEFORE arithmetic (including NaN placeholders).
    target = jnp.where(mask[..., None], target, anchor[:, None])
    prediction = jnp.where(mask[..., None], delta.astype(jnp.float32), 0.)
    target_delta = target - anchor[:, None]
    # Stable algebraic form also avoids cancellation when anchor is large.
    error = prediction - target_delta
    divisor = delta.shape[-1] if feature_reduction == "mean" else 1
    count = mask.sum()
    denom = jnp.maximum(count, 1)
    reconstruction = jnp.where(mask, jnp.square(error).sum(-1) / divisor, 0.).sum() / denom
    delta_loss = reconstruction
    zero_mse = jnp.where(mask, jnp.square(target_delta).mean(-1), 0.).sum() / denom
    mse = jnp.where(mask, jnp.square(error).mean(-1), 0.).sum() / denom
    metrics = {"loss": reconstruction + delta_weight * delta_loss,
               "reconstruction_loss": reconstruction, "delta_loss": delta_loss,
               "delta_mse": mse, "copy_current_mse": zero_mse,
               "delta_nmse": mse / jnp.maximum(zero_mse, 1e-12), "valid_count": count}
    return metrics["loss"], metrics


def sensitivity_horizon_weights(attention, sensitivity, valid, *, beta, action_valid=None,
                                temperature=.1, prediction=None):
    """Per-horizon supervision weights: last-layer usage x action sensitivity.

    Detached, and floored at the uniform distribution: invalid episode-tail
    labels get neither loss nor gradient, and a zero sensitivity (including a
    zero-initialised adapter) falls back to uniform. Extracted so the Euclidean
    and the learned-metric losses weight horizons identically.
    """
    mask = valid.astype(bool)
    count = mask.sum(-1, keepdims=True)
    uniform = mask.astype(jnp.float32) / jnp.maximum(count, 1)
    pred = jnp.where(mask[..., None], (sensitivity if prediction is None else prediction).astype(jnp.float32), 0.)
    if action_valid is None:
        action_valid = jnp.ones(attention.shape[:2], dtype=bool)
    usage = (attention.astype(jnp.float32) * action_valid[..., None]).sum(1)
    usage /= jnp.maximum(action_valid.sum(1, keepdims=True), 1)
    # First-order gradient x input; a saliency heuristic, not causal importance.
    strength = jnp.abs(jnp.sum(jax.lax.stop_gradient(sensitivity.astype(jnp.float32)) * pred, axis=-1))
    strength = jax.lax.stop_gradient(jnp.where(mask, strength, 0.))
    scores = jax.nn.softmax(jnp.where(mask, strength / temperature, -1e9), axis=-1)
    relevance = jnp.where(mask, usage * scores, 0.)
    normalizer = relevance.sum(-1, keepdims=True)
    use_control = (normalizer > 1e-12) & (strength.sum(-1, keepdims=True) > 1e-12)
    control = jnp.where(use_control, relevance / jnp.maximum(normalizer, 1e-12), uniform)
    q = jax.lax.stop_gradient((1 - beta) * uniform + beta * control)
    return q


def control_weighted_delta_loss(delta, anchor, future_target, valid, attention, sensitivity,
                                *, beta, action_valid=None, temperature=.1):
    """Last-layer usage x action sensitivity with a uniform supervision floor.

    q is detached: the model cannot lower loss by moving its own weights toward
    easy horizons. Invalid episode-tail labels get neither loss nor gradient.
    Zero sensitivity (including zero-initialized adapters) falls back to uniform.
    """
    if delta.shape != future_target.shape or valid.shape != delta.shape[:-1]:
        raise ValueError("Con1 label/mask shapes disagree")
    mask = valid.astype(bool)
    count = mask.sum(-1, keepdims=True)
    safe_future = jnp.where(mask[..., None], future_target.astype(jnp.float32), anchor[:, None])
    target = jax.lax.stop_gradient(safe_future - anchor[:, None].astype(jnp.float32))
    pred = jnp.where(mask[..., None], delta.astype(jnp.float32), 0.)
    mse = jnp.square(pred - target).mean(-1)
    q = sensitivity_horizon_weights(attention, sensitivity, valid, beta=beta,
                                    action_valid=action_valid, temperature=temperature,
                                    prediction=pred)
    # Weight examples by valid horizon count; beta=0 is the global masked MSE.
    denom = jnp.maximum(count.sum(), 1)
    loss = jnp.sum(jnp.sum(q * mse, axis=-1) * count[:, 0]) / denom
    plain_mse = jnp.sum(mse) / denom
    zero_mse = jnp.sum(jnp.square(target).mean(-1)) / denom
    return loss, {"con1_delta_nmse": plain_mse / jnp.maximum(zero_mse, 1e-12)}


class DeltaMetric(nn.Module):
    """Bounded diagonal positive-definite metric on the latent-delta residual.

    ``F(r) = mean_d ( m_d * r_d^2 )`` with ``m = 1 + max_scale * tanh(theta)``.

    * **Exactly Euclidean at initialisation** (``theta = 0`` -> ``m = 1``), so
      switching it on cannot jump away from the current behaviour.
    * **Positive definite with a single zero at ``r = 0``** for any ``theta``
      because ``m_d >= 1 - max_scale > 0``. The head therefore can never be
      driven towards a wrong latent; the metric only re-weights *which*
      residual directions are paid for, the target ``delta_z*`` is unchanged.
    * **Non-zero gradient at the identity**, which is what makes it trainable:
      ``m`` enters linearly. Two textbook parameterisations fail here and are
      deliberately not used - ``I + U U^T`` is quadratic in ``U``, and
      ``A = p q^T`` is bilinear in ``(p, q)``; both have an exact saddle at the
      identity, so their parameters would never leave zero.

    A diagonal metric can re-weight latent dimensions but cannot rotate the
    gradient. Rotation needs a non-diagonal metric, which in turn needs a
    parameterisation whose gradient does not vanish at ``I``; that is future
    work, not something to smuggle in with a frozen parameter block.
    """

    latent_dim: int = 2816
    max_scale: float = 0.5

    @nn.compact
    def __call__(self, residual):
        if residual.ndim != 3 or residual.shape[-1] != self.latent_dim:
            raise ValueError(f"Expected residual[B,H,{self.latent_dim}], got {residual.shape}")
        if not 0 <= self.max_scale < 1:
            raise ValueError("max_scale must be in [0, 1) so the metric stays positive definite")
        theta = self.param("theta", nn.initializers.zeros, (self.latent_dim,))
        scale = 1.0 + self.max_scale * jnp.tanh(theta)
        return metric_value(residual.astype(jnp.float32), scale), scale


def metric_value(residual, scale):
    """Per-horizon quadratic form of the diagonal metric."""
    return (jnp.square(residual) * scale).mean(-1)


def metric_gradient(residual, scale):
    """dF/dresidual for the diagonal metric."""
    return 2.0 * scale * residual / residual.shape[-1]


def metric_delta_loss(delta, anchor, future_target, valid, scale, *, weights=None):
    """Masked, optionally sensitivity-weighted loss under a learned metric.

    Drop-in replacement for :func:`control_weighted_delta_loss`: same masking,
    same weighting contract, but the quadratic form is the learned metric
    instead of the identity. ``scale`` is the per-dimension weight produced by
    :class:`DeltaMetric`; with ``scale = 1`` (the initialisation) the value is
    identical to the Euclidean masked MSE. ``weights`` are renormalised per
    sample over valid horizons, so passing a plain mask means uniform weighting.
    """
    if delta.shape != future_target.shape or valid.shape != delta.shape[:-1]:
        raise ValueError("Con1 label/mask shapes disagree")
    mask = valid.astype(bool)
    count = mask.sum(-1, keepdims=True)
    safe_future = jnp.where(mask[..., None], future_target.astype(jnp.float32), anchor[:, None])
    target = jax.lax.stop_gradient(safe_future - anchor[:, None].astype(jnp.float32))
    residual = jnp.where(mask[..., None], delta.astype(jnp.float32) - target, 0.0)
    per_horizon = metric_value(residual, scale)
    if weights is None:
        weights = mask.astype(jnp.float32)
    weights = jax.lax.stop_gradient(weights.astype(jnp.float32) * mask)
    weights = weights / jnp.maximum(weights.sum(-1, keepdims=True), 1e-12)
    denom = jnp.maximum(count.sum(), 1)
    loss = jnp.sum(jnp.sum(weights * per_horizon, axis=-1) * count[:, 0]) / denom
    plain = jnp.sum(jnp.where(mask, per_horizon, 0.0)) / denom
    zero = jnp.sum(jnp.square(target).mean(-1)) / denom
    return loss, {"con1_delta_nmse": plain / jnp.maximum(zero, 1e-12)}


def direction_alignment_loss(delta, anchor, future_target, valid, sensitivity, scale):
    """Train the metric so its gradient points along the action direction.

    ``sensitivity`` is the already-computed ``d(action flow)/d(delta)``. The
    cosine is scale free, so this objective cannot game the magnitude of the
    latent term - only its direction.
    """
    if sensitivity.shape != delta.shape:
        raise ValueError("sensitivity must match the delta shape")
    mask = valid.astype(bool)
    target = jax.lax.stop_gradient(
        jnp.where(mask[..., None], future_target.astype(jnp.float32), anchor[:, None])
        - anchor[:, None].astype(jnp.float32))
    residual = jnp.where(mask[..., None], delta.astype(jnp.float32) - target, 0.0)
    direction = metric_gradient(residual, scale)
    action = jnp.where(mask[..., None], jax.lax.stop_gradient(sensitivity.astype(jnp.float32)), 0.0)
    flat_direction = direction.reshape(direction.shape[0], -1)
    flat_action = action.reshape(action.shape[0], -1)
    denominator = (jnp.linalg.norm(flat_direction, axis=-1) * jnp.linalg.norm(flat_action, axis=-1))
    active = denominator > 1e-12
    cosine = jnp.where(active, jnp.sum(flat_direction * flat_action, axis=-1) / jnp.maximum(denominator, 1e-12), 0.0)
    weight = jnp.where(active, 1.0, 0.0)
    loss = 1.0 - jnp.sum(cosine * weight) / jnp.maximum(weight.sum(), 1.0)
    return loss, {"con1_metric_cosine": jnp.sum(cosine * weight) / jnp.maximum(weight.sum(), 1.0)}


def balanced_latent_weight(flow_gradient, delta_gradient, strength):
    """Gradient-scale balancing factor for the latent term.

    The measured imbalance is large: the latent term contributes ~700x more
    gradient magnitude to the head than the flow term does (see
    ``docs_CON1_DELTA_GRADIENT_ALIGNMENT.md``), so the configured 0.2 vs 1.0
    weights are not the effective balance. This returns a detached factor that
    makes the two contributions comparable, interpolated by ``strength``
    (``0`` keeps the configured weights exactly).
    """
    flow_norm = jnp.linalg.norm(flow_gradient.reshape(-1))
    delta_norm = jnp.linalg.norm(delta_gradient.reshape(-1))
    ratio = jax.lax.stop_gradient(flow_norm / jnp.maximum(delta_norm, 1e-12))
    return (1.0 - strength) + strength * ratio
