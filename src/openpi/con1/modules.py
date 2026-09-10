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

    @nn.compact
    def __call__(self, r_tokens, current_latent):
        if r_tokens.ndim != 3 or current_latent.ndim != 2:
            raise ValueError("Expected R[B,N,E] and current latent[B,D]")
        if r_tokens.shape[0] != current_latent.shape[0] or current_latent.shape[-1] != self.latent_dim:
            raise ValueError("Anchor/batch dimensions disagree")
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
        hidden = hidden + nn.Dense(self.width, name="ff_out")(
            nn.gelu(nn.Dense(2 * self.width, name="ff_in")(nn.LayerNorm(name="ff_norm")(hidden))))
        delta = nn.Dense(self.latent_dim, name="delta_out",
                         kernel_init=nn.initializers.normal(1e-4))(hidden)
        return {"delta": delta, "future": anchor[:, None] + delta}


class ActionDeltaCrossAttention(nn.Module):
    action_width: int
    width: int = 512
    alpha_initial: float = .05

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
        residual = nn.Dense(self.action_width, use_bias=False, name="out",
                            kernel_init=nn.initializers.zeros_init())(attention @ v)
        logit = self.param("alpha_logit", lambda _: jnp.asarray(
            math.log(self.alpha_initial / (1 - self.alpha_initial)), dtype=jnp.float32))
        correction = jax.nn.sigmoid(logit) * residual
        return {"hidden": action_hidden.astype(jnp.float32) + correction,
                "correction": correction, "attention": attention,
                "alpha": jax.nn.sigmoid(logit)}


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
    uniform = mask.astype(jnp.float32) / jnp.maximum(count, 1)
    safe_future = jnp.where(mask[..., None], future_target.astype(jnp.float32), anchor[:, None])
    target = jax.lax.stop_gradient(safe_future - anchor[:, None].astype(jnp.float32))
    pred = jnp.where(mask[..., None], delta.astype(jnp.float32), 0.)
    mse = jnp.square(pred - target).mean(-1)
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
    # Weight examples by valid horizon count; beta=0 is the global masked MSE.
    denom = jnp.maximum(count.sum(), 1)
    loss = jnp.sum(jnp.sum(q * mse, axis=-1) * count[:, 0]) / denom
    plain_mse = jnp.sum(mse) / denom
    zero_mse = jnp.sum(jnp.square(target).mean(-1)) / denom
    return loss, {"con1_delta_nmse": plain_mse / jnp.maximum(zero_mse, 1e-12)}
