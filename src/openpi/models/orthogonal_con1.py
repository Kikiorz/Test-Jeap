"""The paper's action-conditioned retrieval and direct r*s supervision.

This is a separate variant: historical RAPR checkpoints keep their original
architecture and softmax-sensitivity objective for reproducible evaluation.
"""

import flax.nnx as nnx
import jax
import jax.numpy as jnp

_ZERO_INIT = nnx.initializers.zeros_init()
_SMALL_INIT = nnx.initializers.normal(stddev=1e-4)


class OrthogonalDeltaHead(nnx.Module):
    """Current-only H-slot predictor, without channel compression of labels."""

    def __init__(self, input_width, horizon, delta_dim, width, *, rngs):
        self.norm = nnx.LayerNorm(input_width, rngs=rngs)
        self.key = nnx.Linear(input_width, width, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(input_width, width, use_bias=False, rngs=rngs)
        self.temporal_queries = nnx.Param(jax.random.normal(rngs.params(), (horizon, width)) * 0.02)
        self.ff_in = nnx.Linear(width, width * 2, rngs=rngs)
        self.ff_out = nnx.Linear(width * 2, width, rngs=rngs)
        self.out = nnx.Linear(width, delta_dim, kernel_init=_SMALL_INIT, rngs=rngs)

    def __call__(self, current_tokens):
        tokens = self.norm(current_tokens.astype(jnp.float32))
        keys, values = self.key(tokens), self.value(tokens)
        logits = jnp.einsum("md,bkd->bmk", self.temporal_queries.value, keys) / jnp.sqrt(keys.shape[-1])
        hidden = jax.nn.softmax(logits, axis=-1) @ values
        hidden = hidden + self.ff_out(nnx.gelu(self.ff_in(hidden)))
        return self.out(hidden).astype(jnp.float32)


class ActionConditionedQFormer(nnx.Module):
    """One action-conditioned query cross-attention block; M=H, one head.

    Q_A=Q0+Gamma_A(H_A), A=softmax(Q_A Wq (Delta Wk)^T / sqrt(d)).
    W_delta is zero-initialized. No extra relative bias or non-paper
    sensitivity softmax is introduced. Alpha is a continuous residual scale.
    """

    def __init__(self, action_width, delta_dim, horizon, width, *, learnable_alpha, rngs):
        self.horizon = horizon
        self.action_norm = nnx.LayerNorm(action_width, rngs=rngs)
        self.gamma_in = nnx.Linear(action_width, width, rngs=rngs)
        self.gamma_out = nnx.Linear(width, width, rngs=rngs)
        self.query_tokens = nnx.Param(jax.random.normal(rngs.params(), (horizon, width)) * 0.02)
        self.query = nnx.Linear(width, width, use_bias=False, rngs=rngs)
        self.key = nnx.Linear(delta_dim, width, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(delta_dim, width, use_bias=False, rngs=rngs)
        self.out = nnx.Linear(width, action_width, use_bias=False, kernel_init=_ZERO_INIT, rngs=rngs)
        logit = jnp.asarray(-2.944439, dtype=jnp.float32)
        self.alpha_logit = nnx.Param(logit) if learnable_alpha else nnx.Variable(logit)

    def residual(self, action_hidden, delta, *, gate_override=None):
        if action_hidden.shape[-2] != self.horizon or delta.shape[-2] != self.horizon:
            raise ValueError("Paper Con1 uses M=H action-aligned queries and H transitions")
        action = self.action_norm(action_hidden.astype(jnp.float32))
        q_action = self.query_tokens.value + self.gamma_out(nnx.gelu(self.gamma_in(action)))
        q, k, v = self.query(q_action), self.key(delta), self.value(delta)
        logits = jnp.einsum("bmd,bjd->bmj", q, k) / jnp.sqrt(q.shape[-1])
        routes = jax.nn.softmax(logits, axis=-1)
        alpha = jax.nn.sigmoid(self.alpha_logit.value)
        if gate_override is not None:
            alpha = alpha * gate_override
        residual = self.out(routes @ v)
        return alpha * residual.astype(jnp.float32), routes

    def __call__(self, action_hidden, delta, *, gate_override=None):
        residual, routes = self.residual(action_hidden, delta, gate_override=gate_override)
        return action_hidden.astype(jnp.float32) + residual, routes

    def diagnostic_retrieval(self, action_hidden, delta):
        """Counterfactual readouts only; never replace the training forward.

        Uniform retrieval tests learned attention; Q0-only retrieval tests
        action conditioning. Both reuse the same Delta values and readout.
        """
        action = self.action_norm(action_hidden.astype(jnp.float32))
        q0 = jnp.broadcast_to(self.query_tokens.value, (*action.shape[:2], self.query_tokens.value.shape[-1]))
        qa = q0 + self.gamma_out(nnx.gelu(self.gamma_in(action)))
        keys, values = self.key(delta), self.value(delta)
        def attention(queries):
            return jax.nn.softmax(jnp.einsum("bmd,bjd->bmj", self.query(queries), keys)
                                  / jnp.sqrt(keys.shape[-1]), axis=-1)
        full, unconditioned = attention(qa), attention(q0)
        uniform = jnp.full_like(full, 1.0 / self.horizon)
        alpha = jax.nn.sigmoid(self.alpha_logit.value)
        return {"uniform_residual": alpha * self.out(uniform @ values).astype(jnp.float32),
                "unconditioned_residual": alpha * self.out(unconditioned @ values).astype(jnp.float32),
                "condition_kl": (full * (jnp.log(jnp.maximum(full, 1e-30))
                                          - jnp.log(jnp.maximum(unconditioned, 1e-30)))).sum(-1).mean(-1),
                "route_max_mass": full.max(-1).mean(-1)}


def control_weights(routes, delta, action_gradient, *, beta=0.5, valid=None):
    """Exactly r*s normalized, detached; uniform fallback when all s_j=0."""
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0,1)")
    if valid is None:
        valid = jnp.ones(delta.shape[:-1], dtype=jnp.bool_)
    mask = valid.astype(jnp.float32)
    uniform = mask / jnp.maximum(mask.sum(-1, keepdims=True), 1)
    usage = routes.mean(-2)
    sensitivity = jnp.abs(jnp.sum(action_gradient * delta, axis=-1))
    relevance_raw = usage * sensitivity * mask
    mass = relevance_raw.sum(-1, keepdims=True)
    # Do not clamp a small nonzero mass to epsilon: that changes sum(q) to <1.
    relevance = jnp.where(mass > 0, relevance_raw / jnp.where(mass > 0, mass, 1), uniform)
    weights = (1 - beta) * uniform + beta * relevance
    return jax.lax.stop_gradient(weights)


def prediction_feature_divisor(delta, feature_reduction):
    if feature_reduction not in ("sum", "mean"):
        raise ValueError("Prediction feature reduction must be sum or mean")
    return delta.shape[-1] if feature_reduction == "mean" else 1


def prediction_output_gradient(delta, target, weights, *, loss_weight, feature_reduction="sum"):
    """Derivative of the batch-mean weighted prediction objective at Delta.

    q and the teacher are detached in the actual objective. Keep diagnostic
    gradients in the same units as that objective, including feature averaging.
    """
    divisor = prediction_feature_divisor(delta, feature_reduction)
    return 2 * loss_weight * weights[..., None] * (delta - target) / (delta.shape[0] * divisor)


def prediction_metrics(delta, target, weights, valid, *, full_diagnostics=True, feature_reduction="sum"):
    """q-weighted squared error; feature averaging never averages q over H again."""
    divisor = prediction_feature_divisor(delta, feature_reduction)
    target = jax.lax.stop_gradient(target)
    mask = valid.astype(jnp.float32)
    count = jnp.maximum(mask.sum(-1), 1)
    error_energy = jnp.square(delta - target).sum(-1)
    target_energy = jnp.square(target).sum(-1)
    error_mean = (mask * error_energy).sum(-1) / count
    target_mean = (mask * target_energy).sum(-1) / count
    if not full_diagnostics:
        return {
            "rapr_prediction_loss": (weights * error_energy).sum(-1) / divisor,
            "rapr_delta_nmse": error_mean / jnp.maximum(target_mean, 1e-8),
            "rapr_q_uniform_l1": jnp.abs(weights - mask / count[..., None]).sum(-1),
        }
    prediction_energy = jnp.square(delta).sum(-1)
    pred_mean = (mask * prediction_energy).sum(-1) / count
    entropy = -(weights * jnp.log(jnp.maximum(weights, 1e-30))).sum(-1)
    cosine_valid = valid & (target_energy > 1e-16) & (prediction_energy > 1e-16)
    cosine = (delta * target).sum(-1) / jnp.maximum(jnp.sqrt(target_energy * prediction_energy), 1e-16)
    def temporal_spread(value):
        center = (value * mask[..., None]).sum(-2) / count[..., None]
        return jnp.sqrt((jnp.square(value - center[:, None]).sum(-1) * mask).sum(-1) / count)
    result = {
        "rapr_prediction_loss": (weights * error_energy).sum(-1) / divisor,
        "rapr_delta_mse": error_mean / delta.shape[-1],
        "rapr_delta_zero_mse": target_mean / delta.shape[-1],
        "rapr_delta_nmse": error_mean / jnp.maximum(target_mean, 1e-8),
        "rapr_target_rms_norm": jnp.sqrt(target_mean),
        "rapr_prediction_rms_norm": jnp.sqrt(pred_mean),
        "rapr_valid_future_count": mask.sum(-1),
        "rapr_weight_entropy": entropy,
        "rapr_weight_effective_horizon": jnp.where(mask.sum(-1) > 0, jnp.exp(entropy), 0),
        "rapr_delta_cosine": (cosine * cosine_valid).sum(-1) / jnp.maximum(cosine_valid.sum(-1), 1),
        "rapr_delta_cosine_valid_count": cosine_valid.sum(-1).astype(jnp.float32),
        "rapr_prediction_temporal_spread": temporal_spread(delta),
        "rapr_target_temporal_spread": temporal_spread(target),
        "rapr_prediction_target_norm_ratio": jnp.sqrt(pred_mean) / jnp.maximum(jnp.sqrt(target_mean), 1e-8),
        "rapr_q_uniform_l1": jnp.abs(weights - mask / count[..., None]).sum(-1),
    }
    for j in range(delta.shape[-2]):
        result[f"rapr_h{j+1}_error_mse"] = mask[:, j] * error_energy[:, j] / delta.shape[-1]
        result[f"rapr_h{j+1}_zero_mse"] = mask[:, j] * target_energy[:, j] / delta.shape[-1]
        result[f"rapr_h{j+1}_valid"] = mask[:, j]
        result[f"rapr_h{j+1}_q_mass"] = weights[:, j]
    return result
