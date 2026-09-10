"""Pure, JIT-testable three-stage schedules and exact action-layer masks."""
import jax
import jax.numpy as jnp


def stage_values(step, *, warmup=2000, joint=5000, sensitivity=5000, beta_max=.5):
    step = jnp.asarray(step)
    stage = jnp.where(step < warmup, 1, jnp.where(step < warmup + joint, 2, 3))
    progress = jnp.clip((step - warmup - joint) / max(sensitivity - 1, 1), 0., 1.)
    return stage, beta_max * progress


def flow_weight(step, *, initial=2.0, final=1.0, warmup=2000, decay_steps=15000):
    """High early flow weight that cosinely decays to a still-positive floor."""
    step = jnp.asarray(step)
    progress = jnp.clip((step - warmup) / max(decay_steps - 1, 1), 0., 1.)
    weight = final + 0.5 * (initial - final) * (1.0 + jnp.cos(jnp.pi * progress))
    return jnp.where(step < warmup, initial, weight)


def _names(path):
    return tuple(str(getattr(key, "key", getattr(key, "idx", key))) for key in path)


def mask_action_updates(tree, *, freeze_before=14, depth=18, freeze_all=False):
    """Use on gradients AND final AdamW updates (weight decay must not move base)."""
    if not 0 <= freeze_before <= depth:
        raise ValueError("Invalid action-layer split")

    def mask(path, value):
        names = _names(path)
        if "PaliGemma" in names and "llm" in names and any(n.endswith("_1") for n in names):
            if "layers" in names:
                if value.ndim < 1 or value.shape[0] != depth:
                    raise ValueError("Action scan leaf has unexpected depth")
                value = value.at[:freeze_before].set(0)
            value = jnp.where(freeze_all, jnp.zeros_like(value), value)
        if "action_out_proj" in names:
            value = jnp.where(freeze_all, jnp.zeros_like(value), value)
        return value

    return jax.tree_util.tree_map_with_path(mask, tree)


def scale_group_updates(tree, step, *, warmup=2000, fusion_multiplier=1., action_multiplier=.1):
    """Relative to Adam's 1e-5 LR: head 1e-5, fusion 1e-5/5e-6,
    alpha 1e-6, upper action blocks and output at `action_multiplier`.
    """
    def scale(path, value):
        names = _names(path)
        if "con1_delta_head" in names:
            factor = 1.
        elif "con1_cross_attention" in names:
            factor = .1 if "alpha_logit" in names else fusion_multiplier * jnp.where(step < warmup, 1., .5)
        elif "action_out_proj" in names or ("PaliGemma" in names and "llm" in names):
            factor = action_multiplier
        else:
            raise ValueError(f"Unexpected trainable Con1 parameter: {names}")
        return value * factor
    return jax.tree_util.tree_map_with_path(scale, tree)
