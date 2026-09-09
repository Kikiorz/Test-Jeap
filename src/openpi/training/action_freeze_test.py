import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.training.action_freeze import ACTION, SCANNED_ACTION, scope_summary, transform_frozen_action
from openpi.training.config import paper_con1_config


def config():
    return paper_con1_config(joint=True, late2=True, direct_delta=True,
                             no_training_reference=True, train_action_last2=True)


def params():
    var = lambda shape: nnx.VariableState(nnx.Param, jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + 1)
    return nnx.State({"PaliGemma": {"llm": {
        "layers": {"q_einsum_1": {"w": var((18, 3))}, "q_einsum": {"w": var((18, 3))}},
        "final_norm_1": {"scale": var((3,))}}},
        "action_in_proj": {"kernel": var((3, 3))}, "action_out_proj": {"kernel": var((3, 3))},
        "time_mlp_in": {"bias": var((3,))}, "rapr_delta_head": {"kernel": var((3, 3))},
        "rapr_router": {"alpha_logit": var(())}})


def test_stop_gradient_preserves_values_and_only_blocks_frozen_action():
    cfg, state = config(), params()
    stopped = transform_frozen_action(cfg, state, stop_gradient=True)
    for a, b in zip(jax.tree.leaves(stopped), jax.tree.leaves(state), strict=True):
        np.testing.assert_array_equal(a, b)
    grad = jax.jit(jax.grad(lambda tree: sum(jnp.sum(x) for x in jax.tree.leaves(
        transform_frozen_action(cfg, tree, stop_gradient=True)))))(state)
    for path, value in grad.flat_state().items():
        expected = np.ones(value.value.shape)
        if ACTION(path, value):
            if SCANNED_ACTION(path, value):
                expected[:16] = 0
            else:
                expected[...] = 0
        np.testing.assert_array_equal(value.value, expected)


def test_adam_momentum_and_weight_decay_cannot_move_frozen_parameters():
    cfg, state = config(), params()
    tx = optax.adamw(1e-3, weight_decay=.2)
    all_grad = jax.tree.map(jnp.ones_like, state)
    # Simulate restored nonzero Adam momentum from whole-expert training.
    _, opt_state = tx.update(all_grad, tx.init(state), state)
    original = state
    for _ in range(3):
        masked_grad = transform_frozen_action(cfg, all_grad)
        raw_updates, opt_state = tx.update(masked_grad, opt_state, state)
        assert np.any(np.asarray(raw_updates["action_out_proj"]["kernel"].value) != 0)
        updates = transform_frozen_action(cfg, raw_updates)
        state = optax.apply_updates(state, updates)
    for path, before in original.flat_state().items():
        after = state.flat_state()[path].value
        if SCANNED_ACTION(path, before):
            np.testing.assert_array_equal(after[:16], before.value[:16])
            assert np.all(np.asarray(after[16:]) != np.asarray(before.value[16:]))
        elif ACTION(path, before):
            np.testing.assert_array_equal(after, before.value)
    assert float(state["rapr_router"]["alpha_logit"].value) != float(original["rapr_router"]["alpha_logit"].value)


def test_config_preserves_optimizer_tree_dtype_policy_and_schedule():
    old = paper_con1_config(joint=True, late2=True, direct_delta=True, no_training_reference=True)
    new = config()
    for key in ("model", "lr_schedule", "optimizer", "data", "batch_size", "fsdp_devices",
                "num_train_steps", "training_step_offset", "freeze_filter"):
        assert getattr(new, key) == getattr(old, key)
    assert new.exp_name != old.exp_name
    assert new.keep_steps == (3000, 4999, 9999, 14999)
    assert new.rapr_action_freeze_anchor.endswith("/3000/params")
    scope = scope_summary(new, params())
    assert scope["effective_trainable_action_parameters"] == 6
    assert scope["active_action_blocks"] == [16, 17]
    assert scope["con1_and_alpha_parameters"] == 10
    with pytest.raises(ValueError, match="joint direct-delta"):
        paper_con1_config(joint=False, train_action_last2=True)
    assert transform_frozen_action(old, params()) is not None


def test_unexpected_scanned_shape_fails_closed():
    tree = params()
    tree["PaliGemma"]["llm"]["layers"]["q_einsum_1"]["w"].value = jnp.ones((17, 3))
    with pytest.raises(ValueError, match="Unexpected scanned"):
        transform_frozen_action(config(), tree)


def test_restored_optional_none_slots_are_not_counted_as_parameters():
    tree = params()
    tree["optional_slot"] = nnx.VariableState(nnx.Param, None)
    tree["action_time_mlp_out"] = {"bias": nnx.VariableState(nnx.Param, None)}
    assert scope_summary(config(), tree) == scope_summary(config(), params())
    for stop in (False, True):
        changed = transform_frozen_action(config(), tree, stop_gradient=stop)
        assert changed["optional_slot"].value is None
        assert changed["action_time_mlp_out"]["bias"].value is None
