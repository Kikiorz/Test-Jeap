import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import transition_guidance


def _inputs():
    action = jax.random.normal(jax.random.key(1), (2, 10, 32))
    state = jax.random.normal(jax.random.key(2), (2, 32))
    transition = jax.random.normal(jax.random.key(3), (2, 64, 48))
    time = jnp.asarray([0.2, 0.8])
    return action, state, transition, time


def test_guidance_starts_at_exact_zero():
    inputs = _inputs()
    model = transition_guidance.TransitionGuidance(
        action_dim=32, state_dim=32, width=64, depth=2, num_heads=4
    )
    variables = model.init(jax.random.key(4), *inputs)
    output = model.apply(variables, *inputs)
    assert output.shape == inputs[0].shape
    np.testing.assert_allclose(output, 0.0, atol=0.0)


def test_no_transition_control_has_identical_parameter_tree():
    inputs = _inputs()
    model = transition_guidance.TransitionGuidance(
        action_dim=32, state_dim=32, width=64, depth=1, num_heads=4
    )
    variables = model.init(jax.random.key(5), *inputs)
    with_transition = model.apply(variables, *inputs, use_transition=True)
    without_transition = model.apply(variables, *inputs, use_transition=False)
    np.testing.assert_allclose(with_transition, without_transition, atol=0.0)


def test_transition_path_receives_gradient_after_output_opens():
    inputs = _inputs()
    model = transition_guidance.TransitionGuidance(
        action_dim=32, state_dim=32, width=64, depth=1, num_heads=4
    )
    variables = model.init(jax.random.key(6), *inputs)
    params = jax.tree.map(lambda value: value, variables["params"])
    params["guidance_output"]["kernel"] = jnp.ones_like(
        params["guidance_output"]["kernel"]
    ) * 0.01

    def output_sum(transition):
        return model.apply({"params": params}, inputs[0], inputs[1], transition, inputs[3]).sum()

    gradient = jax.grad(output_sum)(inputs[2])
    assert np.isfinite(np.asarray(gradient)).all()
    assert float(jnp.linalg.norm(gradient)) > 0.0
