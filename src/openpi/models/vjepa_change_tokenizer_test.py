import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import vjepa_change_tokenizer as tokenizer


def test_displacement_and_final_teacher_shapes():
    realized = jax.random.normal(jax.random.key(1), (2, 64, 256))
    nochange = jax.random.normal(jax.random.key(2), (2, 64, 256))
    displacement = tokenizer.latent_displacement(realized, nochange)
    model = tokenizer.Stage1Teacher(
        width=64,
        ffn_width=128,
        resampler_depth=1,
        decoder_depth=1,
        num_heads=4,
    )
    variables = model.init(jax.random.key(3), displacement)
    change, action = model.apply(variables, displacement)
    assert change.shape == (2, 16, 128)
    assert action.shape == (2, 10, 7)

    target = jax.random.normal(jax.random.key(4), action.shape)

    def objective(params):
        _, prediction = model.apply({"params": params}, displacement)
        return jnp.mean(jnp.square(prediction - target))

    gradients = jax.grad(objective)(variables["params"])
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(gradients))
    assert any(np.any(np.asarray(leaf) != 0) for leaf in jax.tree.leaves(gradients))


def test_2d_rope_preserves_shape_and_norm():
    value = jax.random.normal(jax.random.key(5), (2, 16, 4, 16))
    positions = tokenizer.square_grid_positions(16)
    rotated = tokenizer.apply_2d_rope(value, positions)
    assert rotated.shape == value.shape
    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(rotated, axis=-1)),
        np.asarray(jnp.linalg.norm(value, axis=-1)),
        rtol=2e-6,
        atol=2e-6,
    )
