"""Regression checks for JAX metadata reconstruction, not model experiments."""

from flax import struct
import jax
import jax.numpy as jnp
from jaxtyping import TypeCheckError
import pytest

from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class _State:
    value: at.Float[at.ArrayLike, "2"]


def test_concrete_invalid_dtype_is_still_rejected():
    with pytest.raises(TypeCheckError):
        _State(jnp.ones(2, dtype=jnp.int32))


def test_jax_aot_metadata_and_execution():
    state = _State(jnp.ones(2, dtype=jnp.float32))
    function = jax.jit(lambda item: item.replace(value=item.value + 1))
    compiled = function.trace(state).lower().compile()
    assert jnp.all(compiled(state).value == 2)


def test_typechecking_context_restores_after_error():
    previous = at.config.jaxtyping_disable
    with pytest.raises(RuntimeError), at.disable_typechecking():
        raise RuntimeError("intentional")
    assert at.config.jaxtyping_disable == previous
