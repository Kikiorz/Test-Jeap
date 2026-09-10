import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config


def test_actual_pi0_con1_gradient_trace():
    # Trace the actual production NNX path (including VJP/SGR) without allocating
    # a full VLM or running an optimizer. Not a full-size GPU numerical audit.
    config = pi0_config.Pi0Config(pi05=True, use_vjepa_aux=True, use_con1=True,
        paligemma_variant="dummy", action_expert_variant="dummy", con1_train_action_layers_from=0,
        con1_width=8, con1_latent_dim=6, action_horizon=10, max_token_len=8)

    def trace(rng):
        model = config.create(rng)
        obs = dataclasses.replace(config.fake_obs(batch_size=1),
            con1_current_latent=jnp.ones((1, 6)), con1_future_latents=jnp.ones((1, 10, 6)),
            con1_future_valid=jnp.ones((1, 10), bool))
        actions = config.fake_act(batch_size=1)
        loss = lambda m: m.compute_con1_loss(rng, obs, actions, beta=.5)[0]
        return nnx.value_and_grad(loss)(model)

    loss, grad = nnx.eval_shape(trace, jax.random.key(0))
    assert loss.shape == ()
    assert len(grad.con1_delta_head.flat_state()) > 0


def test_plain_head_checkpoint_load(tmp_path, monkeypatch):
    from flax import serialization
    from openpi.training import weight_loaders
    head = {"dense": {"kernel": np.ones((2, 3), np.float32)}}
    path = tmp_path / "head.msgpack"
    path.write_bytes(serialization.msgpack_serialize({"params": head, "step": 20000}))
    reference = {"con1_delta_head": {"dense": {"kernel": np.zeros((2, 3), np.float32)}}}
    monkeypatch.setattr(weight_loaders.CheckpointWeightLoader, "load", lambda self, _: reference)
    loaded = weight_loaders.BaseAndCon1HeadWeightLoader("unused", str(path)).load(reference)
    np.testing.assert_array_equal(loaded["con1_delta_head"]["dense"]["kernel"], 1.)
