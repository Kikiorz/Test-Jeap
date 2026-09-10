import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma, pi0_config


def test_suffix_segments_match_original_scan():
    cfg = gemma.Config(width=16, depth=4, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8)
    module = gemma.Module(configs=[cfg, cfg], embed_dtype="float32", adarms=True)
    weights = module.init(jax.random.key(0), use_adarms=[False, True], method=module.init)
    x = jax.random.normal(jax.random.key(1), (1, 3, 16))
    p = jnp.arange(3)[None]
    (_, _), cache = module.apply(weights, [x, None], positions=p, mask=jnp.ones((1, 3, 3), bool))
    h = jax.random.normal(jax.random.key(2), (1, 2, 16))
    pos = jnp.array([[3, 4]])
    mask = jnp.ones((1, 2, 5), bool)
    cond = jnp.ones((1, 16))
    (_, original), _ = module.apply(weights, [None, h], positions=pos, mask=mask,
                                    adarms_cond=[None, cond], kv_cache=cache)
    lower = module.apply(weights, h, pos, mask, cond, cache, start=0, stop=2,
                         frozen=True, method=module.suffix_segment)
    upper = module.apply(weights, lower, pos, mask, cond, cache, start=2, stop=4,
                         method=module.suffix_segment)
    output = module.apply(weights, upper, cond, method=module.normalize_suffix)
    np.testing.assert_allclose(output, original, atol=1e-6, rtol=1e-6)


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
