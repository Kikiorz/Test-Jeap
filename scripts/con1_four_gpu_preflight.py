"""Check real collectives and head updates before a costly cache rebuild."""
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.con1.modules import AnchoredDeltaHead, anchored_loss


def main():
    devices = jax.local_devices()
    assert len(devices) == 4, devices
    started = time.time()
    collective = jax.pmap(lambda x: jax.lax.psum(x, 'd'), axis_name='d')
    values = jnp.arange(4, dtype=jnp.float32)[:, None] * jnp.ones((4, 1024 * 1024))
    for _ in range(5):
        np.testing.assert_array_equal(np.asarray(collective(values)), 6.)
    model = AnchoredDeltaHead(10, 2816, 512)
    r = jax.random.normal(jax.random.key(1), (128, 64, 2048))
    z = jax.random.normal(jax.random.key(2), (128, 2816))
    target = z[:, None, :] + 0.1
    target = jnp.broadcast_to(target, (128, 10, 2816))
    mask = jnp.ones((128, 10), dtype=bool)
    weights = model.init(jax.random.key(42), r[:1], z[:1])['params']
    tx = optax.adamw(5e-5)

    def loss(p, r, z, target, mask):
        delta = model.apply({'params': p}, r, z)['delta']
        return anchored_loss(delta, z, target, mask)[0]

    def step(p, state, r, z, target, mask):
        value, grad = jax.value_and_grad(loss)(p, r, z, target, mask)
        grad = jax.lax.pmean(grad, 'd')
        updates, state = tx.update(grad, state, p)
        return optax.apply_updates(p, updates), state, jax.lax.pmean(value, 'd'), optax.global_norm(grad)

    update = jax.pmap(step, axis_name='d')
    p = jax.device_put_replicated(weights, devices)
    state = jax.device_put_replicated(tx.init(weights), devices)
    batches = [v.reshape((4, 32) + v.shape[1:]) for v in (r, z, target, mask)]
    records = []
    for i in range(5):
        p, state, value, norm = update(p, state, *batches)
        record = {'step': i+1, 'loss': float(value[0]), 'grad_norm': float(norm[0])}
        assert all(np.isfinite(x) for x in record.values()), record
        assert record['grad_norm'] > 0, record
        records.append(record)
    for leaf in jax.tree.leaves(p):
        arr = np.asarray(leaf)
        assert np.isfinite(arr).all()
        for rank in range(1, 4):
            np.testing.assert_allclose(arr[0], arr[rank], rtol=1e-5, atol=1e-6)
    assert records[-1]['loss'] < records[0]['loss'], records
    print(json.dumps({'passed': True, 'devices': list(map(str, devices)),
        'jax': jax.__version__, 'nccl_preload': os.getenv('LD_PRELOAD'),
        'seconds': time.time()-started, 'head_updates': records}), flush=True)


if __name__ == '__main__':
    main()
