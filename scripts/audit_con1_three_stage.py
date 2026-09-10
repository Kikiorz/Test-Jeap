"""Real official-40k, four-device, disposable update audit. Never saves weights."""
import dataclasses
import functools
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

import train
from openpi.training import config as configs, data_loader, sharding

REPORT = Path('/workspace/artifacts/con1/three_stage_preflight.json')


def emit(event, **kwargs):
    print(json.dumps(dict(event=event, unix_time=time.time(), **kwargs)), flush=True)


def difference(before, after):
    old = before.flat_state()
    new = after.flat_state()
    results = {}
    for path, a in old.items():
        b = new[path]
        av, bv = a.value, b.value
        if path[0] == 'con1_delta_head':
            group = 'head'
        elif path[0] == 'con1_cross_attention':
            group = 'cross'
        elif path[0] == 'action_out_proj':
            group = 'action_out'
        elif path[0] == 'PaliGemma' and 'layers' in path and any(str(p).endswith('_1') for p in path):
            assert av.shape[0] == 18
            results['lower_action'] = max(results.get('lower_action', 0),
                float(jnp.max(jnp.abs(av[:14].astype(jnp.float32) - bv[:14].astype(jnp.float32)))))
            av, bv, group = av[14:], bv[14:], 'upper_action'
        else:
            group = 'frozen_other'
        error = float(jnp.max(jnp.abs(av.astype(jnp.float32) - bv.astype(jnp.float32))))
        assert np.isfinite(error), (path, error)
        results[group] = max(results.get(group, 0), error)
    return results


def main():
    train.init_logging()
    assert len(jax.devices()) == 4, jax.devices()
    jax.config.update('jax_compilation_cache_dir', '/workspace/.cache/con1_jax')
    config = dataclasses.replace(configs.get_config('pi05_libero_con1_three_stage_40k'), num_workers=0)
    mesh = sharding.make_mesh(config.fsdp_devices)
    ds = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    rs = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    emit('loading_real_batch', devices=len(jax.devices()), batch_size=config.batch_size)
    loader = data_loader.create_data_loader(config, sharding=ds, shuffle=True)
    batch = next(iter(loader))
    assert batch[0].con1_future_valid is not None
    emit('loading_official_base_and_head')
    state, ss = train.init_train_state(config, jax.random.key(42), mesh, resume=False)
    jax.block_until_ready(state)
    update = jax.jit(functools.partial(train.train_step, config),
        in_shardings=(rs, ss, ds), out_shardings=(ss, rs))
    records = []
    for step in (0, 1, 2000, 11999):
        state = dataclasses.replace(state, step=jnp.asarray(step))
        emit('update_start', step=step)
        start = time.time()
        with sharding.set_mesh(mesh):
            new, metrics = update(jax.random.key(43), state, batch)
        jax.block_until_ready(new)
        metrics = {k: float(v) for k, v in metrics.items()}
        assert all(np.isfinite(v) for v in metrics.values()), metrics
        delta = difference(state.params, new.params)
        assert delta['frozen_other'] == 0 and delta['lower_action'] == 0, delta
        assert delta['head'] > 0 and delta['cross'] > 0, delta
        if step < 2000:
            assert delta['upper_action'] == 0 and delta['action_out'] == 0, delta
        else:
            assert delta['upper_action'] > 0 and delta['action_out'] > 0, delta
        record = dict(step=step, seconds=time.time()-start, metrics=metrics, parameter_max_changes=delta)
        records.append(record)
        emit('update_verified', **record)
        state = new
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(dict(passed=True, records=records,
        disposable_updates=True, production_weights_saved=False), indent=2))
    emit('preflight_passed', report=str(REPORT))


if __name__ == '__main__':
    main()
