"""Bounded LR probe: preserve Adam state; evaluate identical held-out inputs/noise.

Not a LIBERO rollout. The loader restarts its shuffled order on resume, as in
train.py; results must not be described as a paired comparison to the old run.
"""
import argparse
import dataclasses
import functools
import json
import logging
from pathlib import Path
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train
from openpi.training import checkpoints, config as configs, data_loader, sharding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=1000)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--fusion-multiplier", type=float, default=3.)
    parser.add_argument("--validation-batches", type=int, default=4)
    args = parser.parse_args()
    if args.steps <= 0 or args.validation_batches <= 0 or args.fusion_multiplier <= 0:
        raise ValueError("Probe counts and multiplier must be positive")
    logging.basicConfig(level=logging.INFO)
    jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))
    config = dataclasses.replace(
        configs.get_config("pi05_libero_con1_three_stage_40k"),
        exp_name=args.exp_name, checkpoint_base_dir="/workspace/artifacts/checkpoints",
        batch_size=128, num_workers=0, wandb_enabled=False,
        con1_cross_attention_lr_multiplier=args.fusion_multiplier,
    )
    if jax.device_count() != 4:
        raise RuntimeError(f"Expected four GPUs, got {jax.device_count()}")
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    manager, _ = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=1, overwrite=False, resume=False)
    loader = data_loader.create_data_loader(config, sharding=data_shard, shuffle=True)
    train_rng, init_rng = jax.random.split(jax.random.key(config.seed))
    state, state_shard = train.init_train_state(config, init_rng, mesh, resume=True)
    source, resuming = checkpoints.initialize_checkpoint_dir(
        args.source, keep_period=1, overwrite=False, resume=True)
    if not resuming:
        raise ValueError("Source has no checkpoint")
    state = checkpoints.restore_state(source, state, loader, step=args.checkpoint_step)
    source.close()
    start = int(state.step)
    if start + args.steps > config.model.con1_stage1_steps:
        raise ValueError("This probe must stay entirely in stage 1")

    val_data = dataclasses.replace(loader.data_config(), con1_split="validation")
    val_loader = data_loader.create_torch_data_loader(
        val_data, config.model, config.model.action_horizon, config.batch_size,
        sharding=data_shard, shuffle=True, num_batches=args.validation_batches,
        num_workers=0, seed=20260910,
    )
    # Keep the exact transformed examples on CPU; don't retain validation GPU buffers.
    val_batches = [jax.device_get(batch) for batch in val_loader]
    model_def = state.model_def

    def evaluate(params, rng, batch):
        model = nnx.merge(model_def, params)
        model.eval()
        loss, info = model.compute_con1_loss(rng, *batch, beta=0.)
        return dict(info, loss=loss)

    peval = jax.jit(evaluate, in_shardings=(state_shard.params, replicated, data_shard),
                    out_shardings=replicated)

    def validate(label):
        rows = []
        for i, host_batch in enumerate(val_batches):
            batch = jax.device_put(host_batch, data_shard)
            rng = jax.random.fold_in(jax.random.key(701), i)
            with sharding.set_mesh(mesh):
                row = jax.device_get(peval(state.params, rng, batch))
            rows.append({k: float(v) for k, v in row.items()})
        result = dict(label=label, completed_updates=int(state.step), unix_time=time.time(),
                      per_batch=rows, mean={k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
        if not all(np.isfinite(v) for v in result["mean"].values()):
            raise FloatingPointError(result)
        with (config.checkpoint_dir / "validation.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
        print("VALIDATION", json.dumps(result), flush=True)

    manifest = dict(source=args.source, checkpoint_step=args.checkpoint_step,
                    restored_updates=start, additional_steps=args.steps, batch_size=128,
                    fusion_lr=1e-5 * args.fusion_multiplier, head_lr=1e-5, alpha_lr=1e-6,
                    action_expert="frozen", validation_examples=len(val_batches)*128,
                    validation_split_seed=42, validation_shuffle_seed=20260910,
                    validation_noise_seed=701, data_iterator_restored=False)
    (config.checkpoint_dir / "probe_manifest.json").write_text(json.dumps(manifest, indent=2))
    validate("before")
    ptrain = jax.jit(functools.partial(train.train_step, config),
                     in_shardings=(replicated, state_shard, data_shard),
                     out_shardings=(state_shard, replicated), donate_argnums=(1,))
    iterator = iter(loader)
    infos = []
    started = time.time()
    for i in range(args.steps):
        batch = next(iterator)
        with sharding.set_mesh(mesh):
            state, info = ptrain(train_rng, state, batch)
        infos.append(info)
        if (i+1) % 10 == 0 or i+1 == args.steps:
            rows = jax.device_get(infos)
            values = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
            if not all(np.isfinite(v) for v in values.values()):
                raise FloatingPointError(values)
            values.update(completed_updates=int(state.step), probe_updates=i+1,
                          unix_time=time.time(), seconds_per_update=(time.time()-started)/len(infos))
            with (config.checkpoint_dir / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(values)+"\n")
            print("TRAIN", json.dumps(values), flush=True)
            infos = []
            started = time.time()
        if (i+1) % 100 == 0 or i+1 == args.steps:
            validate(f"after_{i+1}")
            started = time.time()
    checkpoints.save_state(manager, state, loader, int(state.step)-1)
    manager.wait_until_finished()
    manager.close()
    print("PROBE_COMPLETE", int(state.step), flush=True)


if __name__ == "__main__":
    main()
