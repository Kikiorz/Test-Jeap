#!/usr/bin/env python3
"""Which parameters a Con1 run actually changed?

Compares the freshly initialised state (base checkpoint + Con1 head, before any
training) with the trained checkpoint, and prints the maximum absolute change
grouped by parameter subtree. This is the ground truth for any claim about
"frozen layers", independent of what the freeze filter was meant to do.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import train
from openpi.training import checkpoints, config as configs, data_loader, sharding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    args = parser.parse_args()

    config = configs.get_config(args.config)
    config = dataclasses.replace(
        config, num_workers=0, wandb_enabled=False, exp_name=args.exp_name,
        checkpoint_base_dir="/workspace/artifacts/checkpoints", resume=True)
    mesh = sharding.make_mesh(config.fsdp_devices)
    _, init_rng = jax.random.split(jax.random.key(config.seed))
    initial, _ = train.init_train_state(config, init_rng, mesh, resume=False)
    del init_rng
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True)
    if not resuming:
        raise ValueError(f"No checkpoint under {config.checkpoint_dir}")
    loader = data_loader.create_data_loader(
        config, sharding=jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)), shuffle=False)
    trained = checkpoints.restore_state(manager, initial, loader, step=args.checkpoint_step)
    jax.block_until_ready(trained.params)

    flat_initial = jax.tree_util.tree_flatten_with_path(initial.params)[0]
    flat_trained = jax.tree_util.tree_flatten_with_path(trained.params)[0]
    grouped = collections.defaultdict(float)
    changed = 0
    total = 0
    for (path, before), (path2, after) in zip(flat_initial, flat_trained):
        assert path == path2
        names = [str(getattr(p, "key", getattr(p, "name", p))) for p in path]
        key = "/".join(names[:3])
        delta = float(jnp.max(jnp.abs(jnp.asarray(before) - jnp.asarray(after))))
        grouped[key] = max(grouped[key], delta)
        total += 1
        if delta > 0:
            changed += 1
    print(f"tensors total={total} changed={changed}")
    for key in sorted(grouped, key=lambda k: -grouped[k]):
        marker = "CHANGED" if grouped[key] > 0 else "frozen "
        print(f"{marker} max|d|={grouped[key]:.6g}  {key}")


if __name__ == "__main__":
    main()
