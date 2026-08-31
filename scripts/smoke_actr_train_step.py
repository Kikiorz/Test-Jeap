#!/usr/bin/env python3
"""Compile and execute a few real ACTR optimizer steps without saving a checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json

import jax
import jax.numpy as jnp
import numpy as np

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import sharding
from scripts import train as _train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_libero_actr_stage1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = dataclasses.replace(
        _config.get_config(args.config),
        batch_size=args.batch_size,
        num_workers=0,
        wandb_enabled=False,
    )
    if config.batch_size % jax.device_count():
        raise ValueError("Smoke-test batch size must be divisible by the visible device count")

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=args.steps,
    )
    batches = iter(loader)
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    state, state_sharding = _train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)
    step = jax.jit(
        functools.partial(_train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )

    metrics = []
    for _ in range(args.steps):
        with sharding.set_mesh(mesh):
            state, info = step(train_rng, state, next(batches))
        info = jax.device_get(info)
        if not all(np.isfinite(np.asarray(value)).all() for value in info.values()):
            raise FloatingPointError(f"Non-finite smoke metrics: {info}")
        metrics.append({key: float(np.asarray(value)) for key, value in info.items()})

    gates = {}
    for path, value in state.params.flat_state().items():
        if "actr" in str(path) and "gate_" in str(path):
            gates[str(path)] = float(np.asarray(jax.device_get(value.value)))
    print(json.dumps({"status": "PASS", "steps": metrics, "gates": gates}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
