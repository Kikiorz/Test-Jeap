#!/usr/bin/env python3
"""Flow loss of the RoboTwin JEPA-WAM base on the official RoboTwin 2.0 data.

The base checkpoint is stored in the training layout, so this goes through
``init_train_state`` (which applies the configured weight loader) and the real
training data pipeline, then measures the flow-matching loss on held-out
episodes. A convention mismatch between the released dataset and the base's
training data shows up directly as an inflated loss.
"""

from __future__ import annotations

import argparse
import dataclasses
import json

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train
from openpi.training import config as configs, data_loader, sharding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_robotwin_con1con2_ctx_20k")
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base = configs.get_config(args.config)
    # The probe measures the *base* policy, so it must not depend on the Con1
    # latent cache (which is only needed to pick the Con1 episode split).
    data = dataclasses.replace(
        base.data,
        con1_latent_root=None,
        base_config=dataclasses.replace(base.data.base_config, con1_latent_root=None))
    config = dataclasses.replace(
        base, data=data,
        model=dataclasses.replace(base.model, use_con1=False, use_con2=False),
        batch_size=args.batch_size, num_workers=0, wandb_enabled=False,
        exp_name="base_loss_probe", resume=False)
    mesh = sharding.make_mesh(config.fsdp_devices)
    _, rng = jax.random.split(jax.random.key(config.seed))
    state, _ = train.init_train_state(config, rng, mesh, resume=False)
    model = nnx.merge(state.model_def, state.params)

    shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = data_loader.create_data_loader(config, sharding=shard, shuffle=True)
    iterator = iter(loader)

    losses = []
    for index in range(args.batches):
        observation, actions = next(iterator)
        out = model.compute_loss(jax.random.fold_in(rng, index), observation, actions, train=True)
        flow = out[0] if isinstance(out, tuple) else out
        losses.append(float(jnp.mean(flow)))
        print(json.dumps({"batch": index, "flow_loss": losses[-1]}), flush=True)

    report = {"config": args.config, "batches": len(losses), "batch_size": args.batch_size,
              "mean_flow_loss": float(np.mean(losses)), "std": float(np.std(losses)),
              "target_dtype": str(actions.dtype)}
    print("RESULT", json.dumps(report, indent=2, sort_keys=True))
    with open(args.out, "w") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
