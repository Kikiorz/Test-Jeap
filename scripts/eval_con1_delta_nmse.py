"""Held-out NMSE of the Con2 delta head, with an action-conditioning ablation.

Restores a checkpoint, evaluates the head on a fixed set of batches, and
repeats with the action chunk shuffled across samples or zeroed. Correct actions
versus shuffled actions isolates whether the *content* of the action matters;
zeroed actions measure how much the head leans on the conditioning at all.

NMSE follows the training metric: masked mean squared error over valid entries
divided by the masked mean squared target delta.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train
from openpi.models import model as _model
from openpi.training import checkpoints, config as configs, data_loader, sharding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="named TrainConfig")
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    jax.config.update("jax_compilation_cache_dir", "/workspace/.cache/con1_jax")
    config = dataclasses.replace(
        configs.get_config(args.config),
        batch_size=args.batch_size, num_workers=0, wandb_enabled=False,
        exp_name=args.exp_name, checkpoint_base_dir="/workspace/artifacts/checkpoints", resume=True,
    )
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True)
    if not resuming:
        raise ValueError(f"No checkpoint under {config.checkpoint_dir}")
    loader = data_loader.create_data_loader(config, sharding=data_shard, shuffle=True)
    _, init_rng = jax.random.split(jax.random.key(config.seed))
    state, _ = train.init_train_state(config, init_rng, mesh, resume=True)
    jax.block_until_ready(state)
    state = checkpoints.restore_state(manager, state, loader, step=args.checkpoint_step)
    model_def = state.model_def
    params = state.params
    cond = bool(config.model.con1_action_conditioning)
    physical = int(config.model.con1_action_dims)

    def run(p, observation, action_chunk):
        model = nnx.merge(model_def, p)
        model.eval()
        obs = _model.preprocess_observation(None, observation, train=False)
        # Use the production prefix path so the VLM pool (when enabled) matches
        # exactly what training and sampling feed the head.
        _, tokens, vlm_context = model._con1_prefix(obs)
        delta = model._con1_delta(tokens, obs.con1_current_latent, action_chunk, vlm_context)
        return delta

    prun = jax.jit(run, out_shardings=replicated)
    totals = {k: [0.0, 0.0] for k in ("true", "shuffled", "zero")}
    rng = np.random.default_rng(args.seed)
    iterator = iter(loader)
    for i in range(args.batches):
        observation, actions = next(iterator)
        host = jax.device_get(observation)
        actions = np.asarray(jax.device_get(actions), np.float32)[..., :physical] if cond else None
        target = np.asarray(host.con1_future_latents, np.float32) - np.asarray(
            host.con1_current_latent, np.float32)[:, None, :]
        valid = np.asarray(host.con1_future_valid, bool)[..., None]
        variants = {"true": actions}
        if cond:
            variants["shuffled"] = actions[rng.permutation(actions.shape[0])]
            variants["zero"] = np.zeros_like(actions)
        else:
            variants = {"true": None}
        for name, chunk in variants.items():
            device_chunk = None if chunk is None else jax.device_put(chunk, replicated)
            with sharding.set_mesh(mesh):
                delta = np.asarray(jax.device_get(prun(params, observation, device_chunk)), np.float32)
            err = np.sum(np.where(valid, (delta - target) ** 2, 0.0))
            base = np.sum(np.where(valid, target**2, 0.0))
            totals.setdefault(name, [0.0, 0.0])
            totals[name][0] += float(err)
            totals[name][1] += float(base)
        print(json.dumps({"batch": i + 1, "of": args.batches}), flush=True)

    report = {
        "config": args.config,
        "exp_name": args.exp_name,
        "checkpoint_dir": str(config.checkpoint_dir),
        "restored_step": int(state.step),
        "samples": args.batches * args.batch_size,
        "action_conditioning": cond,
        "nmse": {k: (v[0] / max(v[1], 1e-12)) for k, v in totals.items() if v[1] > 0},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print("RESULT", json.dumps(report["nmse"]), flush=True)
    print("WROTE", out_path, flush=True)


if __name__ == "__main__":
    main()
