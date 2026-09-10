"""Action ranking for the in-policy Con2 delta head (the pre-ACG design).

``scripts/probe_ac_action_ranking.py`` asks the V-JEPA 2-AC world model which of
a set of candidate action chunks best explains the observed future latent. The
original Con2 head was never scored that way - only with NMSE against a
copy-current baseline, where action conditioning is nearly invisible (+0.002).

This script runs the identical protocol on the in-policy delta head, so the two
Con2 designs can be compared on one axis: how often does the model rank the
action chunk that was actually executed above its perturbations?
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


def build_candidates(actions: np.ndarray, rng, count: int, scale: float) -> list[np.ndarray]:
    """Executed chunk first, then zeroed / shuffled / reversed / perturbed chunks."""
    options = [actions, np.zeros_like(actions), actions[rng.permutation(len(actions))],
               actions[:, ::-1].copy()]
    while len(options) < count:
        options.append(actions + rng.normal(0.0, scale, size=actions.shape).astype(np.float32))
    return options[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260911)
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
    model_def, params = state.model_def, state.params
    physical = int(config.model.con1_action_dims)
    if not config.model.con1_action_conditioning:
        raise ValueError("Ranking needs the action-conditioned head")

    def run(p, observation, action_chunk):
        model = nnx.merge(model_def, p)
        model.eval()
        obs = _model.preprocess_observation(None, observation, train=False)
        _, tokens, vlm_context = model._con1_prefix(obs)
        return model._con1_delta(tokens, obs.con1_current_latent, action_chunk, vlm_context)

    prun = jax.jit(run, out_shardings=replicated)
    rng = np.random.default_rng(args.seed)
    ranks = {h: [] for h in args.horizons}
    gaps = {h: [] for h in args.horizons}
    nmse = {h: [0.0, 0.0] for h in args.horizons}
    scale = 0.1
    iterator = iter(loader)
    for batch_index in range(args.batches):
        observation, actions = next(iterator)
        host = jax.device_get(observation)
        executed = np.asarray(jax.device_get(actions), np.float32)[..., :physical]
        scale = float(executed.std())
        future = np.asarray(host.con1_future_latents, np.float32)
        current = np.asarray(host.con1_current_latent, np.float32)
        valid = np.asarray(host.con1_future_valid, bool)[..., None]
        candidates = build_candidates(executed, rng, args.candidates, scale)
        energies = {h: [] for h in args.horizons}
        for chunk in candidates:
            with sharding.set_mesh(mesh):
                delta = np.asarray(jax.device_get(
                    prun(params, observation, jax.device_put(chunk, replicated))), np.float32)
            predicted = current[:, None, :] + delta
            for h in args.horizons:
                error = np.where(valid[:, h - 1], (predicted[:, h - 1] - future[:, h - 1]) ** 2, 0.0)
                energies[h].append(error.sum(-1))
        for h in args.horizons:
            stacked = np.stack(energies[h])  # [candidates, batch]
            order = np.argsort(stacked, axis=0)
            rank = np.argmax(order == 0, axis=0)
            ranks[h].extend(rank.tolist())
            best = stacked[order[0], np.arange(stacked.shape[1])]
            gaps[h].extend(((best - stacked[0]) / np.maximum(np.abs(stacked[0]), 1e-9)).tolist())
            nmse[h][0] += float(stacked[0].sum())
            nmse[h][1] += float(np.where(valid[:, h - 1], (future[:, h - 1] - current) ** 2, 0.0).sum())
        print(json.dumps({"batch": batch_index + 1, "of": args.batches}), flush=True)

    report = {
        "config": args.config,
        "exp_name": args.exp_name,
        "restored_step": int(state.step),
        "samples": args.batches * args.batch_size,
        "candidates": args.candidates,
        "chance_top1": 1.0 / args.candidates,
        "per_horizon": {
            f"horizon_{h}": {
                "top1_rate": float((np.asarray(ranks[h]) == 0).mean()),
                "mean_rank_percentile": float((np.asarray(ranks[h]) / (args.candidates - 1)).mean()),
                "nmse_vs_copy_current": nmse[h][0] / max(nmse[h][1], 1e-12),
                "windows": len(ranks[h]),
            }
            for h in args.horizons
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("RESULT", json.dumps(report["per_horizon"]), flush=True)


if __name__ == "__main__":
    main()
