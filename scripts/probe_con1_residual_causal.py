"""Deterministic causal audit of the Con1 residual gate, repeated over batches.

Every variant shares the same observations, flow noise, flow time, cached
prefix, and predicted delta. Only the residual gate alpha (and optionally the
residual output scale) changes. Nothing is trained and nothing is saved except
the JSON report. A difference here is a causal property of the frozen
parameters, not a training-curve artefact.

Each batch is internally paired (identical inputs for every alpha), and the
per-batch differences are then aggregated across batches so the effect can be
quoted with a standard error instead of a single-batch anecdote.

Forward only, no training, no writes other than the JSON report. The preamble
mirrors scripts/train.py exactly (same mesh, checkpoint manager, and restore
call) so the restored parameters match what training actually uses.
"""

import argparse
import dataclasses
import json
import math
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train
from openpi.models import model as _model
from openpi.training import checkpoints, config as configs, data_loader, sharding


def _logit(alpha: float) -> float:
    alpha = min(max(alpha, 1e-30), 1 - 1e-9)
    return math.log(alpha / (1 - alpha))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True, help="existing experiment under checkpoint_base_dir")
    parser.add_argument("--config", default="pi05_libero_con1_three_stage_40k",
                        help="named TrainConfig matching the experiment")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--checkpoint-step", type=int, default=None, help="default: latest")
    args = parser.parse_args()

    jax.config.update("jax_compilation_cache_dir", "/workspace/.cache/con1_jax")

    config = dataclasses.replace(
        configs.get_config(args.config),
        batch_size=args.batch_size,
        num_workers=0,
        wandb_enabled=False,
        exp_name=args.exp_name,
        checkpoint_base_dir="/workspace/artifacts/checkpoints",
        resume=True,
    )
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"batch {config.batch_size} not divisible by {jax.device_count()} devices")
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=config.resume
    )
    if not resuming:
        raise ValueError(f"No checkpoint found under {config.checkpoint_dir}")
    loader = data_loader.create_data_loader(config, sharding=data_shard, shuffle=True)
    _, init_rng = jax.random.split(jax.random.key(config.seed))
    state, _ = train.init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(state)
    state = checkpoints.restore_state(checkpoint_manager, state, loader, step=args.checkpoint_step)
    model_def = state.model_def
    pure = state.params.to_pure_dict()

    def context_fn(p, observation):
        model = nnx.merge(model_def, p)
        model.eval()
        return model._con1_context(observation)

    def velocity_fn(p, observation, x_t, time, context, delta):
        model = nnx.merge(model_def, p)
        model.eval()
        return model._con1_velocity(observation, x_t, time, context, delta)

    pcontext = jax.jit(context_fn, out_shardings=replicated)
    pvelocity = jax.jit(velocity_fn, out_shardings=replicated)

    def variant(alpha, residual_scale=1.0, disable_adapter=False):
        p = dict(pure)
        cross = dict(p["con1_cross_attention"])
        cross["alpha_logit"] = jnp.asarray(_logit(alpha), jnp.float32)
        if residual_scale != 1.0:
            out = dict(cross["out"])
            out["kernel"] = out["kernel"] * residual_scale
            cross["out"] = out
        if disable_adapter and "adapter_out" in cross:
            adapter = dict(cross["adapter_out"])
            adapter["kernel"] = jnp.zeros_like(adapter["kernel"])
            cross["adapter_out"] = adapter
        p["con1_cross_attention"] = cross
        return p

    learned_alpha = float(jax.nn.sigmoid(jnp.asarray(pure["con1_cross_attention"]["alpha_logit"])))
    # `alpha_0` disables the action adapter too, so it is the exact frozen base
    # and serves as the common reference for every other variant.
    variants = [
        ("alpha_0", 0.0, 1.0, True),
        ("alpha_learned", learned_alpha, 1.0, False),
        ("adapter_off", learned_alpha, 1.0, True),
        ("alpha_0.25", 0.25, 1.0, False),
        ("alpha_0.5", 0.5, 1.0, False),
        ("alpha_1.0", 1.0, 1.0, False),
        ("alpha_learned_residual_x10", learned_alpha, 10.0, False),
    ]

    names = [name for name, _, _, _ in variants]
    per_batch = {name: [] for name in names}
    iterator = iter(loader)
    for index in range(args.batches):
        batch = next(iterator)
        observation, actions = batch
        observation = _model.preprocess_observation(None, observation, train=False)
        noise_rng, time_rng = jax.random.split(jax.random.fold_in(jax.random.key(args.seed), index))
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions
        action_valid = jnp.concatenate(
            [jnp.ones_like(observation.con1_future_valid[:, :1]), observation.con1_future_valid[:, :-1]], axis=1)
        action_mask = action_valid[..., None] & (jnp.arange(actions.shape[-1]) < config.model.con1_action_dims)
        action_count = jnp.maximum(action_mask.sum(), 1)
        with sharding.set_mesh(mesh):
            context, delta = pcontext(pure, observation)
            jax.block_until_ready(delta)
        reference_velocity = None
        reference_flow = None
        for name, alpha, scale, disable_adapter in variants:
            with sharding.set_mesh(mesh):
                velocity, aux = pvelocity(
                    variant(alpha, scale, disable_adapter), observation, x_t, time, context, delta)
            error = jnp.where(action_mask, velocity.astype(jnp.float32) - u_t.astype(jnp.float32), 0.0)
            flow = float(jax.device_get(jnp.square(error).sum() / action_count))
            if reference_velocity is None:
                reference_velocity, reference_flow = velocity, flow
                row = {"delta_flow": 0.0, "flow": flow, "action_l2": 0.0,
                       "correction_rms": float(jax.device_get(jnp.sqrt(aux["con1_residual_energy"])))}
            else:
                diff = jnp.where(
                    action_mask, velocity.astype(jnp.float32) - reference_velocity.astype(jnp.float32), 0.0)
                row = {
                    "delta_flow": flow - reference_flow,
                    "flow": flow,
                    "action_l2": float(jax.device_get(jnp.sqrt(jnp.square(diff).sum() / action_count))),
                    "correction_rms": float(jax.device_get(jnp.sqrt(aux["con1_residual_energy"]))),
                }
            per_batch[name].append(row)
        print(json.dumps({"batch": index + 1, "of": args.batches}), flush=True)

    report = {
        "exp_name": args.exp_name,
        "checkpoint_dir": str(config.checkpoint_dir),
        "restored_step": int(state.step),
        "batch_size": args.batch_size,
        "batches": args.batches,
        "seed": args.seed,
        "learned_alpha": learned_alpha,
        "action_dims": int(config.model.con1_action_dims),
        "variants": {},
    }
    for name, alpha, scale, disable_adapter in variants:
        rows = per_batch[name]
        entry = {"alpha": alpha, "residual_scale": scale,
                 "adapter_disabled": bool(disable_adapter), "n": len(rows)}
        for key in ("delta_flow", "flow", "action_l2", "correction_rms"):
            values = np.asarray([row[key] for row in rows], np.float64)
            entry[key] = float(values.mean())
            entry[f"{key}_stderr"] = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
        report["variants"][name] = entry
        print(json.dumps({"variant": name, **{k: v for k, v in entry.items() if k != "alpha"}}), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print("WROTE", out_path, flush=True)


if __name__ == "__main__":
    main()
