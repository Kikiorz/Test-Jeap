"""Deterministic causal audit of the Con1 residual gate on one fixed batch.

Every variant shares the same observations, flow noise, flow time, cached
prefix, and predicted delta. Only the residual gate alpha (and optionally the
residual output scale) changes. Nothing is trained and nothing is saved except
the JSON report. A difference here is a causal property of the frozen
parameters, not a training-curve artefact.

Single GPU, forward only. Run with the repo venv and PYTHONPATH=src.
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
    parser.add_argument("--source", required=True, help="experiment checkpoint directory")
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    if jax.device_count() != 1:
        raise RuntimeError(f"This audit expects exactly one visible device, got {jax.device_count()}")
    jax.config.update("jax_compilation_cache_dir", "/workspace/.cache/con1_jax")

    config = dataclasses.replace(
        configs.get_config("pi05_libero_con1_three_stage_40k"),
        batch_size=args.batch_size,
        num_workers=0,
        wandb_enabled=False,
    )
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    loader = data_loader.create_data_loader(config, sharding=data_shard, shuffle=True, num_batches=1)
    batch = next(iter(loader))
    # Build a concrete template by running the real init (loads the named config's
    # base weights), so restored leaves carry explicit shardings. The checkpoint
    # immediately overwrites it; nothing is trained or saved.
    state, _ = train.init_train_state(config, jax.random.key(config.seed), mesh, resume=False)
    jax.block_until_ready(state)
    source, resuming = checkpoints.initialize_checkpoint_dir(
        args.source, keep_period=1, overwrite=False, resume=True
    )
    if not resuming:
        raise ValueError(f"No checkpoint found under {args.source}")
    state = checkpoints.restore_state(source, state, loader, step=args.checkpoint_step)
    source.close()
    model_def = state.model_def
    pure = state.params.to_pure_dict()

    observation, actions = batch
    observation = _model.preprocess_observation(None, observation, train=False)
    noise_rng, time_rng = jax.random.split(jax.random.key(args.seed))
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
    x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
    u_t = noise - actions

    action_valid = jnp.concatenate(
        [jnp.ones_like(observation.con1_future_valid[:, :1]), observation.con1_future_valid[:, :-1]], axis=1
    )
    action_mask = action_valid[..., None] & (jnp.arange(actions.shape[-1]) < config.model.con1_action_dims)
    action_count = jnp.maximum(action_mask.sum(), 1)

    def context_fn(p):
        model = nnx.merge(model_def, p)
        model.eval()
        return model._con1_context(observation)

    def velocity_fn(p, context, delta):
        model = nnx.merge(model_def, p)
        model.eval()
        return model._con1_velocity(observation, x_t, time, context, delta)

    pcontext = jax.jit(context_fn, out_shardings=replicated)
    pvelocity = jax.jit(velocity_fn, out_shardings=replicated)
    with sharding.set_mesh(mesh):
        context, delta = pcontext(pure)
        jax.block_until_ready(delta)

    def variant(alpha, residual_scale=1.0):
        p = dict(pure)
        cross = dict(p["con1_cross_attention"])
        cross["alpha_logit"] = jnp.asarray(_logit(alpha), jnp.float32)
        if residual_scale != 1.0:
            out = dict(cross["out"])
            out["kernel"] = out["kernel"] * residual_scale
            cross["out"] = out
        p["con1_cross_attention"] = cross
        return p

    learned_alpha = float(jax.nn.sigmoid(jnp.asarray(pure["con1_cross_attention"]["alpha_logit"])))
    variants = [
        ("alpha_0", 0.0, 1.0),
        ("alpha_learned", learned_alpha, 1.0),
        ("alpha_0.25", 0.25, 1.0),
        ("alpha_0.5", 0.5, 1.0),
        ("alpha_1.0", 1.0, 1.0),
        ("alpha_learned_residual_x10", learned_alpha, 10.0),
    ]

    report = {
        "source": args.source,
        "checkpoint_step": args.checkpoint_step,
        "restored_step": int(state.step),
        "batch_size": int(actions.shape[0]),
        "seed": args.seed,
        "learned_alpha": learned_alpha,
        "action_dims": int(config.model.con1_action_dims),
        "valid_action_entries": float(jax.device_get(action_count)),
        "variants": {},
    }

    reference_velocity = None
    reference_flow = None
    for name, alpha, scale in variants:
        with sharding.set_mesh(mesh):
            velocity, aux = pvelocity(variant(alpha, scale), context, delta)
        error = jnp.where(action_mask, velocity.astype(jnp.float32) - u_t.astype(jnp.float32), 0.0)
        flow = float(jax.device_get(jnp.square(error).sum() / action_count))
        row = {
            "alpha": alpha,
            "residual_scale": scale,
            "flow_loss": flow,
            "correction_rms": float(jax.device_get(jnp.sqrt(aux["con1_residual_energy"]))),
            "alpha_reported": float(jax.device_get(aux["con1_alpha"])),
            "velocity_rms": float(jax.device_get(jnp.sqrt(jnp.mean(jnp.square(velocity.astype(jnp.float32)))))),
        }
        if reference_velocity is None:
            reference_velocity, reference_flow = velocity, flow
            row["delta_flow_vs_alpha0"] = 0.0
            row["action_l2_vs_alpha0"] = 0.0
        else:
            diff = jnp.where(action_mask, velocity.astype(jnp.float32) - reference_velocity.astype(jnp.float32), 0.0)
            row["delta_flow_vs_alpha0"] = flow - reference_flow
            row["action_l2_vs_alpha0"] = float(
                jax.device_get(jnp.sqrt(jnp.square(diff).sum() / action_count))
            )
        report["variants"][name] = row
        print(json.dumps({"variant": name, **row}), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print("WROTE", out_path, flush=True)


if __name__ == "__main__":
    main()
