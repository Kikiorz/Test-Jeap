"""Is the direct delta-Z gradient a good direction for the *action*?

The reciprocal Con1 coupling has two directions available at the same point:

  g_action = d(action flow loss)/d(delta_z)     (already computed in training as
            the VJP "sensitivity" that SGR weights by)
  g_mse    = d(||delta_z - delta_z*||^2)/d(delta_z)   (the direct latent target)

Correcting delta_z means stepping along one of them. If g_mse were aligned with
g_action, the direct latent loss would already be the optimal correction and a
learned function of (delta_z_hat, delta_z*) could not add anything. This probe
measures how much of the achievable one-step action-flow reduction each
direction actually buys, at an equal delta_z step norm.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=6)
    parser.add_argument("--target-reduction", type=float, default=0.01,
                        help="Reference step: the action-direction step that predicts this "
                             "fractional flow reduction; the latent direction gets the same "
                             "delta_z step norm (that ratio is exactly the cosine).")
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260912)
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

    def forward(p, observation, action_chunk):
        model = nnx.merge(model_def, p)
        model.eval()
        obs = _model.preprocess_observation(None, observation, train=False)
        context, delta = model._con1_context(obs, action_chunk)
        return context, delta, obs

    def flow_value(p, observation, x_t, time, context, delta, action_mask, count, u_t):
        model = nnx.merge(model_def, p)
        model.eval()
        velocity, _ = model._con1_velocity(observation, x_t, time, context, delta)
        error = jnp.where(action_mask, velocity - u_t, 0.0)
        return jnp.square(error).sum() / count

    def flow_and_grads(p, observation, x_t, time, context, delta, action_mask, count, u_t):
        def flow_of(d):
            return flow_value(p, observation, x_t, time, context, d, action_mask, count, u_t)
        flow, pullback = jax.vjp(flow_of, delta)
        return flow, pullback(jnp.ones_like(flow))[0]

    prun = jax.jit(forward, out_shardings=(replicated, replicated, replicated))
    pgrad = jax.jit(flow_and_grads, out_shardings=(replicated, replicated))
    flow_run = jax.jit(flow_value, out_shardings=replicated)

    rng = np.random.default_rng(args.seed)
    records = []
    iterator = iter(loader)
    for batch_index in range(args.batches):
        observation, actions = next(iterator)
        host = jax.device_get(observation)
        action_chunk = np.asarray(jax.device_get(actions), np.float32)[..., :physical]
        noise_rng, time_rng = jax.random.split(jax.random.key(args.seed + batch_index))
        with sharding.set_mesh(mesh):
            context, delta, obs = prun(params, observation, jax.device_put(action_chunk, replicated))
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions
        valid = np.asarray(host.con1_future_valid, bool)[..., None]
        current = np.asarray(host.con1_current_latent, np.float32)
        future = np.asarray(host.con1_future_latents, np.float32)
        action_valid = np.concatenate([np.ones_like(valid[:, :1]), valid[:, :-1]], axis=1)
        action_mask = action_valid & (np.arange(actions.shape[-1])[None, None, :] < physical)
        count = jnp.asarray(max(float(action_mask.sum()), 1.0))
        target_delta = np.where(valid, future - current[:, None, :], 0.0).astype(np.float32)
        device_action_mask = jax.device_put(action_mask, replicated)
        device_u_t = jax.device_put(u_t, replicated)
        device_time = jax.device_put(time, replicated)
        with sharding.set_mesh(mesh):
            flow, g_action = pgrad(params, obs, x_t, device_time, context, delta,
                                   device_action_mask, count, device_u_t)
        flow = float(flow)
        ga = np.asarray(g_action, np.float32).reshape(-1)
        gm = (2.0 * (np.asarray(delta, np.float32) - target_delta)).reshape(-1)
        delta_flat = np.asarray(delta, np.float32).reshape(-1)
        cos = float(np.dot(gm, ga) / max(np.linalg.norm(gm) * np.linalg.norm(ga), 1e-12))
        best_fraction = float(np.dot(gm, ga) / max(np.dot(ga, ga), 1e-12))
        norm_action = float(np.linalg.norm(ga))
        norm_mse = float(np.linalg.norm(gm))
        # The training latent term is a masked mean over valid positions *and*
        # latent dimensions, so its gradient carries a 1/(count*latent_dim)
        # factor that the raw residual above does not. Report both scales: the
        # training-normalised one is what actually competes with the flow term.
        valid_count = float(valid.sum()) * float(np.asarray(delta, np.float32).shape[-1])
        gm_train = gm / max(valid_count, 1.0)
        norm_mse_train = float(np.linalg.norm(gm_train))
        # First-order efficiency at equal delta_z step norm is the cosine, so one
        # reference step fixes both: the action step hits the target reduction and
        # the latent step is scaled to the same delta_z displacement.
        eta_action = args.target_reduction * flow / max(norm_action ** 2, 1e-30)
        eta_mse = eta_action * norm_action / max(norm_mse, 1e-30)
        rows = {"batch": batch_index, "flow": flow, "cos_mse_vs_action": cos,
                "mse_fraction_of_best": best_fraction,
                "norm_ratio_mse_over_action": norm_mse / max(norm_action, 1e-12),
                "norm_ratio_train_normalised": norm_mse_train / max(norm_action, 1e-30),
                "cos_train_normalised_mse_vs_action": float(
                    np.dot(gm_train, ga) / max(norm_mse_train * norm_action, 1e-30))}
        with sharding.set_mesh(mesh):
            flow_mse = float(flow_run(params, obs, x_t, device_time, context,
                                      delta - eta_mse * jnp.asarray(gm.reshape(delta.shape)),
                                      device_action_mask, count, device_u_t))
            flow_action = float(flow_run(params, obs, x_t, device_time, context,
                                         delta - eta_action * g_action,
                                         device_action_mask, count, device_u_t))
        rows["flow_reduction_mse"] = (flow - flow_mse) / flow
        rows["flow_reduction_action"] = (flow - flow_action) / flow
        records.append(rows)
        print(json.dumps(rows), flush=True)

    summary = {
        "config": args.config,
        "exp_name": args.exp_name,
        "restored_step": int(state.step),
        "samples": args.batches * args.batch_size,
        "mean_cos_mse_vs_action": float(np.mean([r["cos_mse_vs_action"] for r in records])),
        "mean_mse_fraction_of_best": float(np.mean([r["mse_fraction_of_best"] for r in records])),
        "mean_norm_ratio": float(np.mean([r["norm_ratio_mse_over_action"] for r in records])),
        "mean_norm_ratio_train_normalised": float(
            np.mean([r["norm_ratio_train_normalised"] for r in records])),
        "target_reduction": args.target_reduction,
        "measured_flow_reduction": {
            "mse": float(np.mean([r["flow_reduction_mse"] for r in records])),
            "action": float(np.mean([r["flow_reduction_action"] for r in records])),
        },
        "mean_efficiency_mse_over_action": float(
            np.mean([r["flow_reduction_mse"] / max(r["flow_reduction_action"], 1e-12) for r in records])),
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print("RESULT", json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main()
