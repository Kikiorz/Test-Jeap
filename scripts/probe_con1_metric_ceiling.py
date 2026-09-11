"""Ceiling of an action-aligned latent metric: can any (linear) M help Con1?

The head's latent term currently uses the Euclidean metric, whose gradient is
orthogonal to the action-improving direction (cos ~ -0.006). The proposed fix is
a learned metric M on the same residual r = delta_z - delta_z*, whose gradient
is 2 M r. At equal delta_z step norm, the fraction of the achievable action-flow
reduction that such a direction buys is exactly

    cos( M r , g_action )

so this probe estimates that ceiling directly: fit M on training episodes,
measure the per-sample cosine on held-out episodes, for a hierarchy of families
(identity, diagonal, low rank, full ridge). If even the best held-out cosine is
near zero, no metric can fix the latent objective and the direction problem has
to be solved somewhere else.
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


def per_sample_cosine(predict, target):
    predict = predict / np.maximum(np.linalg.norm(predict, axis=1, keepdims=True), 1e-30)
    target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-30)
    return float(np.mean(np.sum(predict * target, axis=1)))


def fit_diagonal(r, g):
    numerator = np.sum(r * g, axis=0)
    denominator = np.maximum(np.sum(r * r, axis=0), 1e-30)
    return numerator / denominator


def fit_low_rank(r, g, rank):
    """M = argmin ||M r - g||_F^2 restricted to rank `rank`."""
    # Work in the N x N Gram space: N (samples) << D (latent dim), so this is
    # ~D/N times cheaper than an SVD of the D x D covariance and numerically
    # equivalent for the truncated solution.
    gram = r.T @ r + 1e-6 * np.eye(r.shape[1])
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:rank]
    values = np.maximum(eigenvalues[order], 1e-12)
    basis = eigenvectors[:, order]
    # U = R V / S  (right singular vectors scaled), then M = (G V) (S^-2) U^T
    u = r @ basis / values
    gv = g @ basis
    return (gv / values) @ u.T


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--collect-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-batches", type=int, default=12)
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 16, 64])
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260914)
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

    def collect(p, observation, action_chunk, padded_actions, action_mask, count, noise_rng, time_rng):
        model = nnx.merge(model_def, p)
        model.eval()
        obs = _model.preprocess_observation(None, observation, train=False)
        # The head's conditioning takes the 7-d chunk; the flow path takes the
        # model-width action tensor (the loader pads to 32), which is what
        # embed_suffix/action_in_proj expect.
        context, delta = model._con1_context(obs, action_chunk)
        noise = jax.random.normal(noise_rng, padded_actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, padded_actions.shape[:-2]) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * padded_actions
        u_t = noise - padded_actions

        def flow_of(d):
            velocity, _ = model._con1_velocity(obs, x_t, time, context, d)
            error = jnp.where(action_mask, velocity - u_t, 0.0)
            # Per-sample normaliser, then a batch mean: the VJP is therefore
            # already the per-sample action gradient.
            return jnp.mean(jnp.square(error).sum(axis=(1, 2)) / count)

        flow, pullback = jax.vjp(flow_of, delta)
        g_action = pullback(jnp.ones_like(flow))[0]
        return delta, g_action, obs.con1_current_latent, obs.con1_future_latents

    run = jax.jit(collect, out_shardings=(replicated, replicated, replicated, replicated))
    iterator = iter(loader)
    residuals, gradients, flags = [], [], []
    for batch_index in range(args.collect_batches):
        observation, actions = next(iterator)
        host = jax.device_get(observation)
        padded = np.asarray(jax.device_get(actions), np.float32)
        chunk = padded[..., :physical]
        mask = (np.concatenate([np.ones_like(host.con1_future_valid[:, :1]),
                                host.con1_future_valid[:, :-1]], axis=1)[..., None]
                & (np.arange(padded.shape[-1])[None, None, :] < physical))
        count = jnp.asarray(np.maximum(mask.sum(axis=(1, 2)), 1))
        valid = np.asarray(host.con1_future_valid, bool)[..., None]
        action_valid = np.concatenate([np.ones_like(valid[:, :1]), valid[:, :-1]], axis=1)
        noise_rng, time_rng = jax.random.split(jax.random.key(args.seed + batch_index))
        with sharding.set_mesh(mesh):
            delta, g_action, current, future = run(
                params, observation, jax.device_put(chunk, replicated),
                jax.device_put(padded, replicated),
                jax.device_put(mask, replicated), count, noise_rng, time_rng)
        delta_np = np.asarray(delta, np.float32)
        grad_np = np.asarray(g_action, np.float32)
        current_np = np.asarray(current, np.float32)
        future_np = np.asarray(future, np.float32)
        for sample in range(len(chunk)):
            keep = valid[sample, :, 0] & action_valid[sample, :, 0]
            # Only windows whose horizons are all valid: a ragged tail would make
            # the per-sample feature vectors different lengths, and the fit needs
            # one row per sample with a consistent horizon layout.
            if not keep.all():
                continue
            residual = (delta_np[sample] - (future_np[sample] - current_np[sample][None]))[keep].reshape(-1)
            residuals.append(residual)
            gradients.append(grad_np[sample][keep].reshape(-1))
            flags.append(batch_index)
        print(json.dumps({"collected_batch": batch_index + 1, "of": args.collect_batches}), flush=True)

    cache_path = Path(args.out).with_suffix(".npz")
    if cache_path.exists():
        # The collection is the expensive part (one full policy forward per
        # sample); the fits are cheap with the Gram-space formulation, so a
        # cached collection lets the fit be re-run freely.
        cached = np.load(cache_path)
        r, g, batch_id = cached["r"], cached["g"], cached["batch_id"]
    else:
        r = np.stack(residuals).astype(np.float64)
        g = np.stack(gradients).astype(np.float64)
        batch_id = np.asarray(flags)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, r=r, g=g, batch_id=batch_id)
    train_mask = batch_id < args.train_batches
    val_mask = (batch_id >= args.train_batches) & (batch_id < args.train_batches + args.val_batches)
    test_mask = batch_id >= args.train_batches + args.val_batches

    report = {"config": args.config, "exp_name": args.exp_name, "restored_step": int(state.step),
              "samples": {k: int(v.sum()) for k, v in
                          (("train", train_mask), ("val", val_mask), ("test", test_mask))},
              "baseline_cosine_euclidean": {
                  "val": per_sample_cosine(r[val_mask], g[val_mask]),
                  "test": per_sample_cosine(r[test_mask], g[test_mask])},
              "families": {}}

    diagonal = fit_diagonal(r[train_mask], g[train_mask])
    report["families"]["diagonal"] = {
        "val": per_sample_cosine(r[val_mask] * diagonal, g[val_mask]),
        "test": per_sample_cosine(r[test_mask] * diagonal, g[test_mask]),
        "scale_stats": {"min": float(diagonal.min()), "max": float(diagonal.max()),
                        "mean": float(diagonal.mean())}}
    for rank in args.ranks:
        m = fit_low_rank(r[train_mask].T, g[train_mask].T, rank)
        report["families"][f"low_rank_{rank}"] = {
            "val": per_sample_cosine((m @ r[val_mask].T).T, g[val_mask]),
            "test": per_sample_cosine((m @ r[test_mask].T).T, g[test_mask])}
        print(json.dumps({"fitted": f"low_rank_{rank}"}), flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    np.savez(out_path.with_suffix(".npz"), r=r, g=g, batch_id=batch_id)
    print("RESULT", json.dumps({k: v for k, v in report.items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
