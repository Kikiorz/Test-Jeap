"""Conditioning ablation for the Con2 future-latent predictor.

The anchored delta head only receives the last `vjepa_num_queries` prefix tokens
plus z_t. This probe asks how much the prediction improves when the full PI0.5
prefix (image patches, language, state) is available, so we can separate a
head-optimisation problem from an input-bottleneck problem.

Collected per sample: z_t, the query tokens R_t, and a mask-weighted mean of
every other prefix token. Kernel-ridge probes in the full 2816-d output space
are fit on a held-out split with lambda chosen on a validation split, using the
same NMSE definition as the training metric.
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
from openpi.models.pi0 import make_attn_mask
from openpi.training import checkpoints, config as configs, data_loader, sharding


def _nmse(pred, target):
    err = np.sum((pred - target) ** 2)
    base = np.sum(target**2)
    return float(err / max(base, 1e-12))


def _fit_ridge(train_x, train_y, val_x, val_y, eval_x, eval_y, lambdas, proj_dim, seed):
    """Standardise, random-project to `proj_dim`, then exact ridge.

    The projection keeps the normal equations small (n >> proj_dim) while
    retaining all rows, so every valid (sample, horizon) pair is used.
    """
    mean = train_x.mean(0, keepdims=True)
    std = train_x.std(0, keepdims=True) + 1e-6
    tx = ((train_x - mean) / std).astype(np.float32)
    vx = ((val_x - mean) / std).astype(np.float32)
    ex = ((eval_x - mean) / std).astype(np.float32)
    if tx.shape[1] > proj_dim:
        proj = np.random.default_rng(seed).normal(
            0.0, 1.0 / np.sqrt(proj_dim), size=(tx.shape[1], proj_dim)
        ).astype(np.float32)
        tx, vx, ex = tx @ proj, vx @ proj, ex @ proj
    gram = tx @ tx.T
    rhs = train_y.astype(np.float32)
    best = None
    for lam in lambdas:
        alpha = np.linalg.solve(gram.astype(np.float64) + lam * np.eye(gram.shape[0]), rhs.astype(np.float64))
        score = _nmse(vx @ (tx.T @ alpha), val_y)
        if best is None or score < best[0]:
            best = (score, lam, alpha)
    _, lam, alpha = best
    return _nmse(ex @ (tx.T @ alpha), eval_y), float(lam)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--proj-dim", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    jax.config.update("jax_compilation_cache_dir", "/workspace/.cache/con1_jax")
    rng = np.random.default_rng(args.seed)
    config = dataclasses.replace(
        configs.get_config("pi05_libero_con1_three_stage_40k"),
        batch_size=args.batch_size,
        num_workers=0,
        wandb_enabled=False,
        exp_name=args.exp_name,
        checkpoint_base_dir="/workspace/artifacts/checkpoints",
        resume=True,
    )
    if config.batch_size % jax.device_count() != 0:
        raise ValueError("batch not divisible by devices")
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True
    )
    if not resuming:
        raise ValueError(f"No checkpoint under {config.checkpoint_dir}")
    loader = data_loader.create_data_loader(config, sharding=data_shard, shuffle=True)
    _, init_rng = jax.random.split(jax.random.key(config.seed))
    state, _ = train.init_train_state(config, init_rng, mesh, resume=True)
    jax.block_until_ready(state)
    state = checkpoints.restore_state(manager, state, loader, step=args.checkpoint_step)
    model_def = state.model_def
    params = state.params

    def collect(p, observation):
        model = nnx.merge(model_def, p)
        model.eval()
        obs = _model.preprocess_observation(None, observation, train=False)
        tokens, mask, ar_mask = model.embed_prefix(obs)
        (prefix, _), _cache = model.PaliGemma.llm(
            [tokens, None], mask=make_attn_mask(mask, ar_mask), positions=jnp.cumsum(mask, axis=1) - 1
        )
        q = model.vjepa_num_queries
        queries = prefix[:, -q:]
        context = prefix[:, :-q]
        context_mask = mask[:, :-q].astype(jnp.float32)
        pooled = (context * context_mask[..., None]).sum(1) / jnp.maximum(context_mask.sum(1, keepdims=True), 1.0)
        delta = model.con1_delta_head(queries, obs.con1_current_latent)["delta"]
        return queries, pooled, delta

    pcollect = jax.jit(collect, out_shardings=(replicated, replicated, replicated))
    q_list, ctx_list, head_list, z_list, fut_list, valid_list = [], [], [], [], [], []
    iterator = iter(loader)
    for i in range(args.batches):
        observation, _ = next(iterator)
        with sharding.set_mesh(mesh):
            queries, pooled, head_delta = pcollect(params, observation)
        host = jax.device_get(observation)
        q_list.append(np.asarray(jax.device_get(queries), np.float32))
        ctx_list.append(np.asarray(jax.device_get(pooled), np.float32))
        head_list.append(np.asarray(jax.device_get(head_delta), np.float32))
        z_list.append(np.asarray(host.con1_current_latent, np.float32))
        fut_list.append(np.asarray(host.con1_future_latents, np.float32))
        valid_list.append(np.asarray(host.con1_future_valid, bool))
        print(json.dumps({"collected_batch": i + 1, "of": args.batches}), flush=True)

    queries = np.concatenate(q_list, 0)
    ctx = np.concatenate(ctx_list, 0)
    head_delta = np.concatenate(head_list, 0)
    z0 = np.concatenate(z_list, 0)
    future = np.concatenate(fut_list, 0)
    valid = np.concatenate(valid_list, 0)
    target = future - z0[:, None, :]
    r_mean = queries.mean(1)
    r_max = queries.max(1)
    n, horizon, dim = target.shape
    print(json.dumps({"samples": int(n), "horizon": int(horizon), "latent_dim": int(dim),
                      "r_dim": int(r_mean.shape[1]), "context_dim": int(ctx.shape[1])}), flush=True)

    rows = np.repeat(np.arange(n), horizon)
    cols = np.tile(np.arange(horizon), n)
    keep = valid.reshape(-1)
    rows, cols = rows[keep], cols[keep]
    order = rng.permutation(len(rows))
    rows, cols = rows[order], cols[order]
    n_rows = len(rows)
    n_train, n_val = int(0.6 * n_rows), int(0.2 * n_rows)
    tr = slice(0, n_train)
    va = slice(n_train, n_train + n_val)
    ev = slice(n_train + n_val, n_rows)

    def design(name):
        parts = {
            "z": z0,
            "r_mean": r_mean,
            "r_max": r_max,
            "vlm_ctx": ctx,
        }
        return np.concatenate([parts[k] for k in name.split("+")], axis=1)

    designs = [
        "z",
        "r_mean",
        "z+r_mean",
        "z+r_mean+vlm_ctx",
        "z+r_mean+r_max+vlm_ctx",
        "r_mean+vlm_ctx",
        "vlm_ctx",
    ]
    lambdas = [1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    report = {
        "exp_name": args.exp_name,
        "restored_step": int(state.step),
        "samples": int(n),
        "valid_pairs": int(n_rows),
        "split": {"train": n_train, "val": n_val, "eval": n_rows - n_train - n_val},
        "zero_predictor_nmse": 1.0,
        "head_nmse_eval": float(_nmse(head_delta[rows[ev], cols[ev]], target[rows[ev], cols[ev]])),
        "probes": {},
    }
    print(json.dumps({"probe": "existing_head", "eval_nmse": report["head_nmse_eval"]}), flush=True)
    for name in designs:
        x = design(name)
        score, lam = _fit_ridge(
            x[rows[tr]], target[rows[tr], cols[tr]],
            x[rows[va]], target[rows[va], cols[va]],
            x[rows[ev]], target[rows[ev], cols[ev]],
            lambdas, args.proj_dim, args.seed,
        )
        report["probes"][name] = {
            "eval_nmse": score, "lambda": lam, "dim": int(x.shape[1]), "proj_dim": args.proj_dim
        }
        print(json.dumps({"probe": name, "eval_nmse": score, "lambda": lam, "dim": int(x.shape[1]),
                          "proj_dim": args.proj_dim}), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print("WROTE", out_path, flush=True)


if __name__ == "__main__":
    main()
