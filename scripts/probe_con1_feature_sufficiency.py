"""Is the anchored delta head underfitting, or is R_t the bottleneck?

Collects R_t (PI0.5 query tokens), z_t, and the cached future latents for a
fixed sample, then fits exact kernel-ridge probes in the full 2816-d output
space. Compares, on a held-out split:

  * the model's own delta head
  * a linear probe from z_t alone
  * a linear probe from pooled R_t alone
  * a linear probe from both
  * a random-Fourier-feature (nonlinear) probe from both

NMSE uses the same definition as the training metric: mean squared error over
valid entries divided by the mean squared target delta. A stronger probe that
clearly beats the head means the head is underfit; if every probe stalls at the
same value, R_t/z_t simply do not carry the information.
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


def _nmse(pred, target, weights):
    err = np.sum(weights * np.sum((pred - target) ** 2, axis=-1))
    base = np.sum(weights * np.sum(target**2, axis=-1))
    return float(err / max(base, 1e-12))


def _fit_ridge(train_x, train_y, train_w, val_x, val_y, val_w, eval_x, eval_y, eval_w, lambdas):
    """Exact kernel ridge (dual form) with per-row weights; picks lambda on val."""
    mean = train_x.mean(0, keepdims=True)
    std = train_x.std(0, keepdims=True) + 1e-6
    tx = (train_x - mean) / std
    vx = (val_x - mean) / std
    ex = (eval_x - mean) / std
    sw = np.sqrt(np.maximum(train_w, 0.0)).astype(np.float64)
    gram = (tx * sw[:, None]) @ (tx * sw[:, None]).T
    rhs = (train_y * sw[:, None]).astype(np.float64)
    best = None
    for lam in lambdas:
        alpha = np.linalg.solve(gram + lam * np.eye(gram.shape[0]), rhs)
        val_pred = vx @ ((tx * sw[:, None]).T @ alpha)
        score = _nmse(val_pred, val_y, val_w)
        if best is None or score < best[0]:
            best = (score, lam, alpha)
    _, lam, alpha = best
    eval_pred = ex @ ((tx * sw[:, None]).T @ alpha)
    return _nmse(eval_pred, eval_y, eval_w), float(lam), eval_pred


def _rff(features, dim, seed):
    proj = np.random.default_rng(seed).normal(
        0, 1.0 / np.sqrt(features.shape[1]), size=(features.shape[1], dim)
    ).astype(np.float32)
    return np.concatenate([np.cos(features @ proj), np.sin(features @ proj)], axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--rff-dim", type=int, default=512)
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
        tokens = model.extract_predictive_tokens(obs)
        delta = model.con1_delta_head(tokens, obs.con1_current_latent)["delta"]
        return tokens, delta

    pcollect = jax.jit(collect, out_shardings=(replicated, replicated))

    r_list, z_list, fut_list, valid_list, head_list = [], [], [], [], []
    iterator = iter(loader)
    for i in range(args.batches):
        observation, _ = next(iterator)
        with sharding.set_mesh(mesh):
            tokens, head_delta = pcollect(params, observation)
        host = jax.device_get(observation)
        r_list.append(np.asarray(jax.device_get(tokens), np.float32))
        head_list.append(np.asarray(jax.device_get(head_delta), np.float32))
        z_list.append(np.asarray(host.con1_current_latent, np.float32))
        fut_list.append(np.asarray(host.con1_future_latents, np.float32))
        valid_list.append(np.asarray(host.con1_future_valid, bool))
        print(json.dumps({"collected_batch": i + 1, "of": args.batches}), flush=True)

    r_tokens = np.concatenate(r_list, 0)
    head_delta = np.concatenate(head_list, 0)
    z0 = np.concatenate(z_list, 0)
    future = np.concatenate(fut_list, 0)
    valid = np.concatenate(valid_list, 0)
    target = future - z0[:, None, :]
    r_pool = r_tokens.mean(1)
    n, horizon, dim = target.shape
    print(json.dumps({"samples": int(n), "horizon": int(horizon), "latent_dim": int(dim),
                      "r_pool_dim": int(r_pool.shape[1])}), flush=True)

    rows = np.repeat(np.arange(n), horizon)
    cols = np.tile(np.arange(horizon), n)
    keep = valid.reshape(-1)
    rows, cols = rows[keep], cols[keep]
    # Split by SAMPLE. Splitting (sample, horizon) pairs leaks a sample across
    # train and eval, letting a fitted estimator interpolate between horizons
    # and report a spuriously low NMSE.
    rows = rows.copy()
    n_rows = len(rows)
    n_train_s, n_val_s = int(0.6 * n), int(0.2 * n)
    splits = {
        "train": rows < n_train_s,
        "val": (rows >= n_train_s) & (rows < n_train_s + n_val_s),
        "eval": rows >= n_train_s + n_val_s,
    }

    def design(name):
        if name == "z_only":
            return z0
        if name == "r_only":
            return r_pool
        return np.concatenate([r_pool, z0], axis=1)

    lambdas = [1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    report = {
        "exp_name": args.exp_name,
        "checkpoint_dir": str(config.checkpoint_dir),
        "restored_step": int(state.step),
        "samples": int(n),
        "valid_pairs": int(n_rows),
        "split": {"train_samples": n_train_s, "val_samples": n_val_s,
                  "eval_samples": n - n_train_s - n_val_s, "unit": "sample"},
        "zero_predictor_nmse": 1.0,
        "head_nmse_eval": None,
        "probes": {},
    }
    ev = splits["eval"]
    head_eval = _nmse(head_delta[rows[ev], cols[ev]], target[rows[ev], cols[ev]],
                      np.ones(rows[ev].shape[0], np.float32))
    report["head_nmse_eval"] = float(head_eval)
    print(json.dumps({"probe": "existing_head", "eval_nmse": float(head_eval)}), flush=True)

    for name in ("z_only", "r_only", "both"):
        x = design(name)
        tr, va, ee = splits["train"], splits["val"], splits["eval"]
        score, lam, _ = _fit_ridge(
            x[rows[tr]], target[rows[tr], cols[tr]], np.ones(tr.stop - tr.start, np.float32),
            x[rows[va]], target[rows[va], cols[va]], np.ones(va.stop - va.start, np.float32),
            x[rows[ee]], target[rows[ee], cols[ee]], np.ones(ee.stop - ee.start, np.float32),
            lambdas,
        )
        report["probes"][f"linear_{name}"] = {"eval_nmse": score, "lambda": lam}
        print(json.dumps({"probe": f"linear_{name}", "eval_nmse": score, "lambda": lam}), flush=True)

    xb = design("both")
    xr = _rff(xb, args.rff_dim, args.seed)
    tr, va, ee = splits["train"], splits["val"], splits["eval"]
    score, lam, _ = _fit_ridge(
        xr[rows[tr]], target[rows[tr], cols[tr]], np.ones(tr.stop - tr.start, np.float32),
        xr[rows[va]], target[rows[va], cols[va]], np.ones(va.stop - va.start, np.float32),
        xr[rows[ee]], target[rows[ee], cols[ee]], np.ones(ee.stop - ee.start, np.float32),
        lambdas,
    )
    report["probes"]["rff_both"] = {"eval_nmse": score, "lambda": lam, "rff_dim": args.rff_dim}
    print(json.dumps({"probe": "rff_both", "eval_nmse": score, "lambda": lam}), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print("WROTE", out_path, flush=True)


if __name__ == "__main__":
    main()
