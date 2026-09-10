"""How much does conditioning the Con2 predictor on the action chunk help?

The anchored head currently sees only R_t (PI0.5 query tokens) and z_t, so it
regresses a multi-modal future and is capped by the conditional variance. This
probe adds the ground-truth action chunk a_{t..t+H-1} and measures the NMSE drop
on a held-out split, using the same target definition as training.

Causal variants only expose a_{t..t+j-1} to horizon j, which is what a real head
would be allowed to see. The non-causal `act_all` variant is an optimistic upper
bound reported for reference.
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


def _nmse(pred, target):
    return float(np.sum((pred - target) ** 2) / max(np.sum(target**2), 1e-12))


def _fit_ridge(train_x, train_y, val_x, val_y, eval_x, eval_y, lambdas, proj_dim, seed):
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
    normal = (tx.T @ tx).astype(np.float64)
    rhs = (tx.T @ train_y.astype(np.float32)).astype(np.float64)
    eye = np.eye(normal.shape[0])
    best = None
    for lam in lambdas:
        weight = np.linalg.solve(normal + lam * eye, rhs)
        score = _nmse(vx @ weight, val_y)
        if best is None or score < best[0]:
            best = (score, lam)
    return best[0], float(best[1])


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
        batch_size=args.batch_size, num_workers=0, wandb_enabled=False,
        exp_name=args.exp_name, checkpoint_base_dir="/workspace/artifacts/checkpoints", resume=True,
    )
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
    r_list, head_list, z_list, fut_list, valid_list, act_list = [], [], [], [], [], []
    iterator = iter(loader)
    for i in range(args.batches):
        observation, actions = next(iterator)
        with sharding.set_mesh(mesh):
            tokens, head_delta = pcollect(params, observation)
        host = jax.device_get(observation)
        r_list.append(np.asarray(jax.device_get(tokens), np.float32))
        head_list.append(np.asarray(jax.device_get(head_delta), np.float32))
        z_list.append(np.asarray(host.con1_current_latent, np.float32))
        fut_list.append(np.asarray(host.con1_future_latents, np.float32))
        valid_list.append(np.asarray(host.con1_future_valid, bool))
        act_list.append(np.asarray(jax.device_get(actions), np.float32))
        print(json.dumps({"collected_batch": i + 1, "of": args.batches}), flush=True)

    queries = np.concatenate(r_list, 0)
    head_delta = np.concatenate(head_list, 0)
    z0 = np.concatenate(z_list, 0)
    future = np.concatenate(fut_list, 0)
    valid = np.concatenate(valid_list, 0)
    actions = np.concatenate(act_list, 0)
    target = future - z0[:, None, :]
    r_mean = queries.mean(1)
    n, horizon, dim = target.shape
    a_dim = actions.shape[-1]
    print(json.dumps({"samples": int(n), "horizon": int(horizon), "latent_dim": int(dim),
                      "action_dim": int(a_dim)}), flush=True)

    rows = np.repeat(np.arange(n), horizon)
    cols = np.tile(np.arange(horizon), n)
    keep = valid.reshape(-1)
    rows, cols = rows[keep], cols[keep]
    order = rng.permutation(len(rows))
    rows, cols = rows[order], cols[order]
    n_rows = len(rows)
    n_train, n_val = int(0.6 * n_rows), int(0.2 * n_rows)
    tr, va, ev = slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n_rows)

    step_idx = np.arange(horizon)[None, :]
    # per-row causal prefix over actions a_{t..t+j-1}
    A = actions[rows]
    causal_mask = (step_idx < (cols + 1)[:, None]).astype(np.float32)[:, :, None]
    prefix_sum = (A * causal_mask).sum(1)
    prefix_mean = prefix_sum / np.maximum(causal_mask.sum(1), 1.0)
    prefix_last = A[np.arange(n_rows), np.maximum(cols, 0)]
    prefix_first = A[:, 0, :]
    act_all = actions[rows].reshape(n_rows, -1)
    act_causal = np.concatenate([prefix_mean, prefix_sum, prefix_last, prefix_first], axis=1)

    parts = {
        "z": z0,
        "r_mean": r_mean,
        "act_all": act_all,
        "act_causal": act_causal,
    }

    def design(name):
        return np.concatenate([parts[k] for k in name.split("+")], axis=1)

    designs = [
        "z",
        "z+r_mean",
        "act_all",
        "act_causal",
        "z+act_causal",
        "z+r_mean+act_all",
        "z+r_mean+act_causal",
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
        report["probes"][name] = {"eval_nmse": score, "lambda": lam, "dim": int(x.shape[1])}
        print(json.dumps({"probe": name, "eval_nmse": score, "lambda": lam}), flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print("WROTE", out_path, flush=True)


if __name__ == "__main__":
    main()
