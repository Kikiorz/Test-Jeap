"""Closed-form warm start for the Con2 direct readout.

Fits the exact ridge solution from `[LayerNorm(R).mean(1), z_t]` to the
per-horizon latent delta, choosing lambda on a held-out split, and writes the
result as a msgpack the weight loader can inject into the delta head's
`direct_readout` layer.

This exists because the readout is a ~137M-parameter linear layer trained from
scratch at 2e-5; plain SGD was measured to crawl towards a solution that a
closed-form solve reaches immediately.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import flax.serialization as serialization
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import train
from openpi.models import model as _model
from openpi.training import checkpoints, config as configs, data_loader, sharding


def _nmse(pred, target):
    return float(np.sum((pred - target) ** 2) / max(np.sum(target**2), 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, help="msgpack path for the readout weights")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches", type=int, default=24)
    parser.add_argument("--horizon-agnostic", action="store_true",
                        help="fit one shared feature->delta map instead of one per horizon")
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()

    jax.config.update("jax_compilation_cache_dir", "/workspace/.cache/con1_jax")
    rng = np.random.default_rng(args.seed)
    config = dataclasses.replace(
        configs.get_config(args.config), batch_size=args.batch_size, num_workers=0, wandb_enabled=False)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_shard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    loader = data_loader.create_data_loader(config, sharding=data_shard, shuffle=True)
    _, init_rng = jax.random.split(jax.random.key(config.seed))
    # resume=False: we want the named config's own fresh initialization, which is
    # what training will start from.
    state, _ = train.init_train_state(config, init_rng, mesh, resume=False)
    jax.block_until_ready(state)

    def compute(p, observation):
        model = nnx.merge(state.model_def, p)
        model.eval()
        obs = _model.preprocess_observation(None, observation, train=False)
        tokens = model.extract_predictive_tokens(obs)
        # Matches AnchoredDeltaHead's direct-readout input exactly: the raw
        # predictive-token mean concatenated with the current latent.
        return tokens.mean(1), obs.con1_current_latent

    pcompute = jax.jit(compute, out_shardings=(replicated, replicated))
    params = state.params
    f_list, z_list, fut_list, valid_list = [], [], [], []
    iterator = iter(loader)
    for i in range(args.batches):
        observation, _ = next(iterator)
        with sharding.set_mesh(mesh):
            pooled, latent = pcompute(params, observation)
        host = jax.device_get(observation)
        f_list.append(np.asarray(jax.device_get(pooled), np.float32))
        z_list.append(np.asarray(jax.device_get(latent), np.float32))
        fut_list.append(np.asarray(host.con1_future_latents, np.float32))
        valid_list.append(np.asarray(host.con1_future_valid, bool))
        print(json.dumps({"collected_batch": i + 1, "of": args.batches}), flush=True)

    pooled = np.concatenate(f_list, 0)
    z0 = np.concatenate(z_list, 0)
    future = np.concatenate(fut_list, 0)
    valid = np.concatenate(valid_list, 0)
    target = future - z0[:, None, :]
    x = np.concatenate([pooled, z0], axis=1)
    n, horizon, latent_dim = target.shape
    n_train = int(0.8 * n)
    tr, ev = slice(0, n_train), slice(n_train, n)
    print(json.dumps({"samples": int(n), "feature_dim": int(x.shape[1]),
                      "horizon": int(horizon), "latent_dim": int(latent_dim)}), flush=True)

    mean = x[tr].mean(0, keepdims=True)
    std = x[tr].std(0, keepdims=True) + 1e-6
    xt = ((x - mean) / std).astype(np.float64)
    if args.horizon_agnostic:
        # One weight matrix serves every horizon, which is the estimator that
        # the earlier probe actually measured. Pool all (sample, horizon) pairs.
        rows = np.repeat(np.arange(n), horizon)
        cols = np.tile(np.arange(horizon), n)
        keep = valid.reshape(-1)
        rows, cols = rows[keep], cols[keep]
        train_rows = rows < n_train
        eval_rows = ~train_rows
        flat_x = xt[rows]
        flat_y = target[rows, cols].astype(np.float64)
        kernel = np.zeros((x.shape[1] + 1, latent_dim), np.float32)
        report = {"config": args.config, "samples": int(n), "mode": "horizon_agnostic",
                  "pairs": int(len(rows)), "nmse": {}, "lambda": None, "total_nmse": None}
        best = None
        # Evaluate raw features and a 1024-d random projection side by side; the
        # earlier probe used the projection and reported a far lower NMSE, so
        # the difference has to be attributed explicitly.
        variants = {"raw": np.zeros((flat_x.shape[1], 0), np.float32)}
        proj = np.random.default_rng(args.seed).normal(
            0.0, 1.0 / np.sqrt(1024), size=(flat_x.shape[1], 1024)).astype(np.float32)
        projected = (flat_x @ proj).astype(np.float64)
        for tag, feats in (("raw", flat_x), ("proj1024", projected)):
            for lam in (1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5):
                a = np.concatenate([feats[train_rows], np.ones((int(train_rows.sum()), 1))], axis=1)
                normal = a.T @ a + lam * np.eye(a.shape[1])
                weight = np.linalg.solve(normal, a.T @ flat_y[train_rows])
                b = np.concatenate([feats[eval_rows], np.ones((int(eval_rows.sum()), 1))], axis=1)
                score = _nmse(b @ weight, flat_y[eval_rows])
                report["nmse"][f"{tag}@{lam}"] = score
                print(json.dumps({"variant": tag, "lambda": lam, "heldout_nmse": score}), flush=True)
                if tag == "raw" and (best is None or score < best[0]):
                    best = (score, lam, weight)
        _, lam, weight = best
        w = weight[:-1] / std.T
        bias = weight[-1] - (mean / std) @ weight[:-1]
        kernel[: x.shape[1], :] = w.astype(np.float32)
        kernel[-1, :] = np.asarray(bias, np.float32).reshape(-1)
        report["lambda"] = lam
        report["total_nmse"] = report["nmse"][str(lam)]
        report["warmstart_nmse"] = report["total_nmse"]
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"params": {"shared_readout": {"kernel": kernel[:-1], "bias": kernel[-1]}}}
        out.write_bytes(serialization.msgpack_serialize(payload))
        out.with_suffix(".json").write_text(json.dumps(report, indent=2))
        print("RESULT", json.dumps(report), flush=True)
        print("WROTE", out, flush=True)
        return
    kernel = np.zeros((x.shape[1] + 1, horizon * latent_dim), np.float32)
    kernel[-1, :] = 1.0  # bias row fed by a constant-1 feature
    report = {"config": args.config, "samples": int(n), "lambda": {}, "nmse": {}, "total_nmse": None}
    err = 0.0
    base = 0.0
    best_lambda = None
    for lam in (1.0, 1e2, 1e3, 1e4, 1e5, 1e6):
        pred = np.zeros_like(target[ev], dtype=np.float64)
        for h in range(horizon):
            mask = valid[tr][:, h]
            a = np.concatenate([xt[tr][mask], np.ones((int(mask.sum()), 1))], axis=1)
            y = target[tr][mask, h, :].astype(np.float64)
            normal = a.T @ a + lam * np.eye(a.shape[1])
            weight = np.linalg.solve(normal, a.T @ y)
            mask_e = valid[ev][:, h]
            b = np.concatenate([xt[ev][mask_e], np.ones((int(mask_e.sum()), 1))], axis=1)
            pred[mask_e, h, :] = b @ weight
        keep = valid[ev][..., None]
        score = float(np.sum(np.where(keep, (pred - target[ev]) ** 2, 0.0)) /
                      max(np.sum(np.where(keep, target[ev] ** 2, 0.0)), 1e-12))
        report["nmse"][str(lam)] = score
        print(json.dumps({"lambda": lam, "heldout_nmse": score}), flush=True)
        if best_lambda is None or score < report["nmse"][str(best_lambda)]:
            best_lambda = lam
    lam = best_lambda
    for h in range(horizon):
        mask = valid[tr][:, h]
        a = np.concatenate([xt[tr][mask], np.ones((int(mask.sum()), 1))], axis=1)
        y = target[tr][mask, h, :].astype(np.float64)
        normal = a.T @ a + lam * np.eye(a.shape[1])
        weight = np.linalg.solve(normal, a.T @ y)
        # Undo the input standardisation so the layer consumes raw features.
        w = weight[:-1] / std.T
        b = weight[-1] - (mean / std) @ weight[:-1]
        kernel[: x.shape[1], h * latent_dim:(h + 1) * latent_dim] = w.astype(np.float32)
        kernel[-1, h * latent_dim:(h + 1) * latent_dim] = np.asarray(b, np.float32).reshape(-1)
    report["lambda"] = lam
    report["total_nmse"] = report["nmse"][str(lam)]
    report["warmstart_nmse"] = report["total_nmse"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": {"direct_readout": {"kernel": kernel[:-1], "bias": kernel[-1]}}}
    out.write_bytes(serialization.msgpack_serialize(payload))
    (out.with_suffix(".json")).write_text(json.dumps(report, indent=2))
    print("RESULT", json.dumps(report), flush=True)
    print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
