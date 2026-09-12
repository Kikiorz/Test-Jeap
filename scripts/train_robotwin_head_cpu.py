#!/usr/bin/env python3
"""Train the anchored-delta head alone, on CPU, against the cached features.

`probe_robotwin_latent_ceiling.py` showed a *linear* map from the current latent
reaches 0.659 mean NMSE over the head's horizons while the jointly trained head
sits at 0.86. Two explanations remain:

* optimisation: the head shares a 1e-5 schedule with the whole model and only
  gets the delta term through a weighted sum, so it may simply be undertrained;
* architecture: the attention-over-64-tokens head may be unable to represent the
  map at all.

Training the head by itself, on the cache, with its own schedule separates the
two. `src/openpi/con1/train_head.py` is the LIBERO/4-GPU version of this (it
hardcodes horizon 10, requires the 40k cache and pmaps over every local device),
so this keeps the GPU box free for the running arms and runs on one CPU process.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.con1.data import FeatureDataset, batch
from openpi.con1.modules import AnchoredDeltaHead, anchored_loss


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path,
                   default=Path("/workspace/artifacts/con1/robotwin_clean20_19999_features_v1"))
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--latent-dim", type=int, default=4224)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-frames", type=int, default=32)
    p.add_argument("--out", type=Path,
                   default=Path("/workspace/artifacts/con2/robotwin_head_only_cpu.json"))
    return p.parse_args()


def validation_indices(dataset: FeatureDataset, per_episode: int) -> np.ndarray:
    picked: list[int] = []
    offset = 0
    for episode in dataset.episodes:
        span = episode["length"] - 1
        if span < 1:
            offset += span
            continue
        picked.extend((offset + np.linspace(0, span - 1, min(per_episode, span), dtype=int)).tolist())
        offset += span
    return np.asarray(picked)


def evaluate(params, dataset, indices, model, batch_size: int) -> dict:
    weighted = {"delta_mse": 0.0, "copy_current_mse": 0.0, "loss": 0.0}
    count = 0.0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        values = {k: jnp.asarray(v) for k, v in batch(dataset, chunk).items()}
        out = model.apply({"params": params}, values["r_tokens"], values["anchor"])
        _, metrics = anchored_loss(out["delta"], values["anchor"], values["future_target"],
                                   values["valid"])
        n = float(metrics["valid_count"])
        count += n
        for key in weighted:
            weighted[key] += float(metrics[key]) * n
    record = {k: v / max(count, 1.0) for k, v in weighted.items()}
    record["delta_nmse"] = record["delta_mse"] / max(record["copy_current_mse"], 1e-12)
    record["samples"] = int(count)
    return record


def main() -> None:
    args = parse()
    jax.config.update("jax_platform_name", "cpu")
    started = time.monotonic()

    train = FeatureDataset(args.cache, horizon=args.horizon, split="train", seed=args.seed)
    valid = FeatureDataset(args.cache, horizon=args.horizon, split="validation", seed=args.seed)
    print(f"train episodes {len(train.episodes)} frames {len(train)} | "
          f"validation episodes {len(valid.episodes)} frames {len(valid)}", flush=True)

    model = AnchoredDeltaHead(horizon=args.horizon, latent_dim=args.latent_dim, width=args.width)
    params = model.init(jax.random.key(args.seed),
                        jnp.zeros((1, *train.manifest["r_shape"])),
                        jnp.zeros((1, args.latent_dim)))["params"]
    schedule = optax.linear_schedule(0.0, args.learning_rate, 100)
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    opt_state = tx.init(params)
    indices = validation_indices(valid, args.eval_frames)

    def loss_fn(params, values):
        out = model.apply({"params": params}, values["r_tokens"], values["anchor"])
        return anchored_loss(out["delta"], values["anchor"], values["future_target"], values["valid"])

    @jax.jit
    def update(params, opt_state, values):
        (_, metrics), grad = jax.value_and_grad(loss_fn, has_aux=True)(params, values)
        updates, opt_state = tx.update(grad, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    history = []
    first = evaluate(params, valid, indices, model, args.batch_size)
    history.append({"step": 0, **first})
    print(json.dumps({"step": 0, "delta_nmse": round(first["delta_nmse"], 4)}), flush=True)

    rng = np.random.default_rng(args.seed)
    for step in range(args.steps):
        ids = rng.integers(0, len(train), size=args.batch_size)
        values = {k: jnp.asarray(v) for k, v in batch(train, ids).items()}
        params, opt_state, metrics = update(params, opt_state, values)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            record = evaluate(params, valid, indices, model, args.batch_size)
            entry = {"step": step + 1, "train_loss": float(metrics["loss"]),
                     "train_delta_nmse": float(metrics["delta_mse"] / jnp.maximum(metrics["copy_current_mse"], 1e-12)),
                     "seconds": round(time.monotonic() - started, 1), **record}
            history.append(entry)
            print(json.dumps({"step": entry["step"], "val_delta_nmse": round(record["delta_nmse"], 4),
                              "train_delta_nmse": round(entry["train_delta_nmse"], 4),
                              "s": entry["seconds"]}), flush=True)

    best = min(history, key=lambda row: row["delta_nmse"])
    summary = {"config": vars(args) | {"cache": str(args.cache), "out": str(args.out)},
               "history": history, "best": best,
               "linear_ceiling_mean_nmse_h1_16": 0.6591,
               "joint_run_head_nmse_at_8k": 0.8196,
               "interpretation": ("best < ceiling => optimisation was the limit; "
                                  "best > ceiling => the head architecture cannot represent it")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n")
    print(f"best delta_nmse {best['delta_nmse']:.4f} at step {best['step']}; wrote {args.out}")


if __name__ == "__main__":
    main()
