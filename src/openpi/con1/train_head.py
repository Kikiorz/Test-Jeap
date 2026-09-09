"""Train only the current-anchored Delta Head; the JEPA-WAM base is never touched.

This intentionally consumes an offline feature cache.  The cache must contain
R(o_t) and Phi(o_t), while future Phi values are labels only.
"""
import argparse
import json
from pathlib import Path

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.con1.data import FeatureDataset, batch
from openpi.con1.modules import AnchoredDeltaHead, anchored_loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--latent-dim", type=int, required=True)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--delta-weight", type=float, default=1.)
    args = p.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("steps, batch-size and learning-rate must be positive")
    train = FeatureDataset(args.cache, horizon=args.horizon, split="train", seed=args.seed)
    valid = FeatureDataset(args.cache, horizon=args.horizon, split="validation", seed=args.seed)
    sample = batch(train, np.arange(min(args.batch_size, len(train))))
    model = AnchoredDeltaHead(args.horizon, args.latent_dim, args.width)
    rng = jax.random.key(args.seed)
    variables = model.init(rng, jnp.asarray(sample["r_tokens"]), jnp.asarray(sample["anchor"]))
    tx = optax.adamw(args.learning_rate)
    state = flax.struct.dataclass(type("HeadState", (), {})) if False else None
    opt_state = tx.init(variables["params"])

    def loss_fn(params, values):
        out = model.apply({"params": params}, values["r_tokens"], values["anchor"])
        loss, metrics = anchored_loss(out["delta"], values["anchor"], values["future_target"],
                                      values["valid"], delta_weight=args.delta_weight)
        return loss, metrics

    @jax.jit
    def update(params, opt_state, values):
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, values)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, metrics

    params = variables["params"]
    history = []
    for step in range(args.steps):
        indices = (np.arange(args.batch_size) + step * args.batch_size) % len(train)
        values = {k: jnp.asarray(v) for k, v in batch(train, indices).items()}
        params, opt_state, metrics = update(params, opt_state, values)
        if step == 0 or (step + 1) % 100 == 0 or step + 1 == args.steps:
            history.append({"step": step + 1, **{k: float(v) for k, v in metrics.items() if k != "valid_count"},
                            "valid_count": int(metrics["valid_count"])})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "params.msgpack").write_bytes(flax.serialization.to_bytes(params))
    (args.output / "manifest.json").write_text(json.dumps({
        "schema": "con1-anchored-delta-head-checkpoint-v1", "steps": args.steps,
        "cache_identity": train.identity, "seed": args.seed, "horizon": args.horizon,
        "latent_dim": args.latent_dim, "width": args.width,
        "learning_rate": args.learning_rate, "delta_weight": args.delta_weight,
        "base_parameters_updated": False, "history": history}, indent=2) + "\n")
    print(json.dumps(history[-1]))


if __name__ == "__main__":
    main()
