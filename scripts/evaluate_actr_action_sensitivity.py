#!/usr/bin/env python3
"""Test whether Stage-1 ACTR uses the matching action to predict transition targets."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as _model
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _sign_test_one_sided(positive: int, nonzero: int) -> float:
    if nonzero == 0:
        return 1.0
    return sum(math.comb(nonzero, k) for k in range(positive, nonzero + 1)) / (2**nonzero)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("params", type=pathlib.Path, help="Stage-1 params directory")
    parser.add_argument("--config", default="pi05_libero_actr_stage1")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batches", type=int, default=40)
    parser.add_argument("--min-pair-accuracy", type=float, default=0.55)
    parser.add_argument("--require-gate", action="store_true")
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    if not 0.5 <= args.min_pair_accuracy <= 1.0:
        raise ValueError("--min-pair-accuracy must be in [0.5, 1]")
    config = dataclasses.replace(
        _config.get_config(args.config),
        batch_size=args.batch_size,
        num_workers=0,
        wandb_enabled=False,
    )
    model = config.model.load(_model.restore_params(args.params, dtype=jnp.bfloat16))
    model.eval()
    loader = _data_loader.create_data_loader(
        config,
        shuffle=True,
        num_batches=args.batches,
    )

    @nnx.jit
    def transition_loss(model, rng, observation, actions):
        _, loss = model.compute_loss_components(rng, observation, actions, train=False)
        assert loss is not None
        return loss

    correct_losses: list[np.ndarray] = []
    mismatched_losses: list[np.ndarray] = []
    base_rng = jax.random.key(config.seed + 10_000)
    for batch_index, (observation, actions) in enumerate(loader):
        loss_rng = jax.random.fold_in(base_rng, batch_index)
        # The loader is globally shuffled, so this fixed derangement normally
        # pairs observations with actions from different episodes/tasks while
        # preserving the exact action marginal and tensor shape.
        shift = max(1, actions.shape[0] // 2)
        mismatched_actions = jnp.roll(actions, shift=shift, axis=0)
        correct = transition_loss(model, loss_rng, observation, actions)
        mismatched = transition_loss(model, loss_rng, observation, mismatched_actions)
        correct_losses.append(np.asarray(jax.device_get(correct), dtype=np.float64))
        mismatched_losses.append(np.asarray(jax.device_get(mismatched), dtype=np.float64))

    correct = np.concatenate(correct_losses)
    mismatched = np.concatenate(mismatched_losses)
    margin = mismatched - correct
    finite = np.isfinite(correct) & np.isfinite(mismatched)
    if not finite.all():
        raise FloatingPointError(f"Found {(~finite).sum()} non-finite paired losses")
    positive = int((margin > 0).sum())
    negative = int((margin < 0).sum())
    nonzero = positive + negative
    pair_accuracy = positive / nonzero if nonzero else 0.5
    passed = bool(margin.mean() > 0 and pair_accuracy >= args.min_pair_accuracy)
    report = {
        "status": "PASS" if passed else "FAIL",
        "params": str(args.params.resolve()),
        "samples": int(correct.size),
        "correct_transition_loss": float(correct.mean()),
        "mismatched_transition_loss": float(mismatched.mean()),
        "mismatch_margin": float(margin.mean()),
        "mismatch_margin_standard_error": float(margin.std(ddof=1) / math.sqrt(margin.size)),
        "pair_accuracy": float(pair_accuracy),
        "positive_pairs": positive,
        "negative_pairs": negative,
        "ties": int(correct.size - nonzero),
        "one_sided_sign_test_p": float(_sign_test_one_sided(positive, nonzero)),
        "minimum_pair_accuracy": args.min_pair_accuracy,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if args.require_gate and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
