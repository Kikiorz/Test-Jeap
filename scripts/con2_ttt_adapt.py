#!/usr/bin/env python3
"""Test-time training for the Con2 world model on a stream of new episodes.

Protocol (no leakage): for each episode the model is scored on the second half
*before* any adaptation, is then adapted on the first half only, and is scored
again on the same untouched second half. Adaptation uses only observed
transitions - the same cooldown objective, no labels a policy would not have at
test time - via ``openpi.con2.ac_world_model.adapt``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

VJEPA_ROOT = Path("/workspace/vjepa2")
sys.path.insert(0, str(VJEPA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openpi.con2.ac_world_model import (  # noqa: E402
    TOKENS_PER_FRAME,
    ACWorldModel,
    adapt,
    load_predictor,
    normalize_reps,
)


def sample_windows(tokens, actions, states, context, start, stop, count, rng):
    """Teacher-forced 1-step windows with the base frame inside [start, stop)."""
    low = max(context - 1, start)
    high = min(len(actions) - 1, stop)
    if high <= low:
        return []
    starts = np.unique(rng.integers(low, high, size=count))
    return [int(s) for s in starts]


@torch.no_grad()
def score(model, tokens, actions, states, windows, context, device):
    """NMSE against copy-current for one-step prediction on the given windows."""
    error_model = error_copy = 0.0
    for start in windows:
        window = np.asarray(tokens[start - context + 1:start + 1], dtype=np.float32)
        predicted = model.predict(window, actions[start - context + 1:start + 1],
                                  states[start - context + 1:start + 1]).reshape(-1)
        current = normalize_reps(torch.from_numpy(np.asarray(tokens[start], dtype=np.float32)).to(device)).reshape(-1)
        target = normalize_reps(torch.from_numpy(np.asarray(tokens[start + 1], dtype=np.float32)).to(device)).reshape(-1)
        error_model += float((predicted - target).pow(2).sum())
        error_copy += float((current - target).pow(2).sum())
    return (error_model / error_copy) if error_copy > 0 else float("nan")


def batch_from(tokens, actions, states, windows, context):
    token_batch, action_batch, state_batch = [], [], []
    for start in windows:
        token_batch.append(np.asarray(tokens[start - context + 1:start + 2], dtype=np.float32))
        action_batch.append(np.asarray(actions[start - context + 1:start + 1], dtype=np.float32))
        state_batch.append(np.asarray(states[start - context + 1:start + 1], dtype=np.float32))
    return (torch.from_numpy(np.stack(token_batch)), torch.from_numpy(np.stack(action_batch)),
            torch.from_numpy(np.stack(state_batch)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--episodes", type=int, nargs="+", required=True)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--eval-windows", type=int, default=8)
    parser.add_argument("--buffer-windows", type=int, default=8,
                        help="Windows per adaptation batch; the bool attention mask is materialised.")
    parser.add_argument("--adapt-steps", type=int, default=4)
    parser.add_argument("--auto-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    predictor = load_predictor(args.predictor, root=VJEPA_ROOT, device=device)
    # The released block-causal mask is materialised, so adaptation needs
    # activation checkpointing at any usable batch size.
    predictor.use_activation_checkpointing = True
    model = ACWorldModel(predictor=predictor, device=str(device))
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.learning_rate, weight_decay=0.0)
    rng = np.random.default_rng(args.seed)

    records = []
    started = time.time()
    for episode in args.episodes:
        tokens = np.load(args.cache / "tokens" / f"episode_{episode:06d}.npy", mmap_mode="r")
        actions = np.load(args.cache / "actions" / f"episode_{episode:06d}.npy")
        states = np.load(args.cache / "states" / f"episode_{episode:06d}.npy")
        length = len(actions)
        split = length // 2
        eval_index = sample_windows(tokens, actions, states, args.context, split, length,
                                    args.eval_windows, rng)
        buffer_index = sample_windows(tokens, actions, states, args.context, args.context,
                                      split, args.buffer_windows, rng)
        if not eval_index or not buffer_index:
            continue
        before = score(model, tokens, actions, states, eval_index, args.context, device)
        token_batch, action_batch, state_batch = batch_from(tokens, actions, states, buffer_index, args.context)
        token_batch = token_batch.to(device)
        action_batch = action_batch.to(device)
        state_batch = state_batch.to(device)
        losses = adapt(model, optimizer, token_batch, action_batch, state_batch,
                       steps=args.adapt_steps, auto_steps=args.auto_steps)
        after = score(model, tokens, actions, states, eval_index, args.context, device)
        records.append({"episode": episode, "length": int(length), "windows": len(eval_index),
                        "nmse_before": before, "nmse_after": after,
                        "adapt_loss_start": losses[0], "adapt_loss_end": losses[-1]})
        print(json.dumps(records[-1]), flush=True)

    summary = {
        "predictor": str(args.predictor),
        "episodes": args.episodes,
        "adapt_steps": args.adapt_steps,
        "learning_rate": args.learning_rate,
        "mean_nmse_before": float(np.mean([r["nmse_before"] for r in records])),
        "mean_nmse_after": float(np.mean([r["nmse_after"] for r in records])),
        "improved_episodes": int(sum(r["nmse_after"] < r["nmse_before"] for r in records)),
        "episodes_scored": len(records),
        "seconds": time.time() - started,
        "records": records,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
