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


def sample_windows(tokens, actions, states, context, start, stop, count, rng, stride=1):
    """Teacher-forced 1-step windows with the base frame inside [start, stop)."""
    low = max(stride * (context - 1), start)
    high = min(len(actions) - stride, stop)
    if high <= low:
        return []
    starts = np.unique(rng.integers(low, high, size=count))
    return [int(s) for s in starts]


@torch.no_grad()
def score(model, tokens, actions, states, windows, context, device, stride=1):
    """NMSE against copy-current for one-step prediction on the given windows."""
    error_model = error_copy = 0.0
    for start in windows:
        index = start - stride * np.arange(context - 1, -1, -1)
        window = np.asarray(tokens[index], dtype=np.float32)
        predicted = model.predict(window, actions[index], states[index]).reshape(-1)
        current = normalize_reps(torch.from_numpy(np.asarray(tokens[start], dtype=np.float32)).to(device)).reshape(-1)
        target = normalize_reps(torch.from_numpy(np.asarray(
            tokens[min(start + stride, len(actions) - 1)], dtype=np.float32)).to(device)).reshape(-1)
        error_model += float((predicted - target).pow(2).sum())
        error_copy += float((current - target).pow(2).sum())
    return (error_model / error_copy) if error_copy > 0 else float("nan")


def batch_from(tokens, actions, states, windows, context, stride=1):
    token_batch, action_batch, state_batch = [], [], []
    for start in windows:
        index = start - stride * np.arange(context - 1, -1, -1)
        token_batch.append(np.asarray(tokens[np.concatenate([index, [start + stride]])], dtype=np.float32))
        action_batch.append(np.asarray(actions[index], dtype=np.float32))
        state_batch.append(np.asarray(states[index], dtype=np.float32))
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
    parser.add_argument("--replay-windows", type=int, default=0,
                        help="Windows kept in the cross-episode replay buffer (0 keeps only the "
                             "current episode's first half).")
    parser.add_argument("--frame-stride", type=int, default=2,
                        help="Frames between window steps; 2 matches the released 4-5 fps timebase.")
    parser.add_argument("--no-adapt", type=int, default=0,
                        help="Evaluate with adaptation disabled to quantify the drift floor.")
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
    replay: list[tuple] = []
    for episode in args.episodes:
        tokens = np.load(args.cache / "tokens" / f"episode_{episode:06d}.npy", mmap_mode="r")
        actions = np.load(args.cache / "actions" / f"episode_{episode:06d}.npy")
        states = np.load(args.cache / "states" / f"episode_{episode:06d}.npy")
        length = len(actions)
        split = length // 2
        eval_index = sample_windows(tokens, actions, states, args.context, split, length,
                                    args.eval_windows, rng, args.frame_stride)
        buffer_index = sample_windows(tokens, actions, states, args.context, args.context,
                                      split, args.buffer_windows, rng, args.frame_stride)
        if not eval_index or not buffer_index:
            continue
        before = score(model, tokens, actions, states, eval_index, args.context, device, args.frame_stride)
        losses = []
        if not args.no_adapt:
            for index in buffer_index:
                replay.append(batch_from(tokens, actions, states, [index], args.context, args.frame_stride))
            if args.replay_windows:
                replay = replay[-args.replay_windows:]
            elif len(replay) > len(buffer_index):
                replay = replay[-len(buffer_index):]
            token_batch = torch.cat([r[0] for r in replay])
            action_batch = torch.cat([r[1] for r in replay])
            state_batch = torch.cat([r[2] for r in replay])
            if len(token_batch) > args.buffer_windows:
                keep = rng.permutation(len(token_batch))[:args.buffer_windows]
                token_batch, action_batch, state_batch = (token_batch[keep], action_batch[keep],
                                                          state_batch[keep])
            losses = adapt(model, optimizer, token_batch.to(device), action_batch.to(device),
                           state_batch.to(device), steps=args.adapt_steps, auto_steps=args.auto_steps)
        after = score(model, tokens, actions, states, eval_index, args.context, device, args.frame_stride)
        records.append({"episode": episode, "length": int(length), "windows": len(eval_index),
                        "nmse_before": before, "nmse_after": after,
                        "adapt_loss_start": losses[0] if losses else None,
                        "adapt_loss_end": losses[-1] if losses else None,
                        "replay_size": len(replay)})
        print(json.dumps(records[-1]), flush=True)

    deltas = np.array([r["nmse_after"] - r["nmse_before"] for r in records])
    summary = {
        "predictor": str(args.predictor),
        "episodes": args.episodes,
        "adapt_steps": args.adapt_steps,
        "learning_rate": args.learning_rate,
        "replay_windows": args.replay_windows,
        "no_adapt": bool(args.no_adapt),
        "mean_nmse_before": float(np.mean([r["nmse_before"] for r in records])),
        "mean_nmse_after": float(np.mean([r["nmse_after"] for r in records])),
        "mean_delta": float(deltas.mean()) if len(deltas) else None,
        "stderr_delta": float(deltas.std(ddof=1) / np.sqrt(len(deltas))) if len(deltas) > 1 else None,
        "relative_change": float(deltas.mean() / np.mean([r["nmse_before"] for r in records])),
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
