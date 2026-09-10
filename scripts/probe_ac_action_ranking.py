#!/usr/bin/env python3
"""Does the predictor actually use the action? Energy-landscape / action ranking.

NMSE against the copy baseline is insensitive to action conditioning (a good
one-step predictor can ignore the action and still do well). The released
V-JEPA 2-AC model is instead validated by its action-conditioned energy
landscape: score a set of candidate actions by the predicted latent they
produce, and check whether the action that was really executed is the best one.

Chance level is 1/num_candidates for top-1 and 0.5 for the rank percentile.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

VJEPA_ROOT = Path("/workspace/vjepa2")
sys.path.insert(0, str(VJEPA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openpi.con2.ac_world_model import (  # noqa: E402
    ACTION_DIM,
    ACWorldModel,
    action_candidates,
    load_predictor,
    mean_rank_percentile,
    top1_rate,
)

TOKENS_PER_FRAME = 256
FEATURE_DIM = 1408


def normalize(x):
    return F.layer_norm(x, (x.shape[-1],))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("/workspace/artifacts/con2/ac_tokens_120"))
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--eval-episodes", type=int, nargs="+", default=list(range(100, 120)))
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1, help="Steps to roll out before scoring.")
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--windows", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    predictor = load_predictor(args.predictor, root=VJEPA_ROOT, device=device)
    model = ACWorldModel(predictor=predictor, device=device)
    rng = np.random.default_rng(args.seed)

    action_pool = []
    for episode in args.eval_episodes:
        action_pool.append(np.load(args.cache / "actions" / f"episode_{episode:06d}.npy"))
    pool = np.concatenate(action_pool)
    scale = float(pool.std())

    top1, percentiles, gaps, count = 0, [], [], 0
    for episode in args.eval_episodes:
        tokens = np.load(args.cache / "tokens" / f"episode_{episode:06d}.npy", mmap_mode="r")
        actions = np.load(args.cache / "actions" / f"episode_{episode:06d}.npy")
        states = np.load(args.cache / "states" / f"episode_{episode:06d}.npy")
        length = len(actions)
        if length < args.context + args.horizon + 1:
            continue
        starts = rng.integers(args.context - 1, length - args.horizon - 1, size=args.windows)
        for start in starts:
            start = int(start)
            context = np.asarray(tokens[start - args.context + 1:start + 1], dtype=np.float32)
            action_window = np.stack([actions[i] for i in range(start - args.context + 1, start + 1)])
            state_window = np.stack([states[i] for i in range(start - args.context + 1, start + 1)])
            true_action = actions[start]
            # Un-normalised target frame; score_actions applies the layer norm.
            target = np.asarray(tokens[start + args.horizon], dtype=np.float32)
            options = action_candidates(true_action, pool, args.candidates, rng, scale)
            scores, ranks = model.score_actions(context, action_window, state_window, options,
                                                target, steps=args.horizon)
            rank = int(ranks[0])
            top1 += int(rank == 0)
            percentiles.append(rank / (len(scores) - 1))
            gaps.append((min(scores) - scores[0]) / max(abs(scores[0]), 1e-9))
            count += 1

    report = {
        "predictor": str(args.predictor),
        "horizon": args.horizon,
        "candidates": args.candidates,
        "windows": count,
        "top1_rate": top1 / max(count, 1),
        "chance_top1": 1.0 / args.candidates,
        "mean_rank_percentile": float(np.mean(percentiles)) if percentiles else None,
        "chance_rank_percentile": 0.5,
        "action_scale": scale,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
