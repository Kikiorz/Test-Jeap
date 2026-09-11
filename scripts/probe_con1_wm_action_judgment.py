#!/usr/bin/env python3
"""Judgment measurement: does the world-model gradient improve the *policy's* action?

The proposed Con1-TTT design hinges on one property: taking a V-JEPA 2-AC
gradient step at the chunk the policy actually produced must move that chunk
towards the demonstrated one. If it does, the world model carries a usable
action-side signal and the TTT/coupling design is worth building. If it only
lowers the world model's own error while pushing the action away from the
demonstration, the design is dead and no GPU should be spent on it.

Reported per sample:
  * E(a_policy) vs E(a_refined)  - the world model's own energy (sanity)
  * d_demo(a_policy) vs d_demo(a_refined) - the judgment
  * the world-model rank of the demonstrated chunk among candidates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

VJEPA_ROOT = Path("/workspace/vjepa2")
sys.path.insert(0, str(VJEPA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openpi.con2.ac_world_model import TOKENS_PER_FRAME, load_predictor  # noqa: E402

STRIDE = 2
ACTION_DIMS = 7  # the policy emits the model's padded width; the world model takes 7


def energy(predictor, context, states, executed, actions, target):
    """Roll the world model ``actions.shape[1]`` steps and score the final latent.

    ``context`` is ``[B, C, N, D]`` observed frames, ``states`` the matching
    ``[B, C, 7]``, ``executed`` the ``[B, C-1, 7]`` actions already taken for
    those frames, and ``actions`` the ``[B, H, 7]`` candidate chunk. The world
    model needs equal frame/action/state counts (block k predicts frame k+1),
    exactly as in its training loop.
    """
    batch, context_frames = context.shape[0], context.shape[1]
    hidden = context.reshape(batch, -1, context.shape[-1])
    state_seq = states
    action_seq = torch.cat([executed, actions[:, :1]], dim=1)
    prediction = None
    for step in range(actions.shape[1]):
        if step > 0:
            action_seq = torch.cat([action_seq, actions[:, step:step + 1]], dim=1)
            state_seq = torch.cat([state_seq, state_seq[:, -1:]], dim=1)
        prediction = predictor(hidden, action_seq, state_seq)[:, -TOKENS_PER_FRAME:]
        if step == actions.shape[1] - 1:
            break
        hidden = torch.cat([
            hidden.reshape(batch, -1, TOKENS_PER_FRAME, hidden.shape[-1]),
            prediction.reshape(batch, 1, TOKENS_PER_FRAME, -1),
        ], dim=1).reshape(batch, -1, hidden.shape[-1])
    return (prediction - target).pow(2).sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True, help="npz written by dump_con1_policy_chunks.py")
    parser.add_argument("--cache", type=Path, default=Path("/workspace/artifacts/con2/ac_tokens_440"))
    parser.add_argument("--predictor", type=Path,
                        default=Path("/workspace/artifacts/con2/ft_440_s2_plus3k/predictor.pt"))
    parser.add_argument("--context", type=int, default=8, help="context frames at stride 2")
    parser.add_argument("--horizon", type=int, default=5, help="world-model steps (1 s at 2 s^-1)")
    parser.add_argument("--step", type=float, default=0.1, help="relative step size ||da||/||a||")
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    predictor = load_predictor(args.predictor, root=VJEPA_ROOT, device=device)
    data = np.load(args.chunks)
    episodes, frames = data["episode"], data["frame"]
    policy_chunks, demo_chunks = data["policy_chunk"], data["demo_chunk"]

    rng = np.random.default_rng(0)
    records = []
    for index in range(len(frames)):
        episode, frame = int(episodes[index]), int(frames[index])
        tokens = np.load(args.cache / "tokens" / f"episode_{episode:06d}.npy", mmap_mode="r")
        states = np.load(args.cache / "states" / f"episode_{episode:06d}.npy")
        executed_all = np.load(args.cache / "actions" / f"episode_{episode:06d}.npy")
        context_index = frame - STRIDE * np.arange(args.context - 1, -1, -1)
        target_index = frame + STRIDE * args.horizon
        if context_index[0] < 0 or target_index >= len(states):
            continue
        context = torch.from_numpy(np.asarray(tokens[context_index], dtype=np.float32)).unsqueeze(0).to(device)
        state = torch.from_numpy(np.asarray(states[context_index], dtype=np.float32)).unsqueeze(0).to(device)
        executed = torch.from_numpy(np.asarray(
            executed_all[context_index[:-1]], dtype=np.float32)).unsqueeze(0).to(device)
        target = torch.from_numpy(np.asarray(tokens[target_index], dtype=np.float32)).to(device)

        policy_actions = torch.from_numpy(
            policy_chunks[index][::STRIDE][:args.horizon][:, :ACTION_DIMS]).float().to(device).unsqueeze(0)
        demo_actions = torch.from_numpy(
            demo_chunks[index][::STRIDE][:args.horizon][:, :ACTION_DIMS]).float().to(device).unsqueeze(0)
        if policy_actions.shape[1] != args.horizon or demo_actions.shape[1] != args.horizon:
            continue

        policy_actions = policy_actions.detach().requires_grad_(True)
        with torch.enable_grad():
            e_policy = energy(predictor, context, state, executed, policy_actions, target)
            gradient = torch.autograd.grad(e_policy, policy_actions)[0]
        norm = gradient.norm()
        step = args.step * policy_actions.detach().norm() / norm.clamp_min(1e-12)
        refined = (policy_actions.detach() - step * gradient).detach()
        with torch.no_grad():
            e_refined = energy(predictor, context, state, executed, refined, target)
            candidates = [refined, demo_actions, torch.zeros_like(demo_actions)]
            while len(candidates) < args.candidates:
                noise = torch.from_numpy(rng.normal(0, 0.05, size=demo_actions.shape).astype(np.float32)).to(device)
                candidates.append((demo_actions + noise).detach())
            perturbations = torch.cat(candidates[2:], dim=0)
            energies = torch.stack([
                energy(predictor, context.repeat(len(perturbations), 1, 1, 1),
                       state.repeat(len(perturbations), 1, 1),
                       executed.repeat(len(perturbations), 1, 1), perturbations,
                       target.unsqueeze(0).repeat(len(perturbations), 1, 1)).reshape(1),
                e_refined.reshape(1),
                energy(predictor, context, state, executed, demo_actions, target).reshape(1),
            ])
        distance_policy = float((policy_actions.detach() - demo_actions).norm())
        distance_refined = float((refined - demo_actions).norm())
        records.append({
            "episode": episode, "frame": frame,
            "energy_policy": float(e_policy), "energy_refined": float(e_refined),
            "energy_reduction": float(e_policy - e_refined),
            "distance_policy": distance_policy, "distance_refined": distance_refined,
            "distance_change": distance_refined - distance_policy,
            "demo_rank": int((energies < energies[-1]).sum()),
            "demo_energy": float(energies[-1]),
            "policy_energy": float(energies[-2]),
            "headroom": distance_policy / max(float(demo_actions.norm()), 1e-9),
        })
        print(json.dumps(records[-1]), flush=True)

    deltas = np.asarray([r["distance_change"] for r in records])
    summary = {
        "predictor": str(args.predictor),
        "samples": len(records),
        "step_relative": args.step,
        "horizon_steps": args.horizon,
        "mean_energy_reduction": float(np.mean([r["energy_reduction"] for r in records])) if records else None,
        "mean_distance_policy": float(np.mean([r["distance_policy"] for r in records])) if records else None,
        "mean_distance_refined": float(np.mean([r["distance_refined"] for r in records])) if records else None,
        "mean_distance_change": float(deltas.mean()) if len(deltas) else None,
        "fraction_closer_to_demo": float((deltas < 0).mean()) if len(deltas) else None,
        "chance_fraction": 0.5,
        "mean_demo_rank": float(np.mean([r["demo_rank"] for r in records])) if records else None,
        "mean_headroom": float(np.mean([r["headroom"] for r in records])) if records else None,
        "records": records,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print("RESULT", json.dumps({k: v for k, v in summary.items() if k != "records"}, indent=1))
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
