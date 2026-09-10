#!/usr/bin/env python3
"""Fine-tune the V-JEPA 2-AC predictor on LIBERO (Con2, action-conditioned).

Only the predictor is trained; frames are the frozen AC-encoder tokens produced
by ``scripts/cache_ac_tokens.py``. The objective follows the released cooldown
config (``configs/train/vitg16/droid-256px-8f.yaml``):

  * teacher forcing over the whole window plus ``--auto-steps`` autoregressive
    refinements, summed;
  * smooth L1 (``loss_exp = 1.0``) on layer-normed representations;
  * AdamW with warmup + cosine decay.

Reported metric everywhere is NMSE against the copy-current baseline, i.e. 1.0
means "no better than repeating the current latent".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

VJEPA_ROOT = Path("/workspace/vjepa2")
sys.path.insert(0, str(VJEPA_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import src.hub.backbones as hub  # noqa: E402
from openpi.con2.ac_world_model import load_predictor  # noqa: E402

TOKENS_PER_FRAME = 256
FEATURE_DIM = 1408


class EpisodeStore:
    """Episode-major loader with a small LRU cache; tokens are fp16 on disk."""

    def __init__(self, root: Path, episodes, cache_size: int = 12, action_scale: float = 1.0):
        self.root = root
        self.episodes = list(episodes)
        self.action_scale = action_scale
        self.lengths = {e: int(np.load(root / "actions" / f"episode_{e:06d}.npy", mmap_mode="r").shape[0])
                        for e in self.episodes}
        self.cache_size = cache_size
        self.cache = OrderedDict()

    def get(self, episode):
        if episode not in self.cache:
            tokens = np.load(self.root / "tokens" / f"episode_{episode:06d}.npy", mmap_mode="r")
            actions = np.load(self.root / "actions" / f"episode_{episode:06d}.npy") * self.action_scale
            states = np.load(self.root / "states" / f"episode_{episode:06d}.npy")
            self.cache[episode] = (tokens, actions, states)
            while len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        value = self.cache.pop(episode)
        self.cache[episode] = value
        return value


def sample_batch(store, batch_size, context, generator):
    episodes = [store.episodes[generator.integers(len(store.episodes))] for _ in range(batch_size)]
    tokens, actions, states = [], [], []
    for episode in episodes:
        frames, chunk_actions, chunk_states = store.get(episode)
        length = store.lengths[episode]
        start = int(generator.integers(0, length - context - 1))
        tokens.append(np.asarray(frames[start:start + context + 1], dtype=np.float32))
        actions.append(chunk_actions[start:start + context])
        states.append(chunk_states[start:start + context])
    return (torch.from_numpy(np.stack(tokens)), torch.from_numpy(np.stack(actions)),
            torch.from_numpy(np.stack(states)))


def normalize(x):
    return F.layer_norm(x, (x.shape[-1],))


def predictor_loss(predictor, tokens, actions, states, auto_steps):
    """tokens: [B, K+1, 256, D] teacher-forced window; block k predicts frame k+1."""
    batch, frames, n_tokens, dim = tokens.shape
    context_frames = frames - 1
    hidden = normalize(tokens[:, :context_frames].reshape(batch, context_frames * n_tokens, dim))
    context_actions = actions[:, :context_frames]
    context_states = states[:, :context_frames]
    teacher = normalize(predictor(hidden, context_actions, context_states).reshape(
        batch, context_frames, n_tokens, dim))
    target = normalize(tokens[:, 1:].reshape(batch, context_frames * n_tokens, dim)).reshape(
        batch, context_frames, n_tokens, dim)
    loss = (teacher - target).abs().mean()

    rollout_losses = []
    if auto_steps > 1:
        current = torch.cat([normalize(tokens[:, :1]).reshape(batch, 1, n_tokens, dim),
                             teacher[:, :1].reshape(batch, 1, n_tokens, dim)], dim=1)
        for step in range(1, min(auto_steps, context_frames)):
            flat = current.reshape(batch, current.shape[1] * n_tokens, dim)
            next_hidden = predictor(flat, actions[:, :step + 1], states[:, :step + 1])
            next_hidden = normalize(next_hidden.reshape(batch, step + 1, n_tokens, dim))[:, -1:]
            current = torch.cat([current, next_hidden], dim=1)
            rollout_losses.append((current[:, -1] - target[:, step]).abs().mean())
    if rollout_losses:
        loss = loss + torch.stack(rollout_losses).mean()
    return loss, {"teacher": float((teacher - target).abs().mean())}


@torch.no_grad()
def evaluate(predictor, store, context, horizons, windows_per_episode, device, action_mode="true"):
    """Roll out from a real context and score against the true future frames.

    The context is frames ``start .. start+context-1``; the prediction after
    ``h`` autoregressive steps is compared with frame ``start+context-1+h`` and
    the copy baseline is frame ``start+context-1`` against the same target, so
    "1.0" means no better than repeating the current latent.
    """
    predictor.eval()
    generator = np.random.default_rng(1234)
    horizons = sorted(horizons)
    maximum = horizons[-1]
    results = {f"horizon_{h}": {"windows": 0, "error_model": 0.0, "error_copy": 0.0} for h in horizons}
    for episode in store.episodes:
        frames, actions, states = store.get(episode)
        length = store.lengths[episode]
        if length < context + maximum + 1:
            continue
        starts = generator.integers(0, length - context - maximum - 1, size=windows_per_episode)
        for start in starts:
            start = int(start)
            base = start + context - 1
            window = np.asarray(frames[start:start + context], dtype=np.float32)
            hidden = normalize(torch.from_numpy(window).to(device).reshape(
                1, context * TOKENS_PER_FRAME, FEATURE_DIM))
            action = torch.from_numpy(actions[start:start + context]).unsqueeze(0).to(device)
            state = torch.from_numpy(states[start:start + context]).unsqueeze(0).to(device)
            if action_mode == "shuffled":
                action = action.flip(1)
            elif action_mode == "zero":
                action = torch.zeros_like(action)
            current = normalize(torch.from_numpy(np.asarray(frames[base], dtype=np.float32)).to(device)).reshape(-1)
            for step in range(1, maximum + 1):
                prediction = normalize(predictor(hidden, action, state)[:, -TOKENS_PER_FRAME:])
                single = prediction.reshape(1, 1, TOKENS_PER_FRAME, FEATURE_DIM)
                hidden = torch.cat([hidden.reshape(1, -1, TOKENS_PER_FRAME, FEATURE_DIM), single], dim=1)
                hidden = hidden.reshape(1, hidden.shape[1] * TOKENS_PER_FRAME, FEATURE_DIM)
                action = torch.cat([action, action[:, -1:]], dim=1)
                state = torch.cat([state, state[:, -1:]], dim=1)
                if step in horizons:
                    goal = normalize(torch.from_numpy(np.asarray(frames[base + step], dtype=np.float32)).to(device)).reshape(-1)
                    record = results[f"horizon_{step}"]
                    record["windows"] += 1
                    record["error_model"] += float((prediction.reshape(-1) - goal).pow(2).sum())
                    record["error_copy"] += float((current - goal).pow(2).sum())
    for record in results.values():
        record["nmse_vs_copy_current"] = record["error_model"] / max(record["error_copy"], 1e-9)
    predictor.train()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--eval-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--auto-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--action-scale", type=float, default=1.0,
                        help="Scale the cached actions (LIBERO setpoints are ~0.012x the executed pose delta).")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=VJEPA_ROOT / "vjepa2-ac-vitg.pt")
    parser.add_argument("--init", choices=["pretrained", "random"], default="pretrained")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="Load predictor weights from an earlier run's predictor.pt instead.")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--eval-windows", type=int, default=8)
    parser.add_argument("--action-ablation", nargs="*", default=["shuffled", "zero"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activation-checkpointing", type=int, default=1,
                        help="Recompute blocks in backward; the bool attention mask is materialised.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.init_from is not None:
        predictor = load_predictor(args.init_from, root=VJEPA_ROOT, device=device)
    elif args.init == "pretrained":
        predictor = load_predictor(args.checkpoint, root=VJEPA_ROOT, device=device)
    else:
        _, predictor = hub._make_vjepa2_ac_model(pretrained=False)
        predictor = predictor.to(device)
    if args.activation_checkpointing:
        predictor.use_activation_checkpointing = True
    predictor.train()
    parameters = sum(p.numel() for p in predictor.parameters())

    train_store = EpisodeStore(args.cache, args.train_episodes, action_scale=args.action_scale)
    eval_store = EpisodeStore(args.cache, args.eval_episodes, action_scale=args.action_scale)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, betas=(0.9, 0.95))
    def schedule(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        progress = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    generator = np.random.default_rng(args.seed)
    log_path = args.output / "log.jsonl"
    started = time.time()
    for step in range(args.steps):
        tokens, actions, states = sample_batch(train_store, args.batch_size, args.context, generator)
        tokens, actions, states = tokens.to(device), actions.to(device), states.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, metrics = predictor_loss(predictor, tokens, actions, states, args.auto_steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step % 25 == 0 or step == args.steps - 1:
            record = {"step": step, "loss": float(loss), "lr": scheduler.get_last_lr()[0],
                      "seconds": time.time() - started}
            print(json.dumps(record), flush=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            results = {"true": evaluate(predictor, eval_store, args.context, args.horizons,
                                        args.eval_windows, device)}
            for mode in args.action_ablation:
                results[mode] = evaluate(predictor, eval_store, args.context, args.horizons,
                                         args.eval_windows, device, action_mode=mode)
            record = {"step": step, "eval": results, "seconds": time.time() - started,
                      "parameters": parameters, "init": args.init}
            print(json.dumps(record), flush=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            torch.save({"predictor": predictor.state_dict(), "step": step, "args": vars(args)},
                       args.output / "predictor.pt")
    final = {"true": evaluate(predictor, eval_store, args.context, args.horizons, args.eval_windows, device)}
    for mode in args.action_ablation:
        final[mode] = evaluate(predictor, eval_store, args.context, args.horizons,
                               args.eval_windows, device, action_mode=mode)
    (args.output / "final.json").write_text(json.dumps(
        {"eval": final, "args": {k: str(v) for k, v in vars(args).items()}}, indent=2, sort_keys=True) + "\n")
    torch.save({"predictor": predictor.state_dict(), "step": args.steps, "args": vars(args)},
               args.output / "predictor.pt")


if __name__ == "__main__":
    main()
