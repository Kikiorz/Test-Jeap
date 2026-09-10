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
from openpi.con2.ac_world_model import (  # noqa: E402
    ACWorldModel,
    build_token_adapter,
    load_predictor,
    teacher_forced_loss,
)

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


def sample_batch(store, batch_size, context, generator, stride=1):
    episodes = [store.episodes[generator.integers(len(store.episodes))] for _ in range(batch_size)]
    tokens, actions, states = [], [], []
    for episode in episodes:
        frames, chunk_actions, chunk_states = store.get(episode)
        length = store.lengths[episode]
        span = (context + 1) * stride
        start = int(generator.integers(0, max(1, length - span)))
        index = start + stride * np.arange(context + 1)
        tokens.append(np.asarray(frames[index], dtype=np.float32))
        actions.append(chunk_actions[index[:-1]])
        states.append(chunk_states[index[:-1]])
    return (torch.from_numpy(np.stack(tokens)), torch.from_numpy(np.stack(actions)),
            torch.from_numpy(np.stack(states)))


def normalize(x):
    return F.layer_norm(x, (x.shape[-1],))


@torch.no_grad()
def evaluate(predictor, store, context, horizons, windows_per_episode, device, action_mode="true",
             stride=1, input_mode="free", adapter=None):
    """Roll out from a real context and score against the true future frames.

    The context is ``context`` frames spaced ``stride`` apart (``stride=1`` is
    10 fps, ``stride=2`` is the 5 fps the released model was trained at); the
    prediction after ``h`` autoregressive steps is compared with the frame
    ``stride*h`` beyond the context and the copy baseline is the last context
    frame against the same target, so "1.0" means no better than repeating the
    current latent.

    ``input_mode`` selects what the rollout is conditioned on beyond the context:
    ``free`` repeats the last action and state (a genuine open-loop rollout),
    ``true`` feeds the actions and proprioceptive states that were actually
    executed, which is the setting a policy is in at test time (it always has
    proprioception and its own planned actions) and isolates latent-dynamics
    error from state-propagation error.
    """
    predictor.eval()
    generator = np.random.default_rng(1234)
    horizons = sorted(horizons)
    maximum = horizons[-1]
    results = {f"horizon_{h}": {"windows": 0, "error_model": 0.0, "error_copy": 0.0} for h in horizons}
    for episode in store.episodes:
        frames, actions, states = store.get(episode)
        length = store.lengths[episode]
        span = (context + maximum) * stride + 1
        if length < span:
            continue
        starts = generator.integers(0, length - span, size=windows_per_episode)
        for start in starts:
            start = int(start)
            context_index = start + stride * np.arange(context)
            base = int(context_index[-1])
            window = np.asarray(frames[context_index], dtype=np.float32)
            context_tensor = torch.from_numpy(window).to(device).reshape(
                1, context * TOKENS_PER_FRAME, FEATURE_DIM)
            if adapter is not None:
                context_tensor = adapter(context_tensor)
            hidden = normalize(context_tensor)
            action = torch.from_numpy(actions[context_index]).unsqueeze(0).to(device)
            state = torch.from_numpy(states[context_index]).unsqueeze(0).to(device)
            if action_mode == "shuffled":
                action = action.flip(1)
            elif action_mode == "zero":
                action = torch.zeros_like(action)
            current = normalize(torch.from_numpy(np.asarray(frames[base], dtype=np.float32)).to(device)).reshape(-1)
            for step in range(1, maximum + 1):
                raw = predictor(hidden, action, state)[:, -TOKENS_PER_FRAME:]
                if adapter is not None:
                    raw = adapter(raw)
                prediction = normalize(raw)
                single = prediction.reshape(1, 1, TOKENS_PER_FRAME, FEATURE_DIM)
                hidden = torch.cat([hidden.reshape(1, -1, TOKENS_PER_FRAME, FEATURE_DIM), single], dim=1)
                hidden = hidden.reshape(1, hidden.shape[1] * TOKENS_PER_FRAME, FEATURE_DIM)
                next_index = min(base + stride * step, length - 1)
                if input_mode == "true":
                    action = torch.cat([action, torch.from_numpy(
                        np.asarray(actions[next_index], dtype=np.float32)).reshape(1, 1, -1).to(device)], dim=1)
                    state = torch.cat([state, torch.from_numpy(
                        np.asarray(states[next_index], dtype=np.float32)).reshape(1, 1, -1).to(device)], dim=1)
                else:
                    action = torch.cat([action, action[:, -1:]], dim=1)
                    state = torch.cat([state, state[:, -1:]], dim=1)
                if step in horizons:
                    goal_index = min(base + stride * step, length - 1)
                    goal = normalize(torch.from_numpy(np.asarray(frames[goal_index], dtype=np.float32)).to(device)).reshape(-1)
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
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Frames between consecutive window steps; 2 is 5 fps, matching the "
                             "4 fps the released predictor was trained at.")
    parser.add_argument("--eval-input-mode", choices=["free", "true"], default="free",
                        help="Rollout conditioning: 'free' repeats the last action/state, "
                             "'true' feeds the executed actions and proprioceptive states.")
    parser.add_argument("--token-adapter", type=int, default=0,
                        help="Train a zero-initialised residual adapter on the frozen encoder tokens.")
    parser.add_argument("--adapter-width", type=int, default=512)
    parser.add_argument("--adapter-from", type=Path, default=None)
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
    adapter = None
    if args.token_adapter:
        adapter = build_token_adapter(width=args.adapter_width, device=device)
        if args.adapter_from is not None:
            adapter.load_state_dict(torch.load(args.adapter_from, map_location="cpu", weights_only=False)["adapter"])
        adapter.train()
    world_model = ACWorldModel(predictor=predictor, device=str(device), normalize_reps=True,
                               context_transform=adapter)
    parameters = sum(p.numel() for p in predictor.parameters())

    train_store = EpisodeStore(args.cache, args.train_episodes, action_scale=args.action_scale)
    eval_store = EpisodeStore(args.cache, args.eval_episodes, action_scale=args.action_scale)
    trainable = list(predictor.parameters()) + (list(adapter.parameters()) if adapter is not None else [])
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate,
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
        tokens, actions, states = sample_batch(train_store, args.batch_size, args.context, generator,
                                              args.frame_stride)
        tokens, actions, states = tokens.to(device), actions.to(device), states.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = teacher_forced_loss(world_model, tokens, actions, states, args.auto_steps,
                                       context_transform=adapter)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
                                        args.eval_windows, device, stride=args.frame_stride,
                                        input_mode=args.eval_input_mode, adapter=adapter)}
            for mode in args.action_ablation:
                results[mode] = evaluate(predictor, eval_store, args.context, args.horizons,
                                         args.eval_windows, device, action_mode=mode,
                                         stride=args.frame_stride, input_mode=args.eval_input_mode,
                                         adapter=adapter)
            record = {"step": step, "eval": results, "seconds": time.time() - started,
                      "parameters": parameters, "init": args.init}
            print(json.dumps(record), flush=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            torch.save({"predictor": predictor.state_dict(),
                        "adapter": adapter.state_dict() if adapter is not None else None,
                        "step": step, "args": vars(args)}, args.output / "predictor.pt")
    final = {"true": evaluate(predictor, eval_store, args.context, args.horizons, args.eval_windows,
                              device, stride=args.frame_stride, input_mode=args.eval_input_mode,
                              adapter=adapter)}
    for mode in args.action_ablation:
        final[mode] = evaluate(predictor, eval_store, args.context, args.horizons,
                               args.eval_windows, device, action_mode=mode, stride=args.frame_stride,
                               input_mode=args.eval_input_mode, adapter=adapter)
    (args.output / "final.json").write_text(json.dumps(
        {"eval": final, "args": {k: str(v) for k, v in vars(args).items()}}, indent=2, sort_keys=True) + "\n")
    torch.save({"predictor": predictor.state_dict(),
                "adapter": adapter.state_dict() if adapter is not None else None,
                "step": args.steps, "args": vars(args)}, args.output / "predictor.pt")


if __name__ == "__main__":
    main()
