#!/usr/bin/env python3
"""Adapt the V-JEPA 2-AC encoder's last blocks together with the predictor.

The token cache stores features of the *released* encoder, which makes it the
perfect stop-gradient target: the context stream comes from a partially
unfrozen encoder, the target stream stays the frozen features we cached. That is
the same reason the released cooldown predicts an EMA target encoder instead of
itself - adapting both sides against each other collapses the representation.

Only the last ``--unfreeze-blocks`` transformer blocks plus the final norm are
trained on the encoder side, so the frozen prefix can run under ``no_grad`` and
is captured with a forward pre-hook.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from openpi.con2.ac_world_model import (  # noqa: E402
    TOKENS_PER_FRAME,
    ACWorldModel,
    build_models,
    frame_transform,
    load_predictor,
    normalize_reps,
    teacher_forced_loss_split,
)
from probe_vjepa_ac_libero import load_episode  # noqa: E402


class JointEncoder:
    """Frozen prefix + trainable tail of the AC encoder."""

    def __init__(self, encoder, transform, device, unfreeze_blocks: int):
        self.encoder = encoder
        self.transform = transform
        self.device = device
        self.blocks = list(encoder.blocks)
        self.depth = len(self.blocks)
        self.cut = max(0, self.depth - unfreeze_blocks)
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        for block in self.blocks[self.cut:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        encoder.norm.weight.requires_grad_(True)
        encoder.norm.bias.requires_grad_(True)
        self._captured = {}
        self._handle = self.blocks[self.cut].register_forward_pre_hook(
            lambda module, inputs: self._captured.__setitem__("x", inputs[0])
        )

    def trainable_parameters(self):
        parameters = [p for block in self.blocks[self.cut:] for p in block.parameters() if p.requires_grad]
        parameters += [self.encoder.norm.weight, self.encoder.norm.bias]
        return parameters

    def encode(self, frames: np.ndarray, chunk: int = 8):
        """``[T, H, W, 3]`` uint8 -> ``[T, tokens_per_frame, D]`` context tokens."""
        outputs = []
        for start in range(0, len(frames), chunk):
            clip = np.ascontiguousarray(frames[start:start + chunk])
            batch = self.transform(clip).unsqueeze(0)
            _, _, steps, _, _ = batch.shape
            batch = batch.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            batch = batch.to(self.device)
            with torch.no_grad():
                self.encoder(batch)  # frozen prefix; hook captures the tail input
            hidden = self._captured.pop("x")
            hidden = hidden.detach()
            for block in self.blocks[self.cut:]:
                hidden = block(hidden, mask=None, attn_mask=None, T=1,
                               H_patches=16, W_patches=16)
            hidden = self.encoder.norm(hidden)
            outputs.append(hidden.reshape(steps, TOKENS_PER_FRAME, -1).float())
        return torch.cat(outputs)


class Store:
    """Cached target tokens plus on-disk frames, keyed by episode."""

    def __init__(self, cache: Path, dataset: Path, episodes, camera: str, cache_size: int = 3):
        self.cache = cache
        self.dataset = dataset
        self.camera = camera
        self.episodes = list(episodes)
        self.frames = {}
        self.lengths = {}
        for episode in self.episodes:
            tokens = np.load(cache / "tokens" / f"episode_{episode:06d}.npy", mmap_mode="r")
            actions = np.load(cache / "actions" / f"episode_{episode:06d}.npy")
            states = np.load(cache / "states" / f"episode_{episode:06d}.npy")
            self.frames[episode] = (tokens, actions, states, None)
            self.lengths[episode] = len(actions)
        self.cache_size = cache_size

    def get(self, episode):
        tokens, actions, states, images = self.frames[episode]
        if images is None:
            frames, _, _ = load_episode(self.dataset, episode, self.camera)
            images = frames
            self.frames[episode] = (tokens, actions, states, images)
        return tokens, actions, states, images


def sample(store, batch_size, context, stride, generator):
    episodes = [store.episodes[generator.integers(len(store.episodes))] for _ in range(batch_size)]
    context_frames, targets, actions, states = [], [], [], []
    for episode in episodes:
        tokens, chunk_actions, chunk_states, images = store.get(episode)
        length = store.lengths[episode]
        span = (context + 1) * stride
        start = int(generator.integers(0, max(1, length - span)))
        index = start + stride * np.arange(context + 1)
        context_frames.append(np.asarray(images[index[:context]], dtype=np.uint8))
        targets.append(np.asarray(tokens[index[1:]], dtype=np.float32))
        actions.append(chunk_actions[index[:context]])
        states.append(chunk_states[index[:context]])
    return context_frames, (torch.from_numpy(np.stack(targets)),
                            torch.from_numpy(np.stack(actions)),
                            torch.from_numpy(np.stack(states)))


@torch.no_grad()
def evaluate(model, joint, store, context, horizons, windows, stride, device):
    model.predictor.eval()
    generator = np.random.default_rng(1234)
    maximum = max(horizons)
    results = {f"horizon_{h}": {"windows": 0, "error_model": 0.0, "error_copy": 0.0} for h in horizons}
    for episode in store.episodes:
        tokens, actions, states, images = store.get(episode)
        length = store.lengths[episode]
        span = (context + maximum) * stride + 1
        if length < span:
            continue
        starts = generator.integers(0, length - span, size=windows)
        for start in starts:
            start = int(start)
            index = start + stride * np.arange(context)
            base = int(index[-1])
            window = np.asarray(images[index], dtype=np.uint8)
            hidden = normalize_reps(joint.encode(window).to(device)).reshape(
                1, context * TOKENS_PER_FRAME, -1)
            action = torch.from_numpy(actions[index]).unsqueeze(0).to(device)
            state = torch.from_numpy(states[index]).unsqueeze(0).to(device)
            current = normalize_reps(torch.from_numpy(
                np.asarray(tokens[base], dtype=np.float32)).to(device)).reshape(-1)
            for step in range(1, maximum + 1):
                prediction = normalize_reps(model.predictor(hidden, action, state)[:, -TOKENS_PER_FRAME:])
                hidden = torch.cat([hidden.reshape(1, -1, TOKENS_PER_FRAME, prediction.shape[-1]),
                                    prediction.reshape(1, 1, TOKENS_PER_FRAME, -1)], dim=1)
                hidden = hidden.reshape(1, hidden.shape[1] * TOKENS_PER_FRAME, -1)
                action = torch.cat([action, action[:, -1:]], dim=1)
                state = torch.cat([state, state[:, -1:]], dim=1)
                if step in horizons:
                    goal_index = min(base + stride * step, length - 1)
                    goal = normalize_reps(torch.from_numpy(np.asarray(
                        tokens[goal_index], dtype=np.float32)).to(device)).reshape(-1)
                    record = results[f"horizon_{step}"]
                    record["windows"] += 1
                    record["error_model"] += float((prediction.reshape(-1) - goal).pow(2).sum())
                    record["error_copy"] += float((current - goal).pow(2).sum())
    for record in results.values():
        record["nmse_vs_copy_current"] = record["error_model"] / max(record["error_copy"], 1e-9)
    model.predictor.train()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/workspace/artifacts/datasets/lerobot_libero"))
    parser.add_argument("--checkpoint", type=Path, default=VJEPA_ROOT / "vjepa2-ac-vitg.pt")
    parser.add_argument("--train-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--eval-episodes", type=int, nargs="+", required=True)
    parser.add_argument("--init-from", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera", choices=["agentview", "wrist"], default="agentview")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--auto-steps", type=int, default=4)
    parser.add_argument("--unfreeze-blocks", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--eval-windows", type=int, default=6)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    encoder, _ = build_models(args.checkpoint, root=VJEPA_ROOT, device=device)
    encoder.use_activation_checkpointing = True
    if args.init_from is not None:
        predictor = load_predictor(args.init_from, root=VJEPA_ROOT, device=device)
    else:
        _, predictor = build_models(args.checkpoint, root=VJEPA_ROOT, device=device)
    predictor.use_activation_checkpointing = True
    predictor.train()

    joint = JointEncoder(encoder, frame_transform(VJEPA_ROOT), device, args.unfreeze_blocks)
    model = ACWorldModel(predictor=predictor, device=str(device), normalize_reps=True)
    train_store = Store(args.cache, args.dataset, args.train_episodes, args.camera)
    eval_store = Store(args.cache, args.dataset, args.eval_episodes, args.camera)

    encoder_parameters = joint.trainable_parameters()
    optimizer = torch.optim.AdamW([
        {"params": list(predictor.parameters()), "lr": args.learning_rate},
        {"params": encoder_parameters, "lr": args.encoder_learning_rate},
    ], weight_decay=args.weight_decay, betas=(0.9, 0.95))
    generator = np.random.default_rng(args.seed)
    started = time.time()
    log_path = args.output / "log.jsonl"
    for step in range(args.steps):
        context_frames, (targets, actions, states) = sample(
            train_store, args.batch_size, args.context, args.frame_stride, generator)
        frames = np.concatenate(context_frames, axis=0)
        context_tokens = joint.encode(frames).reshape(
            args.batch_size, args.context, TOKENS_PER_FRAME, -1)
        targets, actions, states = targets.to(device), actions.to(device), states.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = teacher_forced_loss_split(model, context_tokens, targets, actions, states,
                                         auto_steps=args.auto_steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(predictor.parameters()) + encoder_parameters, 1.0)
        optimizer.step()
        if step % 25 == 0 or step == args.steps - 1:
            record = {"step": step, "loss": float(loss.detach()), "seconds": time.time() - started}
            print(json.dumps(record), flush=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            results = evaluate(model, joint, eval_store, args.context, args.horizons,
                               args.eval_windows, args.frame_stride, device)
            record = {"step": step, "eval": results, "seconds": time.time() - started,
                      "unfreeze_blocks": args.unfreeze_blocks, "init_from": str(args.init_from)}
            print(json.dumps(record), flush=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            torch.save({"predictor": predictor.state_dict(),
                        "encoder_tail": [p.detach().cpu() for p in encoder_parameters],
                        "unfreeze_blocks": args.unfreeze_blocks, "step": step},
                       args.output / "joint.pt")


if __name__ == "__main__":
    main()
