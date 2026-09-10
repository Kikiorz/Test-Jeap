"""Action-conditioned latent world model (V-JEPA 2-AC) for Con2.

The released V-JEPA 2-AC checkpoint is an action-conditioned world model: one
token block per frame

    [ action token a_k , state token s_k , 256 patch tokens z_k ]

with block-causal attention over frames, so output block ``k`` predicts frame
``k+1`` from ``(z_k, a_k, s_k)`` plus history. That is exactly the interface
Con2 and test-time training need.

This module is the single source of truth for the parts that are easy to get
quietly wrong, and it is what the ``scripts/probe_ac_*`` reproductions import:

* the LIBERO -> V-JEPA 2-AC input mapping (the two finger joints are mirrored,
  see :func:`map_libero_state`),
* frame encoding in the official ``normalize_reps`` convention,
* teacher-forced and autoregressive rollouts, and
* the action-ranking energy that the AC model is actually validated with
  (NMSE against a copy-current baseline is nearly blind to action
  conditioning; the ranking test is not).

The V-JEPA 2 model definitions live in the official repository, which is cloned
rather than vendored. Point ``VJEPA2_ROOT`` at it (default ``/workspace/vjepa2``)
or pass ``root=`` explicitly; imports are lazy so that ``map_libero_state`` and
the CPU-only tests work without torch models present.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

TOKENS_PER_FRAME = 256
FEATURE_DIM = 1408
ACTION_DIM = 7
STATE_DIM = 7
LIBERO_STATE_DIM = 8
IMAGE_SIZE = 256

"""Robosuite travel of one Panda finger, in metres."""
LIBERO_GRIPPER_TRAVEL = 0.04

"""Per-dimension least-squares scale that maps LIBERO OSC setpoints onto the
executed pose delta (16682 frames of episodes 0-59). V-JEPA 2-AC was trained
where ``action == pose delta`` (``compute_new_pose`` in the official notebook),
while a LIBERO setpoint is ramped by the controller over several control steps;
the raw action is therefore ~80x too large for zero-shot use."""
LIBERO_ACTION_CALIBRATION = np.array(
    [0.01073, 0.01195, 0.01134, 0.01288, 0.01288, 0.01169, 0.00760], dtype=np.float32
)

ACTION_VARIANTS = ("raw", "robosuite", "calibrated")


def normalize_reps(tokens):
    """The official ``normalize_reps`` convention: layer norm over the feature dim.

    The released checkpoint compares layer-normed representations on both sides
    of the loss, so every cached token is normalised before it is used and every
    prediction is normalised before it is compared.
    """
    import torch.nn.functional as F  # noqa: PLC0415

    return F.layer_norm(tokens, (tokens.shape[-1],))


def vjepa_root(root: Path | str | None = None) -> Path:
    """Locate the official V-JEPA 2 checkout."""
    candidate = Path(root) if root is not None else Path(os.environ.get("VJEPA2_ROOT", "/workspace/vjepa2"))
    if not (candidate / "src" / "models" / "ac_predictor.py").exists():
        raise FileNotFoundError(f"Not a V-JEPA 2 checkout: {candidate}")
    return candidate


def _import_vjepa(root: Path | str | None = None):
    repo = vjepa_root(root)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import src.hub.backbones as hub  # noqa: PLC0415
    from app.vjepa_droid.transforms import make_transforms  # noqa: PLC0415

    return hub, make_transforms


def map_libero_state(state: np.ndarray) -> np.ndarray:
    """LIBERO 8-d state -> the 7-d ``[xyz, euler-xyz, gripper]`` pose AC expects.

    LIBERO stores ``[eef_pos(3), eef_axis_angle(3), gripper_qpos(2)]``; the
    released checkpoint encodes state with ``Linear(7, D)`` and the official
    notebook feeds ``[xyz(3), euler_xyz(3), gripper(1)]``. Axis-angle is treated
    as euler-xyz, which agrees for the small rotations seen in these demos.

    The two finger joints are **mirrored** (``qpos[1] ~ -qpos[0]``, correlation
    0.997), so the opening fraction is the mean of their absolute values.
    Averaging them with sign cancels to ~0 and pins the gripper channel at 1.0
    for every frame, which silently removes the gripper from the world model's
    inputs (measured: action conditioning then looks like +0.002 NMSE instead of
    +0.065, and agreement with the executed action drops by more than half).
    """
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != LIBERO_STATE_DIM:
        raise ValueError(f"Expected LIBERO state[B,{LIBERO_STATE_DIM}], got {state.shape}")
    pose = np.empty((len(state), STATE_DIM), dtype=np.float32)
    pose[:, :6] = state[:, :6]
    opening = np.abs(state[:, 6:8]).mean(axis=1) / LIBERO_GRIPPER_TRAVEL
    pose[:, 6] = np.clip(1.0 - opening, 0.0, 1.0)
    return pose


def map_libero_action(action: np.ndarray, variant: str = "raw") -> np.ndarray:
    """LIBERO OSC_POSE action -> the metric deltas V-JEPA 2-AC was trained on.

    ``raw`` keeps the dataset values, ``robosuite`` applies the controller
    ``output_max`` (0.05 m, 0.5 rad) that robosuite uses to turn its normalised
    input into a setpoint, and ``calibrated`` applies
    :data:`LIBERO_ACTION_CALIBRATION`, which matches the executed pose delta.
    """
    if variant not in ACTION_VARIANTS:
        raise ValueError(f"variant must be one of {ACTION_VARIANTS}")
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected action[B,{ACTION_DIM}], got {action.shape}")
    if variant == "robosuite":
        scaled = action.copy()
        scaled[:, :3] *= 0.05
        scaled[:, 3:6] *= 0.5
        return scaled
    if variant == "calibrated":
        return action * LIBERO_ACTION_CALIBRATION
    return action.copy()


def frame_transform(root: Path | str | None = None):
    """Official deterministic 256 px transform (validation branch of the recipe)."""
    _, make_transforms = _import_vjepa(root)
    return make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=IMAGE_SIZE,
    )


def build_models(checkpoint: Path | str, *, encoder_key: str = "encoder",
                 root: Path | str | None = None, device="cpu", dtype=None):
    """Build the released AC encoder + predictor from a downloaded checkpoint.

    ``encoder_key`` selects the online encoder or the EMA ``target_encoder``.
    The released checkpoint trains the predictor on ``target_encoder`` features
    while the official notebook drives it with the online ``encoder``; they are
    nearly identical after the full cooldown and both probes agree, but the knob
    is exposed because the difference is not zero.
    """
    import torch  # noqa: PLC0415

    hub, _ = _import_vjepa(root)
    encoder, predictor = hub._make_vjepa2_ac_model(pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if encoder_key not in state:
        raise KeyError(f"{checkpoint} has no '{encoder_key}' weights")
    encoder.load_state_dict(hub._clean_backbone_key(dict(state[encoder_key])), strict=True)
    predictor.load_state_dict(hub._clean_backbone_key(dict(state["predictor"])), strict=True)
    del state
    if dtype is not None:
        encoder = encoder.to(dtype=dtype)
        predictor = predictor.to(dtype=dtype)
    return encoder.to(device).eval(), predictor.to(device).eval()


def load_predictor(source: Path | str, *, root: Path | str | None = None, device="cpu"):
    """Load only the predictor, from the released checkpoint or a fine-tuned file."""
    import torch  # noqa: PLC0415

    hub, _ = _import_vjepa(root)
    _, predictor = hub._make_vjepa2_ac_model(pretrained=False)
    state = torch.load(source, map_location="cpu", weights_only=False)
    payload = state.get("predictor", state)
    if any(key.startswith("module.") for key in payload):
        payload = {key.replace("module.", ""): value for key, value in payload.items()}
    predictor.load_state_dict(payload, strict=True)
    del state
    return predictor.to(device).eval()


@dataclass
class ACWorldModel:
    """Con2 interface: encode frames, roll the future out, score candidate actions.

    ``predictor`` is anything with the released signature
    ``predictor(context_tokens, actions, states) -> tokens``; tests pass a stub,
    training passes a fine-tuned :func:`load_predictor` module.
    """

    predictor: "object"
    encoder: "object | None" = None
    transform: "object | None" = None
    device: str = "cpu"
    tokens_per_frame: int = TOKENS_PER_FRAME
    normalize_reps: bool = True

    @staticmethod
    def _normalize(tokens):
        return normalize_reps(tokens)

    def _to_tensor(self, value):
        """Accept numpy or torch input; negative strides are common after slicing."""
        import torch  # noqa: PLC0415

        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32, device=self.device)

    def _step(self, context, action, state):
        """One predictor call on already-tensorised inputs."""
        import torch  # noqa: PLC0415

        context = self._normalize(context)
        with torch.no_grad():
            prediction = self.predictor(context, action, state)[:, -self.tokens_per_frame:]
            if self.normalize_reps:
                prediction = self._normalize(prediction)
        return prediction

    def encode(self, frames: np.ndarray, chunk: int = 8):
        """``[T, H, W, 3]`` uint8 frames -> ``[T, tokens_per_frame, FEATURE_DIM]``."""
        import torch  # noqa: PLC0415

        if self.encoder is None or self.transform is None:
            raise ValueError("encode() needs both an encoder and its frame transform")
        outputs = []
        for start in range(0, len(frames), chunk):
            clip = np.ascontiguousarray(frames[start:start + chunk])
            batch = self.transform(clip).unsqueeze(0)
            _, _, steps, _, _ = batch.shape
            # The official recipe encodes each frame independently as a 2-frame
            # clip (tubelet_size=2), never as a temporally attended clip.
            batch = batch.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
            with torch.no_grad():
                tokens = self.encoder(batch.to(self.device))
            outputs.append(tokens.reshape(steps, -1, tokens.shape[-1]).float().cpu())
        return torch.cat(outputs)

    def predict(self, context_tokens, actions, states):
        """One step: predict the next frame's tokens from a context window.

        ``context_tokens`` is ``[T, tokens_per_frame, D]``; the last block of the
        output predicts frame ``T`` (0-indexed) given ``(z_{T-1}, a_{T-1}, s_{T-1})``,
        which is the alignment the released training loop uses.
        """
        import torch  # noqa: PLC0415

        context = self._to_tensor(context_tokens)
        if context.ndim == 3:
            context = context.reshape(1, context.shape[0] * context.shape[1], context.shape[-1])
        action = self._to_tensor(actions).reshape(1, -1, ACTION_DIM)
        state = self._to_tensor(states).reshape(1, -1, STATE_DIM)
        return self._step(context, action, state)

    def rollout(self, context_tokens, actions, states, steps: int):
        """Autoregressively roll ``steps`` frames forward.

        Counterfactual rollouts repeat the new action and the last state, which is
        what the released MPC example does modulo its pose integration.
        """
        import torch  # noqa: PLC0415

        context = self._to_tensor(context_tokens)
        context = context.reshape(1, -1, self.tokens_per_frame, context.shape[-1])
        action = self._to_tensor(actions).reshape(1, -1, ACTION_DIM)
        state = self._to_tensor(states).reshape(1, -1, STATE_DIM)
        prediction = None
        for step in range(steps):
            prediction = self._step(context.reshape(1, -1, context.shape[-1]), action, state)
            if step == steps - 1:
                break
            context = torch.cat([context, prediction.reshape(1, 1, self.tokens_per_frame, -1)], dim=1)
            action = torch.cat([action, action[:, -1:]], dim=1)
            state = torch.cat([state, state[:, -1:]], dim=1)
        return prediction

    def score_actions(self, context_tokens, context_actions, context_states, candidates,
                      target_tokens, steps: int = 1):
        """Energy of each candidate action chunk (lower = closer to the real future).

        Returns ``(energies, ranks)`` where ``ranks[i]`` is the position of
        candidate ``i`` in the ascending energy order. The action-conditioning
        claim lives here: the executed action should rank near the top.
        """
        import torch  # noqa: PLC0415

        target = self._to_tensor(target_tokens)
        if self.normalize_reps:
            target = self._normalize(target)
        target = target.reshape(-1)
        energies = []
        for candidate in candidates:
            actions = np.array(context_actions, dtype=np.float32, copy=True)
            actions[-1] = np.asarray(candidate, dtype=np.float32)
            prediction = self.rollout(context_tokens, actions, context_states, steps)
            energies.append(float((prediction.reshape(-1) - target).pow(2).sum()))
        order = np.argsort(energies)
        ranks = np.empty(len(energies), dtype=int)
        ranks[order] = np.arange(len(energies))
        return energies, ranks


def action_candidates(true_action: np.ndarray, pool: np.ndarray, count: int, rng, scale: float) -> list:
    """Candidate set used by the ranking probe.

    Executed action, zeroed action, the action vector reversed (a scrambled
    action that keeps the magnitude), then a mix of actions sampled from the
    dataset and gaussian perturbations around the executed action at the
    dataset's action scale.
    """
    options = [np.asarray(true_action, dtype=np.float32),
               np.zeros_like(np.asarray(true_action, dtype=np.float32))]
    options.append(np.asarray(true_action, dtype=np.float32)[::-1].copy())
    while len(options) < count:
        if len(pool) and rng.random() < 0.5:
            options.append(np.asarray(pool[int(rng.integers(len(pool)))], dtype=np.float32))
        else:
            noise = rng.normal(0, scale, size=ACTION_DIM).astype(np.float32)
            options.append(np.asarray(true_action, dtype=np.float32) + noise)
    return options[:count]


def teacher_forced_loss(model: "ACWorldModel", tokens, actions, states, auto_steps: int = 2,
                        loss_exp: float = 1.0) -> "object":
    """The released cooldown objective on a batch of windows (training path).

    ``tokens`` is ``[B, K+1, tokens_per_frame, D]``; block ``k`` of the window
    predicts frame ``k+1``. The loss is the window-wide L1/L^``loss_exp`` error on
    layer-normed representations plus the autoregressive refinements, exactly as
    in ``app/vjepa_droid/train.py``.

    Unlike :meth:`ACWorldModel.predict` this keeps the graph and supports a batch
    dimension, so it is what both fine-tuning and test-time adaptation minimise.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    if tokens.ndim != 4:
        raise ValueError(f"Expected tokens[B, K+1, N, D], got {tuple(tokens.shape)}")
    batch, frames, n_tokens, dim = tokens.shape
    context_frames = frames - 1
    if context_frames < 1:
        raise ValueError("A window needs at least two frames")
    hidden = model._normalize(tokens[:, :context_frames].reshape(batch, context_frames * n_tokens, dim))
    teacher = model._normalize(model.predictor(hidden, actions[:, :context_frames],
                                              states[:, :context_frames]).reshape(
        batch, context_frames, n_tokens, dim))
    target = model._normalize(tokens[:, 1:].reshape(batch, context_frames * n_tokens, dim)).reshape(
        batch, context_frames, n_tokens, dim)
    loss = (teacher - target).abs().pow(loss_exp).mean() / loss_exp

    rollout_losses = []
    if auto_steps > 1:
        current = torch.cat([model._normalize(tokens[:, :1]).reshape(batch, 1, n_tokens, dim),
                             teacher[:, :1]], dim=1)
        for step in range(1, min(auto_steps, context_frames)):
            flat = current.reshape(batch, current.shape[1] * n_tokens, dim)
            next_hidden = model._normalize(model.predictor(
                flat, actions[:, :step + 1], states[:, :step + 1]).reshape(
                    batch, step + 1, n_tokens, dim))[:, -1:]
            current = torch.cat([current, next_hidden], dim=1)
            rollout_losses.append((current[:, -1] - target[:, step]).abs().pow(loss_exp).mean() / loss_exp)
    if rollout_losses:
        loss = loss + torch.stack(rollout_losses).mean()
    return loss


def adapt(model: "ACWorldModel", optimizer, tokens, actions, states, *, steps: int = 1,
          auto_steps: int = 2, clip: float = 1.0) -> list:
    """Test-time training: gradient steps on the cooldown objective.

    ``tokens/actions/states`` is a batch of *already observed* windows; nothing
    here is supervised by labels the policy cannot see at test time. Returns the
    per-step losses so a caller can log or early-stop.
    """
    import torch  # noqa: PLC0415

    model_name = type(model.predictor).__name__
    if not hasattr(model.predictor, "parameters"):
        raise TypeError(f"{model_name} has no parameters to adapt")
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = teacher_forced_loss(model, tokens, actions, states, auto_steps=auto_steps)
        loss.backward()
        if clip:
            torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), clip)
        optimizer.step()
        losses.append(float(loss.detach()))
    return losses


def mean_rank_percentile(ranks: Iterable[int], candidates: int) -> float:
    """0 = always best, 0.5 = chance."""
    ranks = np.asarray(list(ranks), dtype=float)
    if not len(ranks) or candidates < 2:
        return float("nan")
    return float((ranks / (candidates - 1)).mean())


def top1_rate(ranks: Sequence[int]) -> float:
    ranks = np.asarray(list(ranks))
    return float((ranks == 0).mean()) if len(ranks) else float("nan")
