# Con2 v2: action-conditioned world model (pretrained V-JEPA 2-AC)

Status: probes complete, fine-tuning in progress. Everything below is measured
on the frozen LIBERO LeRobot conversion (1693 episodes, 40 tasks, 10 fps,
256x256 agentview + wrist, 8-d state, 7-d OSC actions) unless stated otherwise.

## 1. Why the predictor is being replaced

The first Con2 head converged at a held-out NMSE of 0.472 on pooled V-JEPA 2.1
latents, with action conditioning buying only **0.002** NMSE (`+0.012` when the
action is shuffled or zeroed). That is a problem for the paper: the whole point
of Con2 is that the policy's predictive representation is *coupled to the
action*, and the metric that was supposed to show it does not.

Two things are wrong with the old setup:

1. The target is a spatial mean pool of the frame tokens, which averages away
   exactly the structure an action changes.
2. NMSE against a copy-current baseline is nearly blind to action conditioning:
   a predictor that reads only the state and the context frames can already
   place the next latent close to the truth.

Con2 v2 therefore replaces the predictor with the released **V-JEPA 2-AC**
action-conditioned world model and scores it with the test the model was
designed for (action ranking / energy landscape), not only with NMSE.

## 2. What "action-conditioned" means here

The released predictor is a 24-block, 1024-wide, 16-head ViT (305,220,992
parameters) that consumes one token block per frame:

```
frame block k = [ action token a_k , state token s_k , 256 patch tokens z_k ]
```

* `a_k` is the 7-d end-effector delta that was executed at frame `k`, embedded
  by `Linear(7, 1024)`; `s_k` is the 7-d pose `[xyz, euler-xyz, gripper]`,
  embedded by another `Linear(7, 1024)`.
* Attention is block-causal over frames, so output block `k` sees frames
  `<= k`, i.e. it predicts frame `k+1` from `(z_k, a_k, s_k)` plus history.
* Targets are the frozen encoder's layer-normed patch tokens; the loss is
  smooth L1 (`loss_exp = 1.0`) over the whole window plus autoregressive
  refinements. The released cooldown config trains at 256 px, 8 frames, 4 fps.

LIBERO mapping used by every script here:

| AC input | LIBERO source |
|---|---|
| `a_k` (7-d) | dataset `actions` (7-d OSC_POSE); `--action-variant robosuite` additionally applies the robosuite `output_max` scale (0.05 m, 0.5 rad) |
| `s_k` (7-d) | `state[:3]` position, `state[3:6]` axis-angle treated as euler-xyz, `state[6:8]` finger positions collapsed to `1 - mean(|qpos|)/0.04` clipped to `[0, 1]` |
| `z_k` | frozen AC encoder patch tokens of the agentview frame (`--camera wrist` for the wrist view) |

Two mapping details that were measured rather than assumed, because getting
them wrong silently degrades every run:

* The two finger joints are **mirrored** (`qpos[1] ~ -qpos[0]`, correlation
  0.997). Averaging them with sign cancels to ~0 and pins the gripper channel
  at 1.0 for every frame; the opening fraction is the mean of the absolute
  values. After the fix the closedness spans 0.003-0.946 and its per-frame
  change correlates 0.41 with the gripper action, so the channel is live.
* State deltas are ~**0.012x** the action magnitude for position dims
  (least squares over 16682 frames: 0.0107 / 0.0120 / 0.0113). The dataset
  action is a controller setpoint that the OSC controller ramps over several
  control steps, whereas V-JEPA 2-AC was trained where `action == pose delta`
  (`compute_new_pose` in `notebooks/utils/mpc_utils.py`). `--action-variant raw`
  is therefore ~80x too large for zero-shot use and the robosuite `output_max`
  variant is ~4x too large; a calibrated variant is an open item.

Checkpoint: `vjepa2-ac-vitg.pt` (11.76 GB, epoch 315, training loss 0.487) from
`https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt`. The released
`src/hub/backbones.py` points at a localhost URL, so the probe loads the file
directly and rebuilds the model with `_make_vjepa2_ac_model(pretrained=False)`.

## 3. Probe results

### 3.1 The probe itself is correct (in-distribution control)

`notebooks/franka_example_traj.npz`, the trajectory shipped with V-JEPA 2,
scored with the same code path (`scripts/probe_ac_control_franka.py`):

| window | NMSE vs copy-current |
|---|---:|
| 1-step, official Franka trajectory | **0.648** |

So the model does beat the copy baseline where it was trained; the code is not
misaligned with the official usage.

### 3.2 Zero-shot transfer to LIBERO fails

`scripts/probe_vjepa_ac_libero.py`, 2 episodes, 8-frame context, NMSE vs
copy-current (1.0 = repeat the current latent):

| camera | stride | encoder | actions | NMSE | action shuffled | action zeroed |
|---|---:|---|---|---:|---:|---:|
| agentview | 1 (10 fps) | online | raw | 1.187 | 1.189 | 1.173 |
| agentview | 2 (5 fps) | online | raw | 1.104 | 1.106 | 1.099 |
| agentview | 5 (2 fps) | online | raw | 1.035 | 1.035 | 1.037 |
| agentview | 2 | EMA target | raw | 1.104 | 1.106 | 1.099 |
| agentview | 2 | online | robosuite scale | 1.099 | 1.099 | 1.099 |
| wrist | 2 | online | raw | 1.045 | -- | -- |

Every configuration is at or worse than copying, and the action content changes
the result by less than 0.003. Repeated with the fine-tune evaluator on the
held-out episodes (context 8, horizons 1/2/4): 1.067 / 1.097 / 1.233.

### 3.3 The frozen latent space is still usable

Ridge readout on pooled frozen AC features, split by episode
(`scripts/probe_ac_space_sufficiency.py`; 5543 train / 1543 eval windows):

| features | held-out NMSE (h=1) |
|---|---:|
| copy current | 1.000 |
| ridge `lambda=100` | **0.933** (train 0.892) |

The space carries next-frame signal; it is the released predictor that does not
transfer. Fine-tuning is therefore the right move, not a different encoder.

### 3.4 Fine-tuning the predictor on 100 LIBERO episodes

`scripts/cache_ac_tokens.py` writes frozen patch tokens, then
`scripts/finetune_ac_predictor.py` trains the predictor only (encoder frozen),
with the official objective. Steps below are batch 8, context 8, `auto_steps=2`,
AdamW lr 1e-4 with a 100-step warmup, held-out episodes 100-119 (20 episodes),
context 8, horizons 1/2/4.

Before the gripper fix (state channel pinned at 1.0):

| step | h=1 | h=2 | h=4 | shuffled action | zeroed action |
|---:|---:|---:|---:|---:|---:|
| 0 (released weights) | 1.067 | 1.097 | 1.233 | 1.068 | 1.044 |
| 100 | 0.909 | 0.891 | 0.902 | 0.909 | 0.911 |
| 300 | 0.876 | 0.855 | 0.865 | 0.877 | 0.880 |
| 500 | 0.863 | -- | -- | -- | -- |

The action ablation is flat in that table (<=0.005), which is what motivated
the state-mapping audit in section 2.

After the gripper fix, 1000-step run (`ft_fix2_pre`, same data/schedule/seed):

| step | init | h=1 | h=2 | h=4 | action zeroed (h=1 / 2 / 4) |
|---:|---|---:|---:|---:|---|
| 499 | pretrained | 0.844 | 0.815 | 0.824 | 0.879 / 0.861 / 0.859 |
| 999 | pretrained | **0.809** | **0.764** | **0.760** | 0.874 / 0.858 / 0.853 |
| 499 | random | 1.229 | 1.123 | 1.058 | 1.227 / 1.123 / 1.067 |
| 999 | random | 1.119 | 1.026 | 0.962 | 1.128 / 1.055 / 1.026 |

Three conclusions, all from the same held-out episodes:

1. The adapted predictor beats copy-current everywhere (0.76-0.81 against 1.0
   at a 0.1-0.4 s horizon), and it keeps improving to the end of the schedule.
2. **The pretrained initialisation is worth a lot**: from random init the same
   schedule is still worse than copying at h=1-2 after 1000 steps.
3. **Action conditioning is now visible in the metric too**: zeroing the action
   chunk costs +0.065 / +0.094 / +0.093 NMSE (8-12% relative) at step 999,
   against +0.002 before the gripper fix. The gripper channel was the missing
   link that let the model relate the action to the future at all.

### 3.5 Action ranking (energy landscape): the test that does show it

NMSE is blind to action conditioning, so the predictor is scored the way the
V-JEPA 2-AC paper uses it: given the real context, score `N` candidate action
chunks by the latent they predict, and ask where the executed action ranks
(`scripts/probe_ac_action_ranking.py`; candidates = executed action, zero,
time-reversed, action chunks sampled from the dataset, and gaussian
perturbations at the dataset action scale).

Fine-tuned predictor, 12 candidates, horizon 1:

| model | windows | top-1 rate | chance | mean rank percentile |
|---|---:|---:|---:|---:|
| before the gripper fix | 400 | 0.1525 | 0.0833 | 0.253 |
| `ft_fix2_pre` (step 999) | 200 | **0.3100** | 0.0833 | **0.161** |

The executed action is the best-scoring candidate 3.7x more often than chance
and sits, on average, in the top sixth of the ranking, i.e. the predicted future
latent really is action-conditioned. This is the metric to quote for Con2:
NMSE-vs-copy hides action conditioning almost completely, while the ranking test
shows it directly.

## 4. Reproduction

```bash
# weight
aria2c -x 16 -s 16 -c -o vjepa2-ac-vitg.pt \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt

# machinery control (should be < 1.0)
python scripts/probe_ac_control_franka.py --rollout 1

# zero-shot transfer sweep
python scripts/probe_vjepa_ac_libero.py --episodes 0,1 --stride 2 --camera agentview

# frozen-feature sufficiency
python scripts/probe_ac_space_sufficiency.py --horizon 1

# token cache + fine-tune
python scripts/cache_ac_tokens.py --episodes $(seq 0 119) --output <cache>
python scripts/finetune_ac_predictor.py --cache <cache> \
  --train-episodes $(seq 0 99) --eval-episodes $(seq 100 119) \
  --steps 500 --batch-size 8 --device cuda:1 --output <run>

# action ranking / energy landscape
python scripts/probe_ac_action_ranking.py --predictor <run>/predictor.pt \
  --horizon 1 --candidates 12 --windows 20
```

## 5. Open items

1. Scale the fine-tune (440-episode cache, longer schedule) and add the matched
   from-scratch control at the same budget.
2. Longer horizons: the Con2 claim is about a ~1 s (10-step) horizon, so the
   fine-tuned predictor needs h=10 numbers and the ranking test at h=4.
3. Encoder adaptation: only the predictor has been trained; the AC encoder is
   still a DROID model. Unfreezing its last blocks is the obvious next lever if
   token-level NMSE saturates.
4. Wire the adapted predictor into Con2/TTT: the cached tokens already contain
   everything needed, and the action ranking is the objective TTT should
   improve.
