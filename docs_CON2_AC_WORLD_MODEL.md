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

Same recipe on the larger cache (400 training episodes, held-out episodes
400-439, 2000-step schedule):

| step | h=1 | h=2 | h=4 | action zeroed (h=1 / 2 / 4) |
|---:|---:|---:|---:|---|
| 499 | 0.877 | 0.854 | 0.862 | 0.897 / 0.880 / 0.886 |
| 999 | 0.846 | 0.815 | 0.809 | 0.898 / 0.884 / 0.887 |
| 1499 | 0.836 | 0.787 | 0.778 | 0.887 / 0.871 / 0.864 |
| 1999 | **0.831** | **0.785** | **0.772** | 0.891 / 0.876 / 0.870 |

More data needs more steps to pay off (at equal steps the 400-episode run trails
the 100-episode one, on a harder held-out set), but it keeps improving through
2000 steps and the action gap is stable at +0.05 to +0.10 NMSE.

Headline numbers to quote (copy-current = 1.0). The held-out episode set differs
between rows, so compare rows only through the copy-current normalisation:

| setting | held-out episodes | h=1 (0.1 s) | h=2 (0.2 s) | h=4 (0.4 s) |
|---|---|---:|---:|---:|
| released AC weights, zero-shot | 100-119 | 1.067 | 1.097 | 1.233 |
| fine-tuned, 100 episodes / 1000 steps | 100-119 | 0.809 | 0.764 | 0.760 |
| fine-tuned, 400 episodes / 2000 steps | 400-439 | 0.831 | 0.785 | 0.772 |
| from random init, 100 episodes / 1000 steps | 100-119 | 1.119 | 1.026 | 0.962 |

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
| `ft_fix2_pre` step 999, horizon 1 | 200 | **0.3100** | 0.0833 | 0.161 |
| `ft_fix2_pre` step 999, horizon 4 | 100 | **0.3900** | 0.0833 | **0.115** |
| `ft_stride2_ar4`, 5 Hz, horizon 1 (0.2 s) | 200 | **0.4900** | 0.0833 | 0.090 |
| `ft_stride2_ar4`, 5 Hz, horizon 2 (0.4 s) | 200 | **0.6700** | 0.0833 | **0.054** |

The last two rows are the model used with the timebase it was trained at, and
they are the numbers to quote: with 12 candidates the executed action is
ranked best **49% of the time at 0.2 s and 67% at 0.4 s** (chance 8.3%), and it
sits in the top 5-9% of the ranking on average.

### 3.5b The same test on the original Con2 head

`scripts/eval_con1_head_ranking.py` runs the identical protocol on the
in-policy delta head that Con2 used before this change
(`pi05_libero_con1_action_only_40k / con1_action_only_stage1_2k`, the
action-conditioned head), 384 held-out windows, 8 candidates (chance 12.5%):

| model | horizon | top-1 | mean rank percentile | NMSE vs copy |
|---|---:|---:|---:|---:|
| old in-policy delta head | 1 (0.1 s) | **3.4%** | 0.433 | 0.866 |
| old in-policy delta head | 5 (0.5 s) | **8.3%** | 0.410 | 0.551 |
| old in-policy delta head | 10 (1.0 s) | **12.5%** | 0.388 | 0.375 |
| V-JEPA 2-AC world model | 1 (0.2 s) | 49.0% | 0.090 | - |
| V-JEPA 2-AC world model | 2 (0.4 s) | 67.0% | 0.054 | - |

The old head is at or **below chance**: at its shortest horizon the executed
action is the worst-scoring of eight candidates 3.4% of the time, i.e. the head
has learned uses of the action chunk that have nothing to do with which action
was executed - which is exactly why its NMSE gain over an unconditioned head was
only 0.002. Even at h=10, where the copy baseline is weakest and the head's NMSE
is its best (0.375), the ranking is exactly chance.

Caveat: the two candidate sets differ in size (8 vs 12) and perturbation scale,
and the old head was only trained to stage 1 (2k steps). Top-1 rate against
chance is the comparable part, and on that axis the gap is 3.4%/12.5% = 0.27
versus 49%/8.3% = 5.9.

The executed action is the best-scoring candidate 3.7x more often than chance
(4.7x at a 0.4 s horizon) and sits, on average, in the top sixth of the ranking,
i.e. the predicted future latent really is action-conditioned. This is the
metric to quote for Con2: NMSE-vs-copy hides action conditioning almost
completely, while the ranking test shows it directly.

### 3.6 The 1 s horizon is not there yet

The Con2 claim is about a ~1 s horizon, which is 10 autoregressive steps on this
cache. 40 held-out episodes, 4 windows each, identical windows for both rows:

| model | h=1 (0.1 s) | h=4 (0.4 s) | h=10 (1.0 s) |
|---|---:|---:|---:|
| `ft_440_pre` (auto_steps=2) | 0.8355 | 0.7861 | 0.9705 |
| `ft_440_ar4` (+1000 steps, auto_steps=4) | **0.8270** | **0.7597** | **0.9384** |

Short-horizon prediction is solid, but free-running rollout to 1 s decays to
near the copy baseline by h=10, and re-training the schedule for longer rollouts
only recovers ~3%. Two known causes, both addressable:

1. The schedule above trained with `auto_steps = 2`, so the model was never
   optimised for a 10-step rollout; `ft_440_ar4` re-fine-tunes it with
   `auto_steps = 4` on the same cache.
2. A 10-step rollout pushes the context to 18 frames, beyond the 8-frame window
   the released model was trained with.

### 3.7 Action scale

LIBERO setpoints are ~0.012x the executed pose delta (section 2), so the same
100-episode/1000-step schedule was repeated with `--action-scale 0.012`
(`ft_cal_pre`):

| variant | step | h=1 | h=2 | h=4 | action zeroed (h=1 / 2 / 4) |
|---|---:|---:|---:|---:|---|
| raw | 499 | 0.844 | 0.815 | 0.824 | 0.879 / 0.861 / 0.859 |
| raw | 999 | **0.809** | **0.764** | **0.760** | 0.874 / 0.858 / 0.853 |
| calibrated (0.012x) | 499 | 0.849 | 0.820 | 0.824 | 0.855 / 0.830 / 0.831 |
| calibrated (0.012x) | 999 | 0.813 | 0.767 | 0.760 | 0.846 / 0.825 / 0.829 |

Accuracy is the same at the end of the schedule, but the **action ablation gap is
2-3x smaller** under the calibrated scale (+0.033 / +0.058 / +0.070 against
+0.065 / +0.094 / +0.093). Shrinking the action to the size of the pose delta
makes it largely redundant with the state token, so the predictor leans on it
less. The raw scale stays the default.

### 3.8 Test-time training

`scripts/con2_ttt_adapt.py` adapts the predictor online on a stream of new
episodes through ``ACWorldModel.adapt`` (same cooldown objective, observed
transitions only). Protocol with no leakage: score the model on the **second
half** of each episode, adapt on the **first half only**, then score the same
untouched second half again.

| adaptation | episodes | NMSE before | NMSE after | improved |
|---|---:|---:|---:|---:|
| 4 steps, lr 1e-5 | 5 | 0.8494 | 0.8511 | 3/5 |
| 30 steps, lr 1e-4 | 8 | 0.9645 | **0.9465** | 5/8 |

At a trivial budget the effect is noise and one episode regresses; with 30 steps
the held-out error drops ~2%, so the TTT path works but needs a real schedule
(more steps, replay across episodes, early stopping on the observed transition
loss) before it is a paper number.

### 3.8b Timebase: match the model, not the dataset

The released predictor was trained on DROID at **4 fps**, while the cache is
the 10 fps LeRobot conversion, so a "1 s" horizon meant a 10-step rollout of
0.1 s steps - far outside the rate the model learned. Re-running the same recipe
with `--frame-stride 2` (5 fps, so h=5 is one second) fixes this, on the same
100 training episodes and the same held-out episodes:

| schedule | horizon | 0.2 s | 0.4 s | 1.0 s | action zeroed at 1.0 s |
|---|---|---:|---:|---:|---:|
| stride 1 (10 fps), auto_steps=2 | h=10 | - | - | 0.9705 | - |
| stride 1 (10 fps), auto_steps=4 | h=10 | - | - | 0.9384 | - |
| stride 2 (5 fps), auto_steps=2, step 999 | h=1/2/5 | 0.725 | 0.682 | **0.752** | 0.849 |
| stride 2 (5 fps), auto_steps=4, step 999 | h=1/2/5 | **0.710** | **0.656** | **0.731** | 0.829 |
| stride 2 (5 fps), auto_steps=4, 400 episodes | h=1/2/5 | 0.783 | 0.736 | **0.783** | 0.838 |
| stride 2 + executed actions/states | h=1/2/5 | 0.725 | 0.680 | **0.722** | 0.779 |

(The 400-episode row uses held-out episodes 400-439 and is therefore not
comparable row-for-row with the 100-episode rows; the rows above it are.)

Two things improve at once: the 1 s NMSE drops from ~0.97 to 0.74-0.79, and the
action ablation gap grows from +0.03 to +0.09-0.16. Feeding the model the rates
it was trained at is worth far more than any schedule tweak tried above, which
is the strongest argument for keeping the world model's timebase explicit in
the Con2 design.

### 3.8c Representation adaptation (negative result)

The cheap half of encoder adaptation is a trainable residual adapter on the
frozen encoder tokens (``build_token_adapter``, zero-initialised so it starts as
the identity and can only help). Same recipe as the stride-2/auto_steps=4 row
above, 500 steps, adapter width 512, adapter trained jointly with the predictor:

| model | h=1 (0.2 s) | h=2 (0.4 s) | h=5 (1.0 s) | action zeroed at 1 s |
|---|---:|---:|---:|---:|
| no adapter (baseline) | 0.7301 | 0.6863 | **0.7423** | 0.8292 |
| + token adapter | 0.7374 | 0.6905 | 0.7560 | 0.8356 |

The adapter does not help - it is marginally worse at every horizon, so the
frozen representation is not what limits this model at this data scale.

One trap worth recording: the first run of this experiment scored 0.96 at 1 s,
which looked like a collapse. It was a train/eval inconsistency - the
autoregressive branch appended *unadapted* predictions to an adapted context,
so training never saw the context that inference produces. ``teacher_forced_loss``
now applies ``context_transform`` to fed-back predictions as well, and the
experiment above is the corrected one. Anything that transforms tokens must
transform them on both the teacher-forced *and* the fed-back path.

### 3.8d Unfreezing encoder blocks (also negative)

The real version of encoder adaptation: keep the cached frozen-encoder features
as the stop-gradient target (the released EMA argument) and train the last
``--unfreeze-blocks`` transformer blocks plus the final norm from the raw
frames, jointly with the predictor. The frozen prefix runs under ``no_grad``
and is captured with a forward pre-hook
(``scripts/finetune_ac_joint.py``). Starting from the same adapted predictor as
the baseline above, 300 steps, 100 episodes, encoder lr 1e-5:

| model | h=1 (0.2 s) | h=2 (0.4 s) | h=5 (1.0 s) |
|---|---:|---:|---:|
| frozen encoder (paired baseline) | **0.7250** | **0.6705** | **0.7398** |
| + last 4 blocks trainable, step 149 | 0.7625 | 0.7393 | 0.7975 |
| + last 4 blocks trainable, step 299 | 0.7393 | 0.6983 | 0.7487 |

Still worse than freezing, at every horizon and both checkpoints, even though
the training loss does go down (0.47 -> 0.39). Two independent representation
adaptations (3.8c and 3.8d) therefore both fail, which is a useful negative
result: at this data scale the frozen DROID features are **not** what limits the
world model. The remaining error is in the rollout itself, so the levers left
are scale (more episodes/steps), a longer autoregressive curriculum, or a
different objective - not representation surgery.

The last row is the operational setting rather than a free-running rollout: the
policy always has proprioception and its own planned actions at test time, so
conditioning on them removes state-propagation error (0.752 -> 0.722 at 1 s).
The gap is small, so latent-dynamics error - not state drift - is what limits
the 1 s horizon.

## 3.9 Library surface

Everything above is reproduced by the ``scripts/probe_ac_*`` entry points, and
the reusable pieces now live in one place:
``src/openpi/con2/ac_world_model.py`` provides

* ``map_libero_state`` / ``map_libero_action`` (with the mirrored-finger and
  action-scale traps documented and tested),
* ``build_models`` / ``load_predictor`` / ``frame_transform``,
* ``ACWorldModel`` with ``encode``, ``predict``, ``rollout`` and
  ``score_actions`` (the energy/ranking objective TTT would optimise), and
* ``action_candidates``, ``top1_rate``, ``mean_rank_percentile``.

``src/openpi/con2/test_ac_world_model.py`` covers the mappings, the rollout
plumbing, the ranking objective and the candidate set on CPU with a stub
predictor (8 tests, no checkpoint and no GPU). The scripts were refactored onto
this module and re-verified against the numbers above: the control probe returns
0.64850 and the token pipeline returns bit-identical
``mse_copy_current = 314.67804217097733`` before and after the refactor.

## 4. Reproduction

```bash
# library surface + CPU tests (no checkpoint, no GPU)
PYTHONPATH=src python -m pytest src/openpi/con2/test_ac_world_model.py

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

# continue from an adapted predictor and evaluate long horizons
python scripts/finetune_ac_predictor.py --cache <cache> \
  --train-episodes $(seq 0 399) --eval-episodes $(seq 400 439) \
  --init-from <run>/predictor.pt --steps 1 --eval-every 1 \
  --eval-windows 6 --horizons 1 2 4 10 --action-ablation --device cuda:1

# action ranking / energy landscape
python scripts/probe_ac_action_ranking.py --predictor <run>/predictor.pt \
  --horizon 1 --candidates 12 --windows 20
```

The scripts expect the cloned V-JEPA 2 repo (``VJEPA2_ROOT``, default
``/workspace/vjepa2``) and import ``openpi.con2.ac_world_model`` for everything
in section 3.8.

## 5. Open items

1. **1 s rollout**: addressed by the timebase change (section 3.8b) - 0.97 ->
   0.73 on the 100-episode eval set, with `ft_440_stride2_ar4` running on the
   400-episode cache for the headline number. Still open: whether a 16-frame
   context or encoder adaptation pushes it below 0.7.
2. Representation adaptation is closed: both the token adapter (3.8c) and a real
   last-4-block unfreeze against the frozen cached target (3.8d) are worse than
   freezing. Any further accuracy work has to be scale (more episodes/steps), a
   longer autoregressive curriculum, or a different objective.
3. Wire the adapted predictor into Con2/TTT: `ACWorldModel.score_actions` is the
   energy objective TTT would optimise, and the cached tokens already contain
   everything needed. This is the remaining integration step, and it needs a
   decision between test-time action selection (environment-side) and distilling
   a JAX critic for Con1's loss (policy-side).
