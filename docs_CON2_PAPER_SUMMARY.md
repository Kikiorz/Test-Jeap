# Con2 (paper-facing summary)

Everything here is measured on LIBERO (LeRobot conversion, 1693 episodes,
40 tasks, 256x256 agentview + wrist, 8-d state, 7-d OSC actions) with the
released V-JEPA 2-AC checkpoint, and is reproducible from the scripts listed in
section 5. Detailed logs, negative results and traps live in
`docs_CON2_AC_WORLD_MODEL.md`.

## 1. What Con2 is

An action-conditioned latent world model: given the current observation's
V-JEPA patch tokens `z_t`, the executed/proposed action chunk and proprioceptive
states, it predicts the frozen encoder's tokens of future frames.

```
frame block k = [ action token a_k , state token s_k , 256 patch tokens z_k ]
z_hat_{k+1} = P_psi(z_k, a_k, s_k)          (block-causal attention over frames)
```

The predictor is the released V-JEPA 2-AC 24-block / 1024-wide / 305M-parameter
network; only the predictor is fine-tuned (the encoder stays frozen, and its
cached features serve as the stop-gradient target).

Adaptation to LIBERO consists of three things, in order of importance:

1. **Timebase.** The released model was trained at 4 fps on DROID; the dataset
   is 10 fps. Sampling every second frame (5 Hz) is worth more than any other
   change: 1 s NMSE 0.97 -> 0.73.
2. **Input mapping.** Actions are the 7-d OSC setpoints, states are
   `[eef xyz, axis-angle, gripper closedness]`. The two finger joints are
   mirrored, so closedness is `1 - mean(|qpos|)/0.04`; getting this wrong pins
   the gripper channel and makes action conditioning vanish.
3. **Fine-tuning** the predictor with the released objective (teacher forcing +
   autoregressive refinements, smooth L1 on layer-normed reps) on LIBERO
   demonstrations.

The default variant also predicts the *spatial* tokens rather than a mean-pooled
latent, which is what makes action conditioning measurable at all.

## 2. Prediction accuracy (held-out episodes, NMSE vs copy-current baseline)

Copy-current (repeat the current latent) = 1.0. Lower is better. 40 held-out
episodes; the 5 Hz rows use 5 steps = 1 s.

| variant | 0.2 s | 0.4 s | 1.0 s |
|---|---:|---:|---:|
| released weights, zero-shot, 10 fps | 1.067 | 1.097 | 1.233 |
| fine-tuned, 10 fps | 0.844 | 0.824 | 0.970 |
| fine-tuned, 10 fps, longer AR curriculum | - | - | 0.938 |
| **fine-tuned, 5 Hz (recommended)** | **0.710** | **0.656** | **0.731** |
| fine-tuned, 5 Hz, 400 episodes | 0.783 | 0.736 | 0.783 |
| fine-tuned, 5 Hz, 400 episodes, 4000 steps | **0.755** | **0.702** | **0.769** |
| fine-tuned, 5 Hz, executed actions/states as rollout input | 0.725 | 0.680 | 0.722 |
| from random init, same budget | 1.119 | 1.026 | 0.962 |

The last-but-one row is the deployed setting (the policy always has
proprioception and its own planned actions), the others are free-running
rollouts. The random-init row shows the pretrained weights are doing real work.

## 3. Action conditioning

Two measurements, because NMSE alone is nearly blind to it.

**(a) Ablation.** Zeroing the action chunk costs +0.065 / +0.094 / +0.093 NMSE
at 0.2 / 0.4 / 1.0 s in the 10 fps setting, and +0.10 to +0.16 in the 5 Hz
setting (before the gripper-mapping fix it was +0.002, which is how the bug was
found).

**(b) Action ranking (energy landscape).** Score 12 candidate action chunks by
the latent they predict and ask where the executed chunk ranks. Chance = 8.3%.

| model | horizon | top-1 | mean rank percentile |
|---|---:|---:|---:|
| V-JEPA 2-AC world model, 5 Hz | 1 (0.2 s) | **49.0%** | 0.090 |
| V-JEPA 2-AC world model, 5 Hz | 2 (0.4 s) | **67.0%** | 0.054 |
| original in-policy Con2 delta head (8 candidates) | 1 (0.1 s) | 3.4% (chance 12.5%) | 0.433 |
| original in-policy Con2 delta head | 10 (1.0 s) | 12.5% (chance 12.5%) | 0.388 |

The original Con2 head is at or below chance: the action chunk it was
conditioned on carried no information about which action was executed, which is
consistent with its +0.002 NMSE ablation.

## 4. Things that did **not** work (recorded so they are not repeated)

| attempt | result |
|---|---|
| zero-shot transfer of the released predictor | worse than copy-current (1.03-1.23) |
| mean-pooled latent targets (the original Con2) | action conditioning invisible (+0.002) |
| trainable token adapter on frozen features | 1 s NMSE 0.756 vs 0.742 frozen |
| unfreezing the last 4 encoder blocks | 1 s NMSE 0.749 vs 0.740 frozen |
| online TTT with a cross-episode replay buffer | +/-0 (+0.0042 and +0.0002) - the gradient is diluted by other episodes |

The pattern is consistent: at this data scale, adapting the model further
(representation or online) hurts; the value is in the pretrained weights plus
the fine-tune. A no-adaptation control reproduces the metric exactly
(mean delta 0.000000 over 40 episodes), so these deltas are not evaluation
noise.

The one exception is test-time training done per episode rather than across
episodes, which is a genuine gain and is reported separately in section 4b.

## 4b. Test-time training (it does work, per episode)

Same no-leakage protocol: score the model on the second half of each held-out
episode, adapt on the first half only (30 gradient steps, lr 1e-4, observed
transitions only, no labels), score the untouched second half again. 40 held-out
episodes, buffer = the current episode's first half:

| | NMSE before | NMSE after | mean delta | improved |
|---|---:|---:|---:|---:|
| per-episode TTT | 0.9382 | **0.9056** | **-0.0327 +/- 0.0077** | 33/40 |
| no adaptation (control) | - | - | 0.000000 | 0/40 |

That is a 3.5% held-out improvement at 4.2 standard errors, with zero labels and
no environment interaction. The opposite result appears only when the buffer is
shared across episodes (200 windows), where the gradient is dominated by other
episodes and the effect is zero - so the buffer scope, not TTT itself, was the
earlier failure.

## 5. Reproduction

```bash
python scripts/probe_ac_control_franka.py --rollout 1        # machinery check on in-distribution data
python scripts/probe_vjepa_ac_libero.py --episodes 0,1 --stride 2
python scripts/probe_ac_space_sufficiency.py --horizon 1
python scripts/cache_ac_tokens.py --episodes $(seq 0 439) --output <cache>
python scripts/finetune_ac_predictor.py --cache <cache> --train-episodes $(seq 0 99) \
  --eval-episodes $(seq 100 119) --frame-stride 2 --auto-steps 4 --steps 1000 --output <run>
python scripts/probe_ac_action_ranking.py --predictor <run>/predictor.pt --frame-stride 2 \
  --horizon 1 --candidates 12 --windows 10
python scripts/eval_con1_head_ranking.py --config pi05_libero_con1_action_only_40k \
  --exp-name con1_action_only_stage1_2k --out <json>
python scripts/con2_ttt_adapt.py --cache <cache> --predictor <run>/predictor.pt \
  --episodes $(seq 400 439) --adapt-steps 30 --learning-rate 1e-4
PYTHONPATH=src python -m pytest src/openpi/con2/test_ac_world_model.py
```

The library surface used by all of the above is
`src/openpi/con2/ac_world_model.py` (12 CPU tests, no checkpoint or GPU needed).

## 6. Open items

1. Policy-side coupling. Two credible routes, both needing a decision:
   test-time action selection inside a closed-loop LIBERO-Plus evaluation, or
   distilling the world model's energy into a JAX critic so Con1's training can
   use it as a reverse constraint.
2. Scale: the fine-tune still improves with more episodes (0.71 at 100 episodes
   vs 0.78 at 400 on their respective held-out sets), so a larger cache and a
   longer schedule are the obvious remaining accuracy lever.
3. TTT is a gain on prediction error (section 4b) but has never been run inside
   a closed-loop policy rollout, so its effect on success rate is untested.
