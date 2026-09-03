# CoFlow-JEPA

## Joint Transition–Action Flow with Online Sparse Inverse Adaptation

> Status: method specification and implementation contract for `feat/point`.
> The former tracker/point-flow and reciprocal-attention prototypes are not part of this method.
> Base model: the released π0.5 JEPA-WAM checkpoint at step `59999`.

## 1. Research question

JEPA-WAM learns a current-observation-conditioned representation of a future task transition, but its action path does
not have to explain which action realizes that transition. A lower JEPA loss can therefore coexist with weak action
sensitivity or unchanged closed-loop control.

CoFlow-JEPA tests one claim:

> A JEPA transition hypothesis should not be used as a fixed condition for action generation. The transition and action
> hypotheses should be transported together, and their shared coupling should be recalibrated from naturally paired
> executed actions and JEPA transitions observed during deployment.

The complete loop is:

```text
current observation + language
            |
            v
JEPA-WAM transition prior U^P + Gaussian action noise
            |
            v
coupled transition/action rectified flow
            |
            +--------------------+
            v                    v
     executable action A     action-consistent U^AC
            |
            v
execute A and observe o_(t+H)
            |
            v
frozen V-JEPA observed transition U^R
            |
            v
sparse inverse loss on the same transition-to-action coupling
            |
            v
next action chunk uses the adapted coupling
```

There is no external tracker, optical-flow teacher, success image, reward, MPC, DLS controller, or hand-labelled
motion target.

## 2. Shared transition variable

Let the control horizon be exactly `H` environment steps. Every training tuple must be

```text
(o_t, language, proprio_t, A*_[t:t+H-1], o_(t+H)).
```

The released JEPA-WAM predictor produces future-query states from only the current observation and language:

```text
R_t = F_JW(o_t, language)                     # [B, 64, d_model]
Y^P_t = G_align(R_t)                          # [B, 24*24, d_J]
```

The frozen V-JEPA target encoder observes the real pair during training or after execution:

```text
Y^R_t = E_J([o_t, o_(t+H)])                   # [B, 24*24, d_J]
```

`Y^R_t` is not a tracker-generated label. It is the JEPA target representation of the same temporal interval that
contains the executed action chunk. Before execution, the current-only predictor supplies `Y^P_t`; after execution,
the target encoder supplies `Y^R_t`. This predict--act--observe symmetry is the common interface between the offline
joint-flow objective and online adaptation.

Both sides undergo the *same* non-learned spatial reduction and token normalization:

```text
U^P_t = Normalize(Pool_3x3(Y^P_t))            # [B, 64, d_J]
U^R_t = Normalize(Pool_3x3(Y^R_t))            # [B, 64, d_J]
```

`Pool_3x3` maps `24x24` to `8x8`. `Normalize` is tokenwise L2 normalization in float32. This is necessary because the
original auxiliary objective constrains cosine direction rather than raw feature norm; rectified interpolation between
unmatched feature scales would not have a stable meaning.

`U^R` is called an **observed expert transition representation**, not point flow or optical flow. Demonstrations do not provide counterfactual actions,
so it must not be described as a causal transition for arbitrary candidate actions.

### Horizon correction

The released checkpoint was trained with a future offset near 31 frames, while the current LIBERO action chunk has
`H=10`. The main experiment must not pair a 10-step action with a 31-step transition target. Phase 0 therefore creates
new frozen V-JEPA targets at offset `H=10` and lightly adapts only the 64 future queries and alignment head before
training CoFlow. The original VLA and Action Expert remain the initialization.

## 3. Contribution 1: prior-to-outcome action–transition Co-Flow

At flow time `tau`, π0.5 uses `tau=1` for source/noise and `tau=0` for data.

### Action path

```text
A_tau = tau * epsilon_A + (1 - tau) * A*
V*_A  = epsilon_A - A*
```

### Transition path

The transition source is the current-only JEPA prior rather than a second Gaussian:

```text
U_tau = tau * U^P + (1 - tau) * U^R
V*_U  = U^P - U^R
```

The joint vector field predicts both velocities:

```text
(Vhat_A, Vhat_U) = F_theta(A_tau, U_tau, prefix(o_t, language, proprio_t), tau)
```

with the minimal objective

```text
L_CoFlow = mean_square(Vhat_A - V*_A)
         + lambda_U * mean_tokens(||Vhat_U - V*_U||_2^2),

lambda_U = 0.1  # first falsification run
```

The transition term sums over the `d_J` feature axis before averaging tokens and batches. Since `U` is tokenwise
unit-normalized, averaging over `d_J=1408` would artificially suppress this objective by the representation width.
The new transition-velocity output head is zero-initialized, so its initial ODE leaves the JEPA prior unchanged rather
than applying a random high-dimensional drift.

### Two-stream blocks

The first 14 Action Expert blocks retain the released π0.5 computation. The final four blocks are replaced by four
two-stream Co-Flow blocks:

```text
H_A: H action tokens, width d_A
H_U: 64 transition tokens, projected to width d_U
```

Each stream has its own RMSNorm, Q/K/V projections and SwiGLU FFN. A block performs bidirectional joint attention:

```text
Action query     reads [action keys/values; transition keys/values]
Transition query reads [transition keys/values; stopgrad(action keys/values)]
```

The stop-gradient allows the transition loss to use the current action hypothesis without reshaping the pretrained
action representation through `L_U`.

The action-to-transition message has the fixed signal-to-noise weight

```text
rho_A(tau) = 1 - tau.
```

At `tau=1`, pure action noise cannot contaminate the transition stream. As denoising proceeds, an increasingly formed
action hypothesis can refine the transition. Transition-to-action communication is available throughout the flow.
This continuous schedule replaces a learned gate and an arbitrary hard flow-time threshold.

### Initialization and trainable scope

- VLM/vision backbone and the first 14 Action Expert blocks are frozen.
- Action-stream self-attention/FFN weights are initialized from the released final four Action Expert blocks.
- New transition-stream and cross-stream projections are trained.
- Cross-stream output projections are zero-initialized, so the untrained model reproduces the released action path.
- The future queries/alignment head use the horizon-aligned Phase-0 checkpoint and are frozen during the first CoFlow
  test.

At inference:

```text
A_1 = epsilon_A
U_1 = U^P
(A_1, U_1) --same ODE solver--> (A_0, U_0)
```

Only `A_0` is executed. `U_0` is an action-consistent transition hypothesis used for analysis and the later online
loop. No future image or V-JEPA teacher is needed before action execution.

## 4. Contribution 2: action-imprint sparse inverse TTT

After executing `A_exec` and observing `o_(t+H)`, the frozen V-JEPA encoder supplies the same `U^R` representation
used to train the transition stream. There is still no tracker or flow label.

`H` inverse action queries read the 64 observed transition tokens through the same transition key/value projections
used by CoFlow's transition-to-action path. Entmax-1.5 gives an explicitly sparse support:

```text
alpha_k = entmax_1.5(q_k K_phi(U^R)^T / sqrt(d))
z_k     = sum_i alpha_[k,i] V_phi(U^R_i)
Ahat_k  = D_inv(z_k, proprio_t)

L_inv = mean_k Huber(Ahat_k, A_exec_k)
```

The support is interpreted only as **tokens useful for identifying the executed action**, not as a causal segmentation
mask. Future proprioception is excluded so the inverse head cannot solve the task by joint differencing alone.

Rank-4 LoRA parameters are added only to the shared `K_phi,V_phi`. A deployment update modifies those LoRA weights
for one clipped gradient step; all backbones, CoFlow blocks, transition heads and the inverse decoder stay frozen.
Consequently, the update changes the same transition-to-action readout used to generate the next chunk rather than an
isolated auxiliary network.

### Why inverse TTT needs meta-alignment

Reconstructing a bad executed action is not automatically useful. During offline preparation, a support transition
performs one simulated inverse update and a different query sample from the same deployment condition evaluates the
post-update action loss:

```text
Delta_phi_support = -eta * grad_Delta_phi L_inv(support)
L_meta            = L_action(query; phi + Delta_phi_support)
```

First-order meta-training learns the inverse queries, decoder, LoRA initialization and scalar step size. Deployment
uses only `L_inv`; it has no expert action, task reward, value function or policy gradient.

Two protocols must be reported separately:

- episode-reset: online LoRA state resets at every episode;
- persistent-session: online state persists across a predefined sequence, with clean-suite retention measured.

## 5. What is and is not new

The individual ingredients are established: rectified flow, modality-specific joint attention, inverse dynamics,
sparse attention and one-step meta-learning. The paper claim is their single-variable coupling:

> a JEPA-WAM transition prior is used as the source of a transition flow jointly integrated with π0.5 Action Flow,
> while reward-free inverse updates recalibrate the exact transition-to-action coupling used by that joint flow.

This must not be presented as a new generic Transformer, a causal world model, or a new optical-flow estimator.

Relevant precedents include:

- [JEPA-WAM repository](https://github.com/SpriteWithoutIce/openpi_jepawam): released π0.5 code and checkpoint used
  here.
- [WA-JEPA](https://arxiv.org/abs/2608.20974): joint future/action generation in autonomous driving.
- [Scaling Rectified Flow Transformers](https://arxiv.org/abs/2403.03206): modality-specific parameters with
  bidirectional information exchange.
- [Self-Supervised Policy Adaptation during Deployment](https://arxiv.org/abs/2007.04309): reward-free deployment
  adaptation with an auxiliary self-supervised objective.

These precedents motivate the components; they do not establish that the proposed coupling works for robot
manipulation.

## 6. Minimal falsification protocol

The first run is deliberately small. It does not launch full LIBERO-Plus evaluation.

### Gate 0 — temporal/statistical alignment

On a fixed held-out expert subset:

1. Generate V-JEPA targets with exactly `H=10`.
2. Verify `U^P` and `U^R` both have shape `[B,64,1408]`, finite unit-norm tokens and identical pooling.
3. Adapt the query/alignment head on a small training split.
4. Require the matched `U^P/U^R` cosine distance to beat both persistence and within-task shuffled-future controls.

Failure means the transition source and target are not predictive enough; CoFlow is not trained.

### Gate 1 — tiny-batch implementation check

1. Zero cross-stream outputs must reproduce the released action velocity for the same inputs/noise.
2. Overfit 32–64 expert tuples.
3. Both action and transition losses must decrease without NaN/Inf.
4. Both cross directions must receive nonzero gradients after the zero-initialized outputs leave zero.

Failure here is treated as an implementation/optimization bug until shape, mask, gradient and initialization tests are
exhausted.

### Gate 2 — core scientific test

Train only a short fixed-budget run and compare on the same held-out tuples:

1. released JEPA-WAM;
2. fixed `U^P` action conditioning with matched trainable parameters;
3. two independent action/transition flows with no cross-stream communication;
4. full CoFlow.

CoFlow proceeds only if it improves held-out action flow loss over fixed conditioning **and** passes both interventions:

- shuffle `U^P` across the batch: action loss must worsen;
- fix `U^P` and change action noise: the final transition hypothesis must change and the correctly paired result must
  be closer to its observed transition than the shuffled pairing.

These tests distinguish genuine action–transition coupling from an unused transition branch.

### Gate 3 — minimal control result

Only after Gate 2, run paired rollouts on a small sealed LIBERO-Plus subset where the released baseline is not saturated.
Use identical task IDs, environment seeds, inference seeds, action horizon and replanning frequency. The method must
beat both the released checkpoint and matched fixed-conditioning control before scaling evaluation.

### Gate 4 — inverse update usefulness

Before online rollouts, use offline support/query splits from the same perturbation condition. One inverse update must
reduce query action loss relative to:

- no update;
- full-token softmax inverse update;
- sparse inverse update without meta-alignment;
- shuffled executed actions;
- shuffled observed transitions.

If this fails, Con2 is rejected rather than justified through lower inverse reconstruction loss.

## 7. Main evaluation after the gates

Primary benchmark: LIBERO-Plus, with overall, Layout, Camera, Robot Initial State, Noise and Lighting/Background
breakdowns. Clean LIBERO retention is mandatory. All methods use the same action horizon and replanning frequency.

Con1 reports zero-update success. Con2 additionally reports encounter-indexed performance such as `SR@0`, `SR@1`,
`SR@3`, and `SR@5`, plus update latency and retention. RoboTwin 2.0 is a second benchmark only after the complete
method passes LIBERO-Plus.

The existing sealed L4/L5 manifests and former reciprocal-attention results remain historical controls; they are not
relabelled as evidence for CoFlow-JEPA.

## 8. Failure interpretation

| Observation | Interpretation | Next action |
|---|---|---|
| H=10 predictor cannot beat persistence/shuffle | Released transition prior is temporally unsuitable | retrain only future queries/alignment at H=10; do not alter CoFlow |
| Tiny set cannot overfit | code, mask, initialization, or optimizer bug | debug before judging the method |
| Transition loss falls but action is insensitive to `U` | coupling is ignored | reject current CoFlow parameterization; do not add unrelated modules |
| Fixed conditioning equals CoFlow | joint evolution provides no value | Con1 scientific claim is unsupported |
| Offline action loss improves but rollouts do not | behavior-cloning proxy is misaligned with control | stop scaling and inspect paired failure modes |
| Inverse loss falls but post-update query action worsens | naive TTT learns executed errors | meta-alignment is necessary; if it still fails, reject Con2 |

## 9. Planned implementation boundary

The implementation should touch only:

- π0.5 model/config: normalized compact transition projection and four CoFlow blocks;
- data path: horizon-aligned frozen V-JEPA targets;
- training: separate action/transition losses and freeze masks;
- online adapter: shared transition K/V LoRA plus inverse/meta objectives;
- focused tests and reproducible small-run scripts.

No tracker target generator or point-flow dataset is part of the final method. Existing experimental artifacts are kept
read-only for provenance, but new CoFlow checkpoints and logs use separate directories.
