# How to read the RoboTwin arm table

Ten rows, all at the shared step **12,000**, same loader, same seed, same 48
paired batches. `base` = arm A's checkpoint with Con1's correction silenced, so
"vs base" isolates the correction itself.

## Rows and the question each one answers

| row | question |
|---|---|
| `released base` | what the released policy scores with no correction (reference) |
| `base` | arm A minus the correction (the control for every "vs base") |
| `A` | Con1 livecross: does the correction help at all? |
| `A + offline head` | would a better latent predictor help the action? (diagnostic) |
| `A, only the delta path` | is the gain carried by the part that reads the latent? |
| `A, only the adapter` | ...or by the delta-free side channel? |
| `A @ budget 0.05 vs 0.2` | is the correction capped, rather than misinformed? |
| `B` | does Con2 refinement + whole-prefix context add anything? |
| `C` | does conditioning the delta head on actions help? |
| `D` | does training the head properly (5x LR) help? |

## The residual/adapter split

The correction is a sum of two additive parts before the budget is applied:

```
correction = sigmoid(alpha) * residual(delta, hidden)      # the only delta-dependent path
           + adapter_scale  * adapter(hidden)              # never sees delta
```

So "only the delta path" zeroes `adapter_out`, and "only the adapter" zeroes
`out`. Both are compared against arm A, whose row contains both:

| only the delta path | only the adapter | reading |
|---|---|---|
| close to A | close to base | both parts are needed; the gain is shared |
| close to A | close to A | either part alone reproduces the gain (redundant paths) |
| close to base | close to A | the gain is delta-free on RoboTwin - a *departure* from LIBERO (see below), and a reason to look at the architecture |
| close to A | close to base | the delta path carries the gain - improving the latent head is the right lever |

### The LIBERO prior for this table

This decomposition has already been run on LIBERO, closed-loop, 705 paired L5
episodes (`docs_CON1_LIVECROSS_L5.md`):

| variant | pooled delta |
|---|---|
| action adapter only (latent-free) | +0.99pp (47/40) |
| adapter + live latent | **+1.70pp (54/42)** |

So on LIBERO the latent-free adapter carries roughly 58% of the gain **and the
latent branch adds a further +0.71pp on top**, with the largest single-category
effect in the appearance-shift category where the latent branch was expected to
help (Background Textures +3.2 -> +7.4pp). The delta-free adapter is a documented
part of the design (`docs_CON1_METHOD.md` labels it LATENT-FREE), not a defect.

Which means the third row of the table above is **not** the expected outcome: if
RoboTwin shows adapter-only reproducing the whole gain, that is a departure from
the LIBERO result rather than a confirmation, and it would point at something
RoboTwin-specific (e.g. the 2,410-task distribution making the predicted delta
too weak to contribute) rather than at a broken side channel.

## Pre-registered thresholds

Fixed before the numbers arrived, see `docs_CON1_LATENT_CEILING.md`:

* "the head moved": arm D's `con1_delta_nmse` at least 0.05 below arm A's;
* "it bought action accuracy": arm D's paired flow beats arm A by >= 2% with
  |t| >= 3;
* anything in between is reported as inconclusive at this budget.

## The upstream prior that governs the whole table

Two LIBERO measurements predate tonight and constrain how any RoboTwin arm may
be read. Both are in this repo; check them before concluding anything.

**1. `docs_LATENT_ACTION_INFORMATION.md` - what Con1 predicts carries almost no
action information.** Held-out linear probes (26,367 train / 4,562 held out):

| features | R^2 immediate action | R^2 10-step chunk |
|---|---:|---:|
| previous action | **0.962** | **0.705** |
| `z_t` | 0.263 | 0.130 |
| **`Delta z` (the quantity Con1 predicts)** | **0.047** | **0.033** |

The action is almost pure temporal continuity, and adding the latent to the
previous action makes held-out prediction *worse*. The document's own conclusion:
"every coupling that routes this latent into the action is structurally a no-op -
which is exactly what the four earlier experiments measured. The problem is
*upstream of the coupling*."

**2. `docs_CON1_DELTA_GRADIENT_ALIGNMENT.md` - the delta-MSE direction is
orthogonal to the action-improving direction.** At equal `delta_z` step norm, the
action direction reduces flow by +0.97% while the latent-MSE direction reduces it
by -0.028% (efficiency -0.029). Re-weighting cannot fix a Euclidean loss whose
gradient is orthogonal to what helps.

The same document also settles a hypothesis I raised tonight: **the residual
budget does not bind.** On LIBERO, budgets 0.05 / 0.10 / 0.20 give an identical
correction RMS (0.3476) and identical flow (0.34762). So the multi-budget row is
expected to be a no-op, and "the cap is throttling the correction" is not the
explanation for Con2.

### What this means for tonight

Arms C, D and F all try to make the *latent prediction* better. Given (1) and
(2), the **expected** outcome is that they leave the action flow essentially
unchanged. If that is what happens, the correct reading is "consistent with the
root cause", **not** "the head fix failed".

Conversely, if arm D does move the action flow, that would contradict a measured
LIBERO result and would be the more interesting finding of the two.

### Arm B has, in effect, already run this experiment

Worth keeping in view when arm D reports. At 4.5k the two arms differed sharply
in latent quality and not at all in the action:

| arm | `con1_delta_nmse` @4.5k | flow @4.5k |
|---|---:|---:|
| A | ~0.88 | 0.003667 |
| B | 0.773 (0.7065 @9k, **0.6803 @11k**) | 0.003664 (B vs A t = 0.16) |

B has a materially better latent head at equal steps and its action is identical.
That is the same shape of result arm D would produce if the head is not the
mediator. Caveat: B changes two things at once (the Con2 refiner *and* the
whole-prefix context), so it is suggestive rather than a clean single-variable
test - which is precisely what arm D is for.

By 11k B's head reaches **0.6803**, better than the head trained offline in
isolation (0.688) and within 0.02 of the linear ceiling (0.659). So if B's 12k
flow row still matches arm A, "a better latent predictor does not buy action
accuracy" will rest on a head that is the best latent model produced in this
project.

It also sharpens what the RoboTwin Con1 gain can and cannot be: the -11.8% flow
improvement is real, but per (1) it cannot be reaching the action *through latent
information*. The candidates are the latent-free adapter and the cross-attention
acting as an extra conditioning path - which is exactly what the
only-residual / only-adapter rows measure.

## A third prior, about the gradient form of Con2

`docs_CON1_WM_ACTION_JUDGMENT.md` measures whether a world-model *gradient step*
on the policy's own chunk moves it towards the demonstration (80 samples):

| quantity | value |
|---|---:|
| world-model energy reduction | +25,732 (the step works on its own objective) |
| mean distance to the demonstrated chunk | 1.525 -> 1.564 |
| fraction of samples moved closer | **16.3%** (chance is 50%) |

Its conclusion: the world-model gradient is not an action-improvement direction -
it lowers the model's own energy while moving the action away from the expert in
84% of cases. What *is* real is the model's **ranking** ability: with 12
candidates it places the executed chunk first 49% (0.2 s) / 67% (0.4 s) of the
time against 8.3% chance. The document's recommended design is therefore
ranking-based adaptation - sample chunks, score them against observed
transitions, move the adapter towards the better-scoring ones - rather than a
gradient source.

Scope note: this concerns the *gradient* form of Con2 discussed earlier. The Con2
implemented in the openpi model and measured by arm B is a feed-forward refiner,
`delta_used = delta_pred + F(delta_pred, z_t)`, which is a different object and is
not contradicted by this measurement.

## What the table is not

`docs_ROBOTWIN_PROBE_CAVEATS.md`: it is a **training-distribution** measurement
(`con1_holdout_fraction=0.0`), flow loss is a proxy for closed-loop success, and
384 paired samples cannot resolve effects below ~2%.
