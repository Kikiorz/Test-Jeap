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

## What the table is not

`docs_ROBOTWIN_PROBE_CAVEATS.md`: it is a **training-distribution** measurement
(`con1_holdout_fraction=0.0`), flow loss is a proxy for closed-loop success, and
384 paired samples cannot resolve effects below ~2%.
