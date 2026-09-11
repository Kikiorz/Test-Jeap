# Con1: which delta-Z gradient improves the action?

Motivation (user's design): the reciprocal coupling currently corrects `delta_z`
along its own reconstruction error, but "the steepest-descent direction on
`delta_z` is not necessarily the best for the action". The proposal is to push
`(delta_z_hat, delta_z*)` through a *learned function* whose loss produces a
better action update than the direct `delta_z` loss.

This is the measurement that decides whether that function has anything to fix.

## The two directions at the same point

```
g_action = d(action flow loss)/d(delta_z)          # already computed in training
                                                    # as the SGR "sensitivity" VJP
g_mse    = d(||delta_z - delta_z*||^2)/d(delta_z)  # the direct latent target
```

Both are available at the same parameters and the same batch, so the comparison
is exact rather than between two training runs.

## Result (192 samples, Con1 adapter checkpoint step 1001)

`scripts/probe_con1_delta_gradient_alignment.py --config pi05_libero_con1_action_adapter_40k`

| quantity | value |
|---|---:|
| cos(g_mse, g_action) | **-0.0064** (per batch -0.014 ... +0.007) |
| ||g_mse|| / ||g_action|| | 6.0e8 |
| flow reduction at equal delta_z step norm - action direction | **+0.97%** (target was 1%) |
| flow reduction at equal delta_z step norm - latent-MSE direction | **-0.028%** |
| efficiency of the MSE direction vs the action direction | **-0.029** |

Read: at the *same* amount of correction to `delta_z`, stepping along the action
direction buys the full 1% flow reduction, while stepping along the direct
latent-reconstruction direction buys **nothing** (it is orthogonal to within
0.6%, and empirically slightly increases the flow in 5 of 6 batches).

## Consequence for the design

1. **The premise holds, and strongly.** A Euclidean loss on `delta_z` cannot be
   repaired by re-weighting, because re-weighting changes only the magnitude
   across horizons and samples - not the direction, and the direction is
   orthogonal to what helps the action. That is also why the sensitivity-guided
   re-weighting already in Con1 (`sgr_beta`) has had a small effect.
2. **A learned function is the right answer**, and it now has a well-defined
   target: `F_phi(delta_z_hat, delta_z*)` should be trained so that its gradient
   with respect to `delta_z_hat` *is* the action-improving direction, i.e.

   ```
   minimize  1 - cos( grad_{delta_z} F_phi , g_action )        # direction match
   ```

   `g_action` is already computed every training step (the same VJP that the
   SGR weight uses), so this is a supervised distillation of the direction - far
   better conditioned than meta-learning through the policy, and it needs no
   environment interaction.
3. Two paths then become comparable on the same axis:

   | path | loss on (delta_z_hat, delta_z*) | gradient used for the action |
   |---|---|---|
   | A (current) | `||delta_z_hat - delta_z*||^2` (optionally sensitivity-weighted) | orthogonal to g_action, efficiency ~0 |
   | B (proposed) | `F_phi(delta_z_hat, delta_z*)` | trained to align with g_action |

   The evaluation is "which path makes the predicted action better", measured as
   the held-out action flow loss after the same delta_z budget, then by
   closed-loop success.

Caveat: this is measured on the action-adapter checkpoint with one sampled flow
time per batch. The cosine is stable across batches and is scale-free, which is
why it is the number to trust; the absolute flow reductions depend on the step
calibration (the action step was verified to land in the linear regime: 0.97%
measured against 1% predicted).

## Two follow-up measurements (same checkpoint)

### The 5% residual budget is *not* what limits Con1

`scripts/probe_con1_budget_and_conditioning.py` rebuilds the model with a
different `con1_residual_budget` and re-evaluates the *same* parameters, so the
comparison is paired and needs no retraining:

| budget | correction RMS | flow |
|---|---:|---:|
| 0.05 (deployed) | 0.3476 | 0.34762 |
| 0.10 | 0.3476 | 0.34762 |
| 0.20 | 0.3476 | 0.34762 |
| 0 (uncapped) | 0.3476 | 0.34762 |
| 1e-4 (sanity control) | 0.00173 | +3.6e-4 worse |

Identical to the last digit for every budget at or above 5%, while 1e-4 clips
the correction and costs accuracy - so the knob works and the cap simply never
binds. 5% of the action-hidden RMS is larger than the correction the adapter
produces. This **retracts** the earlier claim in
`docs_CON1_3STAGE_TRAINING.md` that "the 5% relative budget is the binding
constraint": what the alpha-sweep invariance actually shows is that the adapter
branch (not the alpha-gated attention branch) dominates the correction, so the
gate has nothing to do.

Consequence: relaxing the retention budget cannot buy accuracy. The limit has
to be the objective, not the magnitude.

### The deployed Con1 head is not action-conditioned at all

`pi05_libero_con1_action_adapter_40k` - the configuration that is trained and
served - sets `con1_action_adapter=True` but leaves `con1_action_conditioning`
at its default **False**. The action chunk is therefore never fed to the delta
head; the action enters only through (i) the adapter's query (the action
expert's hidden state) and (ii) the flow-loss VJP. The variants that do feed the
chunk (`..._action_cond_40k`, `..._action_only_40k`, `..._vlm_ctx_40k`) are the
ones whose action ranking was measured at or below chance.

So "we feed the action in and it still does not improve" has a concrete
explanation at the mechanism level: in the deployed model the action is not fed
to the predictor, and in the models where it is, it is fed with an objective
(Euclidean latent reconstruction) whose direction is orthogonal to the action.
