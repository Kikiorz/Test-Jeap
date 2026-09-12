# What the RoboTwin arm table does and does not measure

Written before the 12k numbers land, so the caveats are not chosen after seeing
the result. Every claim here was checked against the code or the data, not
inferred from the scripts' names.

## 1. It is a training-distribution measurement, not a held-out one

`docs` and the probe docstring call it a "held-out flow probe", and that is wrong
for these configs. The data loader filters episodes through the Con1 split:

```python
split = FeatureDataset(con1_latent_root, horizon=..., split=data_config.con1_split,
                       seed=42, fraction=data_config.con1_holdout_fraction)
selected_episode_ids = [int(e["id"]) for e in split.episodes]
```

and both RoboTwin configs set `con1_holdout_fraction=0.0`, so the "train" split is
all 2,500 episodes. Every arm trained on every episode the probe draws from.

What this does **not** invalidate: the comparison. All seven rows see the same 48
shuffled batches, so the differences are paired and the mechanism comparison
holds.

What it does change: the absolute flow loss is a fit number, not a generalization
number. To get a held-out version, the arms have to be trained with
`con1_holdout_fraction > 0` (e.g. 0.05, ~125 episodes) and the probe run with
`con1_split="validation"`. That is a change for the next training round, not
something that can be recovered from the current checkpoints.

## 2. Flow loss is a proxy

Closed-loop RoboTwin evaluation is deferred (no simulator on this box). Flow loss
is the standard training objective and correlates with action quality, but it is
not success rate.

## 3. `base` and `released base` are designed to be close

`base` is arm A with the correction silenced; `released base` also restores the
released weights. Their difference is only the arm's fine-tuning, which is
~2.3e-2 relative in `action_out_proj` and ~7e-3 in the unfrozen action-expert
groups (see `docs_ROBOTWIN_BASE_DRIFT.md`). A gap of a fraction of a percent is
the expected result, not a broken probe. The meaningful comparison is
`arm X vs base`.

## 4. One row is a diagnostic, not an arm

`arm A + offline head` grafts a head trained separately on the cache onto arm A's
checkpoint. It measures whether a better latent predictor moves the action loss,
but the fusion path was trained against the weaker head's outputs, so a negative
result there is not evidence against mediation on its own.

## 5. Sensitivity

48 batches x 8 = 384 paired samples. The 4.5k pair resolved an 11.8% effect at
t = 7 and a 0.10% effect at t = 0.16, so effects below ~2% are not resolvable
with this budget.
