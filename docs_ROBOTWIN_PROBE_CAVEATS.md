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

## 6. Why the rows are paired even though each row runs a different config

This is worth stating because it is the assumption the whole table rests on and
it is not obvious: arm A, B, C and D each run their *own* training config, so the
loader could in principle shuffle them differently and the paired t statistics
would be meaningless.

Checked in the code rather than assumed:

* `TrainConfig.seed` defaults to 42 and none of the four RoboTwin configs
  overrides it (`grep` over the config blocks: no `seed=` overrides);
* `create_data_loader` forwards `seed=config.seed` to `create_torch_data_loader`,
  which passes it to `TorchDataLoader`, i.e. the shuffle generator;
* the probe fixes `--batch-size` and `--batches` for every row.

So every row walks the same shuffled sequence and batch `i` is the same batch for
all rows. If a future arm ever changes `seed`, batch size or batch count, the
paired statistics must be recomputed rather than reused.

## 7. The zero-correction control used to leak a bias (fixed)

`--zero-correction` zeroes the correction at its output projections so the row
measures the policy with Con1 silenced. It zeroed `out/kernel` and
`adapter_out/kernel`, but `adapter_out` also carries a **bias**, and the Dense
layer is then `adapter = hidden @ 0 + bias = bias` - a constant offset added to
every action token, scaled by `sigmoid(alpha)`.

The evidence was already in the 4.5k numbers: the base row reported
`correction_rms = 0.028` where a silenced correction must report exactly 0.

Checked against a real checkpoint on CPU: the parameter tree contains
`con1_cross_attention/adapter_out/bias/value` (1024,), which the old rule left
alone, and the fixed rule zeroes three leaves -
`adapter_out/bias`, `adapter_out/kernel`, `out/kernel` - while leaving the other
ten Con1 parameters untouched.

Consequence for the 4.5k result: the control was slightly *better* than a true
no-correction baseline, so the reported -11.77% is a slight **underestimate** of
the Con1 gain. The 12k table uses the fixed control.
