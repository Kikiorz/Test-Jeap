# How far is a RoboTwin arm from the released base?

`scripts/compare_arm_to_base.py` reads the released checkpoint (publish layout)
and an arm's training checkpoint (step layout, read through orbax on CPU) and
reports the relative L2 change per subtree. It answers two questions at once:
does the `--base-weights` probe row have anything to measure, and which
parameters the arm actually moved.

Arm A @ 9,000 steps against the released `19999` checkpoint:

| subtree | leaves | relative L2 median | max | moved > 1e-3 |
|---|---|---|---|---|
| `PaliGemma/img` (vision tower) | 23 | 1.66e-03 | 1.79e-03 | 23 |
| `action_in_proj` | 2 | 1.66e-03 | 1.67e-03 | 2 |
| `time_mlp` | 4 | 1.64e-03 | 1.66e-03 | 4 |
| `other` | 11 | 1.67e-03 | 1.83e-03 | 11 |
| `PaliGemma/llm/layers` | 16 | **7.17e-03** | 2.04e-02 | 16 |
| `action_out_proj` | 2 | **2.31e-02** | 2.56e-02 | 2 |

## The uniform 1.66e-3 is a dtype artefact, not learning

Every frozen subtree - including the vision tower, which no arm ever trains -
moved by the same ~1.66e-3 relative amount. That is bfloat16 rounding:
`init_train_state` casts every frozen parameter to bf16, whose unit roundoff is
2^-8 = 3.9e-3, while the released checkpoint is float32.

So when reading this table, only the groups that exceed ~2e-3 carry information:

* `action_out_proj` moved ~2.3e-2
* `PaliGemma/llm/layers` moved ~7e-3 (the `*_1` groups that the freeze filter
  unfreezes, i.e. the action expert's upper blocks)
* everything else is untouched apart from the storage dtype

## What it implies for the probe

1. The `base (Con1 silenced)` and `released base` rows are **not** interchangeable
   - the arm's fine-tuning is real, roughly 4-14x larger than the rounding floor
   - but the expected gap between them is a fraction of a percent, not the tens of
     percent that the Con1 correction is worth. If the two rows come out nearly
     equal, that is the measurement working correctly, not a bug.
2. The arm's own action function is a lightly perturbed base policy: only two
   subtrees move materially, which is what the freeze filter was designed to do
   and means the Con1 effect is not coming from a broadly retrained backbone.

## Reproduce

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_arm_to_base.py \
  --arm /workspace/artifacts/checkpoints/pi05_robotwin_con1_livecross_20k/robotwin_a_con1/9000/params \
  --out /workspace/artifacts/con2/robotwin_armA9k_vs_released.json
```

Raw output: `/workspace/artifacts/con2/robotwin_armA9k_vs_released.json`.
