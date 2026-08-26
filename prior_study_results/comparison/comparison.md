# Joint-limit prior — single-view vs multi-view

fine-tune: **+10 epochs**  |  `joint_limit_regularization` = **100.0**

Each `*_constrained` arm is its `*_reference` checkpoint continued for the same number of epochs with the limit penalty enabled. **All four arms are scored against the same authored ranges**, so the reference rows are the honest "before" number.

> **Caveat — the fine-tuning confound.** The reference arms received zero additional epochs, so any difference below is *the prior plus continued training*, not the prior alone. A `w_limit = 0` control fine-tuned for the same epochs would separate the two; it is not part of this study by design.

Deltas are `constrained - reference`. Lower is better for violations and MPJPE; higher is better for PCK.

## singleview

Authored axes scored: **162**

| metric | reference | constrained | delta |
|---|---|---|---|
| Violating axes (count) | 96 | 5 | -91 (better) |
| Mean violation rate (% frames) | 14.76 | 0.00 | -14.76 (better) |
| Mean overshoot, violating axes (deg) | 2.53 | 0.00 | -2.53 (better) |
| Max overshoot (deg) | 77.70 | 1.13 | -76.57 (better) |
| MPJPE (mm) | 1.09 | 1.88 | +0.79 (worse) |
| Median MPJPE (mm) | 0.92 | 1.59 | +0.66 (worse) |
| PCK@5px native | — | — | — |
| PCK@5px input | — | — | — |

- `sv_reference`: singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth (epoch 386)
- `sv_constrained`: runs/singleview_constrained/checkpoints/checkpoint_epoch_394.pth (epoch 394)

## multiview

Authored axes scored: **162**

| metric | reference | constrained | delta |
|---|---|---|---|
| Violating axes (count) | 76 | 1 | -75 (better) |
| Mean violation rate (% frames) | 13.96 | 0.00 | -13.96 (better) |
| Mean overshoot, violating axes (deg) | 2.65 | 0.00 | -2.65 (better) |
| Max overshoot (deg) | 56.57 | 0.82 | -55.75 (better) |
| MPJPE (mm) | 0.96 | 2.95 | +1.99 (worse) |
| Median MPJPE (mm) | 0.84 | 2.48 | +1.64 (worse) |
| PCK@5px native | — | — | — |
| PCK@5px input | — | — | — |

- `mv_reference`: SMILySTICKS_ViT_model.pth (epoch 345)
- `mv_constrained`: runs/multiview_constrained/checkpoints/checkpoint_epoch_0354.pth (epoch 354)

## Single-view vs multi-view

| metric | single-view delta | multi-view delta |
|---|---|---|
| Violating axes (count) | -91 (better) | -75 (better) |
| Mean violation rate (% frames) | -14.76 (better) | -13.96 (better) |
| Mean overshoot (deg) | -2.53 (better) | -2.65 (better) |
| MPJPE (mm) | +0.79 (worse) | +1.99 (worse) |
| PCK@5px native | — | — |

The hypothesis worth testing here: multi-view already resolves depth ambiguity from geometry, so it should have fewer violations to fix and less to gain from the prior. A single-view delta noticeably larger than the multi-view one supports that reading.
