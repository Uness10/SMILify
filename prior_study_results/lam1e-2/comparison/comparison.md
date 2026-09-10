# Joint-limit prior — single-view vs multi-view

fine-tune: **+25 epochs**  |  `joint_limit_regularization` = **0.01**

Each `*_constrained` arm is its `*_reference` checkpoint continued for the same number of epochs with the limit penalty enabled. **All four arms are scored against the same authored ranges**, so the reference rows are the honest "before" number.

> **Caveat — the fine-tuning confound.** The reference arms received zero additional epochs, so any difference below is *the prior plus continued training*, not the prior alone. A `w_limit = 0` control fine-tuned for the same epochs would separate the two; it is not part of this study by design.

Deltas are `constrained - reference`. Lower is better for violations and MPJPE; higher is better for PCK.

## singleview

Authored axes scored: **162**

| metric | reference | constrained | delta |
|---|---|---|---|
| Violating axes (count) | 96 | 73 | -23 (better) |
| Mean violation rate (% frames) | 14.76 | 0.02 | -14.74 (better) |
| Mean overshoot, violating axes (deg) | 2.53 | 0.00 | -2.53 (better) |
| Max overshoot (deg) | 77.70 | 7.15 | -70.54 (better) |
| MPJPE (mm) | 1.09 | 1.02 | -0.07 (better) |
| Median MPJPE (mm) | 0.92 | 0.87 | -0.05 (better) |
| PCK@5px native | — | — | — |
| PCK@5px input | — | — | — |

- `sv_reference`: /home/mkd34160/test/SMILify/singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth (epoch 386)
- `sv_constrained`: /hpcwork/mkd34160/smilify_runs/singleview_lam1e-2/checkpoints/checkpoint_epoch_435.pth (epoch 435)

## multiview

Authored axes scored: **162**

| metric | reference | constrained | delta |
|---|---|---|---|
| Violating axes (count) | 76 | 57 | -19 (better) |
| Mean violation rate (% frames) | 13.96 | 0.05 | -13.91 (better) |
| Mean overshoot, violating axes (deg) | 2.65 | 0.00 | -2.65 (better) |
| Max overshoot (deg) | 56.57 | 4.90 | -51.68 (better) |
| MPJPE (mm) | 0.96 | 1.14 | +0.18 (worse) |
| Median MPJPE (mm) | 0.84 | 1.02 | +0.18 (worse) |
| PCK@5px native | — | — | — |
| PCK@5px input | — | — | — |

- `mv_reference`: /hpcwork/mkd34160/SMILify/SMILySTICKS_ViT_model.pth (epoch 345)
- `mv_constrained`: /hpcwork/mkd34160/smilify_runs/multiview_lam1e-2/checkpoints/checkpoint_epoch_0369.pth (epoch 369)

## Single-view vs multi-view

| metric | single-view delta | multi-view delta |
|---|---|---|
| Violating axes (count) | -23 (better) | -19 (better) |
| Mean violation rate (% frames) | -14.74 (better) | -13.91 (better) |
| Mean overshoot (deg) | -2.53 (better) | -2.65 (better) |
| MPJPE (mm) | -0.07 (better) | +0.18 (worse) |
| PCK@5px native | — | — |

The hypothesis worth testing here: multi-view already resolves depth ambiguity from geometry, so it should have fewer violations to fix and less to gain from the prior. A single-view delta noticeably larger than the multi-view one supports that reading.
