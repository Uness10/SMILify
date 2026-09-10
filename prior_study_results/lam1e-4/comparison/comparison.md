# Joint-limit prior — single-view vs multi-view

fine-tune: **+25 epochs**  |  `joint_limit_regularization` = **0.0001**

Each `*_constrained` arm is its `*_reference` checkpoint continued for the same number of epochs with the limit penalty enabled. **All four arms are scored against the same authored ranges**, so the reference rows are the honest "before" number.

> **Caveat — the fine-tuning confound.** The reference arms received zero additional epochs, so any difference below is *the prior plus continued training*, not the prior alone. A `w_limit = 0` control fine-tuned for the same epochs would separate the two; it is not part of this study by design.

Deltas are `constrained - reference`. Lower is better for violations and MPJPE; higher is better for PCK.

## singleview

Authored axes scored: **162**

| metric | reference | constrained | delta |
|---|---|---|---|
| Violating axes (count) | 96 | 89 | -7 (better) |
| Mean violation rate (% frames) | 14.76 | 2.73 | -12.03 (better) |
| Mean overshoot, violating axes (deg) | 2.53 | 0.14 | -2.39 (better) |
| Max overshoot (deg) | 77.70 | 50.84 | -26.86 (better) |
| MPJPE (mm) | 1.09 | 0.97 | -0.12 (better) |
| Median MPJPE (mm) | 0.92 | 0.82 | -0.10 (better) |
| PCK@5px native | — | — | — |
| PCK@5px input | — | — | — |

- `sv_reference`: /home/mkd34160/test/SMILify/singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth (epoch 386)
- `sv_constrained`: /hpcwork/mkd34160/smilify_runs/singleview_lam1e-4/checkpoints/checkpoint_epoch_435.pth (epoch 435)

## multiview

Authored axes scored: **162**

| metric | reference | constrained | delta |
|---|---|---|---|
| Violating axes (count) | 76 | 75 | -1 (better) |
| Mean violation rate (% frames) | 13.96 | 4.60 | -9.36 (better) |
| Mean overshoot, violating axes (deg) | 2.65 | 0.44 | -2.21 (better) |
| Max overshoot (deg) | 56.57 | 43.44 | -13.14 (better) |
| MPJPE (mm) | 0.96 | 0.99 | +0.03 (worse) |
| Median MPJPE (mm) | 0.84 | 0.86 | +0.02 (worse) |
| PCK@5px native | — | — | — |
| PCK@5px input | — | — | — |

- `mv_reference`: /hpcwork/mkd34160/SMILify/SMILySTICKS_ViT_model.pth (epoch 345)
- `mv_constrained`: /hpcwork/mkd34160/smilify_runs/multiview_lam1e-4/checkpoints/checkpoint_epoch_0369.pth (epoch 369)

## Single-view vs multi-view

| metric | single-view delta | multi-view delta |
|---|---|---|
| Violating axes (count) | -7 (better) | -1 (better) |
| Mean violation rate (% frames) | -12.03 (better) | -9.36 (better) |
| Mean overshoot (deg) | -2.39 (better) | -2.21 (better) |
| MPJPE (mm) | -0.12 (better) | +0.03 (worse) |
| PCK@5px native | — | — |

The hypothesis worth testing here: multi-view already resolves depth ambiguity from geometry, so it should have fewer violations to fix and less to gain from the prior. A single-view delta noticeably larger than the multi-view one supports that reading.
