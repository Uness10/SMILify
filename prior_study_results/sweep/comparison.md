# Joint-limit prior — single-view vs multi-view

fine-tune: **+50 epochs**

Each `*_constrained` arm is its `*_reference` checkpoint continued for the same number of epochs with the limit penalty enabled. **All four arms are scored against the same authored ranges**, so the reference rows are the honest "before" number.

> **Caveat — the fine-tuning confound.** The reference arms received zero additional epochs, so any difference below is *the prior plus continued training*, not the prior alone. A `w_limit = 0` control fine-tuned for the same epochs would separate the two; it is not part of this study by design.

Deltas are `constrained - reference`. Lower is better for violations and MPJPE; higher is better for PCK.

## singleview

_Only `sv_reference` present — no delta can be computed._

Authored axes scored: **162**

| metric | reference | constrained | delta |
|---|---|---|---|
| Violating axes (count) | 96 | — | — |
| Mean violation rate (% frames) | 14.76 | — | — |
| Mean overshoot, violating axes (deg) | 2.53 | — | — |
| Max overshoot (deg) | 77.70 | — | — |
| MPJPE (mm) | 1.09 | — | — |
| Median MPJPE (mm) | 0.92 | — | — |
| PCK@5px native | — | — | — |
| PCK@5px input | — | — | — |

- `sv_reference`: /home/mkd34160/test/SMILify/singleview_SMILySTICKS_3D_ViT_checkpoints/best_model.pth (epoch 386)

## multiview

_Only `mv_reference` present — no delta can be computed._

Authored axes scored: **162**

| metric | reference | constrained | delta |
|---|---|---|---|
| Violating axes (count) | 76 | — | — |
| Mean violation rate (% frames) | 13.96 | — | — |
| Mean overshoot, violating axes (deg) | 2.65 | — | — |
| Max overshoot (deg) | 56.57 | — | — |
| MPJPE (mm) | 0.96 | — | — |
| Median MPJPE (mm) | 0.84 | — | — |
| PCK@5px native | — | — | — |
| PCK@5px input | — | — | — |

- `mv_reference`: SMILySTICKS_ViT_model.pth (epoch 345)

## Single-view vs multi-view

_Incomplete: missing `sv_constrained`, `mv_constrained`._
