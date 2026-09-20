# Joint-limit prior — single-view vs multi-view

fine-tune: **+50 epochs**  |  `joint_limit_regularization` = **0.0001**

Each `*_constrained` arm is its `*_reference` checkpoint continued for the same number of epochs with the limit penalty enabled. **All four arms are scored against the same authored ranges**, so the reference rows are the honest "before" number.

> **Caveat — the fine-tuning confound.** The reference arms received zero additional epochs, so any difference below is *the prior plus continued training*, not the prior alone. A `w_limit = 0` control fine-tuned for the same epochs would separate the two; it is not part of this study by design.

Deltas are `constrained - reference`. Lower is better for violations and MPJPE; higher is better for PCK.

## singleview

_Only `sv_constrained` present — no delta can be computed._

Authored axes scored: **162**

| metric | reference | constrained | delta |
|---|---|---|---|
| Violating axes (count) | — | 107 | — |
| Mean violation rate (% frames) | — | 5.02 | — |
| Mean overshoot, violating axes (deg) | — | 0.19 | — |
| Max overshoot (deg) | — | 46.33 | — |
| MPJPE (mm) | — | 241.76 | — |
| Median MPJPE (mm) | — | 254.62 | — |
| N-MPJPE (mm), global scale fitted | — | 9.80 | — |
| Median N-MPJPE (mm) | — | 7.56 | — |
| PA-MPJPE (mm), per-frame aligned | — | 4.88 | — |
| Fitted global scale (1.0 = no drift) | — | 2.504 | — |
| PCK@5px native | — | — | — |
| PCK@5px input | — | — | — |

- `sv_constrained`: /hpcwork/mkd34160/smilify_runs/singleview_2d_lam1e-4/checkpoints/best_model.pth (epoch 190)

## multiview

_Neither `mv_reference` nor `mv_constrained` found._

## Single-view vs multi-view

_Incomplete: missing `sv_reference`, `mv_reference`, `mv_constrained`._
