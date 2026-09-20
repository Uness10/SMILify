# Baseline pose study — `sv_constrained`

- Frames analysed: **50630**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 238.3068 |
| Median MPJPE (mm) | 255.5018 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **28 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_1_fe_l | y | 0.0 | 0.0 | 2.1 |
| l_2_fe_l | x | 0.0 | 0.0 | 0.2 |
| l_1_fe_r | x | 0.0 | 0.0 | 0.1 |
| an_1_l | x | 0.0 | 0.0 | 1.6 |
| l_1_co_l | z | 0.0 | 0.0 | 0.6 |
| l_1_co_r | z | 0.0 | 0.0 | 0.8 |
| l_2_fe_r | x | 0.0 | 0.0 | 0.1 |
| l_3_tr_r | y | 0.0 | 0.0 | 0.7 |
| l_3_fe_r | y | 0.0 | 0.0 | 0.7 |
| l_1_tr_r | z | 0.0 | 0.0 | 0.6 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| b_a_3 | y | 116.8 | -18.3 | 98.5 |
| an_1_l | y | 111.7 | -54.6 | 57.1 |
| an_1_r | y | 108.5 | -56.7 | 51.8 |
| l_2_ti_r | x | 106.0 | -46.2 | 59.9 |
| l_2_ti_l | x | 102.9 | -55.2 | 47.7 |
| l_1_ti_r | x | 98.1 | -38.1 | 60.0 |
| l_1_ti_l | x | 95.5 | -53.3 | 42.2 |
| an_1_l | z | 92.0 | -32.5 | 59.5 |
| b_a_2 | y | 89.9 | -17.8 | 72.1 |
| an_1_r | z | 89.7 | -53.0 | 36.7 |
