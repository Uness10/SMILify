# Baseline pose study — `sv_constrained`

- Frames analysed: **50630**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 1.0232 |
| Median MPJPE (mm) | 0.8703 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **73 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_2_co_r | x | 0.2 | 0.0 | 1.8 |
| l_2_tr_r | x | 0.2 | 0.0 | 5.3 |
| l_1_fe_r | x | 0.2 | 0.0 | 2.9 |
| l_2_fe_l | x | 0.2 | 0.0 | 2.9 |
| l_1_co_r | z | 0.1 | 0.0 | 3.8 |
| l_1_fe_l | x | 0.1 | 0.0 | 1.7 |
| l_1_fe_l | z | 0.1 | 0.0 | 0.7 |
| l_3_fe_l | x | 0.1 | 0.0 | 1.9 |
| l_2_co_l | x | 0.1 | 0.0 | 1.9 |
| l_2_tr_l | z | 0.1 | 0.0 | 1.9 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| an_1_r | y | 112.1 | -60.8 | 51.3 |
| an_1_l | y | 109.8 | -57.6 | 52.2 |
| b_a_2 | y | 96.6 | -17.4 | 79.2 |
| l_1_tr_l | x | 87.4 | -51.0 | 36.4 |
| l_2_ta_r | x | 86.8 | -57.6 | 29.2 |
| l_1_co_l | z | 86.3 | -60.4 | 25.8 |
| l_1_co_r | z | 85.1 | -23.8 | 61.3 |
| l_2_ta_l | x | 84.8 | -25.2 | 59.6 |
| l_2_co_r | z | 84.1 | -48.8 | 35.3 |
| l_3_ta_r | x | 82.4 | -39.7 | 42.7 |
