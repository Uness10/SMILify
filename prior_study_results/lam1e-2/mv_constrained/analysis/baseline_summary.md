# Baseline pose study — `mv_constrained`

- Frames analysed: **10126**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 1.1414 |
| Median MPJPE (mm) | 1.0151 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **57 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_2_co_r | x | 1.8 | 0.0 | 2.9 |
| l_1_fe_r | x | 0.6 | 0.0 | 3.7 |
| l_2_fe_l | x | 0.6 | 0.0 | 1.6 |
| l_1_tr_r | z | 0.6 | 0.0 | 2.7 |
| l_2_ti_r | z | 0.5 | 0.0 | 1.8 |
| l_2_co_l | x | 0.4 | 0.0 | 3.5 |
| l_3_fe_l | x | 0.4 | 0.0 | 1.4 |
| l_3_fe_l | y | 0.4 | 0.0 | 2.7 |
| l_2_tr_r | x | 0.3 | 0.0 | 1.3 |
| l_3_fe_r | y | 0.2 | 0.0 | 1.7 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| an_1_r | y | 101.1 | -59.8 | 41.3 |
| b_a_2 | y | 90.8 | -13.9 | 77.0 |
| an_1_l | y | 89.9 | -54.7 | 35.2 |
| l_2_ta_l | x | 86.4 | -22.6 | 63.9 |
| l_2_co_r | z | 84.9 | -51.4 | 33.5 |
| l_1_co_l | z | 83.7 | -60.5 | 23.2 |
| l_1_co_r | z | 82.9 | -22.6 | 60.4 |
| l_3_co_r | x | 76.7 | -31.9 | 44.8 |
| l_1_tr_l | x | 73.7 | -41.0 | 32.6 |
| l_2_ta_r | x | 73.2 | -54.5 | 18.7 |
