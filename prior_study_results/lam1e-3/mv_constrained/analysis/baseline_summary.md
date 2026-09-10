# Baseline pose study — `mv_constrained`

- Frames analysed: **10126**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 1.075 |
| Median MPJPE (mm) | 0.9475 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **73 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_2_fe_l | x | 7.0 | 0.1 | 8.3 |
| l_2_co_r | x | 6.9 | 0.1 | 9.6 |
| l_3_fe_l | x | 5.0 | 0.1 | 5.8 |
| l_1_fe_l | x | 4.9 | 0.0 | 7.2 |
| l_3_fe_l | y | 4.0 | 0.0 | 6.5 |
| l_2_co_l | x | 4.0 | 0.1 | 11.0 |
| l_3_fe_r | x | 3.8 | 0.0 | 17.6 |
| l_2_fe_r | x | 3.8 | 0.0 | 13.5 |
| l_3_fe_r | y | 3.5 | 0.0 | 4.1 |
| l_1_fe_r | x | 3.2 | 0.1 | 13.8 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| an_1_r | y | 103.4 | -62.2 | 41.2 |
| l_2_ta_l | x | 95.8 | -25.5 | 70.3 |
| b_a_2 | y | 92.1 | -14.5 | 77.6 |
| an_1_l | y | 90.6 | -54.0 | 36.7 |
| l_1_co_l | z | 90.3 | -59.9 | 30.4 |
| l_1_co_r | z | 87.9 | -27.3 | 60.6 |
| l_3_co_l | x | 84.4 | -35.8 | 48.5 |
| l_2_co_r | z | 84.1 | -51.1 | 33.0 |
| l_3_co_r | x | 81.2 | -31.0 | 50.2 |
| l_3_ta_r | x | 79.8 | -45.6 | 34.2 |
