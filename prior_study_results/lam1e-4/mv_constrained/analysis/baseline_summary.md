# Baseline pose study — `mv_constrained`

- Frames analysed: **10126**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 0.992 |
| Median MPJPE (mm) | 0.8618 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **75 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_1_fe_r | x | 50.4 | 3.2 | 43.4 |
| l_1_fe_l | x | 47.7 | 2.7 | 34.2 |
| l_2_co_r | x | 47.6 | 3.2 | 40.8 |
| l_2_fe_l | x | 45.6 | 3.0 | 31.6 |
| l_2_fe_r | x | 42.5 | 2.2 | 26.7 |
| l_3_fe_l | x | 40.7 | 2.9 | 37.5 |
| l_3_fe_r | x | 38.8 | 2.7 | 30.2 |
| l_3_fe_l | y | 37.9 | 2.1 | 30.3 |
| l_2_co_l | x | 34.1 | 1.7 | 28.6 |
| l_3_tr_l | z | 30.7 | 1.2 | 9.4 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| l_3_co_l | x | 103.5 | -35.5 | 68.0 |
| l_2_ta_l | x | 103.0 | -30.8 | 72.2 |
| an_1_r | y | 98.9 | -58.8 | 40.1 |
| b_h | z | 95.9 | -45.2 | 50.7 |
| b_a_2 | y | 95.6 | -15.8 | 79.8 |
| l_1_co_l | z | 93.4 | -58.1 | 35.4 |
| an_1_l | y | 90.2 | -53.4 | 36.8 |
| l_3_ta_r | x | 88.9 | -60.8 | 28.1 |
| l_3_co_r | x | 86.7 | -35.3 | 51.4 |
| l_1_fe_r | x | 85.8 | -32.3 | 53.4 |
