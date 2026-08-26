# Baseline pose study — `mv_constrained`

- Frames analysed: **10126**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 2.953 |
| Median MPJPE (mm) | 2.4814 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **1 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_2_tr_l | z | 0.0 | 0.0 | 0.8 |
| b_a_1 | x | 0.0 | 0.0 | 0.0 |
| b_a_1 | y | 0.0 | 0.0 | 0.0 |
| b_a_1 | z | 0.0 | 0.0 | 0.0 |
| b_a_2 | x | 0.0 | 0.0 | 0.0 |
| b_a_2 | y | 0.0 | 0.0 | 0.0 |
| b_a_2 | z | 0.0 | 0.0 | 0.0 |
| b_a_3 | x | 0.0 | 0.0 | 0.0 |
| b_a_3 | y | 0.0 | 0.0 | 0.0 |
| b_a_3 | z | 0.0 | 0.0 | 0.0 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| b_a_2 | y | 63.7 | -4.2 | 59.5 |
| l_2_ta_r | x | 39.4 | -23.1 | 16.3 |
| an_1_r | y | 39.3 | -34.9 | 4.3 |
| l_3_tr_l | x | 35.9 | -21.3 | 14.6 |
| l_1_ta_r | x | 35.9 | -9.9 | 26.0 |
| l_1_co_l | z | 34.8 | -25.5 | 9.3 |
| l_1_tr_r | x | 33.1 | -24.2 | 8.9 |
| l_1_ta_l | x | 31.6 | -15.1 | 16.5 |
| l_2_ta_l | x | 29.5 | -32.5 | -3.0 |
| an_1_l | y | 28.0 | -43.6 | -15.7 |
