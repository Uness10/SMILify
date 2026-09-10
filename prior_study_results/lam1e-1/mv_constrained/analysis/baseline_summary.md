# Baseline pose study — `mv_constrained`

- Frames analysed: **10126**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 1.2111 |
| Median MPJPE (mm) | 1.0624 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **21 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_2_co_l | x | 0.3 | 0.0 | 2.2 |
| l_2_fe_r | x | 0.1 | 0.0 | 1.8 |
| l_1_co_l | z | 0.1 | 0.0 | 2.7 |
| l_3_fe_l | y | 0.1 | 0.0 | 1.2 |
| l_1_fe_r | x | 0.1 | 0.0 | 1.4 |
| l_1_co_r | z | 0.1 | 0.0 | 2.7 |
| l_2_ti_r | z | 0.1 | 0.0 | 0.6 |
| l_2_co_r | x | 0.1 | 0.0 | 1.0 |
| l_2_tr_l | z | 0.1 | 0.0 | 0.7 |
| l_3_tr_l | x | 0.0 | 0.0 | 0.4 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| an_1_r | y | 101.3 | -59.0 | 42.4 |
| an_1_l | y | 90.1 | -52.6 | 37.5 |
| b_a_2 | y | 88.3 | -13.1 | 75.2 |
| l_1_co_r | z | 82.3 | -22.7 | 59.6 |
| l_1_co_l | z | 82.0 | -59.3 | 22.7 |
| l_2_co_r | z | 80.5 | -48.6 | 31.9 |
| l_2_ta_l | x | 72.3 | -22.2 | 50.1 |
| l_1_tr_l | x | 70.9 | -41.8 | 29.1 |
| an_1_r | z | 69.0 | -50.8 | 18.2 |
| l_3_co_r | x | 68.2 | -28.5 | 39.8 |
