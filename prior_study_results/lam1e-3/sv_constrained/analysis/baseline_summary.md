# Baseline pose study — `sv_constrained`

- Frames analysed: **50630**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 1.0075 |
| Median MPJPE (mm) | 0.854 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **85 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_1_fe_r | x | 2.7 | 0.0 | 10.5 |
| l_2_co_r | x | 2.5 | 0.0 | 7.1 |
| l_1_fe_l | x | 2.3 | 0.0 | 7.5 |
| l_3_fe_l | x | 2.2 | 0.0 | 5.2 |
| l_2_tr_r | x | 2.2 | 0.0 | 9.7 |
| l_2_fe_r | x | 2.1 | 0.0 | 9.0 |
| l_2_fe_l | x | 1.7 | 0.0 | 10.5 |
| l_2_co_l | x | 1.5 | 0.0 | 3.8 |
| l_3_fe_r | y | 1.4 | 0.0 | 4.3 |
| l_3_fe_r | x | 1.3 | 0.0 | 18.4 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| an_1_r | y | 115.9 | -65.7 | 50.2 |
| an_1_l | y | 108.9 | -57.4 | 51.5 |
| l_2_tr_l | x | 104.7 | -61.1 | 43.5 |
| l_3_ti_r | x | 101.6 | -49.5 | 52.1 |
| b_a_2 | y | 97.4 | -17.7 | 79.8 |
| l_1_co_l | z | 92.6 | -61.3 | 31.4 |
| l_1_tr_l | x | 91.4 | -51.6 | 39.9 |
| l_1_co_r | z | 91.4 | -30.1 | 61.3 |
| l_2_ta_l | x | 91.1 | -27.4 | 63.6 |
| l_2_ta_r | x | 88.6 | -56.5 | 32.2 |
