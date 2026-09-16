# Baseline pose study — `sv_reference`

- Frames analysed: **50630**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 0.9054 |
| Median MPJPE (mm) | 0.7439 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **94 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_2_co_r | x | 98.9 | 22.7 | 58.2 |
| l_2_co_l | x | 94.2 | 14.7 | 48.8 |
| l_2_fe_r | x | 85.9 | 19.6 | 70.5 |
| l_2_fe_l | x | 84.6 | 17.9 | 57.7 |
| l_1_co_l | x | 81.8 | 8.7 | 32.9 |
| l_1_fe_r | x | 80.9 | 15.8 | 83.2 |
| l_2_ti_l | y | 80.6 | 15.7 | 67.6 |
| l_1_fe_l | x | 75.4 | 13.1 | 59.9 |
| l_1_fe_r | z | 68.1 | 4.0 | 23.7 |
| l_1_ti_r | z | 65.6 | 5.3 | 71.7 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| l_2_ti_l | y | 152.0 | -77.6 | 74.4 |
| l_1_fe_r | x | 142.5 | -93.2 | 49.3 |
| l_3_ta_l | z | 128.8 | -69.5 | 59.3 |
| l_3_ta_r | x | 123.8 | -67.6 | 56.2 |
| l_2_ta_l | x | 121.4 | -42.5 | 78.9 |
| l_2_fe_l | x | 120.3 | -52.6 | 67.7 |
| l_2_fe_r | x | 118.1 | -80.5 | 37.7 |
| l_3_ta_r | z | 118.1 | -48.2 | 69.8 |
| l_1_fe_l | x | 115.8 | -45.9 | 69.9 |
| l_3_ta_l | y | 112.0 | -41.7 | 70.3 |
