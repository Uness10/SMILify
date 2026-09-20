# Baseline pose study — `sv_constrained`

- Frames analysed: **50630**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 242.0276 |
| Median MPJPE (mm) | 253.2117 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **99 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_2_fe_r | x | 91.7 | 46.2 | 132.7 |
| l_2_fe_l | x | 91.4 | 49.2 | 124.6 |
| l_1_fe_l | x | 84.1 | 27.4 | 83.8 |
| l_3_fe_r | x | 82.9 | 26.8 | 89.1 |
| l_1_fe_r | x | 81.8 | 21.9 | 117.6 |
| l_3_fe_l | x | 81.0 | 30.1 | 113.3 |
| l_3_fe_l | y | 75.4 | 30.4 | 111.9 |
| l_3_fe_r | y | 72.4 | 24.9 | 93.7 |
| l_1_fe_r | z | 65.7 | 10.1 | 82.0 |
| l_3_ti_l | z | 65.1 | 8.2 | 39.5 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| l_2_ta_r | x | 326.2 | -175.2 | 151.1 |
| l_1_fe_r | x | 221.1 | -127.6 | 93.5 |
| l_3_fe_l | z | 215.6 | -125.3 | 90.3 |
| l_3_fe_r | z | 191.6 | -89.2 | 102.4 |
| l_3_fe_l | x | 190.9 | -67.6 | 123.3 |
| l_2_fe_r | x | 188.5 | -142.7 | 45.8 |
| l_2_ta_l | x | 186.7 | -90.0 | 96.7 |
| l_1_fe_l | y | 183.2 | -112.6 | 70.6 |
| l_1_ta_l | x | 180.1 | -58.8 | 121.3 |
| l_1_fe_r | y | 177.9 | -109.7 | 68.2 |
