# Baseline pose study — `sv_reference`

- Frames analysed: **50630**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 1.092 |
| Median MPJPE (mm) | 0.9227 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **96 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_2_co_r | x | 96.0 | 18.7 | 53.4 |
| l_2_co_l | x | 87.6 | 10.0 | 41.9 |
| l_2_fe_r | x | 83.5 | 17.7 | 67.9 |
| l_2_ti_l | y | 79.9 | 15.4 | 70.5 |
| l_2_fe_l | x | 79.7 | 14.7 | 53.3 |
| l_1_fe_r | x | 76.2 | 12.8 | 77.7 |
| l_1_fe_l | x | 73.8 | 12.1 | 57.4 |
| l_3_fe_l | x | 67.0 | 8.3 | 45.6 |
| l_3_fe_r | x | 61.7 | 7.9 | 52.1 |
| l_1_co_l | x | 61.4 | 4.3 | 26.7 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| l_2_ti_l | y | 159.7 | -80.5 | 79.1 |
| l_1_fe_r | x | 142.2 | -87.7 | 54.5 |
| l_2_ta_l | x | 132.9 | -44.7 | 88.2 |
| l_3_ta_l | z | 129.2 | -69.3 | 59.9 |
| l_2_ta_r | x | 122.9 | -71.1 | 51.8 |
| l_2_fe_r | x | 121.4 | -77.9 | 43.5 |
| l_3_ti_r | x | 121.2 | -63.0 | 58.1 |
| l_1_fe_l | x | 120.6 | -53.1 | 67.4 |
| l_2_fe_l | x | 118.9 | -55.6 | 63.3 |
| l_3_ta_r | x | 117.8 | -61.7 | 56.1 |
