# Baseline pose study — `mv_reference`

- Frames analysed: **10126**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 0.9639 |
| Median MPJPE (mm) | 0.8382 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **76 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_3_tr_l | z | 100.0 | 10.6 | 22.3 |
| l_3_tr_r | z | 99.6 | 10.6 | 30.4 |
| l_2_co_r | x | 98.0 | 16.8 | 56.6 |
| l_2_co_l | x | 87.6 | 10.0 | 45.8 |
| l_3_fe_r | z | 84.9 | 3.6 | 17.4 |
| l_2_fe_r | x | 83.3 | 16.5 | 51.6 |
| l_2_fe_l | x | 80.5 | 15.2 | 50.7 |
| l_1_fe_r | x | 77.4 | 13.3 | 55.6 |
| l_1_ta_r | z | 76.5 | 4.2 | 15.4 |
| l_1_fe_l | x | 76.1 | 13.1 | 50.3 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| l_1_fe_r | x | 130.1 | -65.6 | 64.5 |
| l_2_ta_l | x | 115.9 | -40.2 | 75.7 |
| l_1_fe_l | x | 111.4 | -51.1 | 60.3 |
| l_2_ta_r | x | 109.8 | -69.7 | 40.1 |
| b_h | z | 106.2 | -49.6 | 56.6 |
| l_3_fe_l | x | 106.1 | -56.7 | 49.4 |
| l_2_fe_l | x | 105.9 | -45.2 | 60.7 |
| l_3_fe_r | x | 102.2 | -56.0 | 46.2 |
| l_1_ta_r | x | 101.1 | -63.9 | 37.3 |
| l_1_co_l | z | 99.2 | -60.4 | 38.8 |
