# Baseline pose study — `mv_reference`

- Frames analysed: **10126**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 0.9042 |
| Median MPJPE (mm) | 0.7805 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **79 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_3_tr_l | z | 100.0 | 10.0 | 21.9 |
| l_3_tr_r | z | 99.6 | 10.5 | 30.1 |
| l_2_co_r | x | 98.5 | 17.7 | 59.1 |
| l_2_co_l | x | 94.0 | 12.6 | 49.2 |
| l_3_fe_r | z | 89.0 | 4.1 | 18.3 |
| l_2_fe_r | x | 84.5 | 17.5 | 53.7 |
| l_2_fe_l | x | 83.5 | 17.2 | 54.5 |
| l_1_fe_r | x | 79.6 | 14.7 | 56.7 |
| l_1_fe_l | x | 77.0 | 13.5 | 50.1 |
| l_1_ta_r | z | 74.6 | 3.9 | 14.5 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| l_1_fe_r | x | 132.2 | -66.7 | 65.5 |
| l_2_ta_l | x | 118.4 | -40.7 | 77.7 |
| l_3_co_l | x | 110.6 | -41.7 | 68.9 |
| l_1_fe_l | x | 106.2 | -46.1 | 60.1 |
| l_2_fe_l | x | 105.4 | -40.8 | 64.5 |
| l_3_fe_l | x | 104.7 | -56.2 | 48.5 |
| l_2_ta_r | x | 102.9 | -62.7 | 40.2 |
| l_3_fe_r | x | 102.9 | -55.2 | 47.7 |
| b_h | z | 102.3 | -46.3 | 56.0 |
| l_2_co_l | x | 102.2 | -59.2 | 42.9 |
