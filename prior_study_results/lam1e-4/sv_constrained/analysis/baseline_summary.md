# Baseline pose study — `sv_constrained`

- Frames analysed: **50630**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 0.9718 |
| Median MPJPE (mm) | 0.8184 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **89 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_1_fe_r | x | 32.4 | 1.3 | 33.7 |
| l_3_fe_l | x | 27.4 | 1.6 | 29.4 |
| l_1_fe_l | x | 26.8 | 0.9 | 25.2 |
| l_2_co_r | x | 26.3 | 0.7 | 18.6 |
| l_2_fe_l | x | 25.2 | 0.7 | 28.9 |
| l_3_fe_r | x | 24.6 | 1.1 | 35.4 |
| l_2_fe_r | x | 24.2 | 0.7 | 22.6 |
| l_3_fe_r | y | 22.5 | 1.1 | 27.1 |
| l_3_fe_l | y | 16.5 | 0.5 | 28.1 |
| l_2_co_l | x | 15.2 | 0.3 | 9.0 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| l_2_tr_l | x | 116.1 | -56.2 | 59.9 |
| an_1_r | y | 113.5 | -64.9 | 48.6 |
| l_3_ti_r | x | 110.7 | -56.3 | 54.4 |
| an_1_l | y | 104.9 | -55.1 | 49.9 |
| l_2_ta_l | x | 103.5 | -30.3 | 73.2 |
| b_a_2 | y | 99.2 | -17.9 | 81.3 |
| l_1_co_r | z | 96.3 | -33.0 | 63.3 |
| l_1_co_l | z | 96.3 | -61.2 | 35.1 |
| b_h | z | 94.1 | -43.0 | 51.1 |
| l_1_tr_l | x | 93.3 | -48.7 | 44.7 |
