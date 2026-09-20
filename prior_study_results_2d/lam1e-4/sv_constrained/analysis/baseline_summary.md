# Baseline pose study — `sv_constrained`

- Frames analysed: **50630**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 241.7587 |
| Median MPJPE (mm) | 254.6216 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **107 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_2_fe_r | x | 37.9 | 0.8 | 17.0 |
| l_2_fe_l | x | 30.1 | 0.4 | 12.0 |
| l_2_co_r | x | 28.9 | 1.3 | 46.3 |
| l_3_fe_l | y | 25.7 | 0.6 | 16.8 |
| l_3_co_l | y | 22.8 | 1.3 | 19.0 |
| l_3_fe_l | x | 22.1 | 1.1 | 37.7 |
| l_3_fe_r | y | 21.0 | 0.5 | 21.3 |
| l_1_fe_r | x | 20.9 | 0.5 | 35.3 |
| l_1_fe_l | x | 20.8 | 0.4 | 21.6 |
| l_3_fe_l | z | 20.4 | 0.3 | 8.3 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| l_1_co_r | y | 150.0 | -82.7 | 67.2 |
| an_1_l | y | 144.5 | -57.8 | 86.7 |
| l_2_co_l | y | 142.8 | -73.7 | 69.2 |
| l_1_co_l | y | 141.0 | -81.9 | 59.1 |
| l_2_co_r | y | 138.0 | -67.8 | 70.2 |
| l_3_ta_r | x | 132.1 | -84.2 | 47.8 |
| an_1_r | y | 131.3 | -52.8 | 78.5 |
| l_3_co_r | y | 122.6 | -64.4 | 58.1 |
| l_3_co_l | y | 122.5 | -59.5 | 63.0 |
| l_2_co_l | z | 121.6 | -62.6 | 59.0 |
