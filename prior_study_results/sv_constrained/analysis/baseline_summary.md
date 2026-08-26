# Baseline pose study — `sv_constrained`

- Frames analysed: **50630**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 1.8775 |
| Median MPJPE (mm) | 1.5873 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **5 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_3_fe_r | x | 0.0 | 0.0 | 1.1 |
| w_1_l | z | 0.0 | 0.0 | 0.2 |
| l_3_co_r | z | 0.0 | 0.0 | 0.3 |
| b_h | x | 0.0 | 0.0 | 0.1 |
| b_h | y | 0.0 | 0.0 | 0.0 |
| b_a_1 | x | 0.0 | 0.0 | 0.0 |
| b_a_1 | y | 0.0 | 0.0 | 0.0 |
| b_a_1 | z | 0.0 | 0.0 | 0.0 |
| b_a_2 | x | 0.0 | 0.0 | 0.0 |
| b_a_2 | y | 0.0 | 0.0 | 0.0 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| b_a_2 | y | 98.8 | -15.5 | 83.4 |
| an_1_r | y | 91.3 | -46.5 | 44.8 |
| an_1_l | y | 83.6 | -39.6 | 44.1 |
| l_1_co_r | z | 68.3 | -19.2 | 49.1 |
| l_1_co_l | z | 63.1 | -48.6 | 14.5 |
| l_2_co_l | z | 60.9 | -19.5 | 41.4 |
| l_3_co_r | y | 57.2 | -29.6 | 27.6 |
| b_a_3 | y | 55.0 | -13.2 | 41.8 |
| b_a_1 | z | 53.1 | -28.6 | 24.5 |
| b_h | z | 52.5 | -22.9 | 29.6 |
