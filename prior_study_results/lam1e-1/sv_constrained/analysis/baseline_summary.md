# Baseline pose study — `sv_constrained`

- Frames analysed: **50630**  |  joints: **55**  |  fps: 30.0
- Authored-limits source: `pkl:SMILy_STICK_limits_authored.pkl:joint_limits`

## Accuracy (from benchmark_report.txt)

| metric | value |
|---|---|
| MPJPE (mm) | 1.0479 |
| Median MPJPE (mm) | 0.8924 |
| PCK@5px (native) | None |
| PCK@5px (input) | None |

## Limit violations (unconstrained model vs authored ranges)

- Joint-axes violating the prior at least once: **41 / 162**

Top 10 most-violated joint axes:

| joint | axis | % frames out | mean overshoot (deg) | max overshoot (deg) |
|---|---|---|---|---|
| l_2_co_r | x | 0.2 | 0.0 | 1.2 |
| l_3_ta_r | z | 0.1 | 0.0 | 2.5 |
| l_3_fe_r | x | 0.0 | 0.0 | 1.6 |
| l_2_tr_r | x | 0.0 | 0.0 | 2.2 |
| l_2_tr_r | y | 0.0 | 0.0 | 0.4 |
| l_1_tr_l | z | 0.0 | 0.0 | 0.2 |
| l_1_co_r | z | 0.0 | 0.0 | 1.2 |
| l_3_tr_l | z | 0.0 | 0.0 | 0.5 |
| l_3_fe_l | x | 0.0 | 0.0 | 0.4 |
| b_h | y | 0.0 | 0.0 | 1.7 |

## Widest range of motion (per axis)

| joint | axis | ROM (deg) | min (deg) | max (deg) |
|---|---|---|---|---|
| l_2_pt_l | x | 145.7 | -55.9 | 89.8 |
| an_1_l | y | 110.2 | -56.6 | 53.6 |
| an_1_r | y | 110.1 | -58.0 | 52.1 |
| b_a_2 | y | 96.7 | -16.7 | 80.0 |
| l_1_tr_l | x | 85.7 | -50.4 | 35.3 |
| l_2_co_r | z | 82.4 | -48.1 | 34.3 |
| l_1_co_r | z | 82.0 | -21.2 | 60.8 |
| an_1_l | z | 80.6 | -25.0 | 55.6 |
| l_2_ta_l | x | 79.3 | -22.0 | 57.4 |
| l_1_co_l | z | 78.8 | -59.4 | 19.3 |
