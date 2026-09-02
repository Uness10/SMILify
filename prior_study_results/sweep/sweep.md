# Joint-limit prior — lambda sweep

`lambda = 0` is the reference checkpoint scored as-is: it received **zero**
additional epochs, so every row below differs from it by *the prior plus the
continuation*, not the prior alone. Comparisons BETWEEN lambdas are clean —
they share the epoch count — so read the sweep as a curve first and the
reference row as context.

| mode | lambda | violating axes | mean viol. rate % | mean overshoot deg | MPJPE mm | PCK@5px (native) |
|---|---|---|---|---|---|---|
| multiview | 0 | 76/162 | 13.96 | 2.65 | 0.96 | — |
| singleview | 0 | 96/162 | 14.76 | 2.53 | 1.09 | — |

Lower is better for violations, overshoot and MPJPE; higher for PCK.
A lambda that lowers violations while leaving MPJPE/PCK flat is the win
condition; one that lowers both violations and PCK is the prior overriding
the data, which is what the low end of this sweep is meant to avoid.
