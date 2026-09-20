# Joint-limit prior — lambda sweep

`lambda = 0` is the reference checkpoint scored as-is: it received **zero**
additional epochs, so every row below differs from it by *the prior plus the
continuation*, not the prior alone. Comparisons BETWEEN lambdas are clean —
they share the epoch count — so read the sweep as a curve first and the
reference row as context.

| mode | lambda | violating axes | mean viol. rate % | mean overshoot deg | MPJPE mm | PCK@5px (native) |
|---|---|---|---|---|---|---|
| singleview | 0 | 99/162 | 16.91 | 5.07 | 242.03 | — |
| singleview | 0.0001 | 107/162 | 5.02 | 0.19 | 241.76 | — |
| singleview | 0.1 | 28/162 | 0.00 | 0.00 | 238.31 | — |

Lower is better for violations, overshoot and MPJPE; higher for PCK.
A lambda that lowers violations while leaving MPJPE/PCK flat is the win
condition; one that lowers both violations and PCK is the prior overriding
the data, which is what the low end of this sweep is meant to avoid.
