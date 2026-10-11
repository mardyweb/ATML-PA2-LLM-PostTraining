Cached batch: 32 rollouts, 8814 response tokens. Clip fraction = tokens with ratio outside [1-eps, 1+eps]; affected = tokens where the clipped term is the one used (gradient zeroed); surrogate gap = unclipped minus clipped surrogate, per token.

| epsilon | clip_frac_midpoint | affected_frac_midpoint | surrogate_gap_midpoint | clip_frac_after_2_steps | affected_frac_after_2_steps | surrogate_gap_after_2_steps |
|---|---|---|---|---|---|---|
| 0.05 | 0.1066 | 0.0523 | 0.005479 | 0.1103 | 0.05412 | 0.005493 |
| 0.2 | 0.008282 | 0.004538 | 0.004071 | 0.008396 | 0.004652 | 0.004075 |
| 0.5 | 0.005559 | 0.00329 | 0.00342 | 0.005559 | 0.00329 | 0.003416 |
