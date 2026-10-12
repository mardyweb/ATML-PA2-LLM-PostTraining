Rebuilt batch check: recomputed vs cached reference log-probs differ by median 0.0043 (mean 0.0264) nats/token; prompts matched {'prompt_id': 32, 'source_index': 0, 'position': 0}; prompt renderings {'chat_template': 29, 'right_truncated': 3}; log-prob temperature 1.0. This is the recomputation noise floor for the midpoint ratios. Advantages are whitened over the whole cached batch (training whitens per update, i.e. per rollout). Cached batch: 32 rollouts, 8814 response tokens. Clip fraction = tokens with ratio outside [1-eps, 1+eps]; affected = tokens where the clipped term is the one used (gradient zeroed); surrogate gap = unclipped minus clipped surrogate, per token.

| epsilon | clip_frac_midpoint | affected_frac_midpoint | clipped_surrogate_midpoint | unclipped_surrogate_midpoint | surrogate_gap_midpoint | clip_frac_after_2_steps | affected_frac_after_2_steps | surrogate_gap_after_2_steps |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.1066 | 0.0523 | -0.005342 | 0.0001379 | 0.005479 | 0.1103 | 0.05412 | 0.005493 |
| 0.2 | 0.008282 | 0.004538 | -0.003933 | 0.0001379 | 0.004071 | 0.008396 | 0.004652 | 0.004075 |
| 0.5 | 0.005559 | 0.00329 | -0.003282 | 0.0001379 | 0.00342 | 0.005559 | 0.00329 | 0.003416 |
