Training budgets differ: standard DPO saw 1446 pairs (91 steps), length-balanced DPO 1442 pairs (91 steps), both one epoch. Held-out preference accuracy per length stratum on the length-stratified eval file, with 95% bootstrap intervals (2000 resamples).

| model | stratum | n | pref_acc | acc_ci_low | acc_ci_high | dpo_loss_beta0.1 | margin_mean | mean_chosen_tokens | mean_rejected_tokens |
|---|---|---|---|---|---|---|---|---|---|
| standard | preferred_longer | 78 | 0.4487 | 0.3458 | 0.5513 | 0.6999 | -0.08381 | 221.3 | 105.3 |
| standard | length_matched | 80 | 0.575 | 0.4625 | 0.675 | 0.6847 | 0.2069 | 121.8 | 122.4 |
| standard | rejected_longer | 79 | 0.6582 | 0.5443 | 0.7595 | 0.6617 | 0.7012 | 101 | 207.1 |
| length_balanced | preferred_longer | 78 | 0.4615 | 0.359 | 0.5769 | 0.7207 | -0.4538 | 221.3 | 105.3 |
| length_balanced | length_matched | 80 | 0.6 | 0.4875 | 0.7 | 0.6775 | 0.3676 | 121.8 | 122.4 |
| length_balanced | rejected_longer | 79 | 0.7975 | 0.7089 | 0.8734 | 0.6275 | 1.481 | 101 | 207.1 |
