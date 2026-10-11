Short forks (8 updates, kl_beta=0.1) from the midpoint. Stability statistic: post_update_kl = k3 estimate of KL(old || new) on each update's rollout after the PPO epochs (bigger = larger policy jump per update).

| epsilon | heldout_reward | reward_sem | kl_per_token | entropy | len_mean | truncation_rate | mean_post_update_kl | max_post_update_kl | mean_clip_fraction | reward_std_over_updates | generated_tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 1.669 | 0.1237 | -3.562e-05 | 0.9928 | 278.4 | 0.0625 | 1.308e-05 | 2.561e-05 | 0 | 0.92 | 2850 |
| 0.2 | 1.722 | 0.1202 | -9.036e-05 | 0.9934 | 291.9 | 0.07812 | 1.367e-05 | 2.202e-05 | 0 | 0.92 | 2850 |
| 0.5 | 1.559 | 0.1295 | -6.126e-05 | 0.9661 | 294.4 | 0.1172 | 1.462e-05 | 3.664e-05 | 0 | 0.92 | 2850 |
