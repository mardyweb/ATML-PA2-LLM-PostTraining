AI-judge labels on all 250 safe and 200 unsafe XSTest prompts, greedy decoding, max 256 new tokens. Safe rates are over safe prompts, unsafe rates over unsafe prompts; ambiguous rate over all prompts. Because the judge does not see the prompt's XSTest class, safe_any_refusal_rate (OVER_REFUSAL or JUSTIFIED_REFUSAL on safe prompts) and unsafe_any_compliance_rate (UNSAFE_COMPLIANCE or SAFE_ANSWER on unsafe prompts) are given as sensitivity checks.

| policy | safe_answer_rate | safe_over_refusal_rate | unsafe_compliance_rate | unsafe_justified_refusal_rate | ambiguous_rate | mean_tokens | std_tokens |
|---|---|---|---|---|---|---|---|
| SFT | 0.580 | 0.000 | 0.000 | 0.880 | 0.000 | 108.049 | 80.575 |
| DPO | 0.556 | 0.000 | 0.000 | 0.875 | 0.000 | 109.944 | 82.137 |
| PPO | 0.580 | 0.000 | 0.000 | 0.875 | 0.000 | 107.960 | 80.652 |
| GRPO | 0.572 | 0.000 | 0.000 | 0.875 | 0.000 | 107.640 | 79.743 |

Full columns (incl. ambiguous by class, parse failures, length by class, reward-model score):

| policy | n_safe | n_unsafe | safe_answer_rate | safe_over_refusal_rate | unsafe_compliance_rate | unsafe_justified_refusal_rate | ambiguous_rate | ambiguous_rate_safe | ambiguous_rate_unsafe | safe_any_refusal_rate | unsafe_any_compliance_rate | parse_failure_rate | mean_tokens | std_tokens | iqr_tokens | frac_hit_max_tokens | mean_tokens_safe | mean_tokens_unsafe | mean_reward_safe | mean_reward_unsafe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT | 250 | 200 | 0.580 | 0.000 | 0.000 | 0.880 | 0.000 | 0.000 | 0.000 | 0.408 | 0.120 | 0.000 | 108.049 | 80.575 | 121.750 | 0.122 | 135.580 | 73.635 | 1.172 | 0.321 |
| DPO | 250 | 200 | 0.556 | 0.000 | 0.000 | 0.875 | 0.000 | 0.000 | 0.000 | 0.436 | 0.125 | 0.000 | 109.944 | 82.137 | 126.000 | 0.116 | 136.136 | 77.205 | 1.192 | 0.367 |
| PPO | 250 | 200 | 0.580 | 0.000 | 0.000 | 0.875 | 0.000 | 0.000 | 0.000 | 0.408 | 0.125 | 0.000 | 107.960 | 80.652 | 115.000 | 0.120 | 136.020 | 72.885 | 1.186 | 0.333 |
| GRPO | 250 | 200 | 0.572 | 0.000 | 0.000 | 0.875 | 0.000 | 0.000 | 0.000 | 0.412 | 0.125 | 0.000 | 107.640 | 79.743 | 115.000 | 0.113 | 135.972 | 72.225 | 1.177 | 0.333 |
