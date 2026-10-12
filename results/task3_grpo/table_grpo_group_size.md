24 prompts x 8 cached completions = 192 completions for every K. Informative = within-group std > 1e-06; informative@0.1sd uses 0.1 x overall reward std (1.557). Bins: difficulty = mean cached reward of the prompt's 8 completions; equal-count terciles, hard = lowest mean reward (edges 1.31, 2.42). Variance of the group-relative signal: var_centred_reward = Var(r - group mean), the unnormalised signal; after std-normalisation every informative group has unit variance, so var_norm_adv just equals the informative rate. Group statistics include completions that hit the cache's generation cap (clipped fraction 0.151), which training would mask.

| K | bin | groups | informative_rate | informative@0.1sd | mean_group_std | var_centred_reward | var_norm_adv |
|---|---|---|---|---|---|---|---|
| 2 | all | 96 | 0.9375 | 0.6354 | 0.3846 | 0.3368 | 0.9375 |
| 2 | hard | 32 | 0.8438 | 0.6875 | 0.4934 | 0.5984 | 0.8437 |
| 2 | medium | 32 | 0.9688 | 0.6562 | 0.4124 | 0.3041 | 0.9687 |
| 2 | easy | 32 | 1 | 0.5625 | 0.2479 | 0.108 | 1 |
| 4 | all | 48 | 0.9583 | 0.8542 | 0.541 | 0.4677 | 0.9583 |
| 4 | hard | 16 | 0.875 | 0.8125 | 0.6615 | 0.7865 | 0.875 |
| 4 | medium | 16 | 1 | 0.875 | 0.5517 | 0.4017 | 1 |
| 4 | easy | 16 | 1 | 0.875 | 0.4098 | 0.2149 | 1 |
| 8 | all | 24 | 0.9583 | 0.9583 | 0.6191 | 0.5599 | 0.9583 |
| 8 | hard | 8 | 0.875 | 0.875 | 0.7634 | 0.8869 | 0.875 |
| 8 | medium | 8 | 1 | 1 | 0.599 | 0.4469 | 1 |
| 8 | easy | 8 | 1 | 1 | 0.4948 | 0.3459 | 1 |
