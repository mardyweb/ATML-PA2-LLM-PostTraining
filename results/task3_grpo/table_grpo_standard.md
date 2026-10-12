Standard continuation: 20 updates x 1 prompt x K=4, 24146 generated tokens, wall-clock 335 s, peak VRAM 14.58 GB on NVIDIA A100-SXM4-40GB. Mean within-group reward std 0.4916, uninformative groups 0, truncated (masked) completions 0.263.

| condition | heldout_reward | reward_sem | kl_per_token | entropy | len_mean | len_std | len_iqr | truncation_rate |
|---|---|---|---|---|---|---|---|---|
| GRPO midpoint (start) | 1.651 | 0.1268 | 2.727e-05 | 0.9564 | 295.6 | 266.5 | 499.8 | 0.1094 |
| GRPO standard (+20 updates) | 1.718 | 0.1178 | 0.0001345 | 0.9738 | 304.4 | 263 | 494.2 | 0.1016 |
