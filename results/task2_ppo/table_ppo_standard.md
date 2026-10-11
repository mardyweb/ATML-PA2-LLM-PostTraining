Standard continuation: 20 updates x 1 prompt, 6455 generated tokens, wall-clock 300 s, peak VRAM 8.15 GB on NVIDIA A100-SXM4-40GB. Mean critic explained variance -13.052.

| condition | heldout_reward | reward_sem | kl_per_token | kl_per_seq | entropy | len_mean | len_std | truncation_rate |
|---|---|---|---|---|---|---|---|---|
| PPO midpoint (start) | 1.757 | 0.121 | -1.064e-05 | -0.003075 | 0.9709 | 288.9 | 259.2 | 0.05469 |
| PPO standard (+20 updates) | 1.735 | 0.1219 | -6.282e-06 | -0.001862 | 0.9983 | 296.5 | 263.7 | 0.07812 |
