# Task 3 - GRPO

Run from the repo root. Finished runs/evaluations are skipped when a command is rerun.

```bash
python -m task3_grpo.check_grpo                                  # sanity checks on advantages + loss
python -m task3_grpo.analyze_group_size                          # K = 2/4/8 on the supplied cache (no training)
python -m task3_grpo.continue_train --run-name standard           # 20 updates from the supplied midpoint
python -m task3_grpo.evaluate --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.evaluate --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.compare_normalization                       # GRPO vs Dr. GRPO forks
python -m task3_grpo.summarize                                   # tables, figures, example candidates
```

**Advantage fix.** `group_relative_advantages` normalised every reward with the mean/std of the
whole batch. It now uses each prompt's own group: A_k = (r_k - mean_group) / (std_group + eps).
`check_grpo.py` checks zero mean / unit std inside each group, invariance to shifting one prompt's
rewards, zero advantage for a constant-reward group, and the two loss normalizations.

**Continuation loop.** Each update samples K = 4 completions for one training prompt (same prompt
order as the PPO task), scores them with the reward model, computes group-relative advantages,
masks completions that hit `max_completion_length`, and takes `policy_epochs` clipped steps with the
k3 KL penalty to the reference. Dropout is off so the single epoch starts at ratio = 1 (which also
means the clip fraction is 0 with one epoch). A group is uninformative when its reward std is
below 1e-6, the same eps as the advantage function. For the normalization study every completion's
share of the policy loss is back-propagated on its own to get a per-completion gradient norm.

**Group-size study.** Each prompt's first 8 cached completions (generation_index order) are cut into
8/K disjoint groups, so every K uses the same completions. Difficulty bins are equal-count terciles
of a prompt's mean cached reward (hard = lowest).

**Normalization study.** Two 8-update forks from the same midpoint that differ only in
`loss_type` (`grpo` divides each completion by its own length, `dr_grpo` by the constant
`max_completion_length`). Advantage normalization is unchanged in both, so only the length
normalization differs.

**Held-out evaluation** is the shared protocol in `common/rl.py` (first 128 eval prompts that fit in 256 prompt tokens, seed 6304,
temperature 0.7, top-p 0.9), with a 768-token cap to match the PPO evaluation.

**Notes.** Entropy is the full next-token distribution entropy (see the Task 2 README); in the
training log it is averaged per completion then across the K completions, while held-out entropy is
pooled over tokens. Prompts longer than 256 tokens are skipped, as in Task 2. The standard run also
computes one extra backward pass per completion for the gradient-norm diagnostic, so its wall-clock
time includes that overhead. Dropout is off during updates, as in Task 2.
