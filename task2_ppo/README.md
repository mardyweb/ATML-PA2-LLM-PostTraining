# Task 2 - PPO

Run from the repo root. Finished runs/evaluations are skipped when a command is rerun.

```bash
python -m task2_ppo.check_ppo                                   # sanity checks on the objective + GAE
python -m task2_ppo.continue_train --run-name standard           # 20 updates from the supplied midpoint
python -m task2_ppo.evaluate --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping                             # cached-batch study + eps forks
python -m task2_ppo.ablate_kl                                    # kl_beta forks
python -m task2_ppo.summarize                                    # tables, figures, example candidates
```

**Objective fix.** `ppo_policy_loss` used `torch.maximum` of the two surrogates; PPO takes the
minimum. `check_ppo.py` checks the capped cases (A > 0 with a large ratio, A < 0 with a small
ratio), that the unclipped side keeps its gradient, agreement with the formula, and GAE against a
brute-force implementation. The starter version fails the clipping checks.

**Continuation loop.** Each update samples one response per prompt (prompts taken in file order
from the training pool, skipping any longer than `max_prompt_length`), scores it with the reward
model (minus `missing_eos_penalty` if it never emits EOS), adds the per-token KL penalty, computes
GAE with the midpoint critic, normalises advantages and runs `ppo_epochs` passes over the rollout
updating policy LoRA and critic (LoRA + head) together. Dropout is off so the first epoch starts
at ratio = 1. Clip/affected fractions are reported for the last epoch, and `post_update_approx_kl`
(k3 estimate of KL(old || new) after the epochs) is used as the per-update stability statistic.
Every fork starts from the same midpoint policy/critic, with the same prompts and seed.

**Cached clipping study.** The cached batch is rebuilt on the policy tokenizer, advantages come
from its stored old/reference log-probs, its reward (reward model if missing) and the midpoint
critic. For each epsilon it reports the clip fraction (ratio outside the band), affected-token
fraction (tokens where the clipped term is selected, so the gradient is zero) and the surrogate gap
at the midpoint and after each PPO step on that batch.

**Held-out evaluation** (`common/rl.py`, shared with Task 3): first 128 eval prompts, seed 6304,
temperature 0.7, top-p 0.9, up to `eval_max_response_length` tokens; reward model score, KL to the
reference on the sampled tokens, full-distribution entropy, length and truncation rate.
