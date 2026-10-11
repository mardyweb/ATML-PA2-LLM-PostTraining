# Task 1 - DPO

Run from the repo root. Each step skips work that's already done, and an interrupted
training run picks up from its last checkpoint.

```bash
python -m task1_dpo.check_dpo_loss                      # sanity checks on the loss
python -m task1_dpo.train --run-name standard            # 1 epoch on the standard set
python -m task1_dpo.evaluate --adapter none --name sft   # base model baseline
python -m task1_dpo.evaluate --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta                          # beta = 0.03 / 0.10 / 0.30 short runs
python -m task1_dpo.analyze_length                       # length-balanced run + dataset length stats
python -m task1_dpo.summarize                            # tables, figures, example candidates
```

**Loss fix.** `dpo.py` originally used `beta * (policy_margin + ref_margin)`; it should be
`beta * (policy_margin - ref_margin)`. `check_dpo_loss.py` shows the fixed version gives log 2
when policy = reference and matches the formula, and that the original version doesn't.

**Evaluation settings (same for every model).** Held-out pairs are scored with summed
response log-probs under the policy and the reference (same model, LoRA disabled).
Generations use the first 300 unique held-out prompts, seed 6304, temperature 0.7,
top-p 0.9, max 256 new tokens, batch size 32. Reward comes from the course reward model;
KL is log pi - log ref on the sampled tokens (per-token mean, plus per-sequence sum).
Word-limit compliance uses the 10 shared prompts with 4 samples each.

**Outputs.** Per run in `results/task1_dpo/<run>/` (training log, data ids, summary, per-example
files). Tables, figures and `qualitative_candidates.md` in `results/task1_dpo/`.
Adapters go to `outputs/task1_dpo/` (not committed).
