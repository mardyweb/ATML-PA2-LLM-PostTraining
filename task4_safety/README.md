# Task 4 - Safety calibration

Run from the repo root, after the standard Task 1-3 adapters exist.

```bash
python -m task4_safety.generate_responses     # SFT/DPO/PPO/GRPO on all XSTest prompts (+ reward-model score)
python -m task4_safety.judge_responses        # fixed AI judge, cached per response (resumable)
python -m task4_safety.make_audit_sheet       # blind sheet: 60 fixed prompts x 4 policies = 240 rows
#   ... fill in manual_label, save as results/task4_safety/manual_audit_labels.xlsx (or .csv) ...
python -m task4_safety.evaluate_safety        # rates, categories, length, agreement, figures
```

**Policies.** Exactly the four fixed ones: base Qwen2.5-1.5B-Instruct (SFT), and the standard
adapters from Task 1 step 1, Task 2 step 1 and Task 3 step 1 (`configs/feedback.yaml`). No ablation
checkpoints are used.

**Generation.** One greedy response per policy and prompt with the supplied `generate_for_policy`
(no sampling, `safety_max_new_tokens`), in XSTest csv order, batch size 16 for every policy. Each
response is also scored with the course reward model, to look at reward vs safety label.

**Judging.** The supplied judge prompt, parser and `judge_one`, applied to every response one at a
time. Labels are appended to `judged_<policy>.jsonl` as they come, so a rerun continues where it
stopped. Parse failures stay as the parser's AMBIGUOUS and are counted separately.

**Manual audit.** `make_audit_sheet` keeps the supplied `fixed_audit_ids` (30 safe + 30 unsafe
prompts, seeded) and writes one row per (prompt, policy response) pair: 60 prompts x 4 policies = 240
rows, as `manual_audit_sheet.csv` and `.xlsx` (the Excel copy has a label dropdown). Prompts are in
random order and each prompt's four responses are shuffled; neither the policy nor the AI label is
shown (`manual_audit_key.csv` maps rows back to policies). Accepted labels: the five full names, or
SA / JR / UC / OR / AMB. The filled-in file goes to `results/task4_safety/manual_audit_labels.xlsx`
(or `.csv`); only its `item_id`, `manual_label` and `notes` columns are read. `--dedup` lists
identical responses to the same prompt once instead.

**Metrics.** Safe-answer and over-refusal rates over safe prompts; unsafe-compliance and
justified-refusal rates over unsafe prompts; ambiguous rate overall and by class; full label
distribution by class and by XSTest category; mean response length; reward-model score by label.
Audit (unit = prompt-response pair, 60 prompts x 4 policies = 240): agreement, Cohen's kappa, confusion matrix (manual x
AI), agreement by class and policy, ambiguous rates, each policy's rates on the audited prompts
computed with AI labels vs manual labels, and how consistently identical responses were labelled.
