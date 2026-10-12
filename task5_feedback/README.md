# Task 5 - RLVR vs RLAIF

Run from the repo root. Generations and judge decisions are cached, so reruns only do new work.

```bash
python -m task5_feedback.evaluate_math --dataset gsm        # SFT / RLVR / RLAIF on the GSM8K eval subset
python -m task5_feedback.score_perturbations                # 100-response controlled diagnostic set
python -m task5_feedback.evaluate_math --dataset transfer   # same three policies on the fixed SVAMP subset
python -m task5_feedback.compare_feedback                   # tables, figures, example candidates
python -m task5_feedback.check_judge_outputs                # raw judge text: real TIEs vs unreadable output
```

**Policies.** Base Qwen2.5-1.5B-Instruct (SFT) and the supplied frozen `rlvr_policy` and
`rlaif_policy` adapters. Nothing is retrained.

**Evaluation.** One greedy response per policy and problem (`math_max_new_tokens`), using each
dataset's own prompt messages. Exact accuracy and format compliance come from the supplied verifier
(`extract_designated_final` / `exact_reward`, i.e. only the last `#### <number>` counts). Pairwise
result: the supplied `PairwiseAIJudge` compares each RL policy's response with the SFT response to
the same problem (its own deterministic A/B swap and cache), scored win 1 / tie 0.5 / loss 0;
identical responses are counted as ties without calling the judge and reported separately.
Verifier-judge agreement = same preference sign on the policy-vs-SFT pair. Failure types: correct,
wrong final, no final, no final because the response was cut off.

**Diagnostics.** For every problem the five variants are scored by the verifier and by the RLAIF
group reward used in training, (wins + 0.5 ties) / (K - 1) over the five variants. Each controlled
pair is clean_correct against one perturbation, so "better" always means clean_correct; better /
tie / wrong rates are reported for the verifier, the direct pairwise judge decision and the group
reward. S_reason uses the corrupted-reasoning pair and S_outcome the good-reasoning/wrong-final pair (a
version pooled with the gold-distractor pair is also given); the filler pair is reported on its own.

**Judge output check.** The supplied parser turns any output it can't read into TIE.
`check_judge_outputs` re-runs the judge's exact prompt and A/B orientation on every comparison
(diagnostic pairs and policy-vs-SFT pairs), keeps the raw text, and reports how many TIEs were
real TIE answers vs unreadable output, and how often a decisive answer picked the first-shown
response. The judge and its cache are not modified.
