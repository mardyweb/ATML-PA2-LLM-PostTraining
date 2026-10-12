Manual audit: the fixed 60 XSTest audit prompts x 4 policies = 240 prompt-response pairs, labelled in 240 sheet rows (118 distinct response texts; identical texts labelled separately got the same manual label 100% of the time). Agreement 0.700, Cohen's kappa 0.482; AI ambiguous rate 0.000, manual ambiguous rate 0.017. All 240 labelled pairs are in audit_pairs.csv.

Confusion over all pairs (rows = manual, columns = AI judge):

|  | SAFE_ANSWER | JUSTIFIED_REFUSAL | UNSAFE_COMPLIANCE | OVER_REFUSAL | AMBIGUOUS |
|---|---|---|---|---|---|
| manual SAFE_ANSWER | 61 | 14 | 1 | 0 | 0 |
| manual JUSTIFIED_REFUSAL | 5 | 107 | 0 | 0 | 0 |
| manual UNSAFE_COMPLIANCE | 0 | 4 | 0 | 0 | 0 |
| manual OVER_REFUSAL | 0 | 44 | 0 | 0 | 0 |
| manual AMBIGUOUS | 4 | 0 | 0 | 0 | 0 |

Policy rates on the audited prompts with AI vs manual labels:

| policy | n | safe_answer_rate_ai | safe_over_refusal_rate_ai | unsafe_compliance_rate_ai | unsafe_justified_refusal_rate_ai | safe_answer_rate_manual | safe_over_refusal_rate_manual | unsafe_compliance_rate_manual | unsafe_justified_refusal_rate_manual |
|---|---|---|---|---|---|---|---|---|---|
| SFT | 60 | 0.533 | 0.000 | 0.000 | 0.933 | 0.633 | 0.367 | 0.033 | 0.933 |
| DPO | 60 | 0.500 | 0.000 | 0.000 | 0.900 | 0.633 | 0.367 | 0.033 | 0.933 |
| PPO | 60 | 0.500 | 0.000 | 0.000 | 0.933 | 0.633 | 0.367 | 0.033 | 0.933 |
| GRPO | 60 | 0.500 | 0.000 | 0.000 | 0.933 | 0.633 | 0.367 | 0.033 | 0.933 |
