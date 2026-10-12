20 problems x 5 variants. Every controlled pair is clean_correct (the better response) against one perturbation; better / tie / wrong = how often each mechanism prefers clean_correct, ties, or prefers the perturbed response. S_reason uses the corrupted-reasoning pair; S_outcome uses the good-reasoning/wrong-final pair (reasoning kept, designated final changed), and S_outcome_incl_distractor also counts the gold-distractor pair. RLAIF (direct pairwise) is the supplied judge's decision on each controlled pair; RLAIF (group) ranks by the training-time group win rate over the 5 variants (K = 5).

| mechanism | S_reason | reason_tie | reason_wrong | S_outcome | outcome_tie | outcome_wrong | S_outcome_incl_distractor |
|---|---|---|---|---|---|---|---|
| RLVR (exact verifier) | 0.000 | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| RLAIF (direct pairwise) | 0.000 | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.125 |
| RLAIF (group win-rate reward) | 0.100 | 0.750 | 0.150 | 0.300 | 0.550 | 0.150 | 0.350 |

All five variants (clean_correct is the reference in every pair):

| variant | mean_rlvr_reward | mean_rlaif_group_reward | mean_words | rlvr_better/tie/wrong | rlaif_pairwise_better/tie/wrong |
|---|---|---|---|---|---|
| clean_correct | 1.000 | 0.494 | 49.700 | reference (the better response) | reference (the better response) |
| corrupt_reasoning_correct_final | 1.000 | 0.500 | 49.700 | 0.00 / 1.00 / 0.00 | 0.00 / 1.00 / 0.00 |
| good_reasoning_wrong_final | 0.000 | 0.475 | 49.700 | 1.00 / 0.00 / 0.00 | 0.00 / 1.00 / 0.00 |
| persuasive_filler_correct | 1.000 | 0.588 | 59.700 | 0.00 / 1.00 / 0.00 | 0.00 / 0.75 / 0.25 |
| gold_distractor_wrong_final | 0.000 | 0.444 | 17.000 | 1.00 / 0.00 / 0.00 | 0.25 / 0.70 / 0.05 |

Per perturbation, long form:

| perturbation | mechanism | better | tie | wrong | n |
|---|---|---|---|---|---|
| corrupt_reasoning_correct_final | RLVR (exact verifier) | 0.000 | 1.000 | 0.000 | 20 |
| corrupt_reasoning_correct_final | RLAIF (direct pairwise) | 0.000 | 1.000 | 0.000 | 20 |
| corrupt_reasoning_correct_final | RLAIF (group win-rate reward) | 0.100 | 0.750 | 0.150 | 20 |
| good_reasoning_wrong_final | RLVR (exact verifier) | 1.000 | 0.000 | 0.000 | 20 |
| good_reasoning_wrong_final | RLAIF (direct pairwise) | 0.000 | 1.000 | 0.000 | 20 |
| good_reasoning_wrong_final | RLAIF (group win-rate reward) | 0.300 | 0.550 | 0.150 | 20 |
| persuasive_filler_correct | RLVR (exact verifier) | 0.000 | 1.000 | 0.000 | 20 |
| persuasive_filler_correct | RLAIF (direct pairwise) | 0.000 | 0.750 | 0.250 | 20 |
| persuasive_filler_correct | RLAIF (group win-rate reward) | 0.100 | 0.500 | 0.400 | 20 |
| gold_distractor_wrong_final | RLVR (exact verifier) | 1.000 | 0.000 | 0.000 | 20 |
| gold_distractor_wrong_final | RLAIF (direct pairwise) | 0.250 | 0.700 | 0.050 | 20 |
| gold_distractor_wrong_final | RLAIF (group win-rate reward) | 0.400 | 0.500 | 0.100 | 20 |
