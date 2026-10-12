GSM8K eval: 300 problems; SVAMP transfer: 100 problems. Greedy decoding, max 512 new tokens. Win rate = AI judge vs the SFT response on the same problem (win 1, tie 0.5, loss 0); identical responses count as ties. Agreement = verifier and judge give the same preference sign for the policy-vs-SFT pair. Judge cost: about 0.26 s per new comparison on the GPU. Columns ending _nonidentical drop the pairs where the policy and SFT wrote the same text.

| policy | gsm_accuracy | gsm_format | gsm_mean_tokens | gsm_std_tokens | gsm_truncation | gsm_win_vs_sft | gsm_judge_ties | gsm_identical_to_sft | gsm_verifier_judge_agree | svamp_accuracy | svamp_format | svamp_mean_tokens | svamp_std_tokens | svamp_truncation | svamp_win_vs_sft | svamp_judge_ties | svamp_identical_to_sft | svamp_verifier_judge_agree | accuracy_drop | win_rate_change | gsm_win_vs_sft_nonidentical | gsm_agree_nonidentical | svamp_win_vs_sft_nonidentical | svamp_agree_nonidentical |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SFT | 0.280 | 0.350 | 275.860 | 88.049 | 0.020 |  |  |  |  | 0.440 | 0.570 | 164.690 | 73.015 | 0.000 |  |  |  |  | -0.160 |  |  |  |  |  |
| RLVR | 0.297 | 0.370 | 276.117 | 88.244 | 0.023 | 0.500 | 56.000 | 238.000 | 0.950 | 0.460 | 0.590 | 164.040 | 72.823 | 0.000 | 0.505 | 7.000 | 90.000 | 0.950 | -0.163 | 0.005 | 0.500 | 0.758 | 0.550 | 0.500 |
| RLAIF | 0.280 | 0.360 | 274.303 | 85.825 | 0.017 | 0.490 | 54.000 | 236.000 | 0.940 | 0.420 | 0.550 | 163.550 | 70.667 | 0.000 | 0.505 | 10.000 | 87.000 | 0.950 | -0.140 | 0.015 | 0.453 | 0.719 | 0.538 | 0.615 |

Failure types (counts):

| dataset | policy | no_final | correct | wrong_final | no_final_truncated |
|---|---|---|---|---|---|
| gsm | SFT | 189 | 84 | 21 | 6 |
| gsm | RLVR | 182 | 89 | 22 | 7 |
| gsm | RLAIF | 187 | 84 | 24 | 5 |
| svamp | SFT | 43 | 44 | 13 | 0 |
| svamp | RLVR | 41 | 46 | 13 | 0 |
| svamp | RLAIF | 45 | 42 | 13 | 0 |

Verifier vs judge on non-identical policy-vs-SFT pairs (counts):

| dataset | policy | verifier | n | judge_prefers_policy | judge_tie | judge_prefers_sft |
|---|---|---|---|---|---|---|
| gsm | RLVR | policy correct, SFT wrong | 9 | 1 | 6 | 2 |
| gsm | RLVR | same correctness | 49 | 2 | 46 | 1 |
| gsm | RLVR | SFT correct, policy wrong | 4 | 0 | 4 | 0 |
| gsm | RLAIF | policy correct, SFT wrong | 5 | 0 | 3 | 2 |
| gsm | RLAIF | same correctness | 54 | 2 | 46 | 6 |
| gsm | RLAIF | SFT correct, policy wrong | 5 | 0 | 5 | 0 |
| svamp | RLVR | policy correct, SFT wrong | 2 | 0 | 2 | 0 |
| svamp | RLVR | same correctness | 8 | 2 | 5 | 1 |
| svamp | RLVR | SFT correct, policy wrong | 0 | 0 | 0 | 0 |
| svamp | RLAIF | policy correct, SFT wrong | 1 | 0 | 1 | 0 |
| svamp | RLAIF | same correctness | 9 | 2 | 7 | 0 |
| svamp | RLAIF | SFT correct, policy wrong | 3 | 0 | 2 | 1 |
