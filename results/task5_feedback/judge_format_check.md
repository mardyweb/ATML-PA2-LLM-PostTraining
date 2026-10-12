Raw output of the fixed pairwise judge (same prompt, orientation and greedy decoding as PairwiseAIJudge.compare). The supplied parser maps unreadable output to TIE.

| source | comparisons | judge_said_A_or_B | judge_said_TIE | unparseable_counted_as_TIE | share_of_TIEs_that_are_parse_failures | exact_single_label_output | decisive_picks_first_shown | matches_cached_result |
|---|---|---|---|---|---|---|---|---|
| diagnostics | 200 | 0.140 | 0.860 | 0.000 | 0.000 | 0.965 | 0.250 | 1.000 |
| gsm_rlvr_vs_sft | 62 | 0.097 | 0.903 | 0.000 | 0.000 | 0.968 | 0.667 | 1.000 |
| gsm_rlaif_vs_sft | 64 | 0.156 | 0.844 | 0.000 | 0.000 | 0.953 | 0.600 | 1.000 |
| transfer_rlvr_vs_sft | 10 | 0.300 | 0.700 | 0.000 | 0.000 | 0.700 | 1.000 | 1.000 |
| transfer_rlaif_vs_sft | 13 | 0.231 | 0.769 | 0.000 | 0.000 | 0.846 | 1.000 | 1.000 |

Most common raw outputs, diagnostics: 'TIE' x172, 'B' x21, 'A, B' x7

Most common raw outputs, gsm_rlvr_vs_sft: 'TIE' x56, 'A' x2, 'B' x2, 'A, B' x2

Most common raw outputs, gsm_rlaif_vs_sft: 'TIE' x54, 'B' x4, 'A, B' x3, 'A' x3

Most common raw outputs, transfer_rlvr_vs_sft: 'TIE' x7, 'A, B,' x2, 'A, B' x1

Most common raw outputs, transfer_rlaif_vs_sft: 'TIE' x10, 'A, B,' x1, 'A, B' x1, 'A' x1
