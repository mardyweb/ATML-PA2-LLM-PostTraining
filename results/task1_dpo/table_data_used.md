Pairs whose prompt alone does not fit in max_sequence_length=768 tokens are dropped (the course encode_prompt_response cannot encode them). Dropped IDs are in dataset_length_profiles.json. Length statistics are on the pairs used.

| split | rows_in_file | pairs_used | dropped_overlong_prompt | strata_used | strata_dropped | frac_chosen_longer | mean_len_diff_chosen_minus_rejected |
|---|---|---|---|---|---|---|---|
| standard_train | 1500 | 1446 | 54 | all=1446 | all=54 | 0.5712 | 45.91 |
| beta_fork_subset | 600 | 581 | 19 | all=581 | all=19 | 0.5766 | 52.26 |
| standard_eval | 300 | 290 | 10 | all=290 | all=10 | 0.531 | 29.04 |
| length_balanced_train | 1500 | 1442 | 58 | rejected_longer=482 length_matched=482 preferred_longer=478 | preferred_longer=22 rejected_longer=18 length_matched=18 | 0.4903 | 4.728 |
| length_stratified_eval | 246 | 237 | 9 | rejected_longer=79 preferred_longer=78 length_matched=80 | preferred_longer=4 rejected_longer=3 length_matched=2 | 0.481 | 2.646 |

Fraction of used pairs whose response is cut at max_sequence_length:

- standard_train: all: chosen 0.12 / rejected 0.09
- beta_fork_subset: all: chosen 0.12 / rejected 0.08
- standard_eval: all: chosen 0.12 / rejected 0.11
- length_balanced_train: rejected_longer: chosen 0.01 / rejected 0.03, length_matched: chosen 0.02 / rejected 0.02, preferred_longer: chosen 0.03 / rejected 0.01
- length_stratified_eval: rejected_longer: chosen 0.00 / rejected 0.04, preferred_longer: chosen 0.05 / rejected 0.04, length_matched: chosen 0.00 / rejected 0.00
