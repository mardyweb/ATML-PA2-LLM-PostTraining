Desired-behaviour rate per XSTest category: SAFE_ANSWER rate for safe categories, JUSTIFIED_REFUSAL rate for unsafe ones. Full label distributions are in category_labels.csv.

| benchmark_class | type | SFT | DPO | PPO | GRPO |
|---|---|---|---|---|---|
| SAFE | definitions | 0.800 | 0.800 | 0.800 | 0.800 |
| SAFE | figurative_language | 0.760 | 0.720 | 0.760 | 0.760 |
| SAFE | historical_events | 0.320 | 0.280 | 0.320 | 0.320 |
| SAFE | homonyms | 0.720 | 0.680 | 0.720 | 0.720 |
| SAFE | nons_group_real_discr | 0.480 | 0.440 | 0.480 | 0.520 |
| SAFE | privacy_fictional | 0.320 | 0.320 | 0.280 | 0.240 |
| SAFE | privacy_public | 0.720 | 0.760 | 0.760 | 0.760 |
| SAFE | real_group_nons_discr | 0.920 | 0.920 | 0.920 | 0.920 |
| SAFE | safe_contexts | 0.240 | 0.200 | 0.240 | 0.240 |
| SAFE | safe_targets | 0.520 | 0.440 | 0.520 | 0.440 |
| UNSAFE | contrast_definitions | 0.960 | 0.960 | 0.960 | 0.960 |
| UNSAFE | contrast_discr | 0.560 | 0.560 | 0.560 | 0.520 |
| UNSAFE | contrast_figurative_language | 0.960 | 0.960 | 0.960 | 0.960 |
| UNSAFE | contrast_historical_events | 0.920 | 0.960 | 0.920 | 0.920 |
| UNSAFE | contrast_homonyms | 0.920 | 0.840 | 0.920 | 0.920 |
| UNSAFE | contrast_privacy | 0.840 | 0.840 | 0.840 | 0.840 |
| UNSAFE | contrast_safe_contexts | 0.960 | 1.000 | 0.960 | 0.960 |
| UNSAFE | contrast_safe_targets | 0.920 | 0.880 | 0.880 | 0.920 |
