Generated length and explicit word-limit compliance on the common word-limit prompt set (10 prompts x 4 samples, course parse_word_limit / word_limit_compliance), plus generated length on the held-out prompts. Same decoding for every model.

| model | word_limit_compliance | wl_words_mean | wl_words_std | wl_tokens_mean | wl_tokens_std | wl_mean_words_over_limit | heldout_tokens_mean | heldout_tokens_std | heldout_tokens_iqr |
|---|---|---|---|---|---|---|---|---|---|
| sft | 0.575 | 40.6 | 19.55 | 47.98 | 22.59 | 8.525 | 156.3 | 103.6 | 210 |
| standard | 0.55 | 39.62 | 18.31 | 47.1 | 21.7 | 7.675 | 157.3 | 103.9 | 210 |
| length_balanced | 0.575 | 39.1 | 18.8 | 46.35 | 22.57 | 7.25 | 156.2 | 105.3 | 216 |
