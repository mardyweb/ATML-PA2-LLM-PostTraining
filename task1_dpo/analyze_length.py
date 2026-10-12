"""Length study: measure the length bias in each preference file, train on the length-balanced
set with the standard settings, and evaluate it the same way as the standard model.
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, preference_responses, prompt_messages_from_preference, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import clear_gpu, load_tokenizer
from task1_dpo.evaluate import STRATUM_KEYS, evaluate_model, stratum_of
from task1_dpo.train import adapter_done, filter_encodable, row_id, run_training


def dataset_length_profile(rows, tokenizer, max_len):
    """Length structure of a preference file, on the pairs that are actually used (over-long prompts dropped)."""
    kept_idx, dropped_idx = filter_encodable(rows, tokenizer, max_len)
    lc, lr, strata, trunc = [], [], {}, {}
    for i in kept_idx:
        row = rows[i]
        yc, yr = preference_responses(row)
        a = len(tokenizer(yc, add_special_tokens=False)["input_ids"])
        b = len(tokenizer(yr, add_special_tokens=False)["input_ids"])
        n_prompt = len(tokenizer.apply_chat_template(prompt_messages_from_preference(row), tokenize=True,
                                                     add_generation_prompt=True))
        lc.append(a)
        lr.append(b)
        s = stratum_of(row, tokenizer)[0] if any(k in row for k in STRATUM_KEYS) else "all"
        strata[s] = strata.get(s, 0) + 1
        # +1 for the EOS that encode_prompt_response appends
        t = trunc.setdefault(s, [0, 0])
        t[0] += int(n_prompt + a + 1 > max_len)
        t[1] += int(n_prompt + b + 1 > max_len)
    dropped_strata = {}
    for i in dropped_idx:
        s = stratum_of(rows[i], tokenizer)[0] if any(k in rows[i] for k in STRATUM_KEYS) else "all"
        dropped_strata[s] = dropped_strata.get(s, 0) + 1
    lc, lr = np.array(lc, float), np.array(lr, float)
    d = lc - lr
    return {
        "rows_in_file": len(rows),
        "num_pairs_used": int(len(lc)),
        "num_dropped_overlong_prompt": len(dropped_idx),
        "dropped_ids": [row_id(rows[i], i) for i in dropped_idx],
        "chosen_tokens_mean": float(lc.mean()),
        "rejected_tokens_mean": float(lr.mean()),
        "chosen_tokens_median": float(np.median(lc)),
        "rejected_tokens_median": float(np.median(lr)),
        "mean_length_diff_chosen_minus_rejected": float(d.mean()),
        "frac_chosen_longer": float((d > 0).mean()),
        "frac_rejected_longer": float((d < 0).mean()),
        "frac_equal_length": float((d == 0).mean()),
        "median_length_ratio_chosen_over_rejected": float(np.median(lc / np.maximum(lr, 1))),
        "strata_counts_used": strata,
        "strata_counts_dropped": dropped_strata,
        "truncated_at_max_len": {s: {"chosen": v[0] / strata[s], "rejected": v[1] / strata[s]} for s, v in trunc.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--num-gen-prompts", type=int, default=300)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    results = repo_path(cfg["results_dir"])

    balanced = read_jsonl(cfg["paths"]["dpo_length_train"])
    stratified = read_jsonl(cfg["paths"]["dpo_length_eval"])
    print("Length-balanced train rows:", len(balanced))
    print("Length-stratified eval rows:", len(stratified))

    # length stats of the datasets themselves
    tok = load_tokenizer(cfg["base_model"])
    max_len = int(cfg["max_sequence_length"])
    std_train = read_jsonl(cfg["paths"]["dpo_standard_train"])
    profiles = {
        "standard_train": dataset_length_profile(std_train, tok, max_len),
        "beta_fork_subset": dataset_length_profile(std_train[: int(cfg["short_ablation_examples"])], tok, max_len),
        "standard_eval": dataset_length_profile(read_jsonl(cfg["paths"]["dpo_standard_eval"]), tok, max_len),
        "length_balanced_train": dataset_length_profile(balanced, tok, max_len),
        "length_stratified_eval": dataset_length_profile(stratified, tok, max_len),
    }
    save_json(results / "dataset_length_profiles.json", profiles)
    for k, v in profiles.items():
        print(f"  {k:24s} used={v['num_pairs_used']:5d}/{v['rows_in_file']:5d} chosen-longer={v['frac_chosen_longer']:.2f} "
              f"mean diff={v['mean_length_diff_chosen_minus_rejected']:+.1f} tok  strata={v['strata_counts_used']}")

    # same settings as the standard run, just a different training file
    out = repo_path(cfg["length_output"])
    if args.force or not adapter_done(out):
        run_training(args.config, "length_balanced", dataset_path=cfg["paths"]["dpo_length_train"],
                     output_path=str(out))
        clear_gpu()
    else:
        print("[length_balanced] adapter exists, skipping training")

    # make sure both models are evaluated
    if args.force or not (results / "length_balanced" / "summary.json").exists():
        evaluate_model(args.config, str(out), "length_balanced", num_gen_prompts=args.num_gen_prompts)
        clear_gpu()
    std_adapter = repo_path(cfg["standard_output"])
    if not (results / "standard" / "summary.json").exists():
        if not adapter_done(std_adapter):
            raise SystemExit("Standard DPO adapter not found - run task1_dpo.train first.")
        evaluate_model(args.config, str(std_adapter), "standard", num_gen_prompts=args.num_gen_prompts)
        clear_gpu()

    print("\nLength study done. Build tables/figures with: python -m task1_dpo.summarize")


if __name__ == "__main__":
    main()
