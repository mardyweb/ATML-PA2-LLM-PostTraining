"""Length study: measure the length bias in each preference file, train on the length-balanced
set with the standard settings, and evaluate it the same way as the standard model.
"""
from __future__ import annotations

import argparse

import numpy as np

from common.data import load_yaml, preference_responses, read_jsonl, repo_path
from common.logging_utils import save_json
from common.models import clear_gpu, load_tokenizer
from task1_dpo.evaluate import STRATUM_KEYS, evaluate_model, stratum_of
from task1_dpo.train import adapter_done, run_training


def dataset_length_profile(rows, tokenizer):
    lc, lr, strata = [], [], {}
    for row in rows:
        yc, yr = preference_responses(row)
        a = len(tokenizer(yc, add_special_tokens=False)["input_ids"])
        b = len(tokenizer(yr, add_special_tokens=False)["input_ids"])
        lc.append(a)
        lr.append(b)
        if any(k in row for k in STRATUM_KEYS):
            s, _ = stratum_of(row, tokenizer)
            strata[s] = strata.get(s, 0) + 1
    lc, lr = np.array(lc, float), np.array(lr, float)
    d = lc - lr
    return {
        "num_pairs": int(len(lc)),
        "chosen_tokens_mean": float(lc.mean()),
        "rejected_tokens_mean": float(lr.mean()),
        "chosen_tokens_median": float(np.median(lc)),
        "rejected_tokens_median": float(np.median(lr)),
        "mean_length_diff_chosen_minus_rejected": float(d.mean()),
        "frac_chosen_longer": float((d > 0).mean()),
        "frac_rejected_longer": float((d < 0).mean()),
        "frac_equal_length": float((d == 0).mean()),
        "median_length_ratio_chosen_over_rejected": float(np.median(lc / np.maximum(lr, 1))),
        "strata_counts": strata,
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
    profiles = {
        "standard_train": dataset_length_profile(read_jsonl(cfg["paths"]["dpo_standard_train"]), tok),
        "standard_eval": dataset_length_profile(read_jsonl(cfg["paths"]["dpo_standard_eval"]), tok),
        "length_balanced_train": dataset_length_profile(balanced, tok),
        "length_stratified_eval": dataset_length_profile(stratified, tok),
    }
    save_json(results / "dataset_length_profiles.json", profiles)
    for k, v in profiles.items():
        print(f"  {k:24s} pairs={v['num_pairs']:5d} chosen-longer={v['frac_chosen_longer']:.2f} "
              f"mean diff={v['mean_length_diff_chosen_minus_rejected']:+.1f} tok  strata={v['strata_counts']}")

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
