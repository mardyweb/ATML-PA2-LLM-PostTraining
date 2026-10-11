"""Short DPO runs for each beta in the config, all from the same base model and data subset.

Runs that already have a saved adapter / summary are skipped, so it's safe to rerun after a disconnect.
"""
from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate_model
from task1_dpo.train import adapter_done, run_training


def fork_name(beta: float) -> str:
    return f"beta_{beta:g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--betas", type=float, nargs="*", help="override the beta list from the config")
    ap.add_argument("--num-gen-prompts", type=int, default=300)
    ap.add_argument("--force", action="store_true", help="retrain/re-evaluate even if outputs exist")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    betas = args.betas or [float(b) for b in cfg["betas"]]
    n = int(cfg["short_ablation_examples"])
    print("Required beta values:", betas)
    print("Short-run examples per condition:", n)

    for beta in betas:
        name = fork_name(beta)
        out = repo_path(f"outputs/task1_dpo/{name}")
        if args.force or not adapter_done(out):
            run_training(args.config, name, output_path=str(out), beta=beta, max_examples=n)
            clear_gpu()
        else:
            print(f"[{name}] adapter exists, skipping training")
        summary = repo_path(cfg["results_dir"]) / name / "summary.json"
        if args.force or not summary.exists():
            evaluate_model(args.config, str(out), name, beta=beta, num_gen_prompts=args.num_gen_prompts)
            clear_gpu()
        else:
            print(f"[{name}] evaluation exists, skipping")

    print("\nBeta forks done. Build tables/figures with: python -m task1_dpo.summarize")


if __name__ == "__main__":
    main()
