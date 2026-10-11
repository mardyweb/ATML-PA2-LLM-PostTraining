"""Canonical GRPO (1/T_k per completion) vs Dr. GRPO-style (constant 1/max_len) normalization.
Two short forks from the same GRPO midpoint; only loss_type differs (same prompts, seed, K,
reward, beta, eps, generation cap). Then the shared held-out evaluation for both."""
from __future__ import annotations

import argparse

from common.data import load_yaml
from task3_grpo.continue_train import adapter_done, run_grpo
from task3_grpo.evaluate import evaluate_grpo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    for loss_type in ("grpo", "dr_grpo"):
        name = f"fork_{loss_type}"
        out = f"outputs/task3_grpo/{name}"
        if args.force or not adapter_done(out):
            run_grpo(args.config, output=out, updates=int(cfg["fork_updates"]), loss_type=loss_type, run_name=name)
        else:
            print(f"[{name}] already trained - skipping")
        evaluate_grpo(args.config, out, name, num_prompts=args.num_prompts, force=args.force)
    print("\nNormalization study done. Tables/figures: python -m task3_grpo.summarize")


if __name__ == "__main__":
    main()
