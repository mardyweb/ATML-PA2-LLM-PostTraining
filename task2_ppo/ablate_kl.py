"""KL-pressure study: short forks from the PPO midpoint for each kl_beta in the config, same
epsilon / prompts / update budget, then the shared held-out evaluation. The eps=0.2, beta=0.1
fork is the same run as in the clipping study, so it's only trained once."""
from __future__ import annotations

import argparse

from common.data import load_yaml
from task2_ppo.analyze_clipping import fork_name
from task2_ppo.continue_train import adapter_done, run_ppo
from task2_ppo.evaluate import evaluate_ppo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])

    eps = float(cfg["clip_epsilon"])
    for beta in [float(b) for b in cfg["kl_values"]]:
        name = fork_name(eps, beta)
        out = f"outputs/task2_ppo/{name}"
        if args.force or not adapter_done(out):
            run_ppo(args.config, output=out, updates=int(cfg["fork_updates"]), clip_epsilon=eps,
                    kl_beta=beta, run_name=name)
        else:
            print(f"[{name}] already trained - skipping")
        evaluate_ppo(args.config, out, name, num_prompts=args.num_prompts, force=args.force)
    print("\nKL study done. Tables/figures: python -m task2_ppo.summarize")


if __name__ == "__main__":
    main()
