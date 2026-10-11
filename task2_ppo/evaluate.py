from __future__ import annotations

import argparse

from common.data import load_yaml, repo_path
from common.rl import evaluate_heldout


def evaluate_ppo(config_path: str, adapter: str, name: str, num_prompts: int = 128, force: bool = False):
    cfg = load_yaml(config_path)
    out_dir = repo_path(cfg["results_dir"]) / name
    if (out_dir / "eval_summary.json").exists() and not force:
        print(f"[{name}] already evaluated - skipping")
        return None
    print(f"[{name}] held-out evaluation of {adapter}", flush=True)
    return evaluate_heldout(cfg, adapter, out_dir, num_prompts=num_prompts,
                            max_new_tokens=int(cfg["eval_max_response_length"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    evaluate_ppo(args.config, args.adapter, args.name, args.num_prompts, args.force)


if __name__ == "__main__":
    main()
