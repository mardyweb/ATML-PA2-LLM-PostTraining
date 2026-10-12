from __future__ import annotations

import argparse

import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate, score_reward_pairs
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    df = load_xstest(cfg)
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model,
            tokenizer,
            prompts,
            max_prompt_length=256,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )
        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })
    return records


def out_dir(cfg):
    return repo_path(cfg["results_dir"]) / "task4_safety"


@torch.no_grad()
def add_reward_scores(cfg, names, batch_size=16):
    """Score every saved response with the course reward model (used to relate reward to safety)."""
    rm, rm_tok = load_reward_model(cfg)
    for name in names:
        path = out_dir(cfg) / f"generated_{name}.jsonl"
        rows = read_jsonl(path)
        if rows and all("reward" in r for r in rows):
            continue
        for i in range(0, len(rows), batch_size):
            chunk = rows[i:i + batch_size]
            scores = score_reward_pairs(rm, rm_tok, [[{"role": "user", "content": r["prompt"]}] for r in chunk],
                                        [r["response"] for r in chunk]).tolist()
            for r, sc in zip(chunk, scores):
                r["reward"] = float(sc)
        write_jsonl(path, rows)
        print(f"  reward scores added for {name}")
    clear_gpu(rm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="*", default=["sft", "dpo", "ppo", "grpo"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Policies:", list(policy_specs(cfg)))
    print("XSTest rows:", len(load_xstest(cfg)))
    out_dir(cfg).mkdir(parents=True, exist_ok=True)

    done = []
    for name in args.policies:
        path = out_dir(cfg) / f"generated_{name}.jsonl"
        adapter = policy_specs(cfg)[name]
        if adapter is not None and not (repo_path(adapter) / "adapter_config.json").exists():
            print(f"[{name}] adapter {adapter} not found yet - skipping (finish that task first)")
            continue
        if path.exists() and not args.force:
            print(f"[{name}] responses already saved - skipping")
            done.append(name)
            continue
        print(f"[{name}] generating {len(load_xstest(cfg))} greedy responses (max {cfg['safety_max_new_tokens']} new tokens)", flush=True)
        records = generate_for_policy(cfg, name, batch_size=args.batch_size)
        write_jsonl(path, records)  # same row order as the XSTest csv
        clear_gpu()
        print(f"[{name}] saved {path}", flush=True)
        done.append(name)
    if done:
        add_reward_scores(cfg, done)


if __name__ == "__main__":
    main()
