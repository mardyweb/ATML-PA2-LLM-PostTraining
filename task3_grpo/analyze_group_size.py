"""Group-size study on the supplied K=8 completion cache (no training).

Each prompt's first 8 cached completions (in generation_index order) are split into 8/K disjoint
groups of size K, so K = 2, 4 and 8 all use exactly the same completions (equal generation budget).
Prompt difficulty = mean cached reward over those 8 completions, split into equal-count terciles
(hard = lowest third of mean reward).
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages

STD_TOL = 1e-6
CACHE_PER_PROMPT = 8


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int):
    """Return K-sized groups while keeping total cached completions fixed.

    Every prompt contributes its first 8 completions, cut into consecutive blocks of K, so the
    total number of completions is (#prompts x 8) for every K.
    """
    if CACHE_PER_PROMPT % k:
        raise ValueError(f"K={k} must divide {CACHE_PER_PROMPT}")
    groups = []
    for pid, rows in by_prompt.items():
        rows = rows[:CACHE_PER_PROMPT]
        for start in range(0, CACHE_PER_PROMPT, k):
            groups.append({"prompt": pid, "block": start // k, "rows": rows[start:start + k]})
    return groups


def group_metrics(groups, reward_scale):
    rewards = torch.tensor([float(r["reward"]) for g in groups for r in g["rows"]])
    gid = torch.tensor([i for i, g in enumerate(groups) for _ in g["rows"]])
    adv = group_relative_advantages(rewards, gid)
    stds = np.array([np.std([float(r["reward"]) for r in g["rows"]]) for g in groups])
    centred = np.concatenate([[float(r["reward"]) - np.mean([float(x["reward"]) for x in g["rows"]]) for r in g["rows"]]
                              for g in groups])
    out = {
        "num_groups": len(groups),
        "num_completions": int(rewards.numel()),
        "informative_rate": float(np.mean(stds > STD_TOL)),
        "uninformative_fraction": float(np.mean(stds <= STD_TOL)),
        "mean_within_group_std": float(stds.mean()),
        "median_within_group_std": float(np.median(stds)),
        "var_normalized_advantage": float(adv.var(unbiased=False)),
        "var_centred_reward": float(np.var(centred)),
        "mean_abs_centred_reward": float(np.mean(np.abs(centred))),
        "frac_completions_nonzero_adv": float((adv.abs() > 1e-6).float().mean()),
    }
    # informative rate at stricter thresholds, as fractions of the overall reward std
    out["informative_rate_by_threshold"] = {f"{f:g}": float(np.mean(stds > f * reward_scale)) for f in (0.05, 0.1, 0.25, 0.5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    all_rewards = np.array([float(r["reward"]) for rows in by_prompt.values() for r in rows[:CACHE_PER_PROMPT]])
    scale = float(all_rewards.std())
    mean_r = {pid: float(np.mean([float(r["reward"]) for r in rows[:CACHE_PER_PROMPT]])) for pid, rows in by_prompt.items()}
    edges = np.quantile(list(mean_r.values()), [1 / 3, 2 / 3])
    names = ("hard", "medium", "easy")
    bin_of = {pid: names[int(np.searchsorted(edges, m, side="right"))] for pid, m in mean_r.items()}

    result = {
        "num_prompts": len(by_prompt), "completions_per_prompt": CACHE_PER_PROMPT,
        "total_completions": int(len(all_rewards)), "reward_mean": float(all_rewards.mean()), "reward_std": scale,
        "std_tolerance": STD_TOL,
        "binning_rule": "difficulty = mean cached reward of the prompt's 8 completions; equal-count terciles, "
                        "hard = lowest mean reward",
        "bin_edges_mean_reward": [float(e) for e in edges],
        "bin_counts": {n: sum(v == n for v in bin_of.values()) for n in names},
        "per_k": {}, "per_k_per_bin": {},
    }
    for k in [int(x) for x in cfg["group_sizes"]]:
        groups = regroup_equal_generation_budget(by_prompt, k)
        result["per_k"][str(k)] = group_metrics(groups, scale)
        result["per_k_per_bin"][str(k)] = {n: group_metrics([g for g in groups if bin_of[g["prompt"]] == n], scale)
                                          for n in names}
        m = result["per_k"][str(k)]
        print(f"  K={k}: groups={m['num_groups']} informative={m['informative_rate']:.3f} "
              f"mean std={m['mean_within_group_std']:.4f} var(centred)={m['var_centred_reward']:.4f}")

    # a few groups to look at for the qualitative section
    pool = read_jsonl(cfg["paths"]["rl_prompt_train"]) + read_jsonl(cfg["paths"]["rl_prompt_eval"])
    by_pid = {str(r["prompt_id"]): r for r in pool if "prompt_id" in r}
    by_source = {str(r["source_index"]): r for r in pool if "source_index" in r}
    spread = sorted(by_prompt, key=lambda pid: np.std([float(r["reward"]) for r in by_prompt[pid][:CACHE_PER_PROMPT]]))
    examples = []
    for tag, pid in (("lowest reward spread", spread[0]), ("highest reward spread", spread[-1])):
        rows = by_prompt[pid][:CACHE_PER_PROMPT]
        src = by_pid.get(str(rows[0].get("prompt_id"))) or by_source.get(pid)
        examples.append({
            "kind": tag, "source_index": pid, "difficulty_bin": bin_of[pid],
            "prompt": prompt_messages(src)[-1]["content"][:400] if src else None,
            "completions": [{"reward": float(r["reward"]), "clipped_at_max": r.get("clipped_at_max"),
                             "text": str(r.get("completion", ""))[:300]} for r in rows],
        })
    # completions that hit the generation cap, per difficulty bin
    clipped = [r for rows in by_prompt.values() for r in rows[:CACHE_PER_PROMPT] if r.get("clipped_at_max")]
    result["clipped_at_max_fraction"] = len(clipped) / max(len(all_rewards), 1)
    result["uninformative_k8_prompts"] = [
        {"source_index": pid, "rewards": [float(r["reward"]) for r in rows[:CACHE_PER_PROMPT]],
         "clipped_at_max": [bool(r.get("clipped_at_max")) for r in rows[:CACHE_PER_PROMPT]]}
        for pid, rows in by_prompt.items() if np.std([float(r["reward"]) for r in rows[:CACHE_PER_PROMPT]]) <= STD_TOL]
    result["examples"] = examples
    out = repo_path(cfg["results_dir"]) / "group_size_study.json"
    save_json(out, result)
    print("  wrote", out)


if __name__ == "__main__":
    main()
