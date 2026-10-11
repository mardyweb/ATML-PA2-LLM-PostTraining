"""Helpers shared by the PPO and GRPO tasks: prompt pools, critic values, entropy, held-out eval."""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from common.data import prompt_messages, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed, wall_timer
from common.metrics import safe_corr, word_count
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode


def prompt_id(row: dict, index: int):
    for key in ("prompt_id", "id", "source_index", "example_id"):
        if key in row and row[key] is not None:
            return row[key]
    return index


def load_prompt_pool(path, tokenizer, max_prompt_tokens: int):
    """Prompts in file order, minus any that would get cut off by the prompt length limit."""
    pool, dropped = [], []
    for i, row in enumerate(read_jsonl(path)):
        msgs = prompt_messages(row)
        n = len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        item = {"id": prompt_id(row, i), "index": i, "messages": msgs, "row": row}
        (pool if n <= max_prompt_tokens else dropped).append(item)
    return pool, dropped


def entropy_from_logits(logits, mask):
    """Per-token entropy of the full next-token distribution, averaged over valid tokens per row."""
    logp = F.log_softmax(logits.float(), dim=-1)
    ent = -(logp.exp() * logp).sum(-1)
    return (ent * mask).sum(-1) / mask.sum(-1).clamp_min(1.0), ent


def critic_values(value_model, input_ids, attention_mask):
    """Per-position values. Same idea as common.models.token_values, but casts the hidden
    states to the head's dtype (the trainable head is kept in fp32)."""
    inner = value_model.get_base_model() if hasattr(value_model, "get_base_model") else value_model
    backbone = getattr(inner, inner.base_model_prefix)
    hidden = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                      return_dict=True).last_hidden_state
    head = inner.score if hasattr(inner, "score") else inner.classifier
    # with modules_to_save the head holds a frozen original plus the trainable copy; match the copy
    params = list(head.parameters())
    w = next((p for p in params if p.requires_grad), params[0])
    return head(hidden.to(w.dtype)).squeeze(-1).float()


def length_stats(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(med),
            "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1), "n": int(a.size)}


def write_jsonl(path, rows):
    path = repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@torch.no_grad()
def evaluate_heldout(cfg, adapter, out_dir, num_prompts=128, batch_size=16, max_new_tokens=None,
                     prompt_path=None, chunk=4):
    """Same held-out protocol for every RL policy: fixed prompts, fixed seed and decoding,
    reward model score, sampled KL to the reference, entropy and length."""
    out_dir = repo_path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timer = wall_timer()
    tok = load_tokenizer(cfg["base_model"])
    max_prompt = int(cfg.get("max_prompt_length", 256))
    max_new = int(max_new_tokens or cfg.get("eval_max_response_length", cfg.get("max_completion_length", 512)))
    pool, dropped = load_prompt_pool(prompt_path or cfg["paths"]["rl_prompt_eval"], tok, max_prompt)
    pool = pool[:num_prompts]

    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    rm, rm_tok = load_reward_model(cfg)
    gen = cfg.get("generation", {})
    penalty = float(cfg.get("missing_eos_penalty", 0.0))
    set_seed(int(cfg["seed"]))

    records, kl_sum, kl_n, ent_sum = [], 0.0, 0.0, 0.0
    for i in range(0, len(pool), batch_size):
        items = pool[i:i + batch_size]
        prompts = [it["messages"] for it in items]
        g = batch_generate(policy, tok, prompts, max_prompt_length=max_prompt, max_new_tokens=max_new,
                           temperature=float(gen.get("temperature", 0.7)), top_p=float(gen.get("top_p", 0.9)),
                           do_sample=bool(gen.get("do_sample", True)))
        kl_seq, kl_tok, ents = [], [], []
        for j in range(0, len(items), chunk):
            sl = slice(j, j + chunk)
            seq, attn, rids, m = g["sequences"][sl], g["attention_mask"][sl], g["response_ids"][sl], g["response_mask"][sl]
            pol, logits = response_token_logprobs(policy, seq, attn, g["prompt_width"], rids)
            ent_row, ent_tok = entropy_from_logits(logits, m)
            del logits
            with reference_mode(policy):
                ref, _ = response_token_logprobs(policy, seq, attn, g["prompt_width"], rids)
            d = (pol - ref) * m
            kl_sum += float(d.sum())
            kl_n += float(m.sum())
            ent_sum += float((ent_tok * m).sum())
            kl_seq += d.sum(-1).tolist()
            kl_tok += (d.sum(-1) / m.sum(-1).clamp_min(1)).tolist()
            ents += ent_row.tolist()
        scores = score_reward_pairs(rm, rm_tok, prompts, g["responses"],
                                    max_length=int(cfg.get("reward_max_length", 1280))).tolist()
        for k, it in enumerate(items):
            records.append({
                "id": it["id"], "index": it["index"], "prompt": it["messages"][-1]["content"],
                "response": g["responses"][k], "response_tokens": g["response_lengths"][k],
                "response_words": word_count(g["responses"][k]),
                "terminated_with_eos": g["terminated_with_eos"][k], "truncated": g["truncated"][k],
                "reward": scores[k],
                "reward_with_eos_penalty": scores[k] - (0.0 if g["terminated_with_eos"][k] else penalty),
                "kl_seq_sum": kl_seq[k], "kl_token_mean": kl_tok[k], "entropy": ents[k],
            })
        print(f"  eval {min(i + batch_size, len(pool))}/{len(pool)}", flush=True)

    rewards = [r["reward"] for r in records]
    summary = {
        "adapter": str(adapter),
        "num_prompts": len(records),
        "num_dropped_long_prompts": len(dropped),
        "max_new_tokens": max_new,
        "decoding": gen,
        "seed": int(cfg["seed"]),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_sem": float(np.std(rewards) / max(np.sqrt(len(rewards)), 1)),
        "reward_with_eos_penalty_mean": float(np.mean([r["reward_with_eos_penalty"] for r in records])),
        "kl_token_mean": kl_sum / max(kl_n, 1.0),
        "kl_seq_sum_mean": float(np.mean([r["kl_seq_sum"] for r in records])),
        "entropy_token_mean": ent_sum / max(kl_n, 1.0),
        "length_tokens": length_stats([r["response_tokens"] for r in records]),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])),
        "corr_reward_vs_length": safe_corr(rewards, [r["response_tokens"] for r in records]),
        "eval_wall_clock_s": round(timer(), 1),
    }
    write_jsonl(out_dir / "eval_generations.jsonl", records)
    save_json(out_dir / "eval_summary.json", summary)
    print(f"  held-out reward={summary['reward_mean']:.3f} KL/token={summary['kl_token_mean']:.4f} "
          f"entropy={summary['entropy_token_mean']:.3f} len={summary['length_tokens']['mean']:.0f} "
          f"trunc={summary['truncation_rate']:.2f}", flush=True)
    clear_gpu(policy, rm)
    del policy, rm
    clear_gpu()
    return summary
