"""Evaluate one DPO adapter (or the base model with --adapter none).

Held-out pair metrics, per-stratum accuracy on the length eval set, reward / KL / length on
sampled generations, and word-limit compliance. Saved under results/task1_dpo/<name>/.
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed, wall_timer
from common.metrics import parse_word_limit, safe_corr, word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.train import concat_pair_batches, filter_encodable, row_id

STRATUM_KEYS = ("length_stratum", "stratum", "strata", "length_bucket", "bucket", "length_group",
                "length_category", "category", "group", "length_relation")
MAX_GEN_PROMPT_TOKENS = 1024
REWARD_MAX_LENGTH = 1280


def _write_jsonl(path, rows):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def length_stats(values):
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {}
    q1, med, q3, p90 = np.percentile(a, [25, 50, 75, 90])
    return {"mean": float(a.mean()), "std": float(a.std()), "median": float(med),
            "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1), "p90": float(p90), "n": int(a.size)}


def stratum_of(row: dict, tokenizer=None):
    for k in STRATUM_KEYS:
        if k in row and row[k] not in (None, ""):
            return str(row[k]), k
    # only used if the file has no stratum column
    yc, yr = preference_responses(row)
    lc = len(tokenizer(yc, add_special_tokens=False)["input_ids"])
    lr = len(tokenizer(yr, add_special_tokens=False)["input_ids"])
    ratio = lc / max(lr, 1)
    if ratio > 1.25:
        return "preferred_longer(derived)", None
    if ratio < 0.8:
        return "rejected_longer(derived)", None
    return "matched(derived)", None


@torch.no_grad()
def pair_logprobs(model, tokenizer, rows, max_len, batch_size):
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        enc_c, enc_r = [], []
        for row in chunk:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            enc_c.append(encode_prompt_response(tokenizer, prompt, yc, max_len))
            enc_r.append(encode_prompt_response(tokenizer, prompt, yr, max_len))
        n = len(chunk)
        batch = concat_pair_batches(pad_batch(tokenizer, enc_c), pad_batch(tokenizer, enc_r), tokenizer.pad_token_id)
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        pol, _, _ = response_sequence_logprobs(model, batch)
        with reference_mode(model):
            ref, _, _ = response_sequence_logprobs(model, batch)
        ntok = batch["response_mask"].sum(-1)
        for j in range(n):
            out.append({
                "policy_chosen_logp": float(pol[j]), "policy_rejected_logp": float(pol[n + j]),
                "ref_chosen_logp": float(ref[j]), "ref_rejected_logp": float(ref[n + j]),
                "chosen_tokens_scored": int(ntok[j]), "rejected_tokens_scored": int(ntok[n + j]),
            })
    return out


def evaluate_pairs(model, tokenizer, rows, cfg, betas_for_loss, batch_size, with_strata=False):
    max_len = int(cfg["max_sequence_length"])
    kept, dropped = filter_encodable(rows, tokenizer, max_len)
    rows_k = [rows[i] for i in kept]
    lp = pair_logprobs(model, tokenizer, rows_k, max_len, batch_size)
    records = []
    for idx, row, r in zip(kept, rows_k, lp):
        yc, yr = preference_responses(row)
        margin = (r["policy_chosen_logp"] - r["ref_chosen_logp"]) - (r["policy_rejected_logp"] - r["ref_rejected_logp"])
        rec = {
            "id": row_id(row, idx),
            "index": idx,
            **r,
            "margin": margin,
            "correct": margin > 0,
            "chosen_tokens_full": len(tokenizer(yc, add_special_tokens=False)["input_ids"]),
            "rejected_tokens_full": len(tokenizer(yr, add_special_tokens=False)["input_ids"]),
            "chosen_words": word_count(yc),
            "rejected_words": word_count(yr),
        }
        if with_strata:
            rec["stratum"], rec["stratum_key"] = stratum_of(row, tokenizer)
        records.append(rec)

    m = torch.tensor([r["margin"] for r in records], dtype=torch.float64)
    summary = {
        "num_pairs": len(records),
        "num_dropped_overlong_prompt": len(dropped),
        "preference_accuracy": float((m > 0).double().mean()),
        "margin_mean": float(m.mean()),
        "margin_std": float(m.std()) if len(m) > 1 else 0.0,
    }
    for b in betas_for_loss:
        summary[f"dpo_loss_beta{b:g}"] = float(-F.logsigmoid(b * m).mean())
    len_diff = [r["chosen_tokens_full"] - r["rejected_tokens_full"] for r in records]
    summary["frac_chosen_longer"] = float(np.mean([d > 0 for d in len_diff]))
    summary["longer_heuristic_accuracy"] = summary["frac_chosen_longer"]  # "always pick the longer one" baseline
    summary["corr_margin_vs_length_diff"] = safe_corr(m.numpy(), len_diff)

    if with_strata:
        strata = {}
        for s in sorted({r["stratum"] for r in records}):
            sub = [r for r in records if r["stratum"] == s]
            sm = np.array([r["margin"] for r in sub])
            strata[s] = {
                "n": len(sub),
                "preference_accuracy": float((sm > 0).mean()),
                "margin_mean": float(sm.mean()),
                "mean_chosen_tokens": float(np.mean([r["chosen_tokens_full"] for r in sub])),
                "mean_rejected_tokens": float(np.mean([r["rejected_tokens_full"] for r in sub])),
            }
        summary["strata"] = strata
        summary["stratum_label_key"] = records[0]["stratum_key"] if records else None
    return summary, records


def generation_prompts(cfg, tokenizer, n):
    rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
    seen, out = set(), []
    for idx, row in enumerate(rows):
        msgs = prompt_messages_from_preference(row)
        key = str(msgs)
        if key in seen:
            continue
        seen.add(key)
        ntok = len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))
        if ntok > MAX_GEN_PROMPT_TOKENS:
            continue
        out.append({"id": row_id(row, idx), "index": idx, "messages": msgs})
        if len(out) >= n:
            break
    return out


@torch.no_grad()
def generate_and_score(model, tokenizer, rm, rm_tok, items, cfg, batch_size, max_new_tokens, kl_chunk=8):
    gen_cfg = cfg.get("generation", {})
    records = []
    kl_tok_sum, kl_tok_n = 0.0, 0.0
    for i in range(0, len(items), batch_size):
        chunk = items[i:i + batch_size]
        g = batch_generate(
            model, tokenizer, [it["messages"] for it in chunk],
            max_prompt_length=MAX_GEN_PROMPT_TOKENS, max_new_tokens=max_new_tokens,
            temperature=float(gen_cfg.get("temperature", 0.7)), top_p=float(gen_cfg.get("top_p", 0.9)),
            do_sample=bool(gen_cfg.get("do_sample", True)),
        )
        # KL estimate on the sampled tokens: log pi - log ref
        kl_seq, kl_mean = [], []
        for j in range(0, len(chunk), kl_chunk):
            sl = slice(j, j + kl_chunk)
            seqs, attn = g["sequences"][sl], g["attention_mask"][sl]
            rids, rmask = g["response_ids"][sl], g["response_mask"][sl]
            pol, _ = response_token_logprobs(model, seqs, attn, g["prompt_width"], rids)
            with reference_mode(model):
                ref, _ = response_token_logprobs(model, seqs, attn, g["prompt_width"], rids)
            diff = (pol - ref) * rmask
            kl_tok_sum += float(diff.sum())
            kl_tok_n += float(rmask.sum())
            kl_seq.extend(diff.sum(-1).tolist())
            kl_mean.extend((diff.sum(-1) / rmask.sum(-1).clamp_min(1)).tolist())
        scores = score_reward_pairs(rm, rm_tok, [it["messages"] for it in chunk], g["responses"],
                                    max_length=REWARD_MAX_LENGTH).tolist()
        for k, it in enumerate(chunk):
            records.append({
                "id": it["id"], "index": it.get("index"),
                "prompt": it["messages"][-1]["content"],
                "response": g["responses"][k],
                "response_tokens": g["response_lengths"][k],
                "response_words": word_count(g["responses"][k]),
                "terminated_with_eos": g["terminated_with_eos"][k],
                "truncated": g["truncated"][k],
                "reward": scores[k],
                "kl_seq_sum": kl_seq[k],
                "kl_token_mean": kl_mean[k],
                **({"sample": it["sample"]} if "sample" in it else {}),
            })
        print(f"  generated {min(i + batch_size, len(items))}/{len(items)}", flush=True)
    return records, (kl_tok_sum / max(kl_tok_n, 1.0))


def _gen_summary(records, kl_token_level):
    rewards = [r["reward"] for r in records]
    return {
        "num_prompts": len(records),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "kl_token_mean": float(kl_token_level),  # averaged over all response tokens
        "kl_seq_sum_mean": float(np.mean([r["kl_seq_sum"] for r in records])),
        "length_tokens": length_stats([r["response_tokens"] for r in records]),
        "length_words": length_stats([r["response_words"] for r in records]),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])),
        "corr_reward_vs_length": safe_corr(rewards, [r["response_tokens"] for r in records]),
    }


def evaluate_model(
    config_path: str,
    adapter: str | None,
    name: str,
    beta: float | None = None,
    num_gen_prompts: int = 300,
    gen_batch_size: int = 32,
    pair_batch_size: int = 4,
    word_limit_samples: int = 4,
    skip_generation: bool = False,
):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    out_dir = repo_path(cfg.get("results_dir", "results/task1_dpo")) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = None if adapter in (None, "", "none", "None", "sft") else adapter

    if beta is None:
        meta_path = out_dir / "train_meta.json"
        beta = float(load_json(meta_path)["beta"]) if meta_path.exists() else float(cfg["beta"])
    betas_for_loss = sorted({float(beta), float(cfg["beta"])})

    timer = wall_timer()
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    summary = {"name": name, "adapter": adapter or "none (reference/SFT policy)", "beta": beta,
               "seed": seed, "decoding": cfg.get("generation", {}),
               "max_generation_tokens": int(cfg["max_generation_tokens"])}

    if adapter is None:
        # base model is the reference, so every margin is 0 and the loss is just log 2
        note = {"note": "policy is the reference: margin = 0 for every pair, DPO loss = log 2",
                "preference_accuracy": float("nan"), "margin_mean": 0.0,
                **{f"dpo_loss_beta{b:g}": math.log(2) for b in betas_for_loss}}
        summary["heldout_standard"] = dict(note)
        summary["heldout_length_stratified"] = dict(note)
    else:
        print(f"[{name}] held-out preference pairs ...", flush=True)
        std_rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
        s, recs = evaluate_pairs(model, tokenizer, std_rows, cfg, betas_for_loss, pair_batch_size)
        summary["heldout_standard"] = s
        _write_jsonl(out_dir / "pairs_standard_eval.jsonl", recs)

        print(f"[{name}] length-stratified pairs ...", flush=True)
        len_rows = read_jsonl(cfg["paths"]["dpo_length_eval"])
        s, recs = evaluate_pairs(model, tokenizer, len_rows, cfg, betas_for_loss, pair_batch_size, with_strata=True)
        summary["heldout_length_stratified"] = s
        _write_jsonl(out_dir / "pairs_length_eval.jsonl", recs)

    if not skip_generation:
        rm, rm_tok = load_reward_model(cfg)
        max_new = int(cfg["max_generation_tokens"])

        print(f"[{name}] generating on held-out prompts ...", flush=True)
        items = generation_prompts(cfg, tokenizer, num_gen_prompts)
        set_seed(seed)
        gens, kl_tok = generate_and_score(model, tokenizer, rm, rm_tok, items, cfg, gen_batch_size, max_new)
        summary["generation"] = _gen_summary(gens, kl_tok)
        _write_jsonl(out_dir / "generations.jsonl", gens)

        print(f"[{name}] word-limit prompts ...", flush=True)
        wl_rows = read_jsonl(cfg["paths"]["word_limit_prompts"])
        wl_items = []
        for k in range(word_limit_samples):
            for idx, row in enumerate(wl_rows):
                wl_items.append({"id": row.get("prompt_id", idx), "index": idx, "messages": prompt_messages(row), "sample": k})
        set_seed(seed)
        wl, wl_kl = generate_and_score(model, tokenizer, rm, rm_tok, wl_items, cfg, gen_batch_size, max_new)
        for r, it in zip(wl, wl_items):
            r["word_limit"] = parse_word_limit(r["prompt"])
            r["compliant"] = word_limit_compliance(r["prompt"], r["response"])
        valid = [r for r in wl if r["compliant"] is not None]
        summary["word_limit"] = {
            "num_prompts": len(wl_rows),
            "samples_per_prompt": word_limit_samples,
            "num_scored": len(valid),
            "compliance_rate": float(np.mean([r["compliant"] for r in valid])) if valid else float("nan"),
            "words": length_stats([r["response_words"] for r in wl]),
            "mean_words_over_limit": float(np.mean([max(0, r["response_words"] - r["word_limit"]) for r in valid])) if valid else float("nan"),
            "per_prompt_compliance": {
                str(pid): float(np.mean([r["compliant"] for r in valid if r["id"] == pid]))
                for pid in dict.fromkeys(r["id"] for r in valid)
            },
            "kl_token_mean": float(wl_kl),
            "reward_mean": float(np.mean([r["reward"] for r in wl])),
        }
        _write_jsonl(out_dir / "word_limit_generations.jsonl", wl)
        clear_gpu(rm)
        del rm

    summary["eval_wall_clock_s"] = round(timer(), 1)
    save_json(out_dir / "summary.json", summary)
    hs = summary["heldout_standard"]
    print(f"[{name}] held-out acc={hs['preference_accuracy']:.3f} "
          f"loss(beta={beta:g})={hs[f'dpo_loss_beta{beta:g}']:.4f}", end="")
    if "generation" in summary:
        g = summary["generation"]
        print(f" | RM={g['reward_mean']:.3f} KL/token={g['kl_token_mean']:.4f} "
              f"len={g['length_tokens']['mean']:.1f} | word-limit={summary['word_limit']['compliance_rate']:.2f}")
    else:
        print()
    clear_gpu(model)
    del model
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True, help="adapter directory, or 'none' for the SFT/reference policy")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, help="beta used for the reported loss (defaults to the run's beta)")
    ap.add_argument("--num-gen-prompts", type=int, default=300)
    ap.add_argument("--gen-batch-size", type=int, default=32)
    ap.add_argument("--pair-batch-size", type=int, default=4)
    ap.add_argument("--word-limit-samples", type=int, default=4)
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-evaluate even if summary.json exists")
    args = ap.parse_args()
    done = repo_path(load_yaml(args.config).get("results_dir", "results/task1_dpo")) / args.name / "summary.json"
    if done.exists() and not args.force:
        print(f"[{args.name}] already evaluated ({done}) - skipping (use --force to redo)")
        return
    evaluate_model(args.config, args.adapter, args.name, args.beta, args.num_gen_prompts,
                   args.gen_batch_size, args.pair_batch_size, args.word_limit_samples, args.skip_generation)


if __name__ == "__main__":
    main()
