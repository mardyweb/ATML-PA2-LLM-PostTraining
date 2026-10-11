"""Clipping study.

Part 1 uses the supplied cached rollout batch: rebuild it, compute advantages from its stored
rewards, values and old/ref log-probs, then for each epsilon measure the clipped surrogate, clip
fraction and affected-token fraction at the midpoint and after each PPO epoch on that batch.
Part 2 runs the short continuation forks (one per epsilon) and evaluates them on held-out prompts.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.models import (clear_gpu, load_policy, load_reward_model, load_tokenizer, load_value_model,
                           reference_mode, trainable_parameters)
from common.rl import critic_values
from task2_ppo.continue_train import adapter_done, run_ppo
from task2_ppo.evaluate import evaluate_ppo
from task2_ppo.ppo import compute_gae, normalize_advantages, shaped_rewards


def fork_name(eps: float, beta: float) -> str:
    return f"fork_eps{eps:g}_kl{beta:g}"


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def _flat(x):
    return torch.as_tensor(x, dtype=torch.float32).flatten()


def _first(row, keys):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _int_list(x):
    if isinstance(x, (list, tuple)) and all(isinstance(t, (int, np.integer)) for t in x):
        return [int(t) for t in x]
    if torch.is_tensor(x) and not torch.is_floating_point(x):
        return [int(t) for t in x.flatten().tolist()]
    return None


def prompt_variants(tok, messages, max_prompt):
    """Ways the prompt could have been tokenized when the cache was made. The frozen reference
    log-probs tell us which one was actually used."""
    full = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    out = {"chat_template": full}
    if len(full) > max_prompt:
        out["right_truncated"] = full[:max_prompt]  # what batch_generate's truncation does
        out["left_truncated"] = full[-max_prompt:]
    if messages[0].get("role") != "system":
        text = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"
        out["no_system"] = tok(text, add_special_tokens=False)["input_ids"]
        out["helpful_system"] = tok.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."}] + list(messages),
            tokenize=True, add_generation_prompt=True)
    return out


def rebuild_cached_batch(cfg, tok, rows):
    """Turn cache rows back into (prompt ids, response ids, old/ref log-probs) on the policy tokenizer."""
    pools = read_jsonl(cfg["paths"]["rl_prompt_train"]) + read_jsonl(cfg["paths"]["rl_prompt_eval"])
    by_pid = {str(r["prompt_id"]): r for r in pools if "prompt_id" in r}
    by_source = {str(r["source_index"]): r for r in pools if "source_index" in r}
    eos = tok.eos_token_id
    penalty = float(cfg.get("missing_eos_penalty", 0.0))
    batch, mismatched, how = [], 0, {"prompt_id": 0, "source_index": 0, "position": 0}
    for row in rows:
        if "prompt_id" in row and str(row["prompt_id"]) in by_pid:
            src, key = by_pid[str(row["prompt_id"])], "prompt_id"
        elif str(row["source_index"]) in by_source:
            src, key = by_source[str(row["source_index"])], "source_index"
        else:
            src, key = pools[int(row["source_index"])], "position"
        how[key] += 1
        messages = prompt_messages(src)
        variants = prompt_variants(tok, messages, int(cfg["max_prompt_length"]))
        prompt_ids = variants["chat_template"]

        old, ref = _flat(row["old_logprobs"]), _flat(row["ref_logprobs"])
        resp_ids = _int_list(_first(row, ("response_tokens", "response_ids", "response_token_ids")))
        if resp_ids is None or len(resp_ids) != len(old):
            resp_ids = tok(row["response"], add_special_tokens=False)["input_ids"]
            if len(resp_ids) + 1 == len(old) and eos is not None:
                resp_ids = resp_ids + [eos]
        n = min(len(resp_ids), len(old), len(ref))
        if not (len(resp_ids) == len(old) == len(ref)):
            mismatched += 1

        terminated = row.get("terminated_with_eos")
        if terminated is None:
            terminated = bool(eos is not None and resp_ids[:n] and resp_ids[n - 1] == eos)
        reward = _first(row, ("effective_terminal_reward",))
        if reward is None:
            raw = _first(row, ("raw_terminal_reward", "reward", "task_reward", "score"))
            reward = None if raw is None else float(_flat(raw)[0]) - (0.0 if terminated else penalty)
        values = row.get("values")
        values = None if values is None else _flat(values)
        batch.append({
            "source_index": row["source_index"], "messages": messages, "prompt_ids": prompt_ids, "variants": variants,
            "response_ids": resp_ids[:n], "old": old[:n], "ref": ref[:n],
            "task_reward": None if reward is None else float(_flat(reward)[0]),
            "terminated": bool(terminated), "values": values[:n] if values is not None and len(values) >= n else None,
            "response": row["response"],
        })
    return batch, mismatched, how


def token_logprobs(model, seq, attn, pw, rids, temperature=1.0):
    out = model(input_ids=seq, attention_mask=attn, use_cache=False, return_dict=True)
    logits = out.logits[:, pw - 1:-1, :][:, : rids.shape[1], :].float() / temperature
    return torch.gather(F.log_softmax(logits, dim=-1), -1, rids.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def score_cached(rm, item, tok, max_length):
    """RM score for a cached rollout that doesn't store its reward."""
    rm_model, rm_tok = rm
    return float(score_reward_pairs(rm_model, rm_tok, [item["messages"]], [item["response"]], max_length=max_length)[0])


def _forward(policy, item, device):
    seq = torch.tensor([item["prompt_ids"] + item["response_ids"]], device=device)
    attn = torch.ones_like(seq)
    rids = seq[:, len(item["prompt_ids"]):]
    return seq, attn, rids


def _stats(new_list, batch, adv_list, eps):
    clip, aff, tok_n, s_clip, s_unclip, logr = 0.0, 0.0, 0.0, 0.0, 0.0, []
    for new, it, adv in zip(new_list, batch, adv_list):
        ratio = torch.exp(new - it["old"])
        outside = (ratio < 1 - eps) | (ratio > 1 + eps)
        affected = ((adv > 0) & (ratio > 1 + eps)) | ((adv < 0) & (ratio < 1 - eps))
        clip += float(outside.sum())
        aff += float(affected.sum())
        tok_n += ratio.numel()
        s_unclip += float((ratio * adv).sum())
        s_clip += float(torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv).sum())
        logr.append((new - it["old"]).flatten())
    logr = torch.cat(logr)
    return {"clip_fraction": clip / tok_n, "affected_fraction": aff / tok_n,
            "surrogate_clipped": s_clip / tok_n, "surrogate_unclipped": s_unclip / tok_n,
            "surrogate_gap": (s_unclip - s_clip) / tok_n,
            "log_ratio_mean": float(logr.mean()), "log_ratio_std": float(logr.std()),
            "ratio_max": float(logr.exp().max()), "ratio_min": float(logr.exp().min()), "tokens": int(tok_n)}


def cached_clipping_study(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tok = load_tokenizer(cfg["base_model"])
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    print("Cached PPO rollouts:", len(rows))
    print("Cache keys:", sorted(rows[0].keys()))
    batch, mismatched, how = rebuild_cached_batch(cfg, tok, rows)
    print(f"  prompts matched by: {how}")
    if mismatched:
        print(f"  warning: {mismatched} rows had response/log-prob length mismatches (trimmed to the shorter one)")

    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    for p in trainable_parameters(policy):
        if p.dtype != torch.float32:
            p.data = p.data.float()
    policy.eval()
    device = next(policy.parameters()).device
    start_state = {n: p.detach().clone() for n, p in policy.named_parameters() if p.requires_grad}

    # The frozen reference is known exactly, so recomputing its log-probs tells us whether the cache
    # was rebuilt correctly and whether its log-probs were taken at temperature 1 or the sampling temperature.
    gen_temp = float(cfg.get("generation", {}).get("temperature", 0.7))
    temps = sorted({1.0, gen_temp})
    best = {t: [] for t in temps}  # per row: (diff vs cached ref, variant name)
    with torch.no_grad():
        for it in batch:
            rids = torch.tensor([it["response_ids"]], device=device)
            for t in temps:
                cands = []
                for vname, pids in it["variants"].items():
                    seq = torch.tensor([pids + it["response_ids"]], device=device)
                    with reference_mode(policy):
                        r = token_logprobs(policy, seq, torch.ones_like(seq), len(pids), rids, t)[0].cpu()
                    cands.append((float((r - it["ref"]).abs().mean()), vname))
                best[t].append(min(cands))
    temp = min(temps, key=lambda t: float(np.mean([d for d, _ in best[t]])))
    for it, (d, vname) in zip(batch, best[temp]):
        it["prompt_ids"], it["prompt_variant"], it["ref_diff"] = it["variants"][vname], vname, d

    # how well does the rebuilt batch reproduce the cached log-probs?
    ref_d, old_d = [], []
    with torch.no_grad():
        for it in batch:
            seq, attn, rids = _forward(policy, it, device)
            pw = len(it["prompt_ids"])
            with reference_mode(policy):
                ref_d.append((token_logprobs(policy, seq, attn, pw, rids, temp)[0].cpu() - it["ref"]).abs())
            old_d.append((token_logprobs(policy, seq, attn, pw, rids, temp)[0].cpu() - it["old"]).abs())
    ref_d, old_d = torch.cat(ref_d), torch.cat(old_d)
    variant_counts = {}
    for it in batch:
        variant_counts[it["prompt_variant"]] = variant_counts.get(it["prompt_variant"], 0) + 1
    agreement = {
        "temperature_mean_ref_diff": {f"T={t:g}": float(np.mean([d for d, _ in best[t]])) for t in temps},
        "prompt_variant_counts": variant_counts,
        "ref_mean_abs_diff": float(ref_d.mean()), "ref_median_abs_diff": float(ref_d.median()),
        "ref_frac_tokens_diff_over_0.05": float((ref_d > 0.05).float().mean()),
        "old_mean_abs_diff": float(old_d.mean()), "old_median_abs_diff": float(old_d.median()),
        "per_row_ref_diff": [round(it["ref_diff"], 4) for it in batch],
    }
    print(f"  log-prob temperature: {temp:g}  (mean |ref diff| per temperature: {agreement['temperature_mean_ref_diff']})")
    print(f"  prompt rendering that matches the cache: {variant_counts}")
    print(f"  recomputed vs cached reference log-probs: mean |diff| {agreement['ref_mean_abs_diff']:.4f}, "
          f"median {agreement['ref_median_abs_diff']:.4f}, tokens off by >0.05: {agreement['ref_frac_tokens_diff_over_0.05']:.3f}")
    print(f"  per-row mean |ref diff|: {agreement['per_row_ref_diff']}")
    if agreement["ref_median_abs_diff"] > 0.02:
        print("  note: reference log-probs don't reproduce exactly; treat part of the midpoint ratio spread as "
              "recomputation noise (see logprob_agreement in the json)")

    use_cached_values = all(it["values"] is not None for it in batch)
    value_model = None if use_cached_values else load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
    need_rm = any(it["task_reward"] is None for it in batch)
    rm = load_reward_model(cfg) if need_rm else None

    # advantages from the cached old/ref log-probs, reward and values (critic only if values are missing)
    rewards_used, adv_rows, masks = [], [], []
    with torch.no_grad():
        for it in batch:
            seq, attn, rids = _forward(policy, it, device)
            pw = len(it["prompt_ids"])
            if use_cached_values:
                values = it["values"][None].float()
            else:
                values = critic_values(value_model, seq, attn)[:, pw - 1:-1][:, : rids.shape[1]].cpu()
            if it["task_reward"] is None:
                r = score_cached(rm, it, tok, int(cfg["reward_max_length"]))
                it["task_reward"] = r - (0.0 if it["terminated"] else float(cfg.get("missing_eos_penalty", 0.0)))
            r = it["task_reward"]
            rewards_used.append(r)
            m = torch.ones(1, len(it["old"]))
            shaped = shaped_rewards(torch.tensor([r]), it["old"][None], it["ref"][None], m, float(cfg["kl_beta"]))
            adv, _ = compute_gae(shaped, values, m, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
            adv_rows.append(adv[0])
            masks.append(m[0])
    if value_model is not None:
        clear_gpu(value_model)
        del value_model
    if rm is not None:
        clear_gpu(rm[0])
        del rm

    # normalise across the whole cached batch, like one PPO batch
    T = max(len(a) for a in adv_rows)
    A = torch.zeros(len(adv_rows), T)
    M = torch.zeros(len(adv_rows), T)
    for i, a in enumerate(adv_rows):
        A[i, : len(a)] = a
        M[i, : len(a)] = 1
    A = normalize_advantages(A, M)
    adv_rows = [A[i, : len(a)] for i, a in enumerate(adv_rows)]

    results = {"num_rollouts": len(batch), "num_tokens": int(M.sum()), "kl_beta": float(cfg["kl_beta"]),
               "length_mismatches": mismatched, "prompt_match": how, "logprob_agreement": agreement,
               "logprob_temperature": temp, "values_source": "cache" if use_cached_values else "midpoint critic",
               "mean_task_reward": float(np.mean(rewards_used)),
               "ppo_epochs": int(cfg["ppo_epochs"]), "learning_rate": float(cfg["policy_learning_rate"]),
               "per_epsilon": {}}
    use_scaler = torch.cuda.is_available()
    for eps in [float(e) for e in cfg["clip_values"]]:
        with torch.no_grad():
            for n, p in policy.named_parameters():
                if n in start_state:
                    p.copy_(start_state[n])
        opt = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler, init_scale=1024.0, growth_interval=10**9)
        curve = []
        for epoch in range(int(cfg["ppo_epochs"]) + 1):
            last = epoch == int(cfg["ppo_epochs"])
            new_list = []
            opt.zero_grad(set_to_none=True)
            for it, adv in zip(batch, adv_rows):
                seq, attn, rids = _forward(policy, it, device)
                with torch.set_grad_enabled(not last):
                    new = token_logprobs(policy, seq, attn, len(it["prompt_ids"]), rids, temp)[0]
                    if not last:
                        ratio = torch.exp(new - it["old"].to(device))
                        a = adv.to(device)
                        obj = torch.minimum(ratio * a, ratio.clamp(1 - eps, 1 + eps) * a)
                        loss = -obj.sum() / results["num_tokens"]
                        scaler.scale(loss).backward()
                new_list.append(new.detach().float().cpu())
            stats = _stats(new_list, batch, adv_rows, eps)
            stats["epoch"] = epoch
            stats["when"] = "midpoint (before any step)" if epoch == 0 else f"after {epoch} step(s)"
            curve.append(stats)
            if not last:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
                scaler.step(opt)
                scaler.update()
        results["per_epsilon"][f"{eps:g}"] = curve
        c0, cl = curve[0], curve[-1]
        print(f"  eps={eps:g}: clip frac {c0['clip_fraction']:.4f} -> {cl['clip_fraction']:.4f}, "
              f"affected {c0['affected_fraction']:.4f} -> {cl['affected_fraction']:.4f}, "
              f"surrogate gap {c0['surrogate_gap']:.5f} -> {cl['surrogate_gap']:.5f}", flush=True)

    # log-ratio distribution at the midpoint (same for every epsilon)
    with torch.no_grad():
        for n, p in policy.named_parameters():
            if n in start_state:
                p.copy_(start_state[n])
        lr_all = []
        for it in batch:
            seq, attn, rids = _forward(policy, it, device)
            new = token_logprobs(policy, seq, attn, len(it["prompt_ids"]), rids, temp)
            lr_all.append((new[0].float().cpu() - it["old"]))
        results["log_ratio_at_midpoint"] = [round(float(x), 5) for x in torch.cat(lr_all)]

    out = repo_path(cfg["results_dir"]) / "clipping_cached.json"
    save_json(out, results)
    print("  wrote", out)
    clear_gpu(policy)
    del policy
    clear_gpu()
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--skip-forks", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])

    if args.force or not (res / "clipping_cached.json").exists():
        cached_clipping_study(args.config)
    else:
        print("cached clipping study already done - skipping")
    if args.skip_forks:
        return

    print("Required epsilon values:", cfg["clip_values"], "| fork updates:", cfg["fork_updates"])
    beta = float(cfg["kl_beta"])
    for eps in [float(e) for e in cfg["clip_values"]]:
        name = fork_name(eps, beta)
        out = f"outputs/task2_ppo/{name}"
        if args.force or not adapter_done(out):
            run_ppo(args.config, output=out, updates=int(cfg["fork_updates"]), clip_epsilon=eps,
                    kl_beta=beta, run_name=name)
        evaluate_ppo(args.config, out, name, num_prompts=args.num_prompts, force=args.force)
    print("\nClipping study done. Tables/figures: python -m task2_ppo.summarize")


if __name__ == "__main__":
    main()
