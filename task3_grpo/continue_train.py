from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import sampled_kl
from common.models import (clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode,
                           trainable_parameters)
from common.rl import entropy_from_logits, load_prompt_pool, write_jsonl
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences

STD_TOL = 1e-6  # same eps as group_relative_advantages: a group is uninformative if its std is below this


def adapter_done(path) -> bool:
    return (repo_path(path) / "adapter_config.json").exists()


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True)
    for p in trainable_parameters(policy):
        if p.dtype != torch.float32:
            p.data = p.data.float()
    policy.eval()  # no dropout, so old and new log-probs match on the first pass
    reward_model, reward_tokenizer = load_reward_model(cfg)
    pool, dropped = load_prompt_pool(cfg["paths"]["rl_prompt_train"], tokenizer, int(cfg["max_prompt_length"]))
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg, "tokenizer": tokenizer, "policy": policy,
        "reward_model": reward_model, "reward_tokenizer": reward_tokenizer,
        "prompt_pool": pool, "dropped_prompts": dropped, "optimizer": optimizer,
    }


def per_completion_grad_norms(policy, new_logp, old_logp, adv, token_mask, eps, loss_type, max_len, scale):
    """Gradient norm of each completion's share of the policy term (used for the length analysis)."""
    params = trainable_parameters(policy)
    ratio = torch.exp(new_logp - old_logp)
    a = adv[:, None]
    obj = torch.minimum(ratio * a, ratio.clamp(1 - eps, 1 + eps) * a)
    token_sum = (obj * token_mask).sum(-1)
    denom = token_mask.sum(-1).clamp_min(1.0) if loss_type == "grpo" else torch.full_like(token_sum, float(max_len))
    pieces = -(token_sum / denom) / new_logp.shape[0]
    norms = []
    for k in range(new_logp.shape[0]):
        if float(token_mask[k].sum()) == 0:
            norms.append(0.0)
            continue
        grads = torch.autograd.grad(pieces[k] * scale, params, retain_graph=True, allow_unused=True)
        sq = sum(float((g.float() / scale).pow(2).sum()) for g in grads if g is not None)
        norms.append(sq ** 0.5)
    return norms


def run_grpo(config_path: str, output: str | None = None, updates: int | None = None, loss_type: str = "grpo",
             run_name: str = "standard", prompt_offset: int = 0, track_grad_norms: bool = True):
    b = prepare_grpo_continuation(config_path)
    cfg = b["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    res = repo_path(cfg["results_dir"]) / run_name
    res.mkdir(parents=True, exist_ok=True)
    log_path = res / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    policy, tok, opt = b["policy"], b["tokenizer"], b["optimizer"]
    n_updates, ppu, K = int(cfg["updates"]), int(cfg["prompts_per_update"]), int(cfg["num_generations"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_len = int(cfg["max_completion_length"])
    gen = cfg.get("generation", {})
    pool = b["prompt_pool"]
    print(f"[{run_name}] {n_updates} updates x {ppu} prompt(s) x K={K}, loss_type={loss_type}, eps={eps}, "
          f"beta={beta}, pool={len(pool)} (dropped {len(b['dropped_prompts'])} long prompts)", flush=True)

    use_scaler = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler, init_scale=1024.0, growth_interval=10**9)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    set_seed(int(cfg["seed"]))
    rollouts, used_ids, gen_tokens = [], [], 0

    for u in range(n_updates):
        items = [pool[(prompt_offset + u * ppu + j) % len(pool)] for j in range(ppu)]
        used_ids += [it["id"] for it in items]
        prompts = [it["messages"] for it in items for _ in range(K)]
        group_ids = torch.tensor([j for j in range(ppu) for _ in range(K)])

        g = batch_generate(policy, tok, prompts, max_prompt_length=int(cfg["max_prompt_length"]), max_new_tokens=max_len,
                           temperature=float(gen.get("temperature", 0.7)), top_p=float(gen.get("top_p", 0.9)),
                           do_sample=bool(gen.get("do_sample", True)))
        seq, attn, rids, rmask = (g[k].clone() for k in ("sequences", "attention_mask", "response_ids", "response_mask"))
        pw = g["prompt_width"]
        with torch.no_grad():
            old_logp, logits = response_token_logprobs(policy, seq, attn, pw, rids)
            ent_row, _ = entropy_from_logits(logits, rmask)
            del logits
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(policy, seq, attn, pw, rids)
            rewards = score_reward_pairs(b["reward_model"], b["reward_tokenizer"], prompts, g["responses"],
                                         max_length=int(cfg.get("reward_max_length", 1280))).float().cpu()
        adv = group_relative_advantages(rewards, group_ids).to(seq.device)
        token_mask = mask_truncated_sequences(rmask, g["truncated"]) if cfg.get("mask_truncated_completions", True) else rmask

        group_std = [float(rewards[group_ids == j].std(unbiased=False)) for j in range(ppu)]
        grad_norms = None
        for epoch in range(int(cfg["policy_epochs"])):
            new_logp, _ = response_token_logprobs(policy, seq, attn, pw, rids)
            loss, st = grpo_policy_loss(new_logp, old_logp, adv, token_mask, ref_logp, eps, beta,
                                        loss_type=loss_type, max_completion_length=max_len)
            if epoch == 0 and track_grad_norms:
                grad_norms = per_completion_grad_norms(policy, new_logp, old_logp, adv, token_mask, eps,
                                                       loss_type, max_len, scaler.get_scale())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), float(cfg["max_grad_norm"]))
            scaler.step(opt)
            scaler.update()

        lengths = rmask.sum(-1)
        gen_tokens += int(lengths.sum())
        comps = []
        for k in range(len(prompts)):
            T = int(lengths[k])
            comps.append({"group": int(group_ids[k]), "length": T, "reward": float(rewards[k]), "advantage": float(adv[k]),
                          "truncated": bool(g["truncated"][k]), "masked": float(token_mask[k].sum()) == 0,
                          "grad_norm": None if grad_norms is None else grad_norms[k],
                          # how much total weight the sequence's tokens get in the loss
                          "seq_weight": 0.0 if float(token_mask[k].sum()) == 0 else
                          abs(float(adv[k])) * (1.0 if loss_type == "grpo" else T / max_len)})
        rec = {
            "update": u + 1, "prompt_ids": [it["id"] for it in items], "loss_type": loss_type,
            "reward": float(rewards.mean()),
            "group_reward_std": float(np.mean(group_std)),
            "uninformative_fraction": float(np.mean([s <= STD_TOL for s in group_std])),
            "kl_ref_token_mean": float(sampled_kl(old_logp, ref_logp, rmask)),
            "kl_k3_loss_term": float(st["sampled_kl"]),
            "policy_loss": float(st["policy_term"]), "loss": float(loss.detach()),
            "grad_norm": float(gnorm), "clip_fraction": float(st["clip_fraction"]),
            "entropy": float(ent_row.mean()), "response_length": float(lengths.float().mean()),
            "truncated_fraction": float(np.mean(g["truncated"])),
            "completions": comps, "generated_tokens_so_far": gen_tokens,
            "elapsed_s": round(elapsed(), 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0,
        }
        append_jsonl(log_path, rec)
        for k, resp in enumerate(g["responses"]):
            rollouts.append({"update": u + 1, "id": items[int(group_ids[k])]["id"], "prompt": prompts[k][-1]["content"],
                             "response": resp, "reward": float(rewards[k]), "advantage": float(adv[k]),
                             "length": int(lengths[k]), "truncated": bool(g["truncated"][k])})
        print(f"[{run_name}] update {u + 1}/{n_updates} reward={rec['reward']:.3f} grp_std={rec['group_reward_std']:.3f} "
              f"kl={rec['kl_ref_token_mean']:.4f} pg={rec['policy_loss']:.4f} gnorm={rec['grad_norm']:.3f} "
              f"H={rec['entropy']:.3f} len={rec['response_length']:.0f} trunc={rec['truncated_fraction']:.2f} "
              f"t={rec['elapsed_s']:.0f}s", flush=True)

    policy.save_pretrained(str(out))
    write_jsonl(res / "train_rollouts.jsonl", rollouts)
    meta = {
        "run_name": run_name, "loss_type": loss_type, "updates": n_updates, "prompts_per_update": ppu,
        "num_generations": K, "policy_epochs": int(cfg["policy_epochs"]), "learning_rate": float(cfg["learning_rate"]),
        "clip_epsilon": eps, "kl_beta": beta, "max_completion_length": max_len,
        "mask_truncated_completions": bool(cfg.get("mask_truncated_completions", True)),
        "seed": int(cfg["seed"]), "prompt_offset": prompt_offset, "prompt_ids": used_ids,
        "generated_tokens": gen_tokens, "wall_clock_s": round(elapsed(), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "start_policy": cfg["paths"]["grpo_midpoint_policy"], "adapter_path": str(out),
        "per_completion_grad_norms": track_grad_norms,
    }
    save_json(res / "train_meta.json", meta)
    print(f"[{run_name}] done in {meta['wall_clock_s']:.0f}s, peak VRAM {meta['peak_vram_gb']} GB, saved {out}", flush=True)
    clear_gpu()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--prompt-offset", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = args.output or load_yaml(args.config)["output"]
    if adapter_done(out) and not args.force:
        print(f"[{args.run_name}] adapter already at {out} - skipping (use --force to rerun)")
        return
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.prompt_offset)


if __name__ == "__main__":
    main()
