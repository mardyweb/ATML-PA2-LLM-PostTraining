from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import masked_mean
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    trainable_parameters,
    value_parameter_groups,
)
from common.rl import critic_values, entropy_from_logits, load_prompt_pool, write_jsonl
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards, value_mse_loss


def adapter_done(path) -> bool:
    return (repo_path(path) / "adapter_config.json").exists()


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=True)
    value_model = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"],
                                   train_mode=cfg.get("value_train_mode", "head_only"))
    # keep trainable weights in fp32 so the fp16 loss scaler can unscale them
    for p in list(trainable_parameters(policy)) + list(trainable_parameters(value_model)):
        if p.dtype != torch.float32:
            p.data = p.data.float()
    # dropout off: old and new log-probs then agree exactly before the first update
    policy.eval()
    value_model.eval()

    reward_model, reward_tokenizer = load_reward_model(cfg)
    pool, dropped = load_prompt_pool(cfg["paths"]["rl_prompt_train"], tokenizer, int(cfg["max_prompt_length"]))

    policy_optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["policy_learning_rate"]))
    value_optimizer = AdamW(
        value_parameter_groups(value_model, lora_lr=float(cfg["value_lora_learning_rate"]),
                               head_lr=float(cfg["value_head_learning_rate"])),
        weight_decay=0.0,
    )
    return {
        "cfg": cfg, "tokenizer": tokenizer, "policy": policy, "value_model": value_model,
        "reward_model": reward_model, "reward_tokenizer": reward_tokenizer,
        "prompt_pool": pool, "dropped_prompts": dropped,
        "policy_optimizer": policy_optimizer, "value_optimizer": value_optimizer,
    }


def collect_rollout(b, items):
    """Sample one response per prompt and compute everything PPO needs from the old policy."""
    cfg, tok, policy = b["cfg"], b["tokenizer"], b["policy"]
    gen = cfg.get("generation", {})
    prompts = [it["messages"] for it in items]
    g = batch_generate(policy, tok, prompts, max_prompt_length=int(cfg["max_prompt_length"]),
                       max_new_tokens=int(cfg["max_response_length"]),
                       temperature=float(gen.get("temperature", 0.7)), top_p=float(gen.get("top_p", 0.9)),
                       do_sample=bool(gen.get("do_sample", True)))
    # generate() runs under inference_mode; clone so these can be used in the backward pass
    seq, attn, rids, mask = (g[k].clone() for k in ("sequences", "attention_mask", "response_ids", "response_mask"))
    pw = g["prompt_width"]
    with torch.no_grad():
        old_logp, logits = response_token_logprobs(policy, seq, attn, pw, rids)
        entropy, _ = entropy_from_logits(logits, mask)
        del logits
        with reference_mode(policy):
            ref_logp, _ = response_token_logprobs(policy, seq, attn, pw, rids)
        old_values = critic_values(b["value_model"], seq, attn)[:, pw - 1:-1][:, : rids.shape[1]]
        scores = score_reward_pairs(b["reward_model"], b["reward_tokenizer"], prompts, g["responses"],
                                    max_length=int(cfg["reward_max_length"])).to(seq.device)
    terminated = torch.tensor(g["terminated_with_eos"], device=seq.device)
    task_reward = scores - float(cfg.get("missing_eos_penalty", 0.0)) * (~terminated).float()
    return {"g": g, "seq": seq, "attn": attn, "pw": pw, "rids": rids, "mask": mask,
            "old_logp": old_logp, "ref_logp": ref_logp, "old_values": old_values,
            "scores": scores, "task_reward": task_reward, "entropy": entropy}


def ppo_update(b, ro, eps, kl_beta, scaler):
    cfg = b["cfg"]
    policy, value_model = b["policy"], b["value_model"]
    mask = ro["mask"]

    rewards = shaped_rewards(ro["task_reward"], ro["old_logp"], ro["ref_logp"], mask, kl_beta)
    adv, returns = compute_gae(rewards, ro["old_values"], mask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
    adv_norm = normalize_advantages(adv, mask)

    valid = mask.bool()
    ret_v, val_v = returns[valid], ro["old_values"][valid]
    explained_var = float(1 - (ret_v - val_v).var(unbiased=False) / ret_v.var(unbiased=False).clamp_min(1e-8))

    per_epoch = []
    for _ in range(int(cfg["ppo_epochs"])):
        new_logp, _ = response_token_logprobs(policy, ro["seq"], ro["attn"], ro["pw"], ro["rids"])
        pg_loss, ratio, clip_frac = ppo_policy_loss(new_logp, ro["old_logp"], adv_norm, mask, eps=eps)
        new_values = critic_values(value_model, ro["seq"], ro["attn"])[:, ro["pw"] - 1:-1][:, : ro["rids"].shape[1]]
        v_loss = value_mse_loss(new_values, returns, mask)
        loss = pg_loss + float(cfg["value_coef"]) * v_loss

        # tokens where the clip actually zeroes the gradient
        affected = ((adv_norm > 0) & (ratio > 1 + eps)) | ((adv_norm < 0) & (ratio < 1 - eps))
        log_ratio = new_logp.detach() - ro["old_logp"]
        approx_kl = masked_mean(torch.exp(log_ratio) - 1 - log_ratio, mask)  # k3 estimator, old -> new

        b["policy_optimizer"].zero_grad(set_to_none=True)
        b["value_optimizer"].zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(b["policy_optimizer"])
        scaler.unscale_(b["value_optimizer"])
        max_norm = float(cfg["max_grad_norm"])
        gn_pol = torch.nn.utils.clip_grad_norm_(trainable_parameters(policy), max_norm)
        gn_val = torch.nn.utils.clip_grad_norm_(trainable_parameters(value_model), max_norm)
        scale_before = scaler.get_scale()
        scaler.step(b["policy_optimizer"])
        scaler.step(b["value_optimizer"])
        scaler.update()

        rv = ratio[valid]
        per_epoch.append({
            "policy_loss": float(pg_loss.detach()), "value_loss": float(v_loss.detach()),
            "clip_fraction": float(clip_frac), "affected_fraction": float(masked_mean(affected.float(), mask)),
            "approx_kl_old_new": float(approx_kl), "ratio_max": float(rv.max()), "ratio_min": float(rv.min()),
            "grad_norm_policy": float(gn_pol), "grad_norm_value": float(gn_val),
            "skipped_overflow": bool(scaler.get_scale() < scale_before),
        })

    with torch.no_grad():
        new_logp, _ = response_token_logprobs(policy, ro["seq"], ro["attn"], ro["pw"], ro["rids"])
        lr_final = new_logp - ro["old_logp"]
        final_kl = float(masked_mean(torch.exp(lr_final) - 1 - lr_final, mask))
        final_clip = float(masked_mean(((torch.exp(lr_final) < 1 - eps) | (torch.exp(lr_final) > 1 + eps)).float(), mask))

    last = per_epoch[-1]
    return {
        # loss/clip numbers come from the last epoch, the first one always has ratio == 1
        **{k: last[k] for k in ("policy_loss", "value_loss", "clip_fraction", "affected_fraction",
                                "approx_kl_old_new", "ratio_max", "ratio_min")},
        "grad_norm_policy": float(np.mean([e["grad_norm_policy"] for e in per_epoch])),
        "grad_norm_value": float(np.mean([e["grad_norm_value"] for e in per_epoch])),
        "skipped_overflow_steps": sum(e["skipped_overflow"] for e in per_epoch),
        "post_update_approx_kl": final_kl,
        "post_update_clip_fraction": final_clip,
        "explained_variance": explained_var,
        "value_first_token": float(ro["old_values"][:, 0].mean()),
        "return_first_token": float(returns[:, 0].mean()),
        "adv_raw_mean": float(adv[valid].mean()),
        "per_epoch": per_epoch,
    }


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None,
            clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard",
            prompt_offset: int = 0):
    b = prepare_ppo_continuation(config_path)
    cfg = b["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    res = repo_path(cfg["results_dir"]) / run_name
    res.mkdir(parents=True, exist_ok=True)
    log_path = res / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    n_updates, ppu = int(cfg["updates"]), int(cfg["prompts_per_update"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    pool = b["prompt_pool"]
    print(f"[{run_name}] {n_updates} updates x {ppu} prompt(s), eps={eps}, kl_beta={beta}, "
          f"prompt pool={len(pool)} (dropped {len(b['dropped_prompts'])} over {cfg['max_prompt_length']} tokens)", flush=True)

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
        ro = collect_rollout(b, items)
        stats = ppo_update(b, ro, eps, beta, scaler)
        mask = ro["mask"]
        lengths = mask.sum(-1)
        gen_tokens += int(lengths.sum())
        kl_tok = masked_mean(ro["old_logp"] - ro["ref_logp"], mask)
        rec = {
            "update": u + 1,
            "prompt_ids": [it["id"] for it in items],
            "reward": float(ro["scores"].mean()),
            "reward_with_eos_penalty": float(ro["task_reward"].mean()),
            "kl_ref_token_mean": float(kl_tok),
            "kl_ref_seq_sum": float(((ro["old_logp"] - ro["ref_logp"]) * mask).sum(-1).mean()),
            "entropy": float(ro["entropy"].mean()),
            "response_length": float(lengths.float().mean()),
            "terminated_with_eos": float(np.mean(ro["g"]["terminated_with_eos"])),
            **{k: v for k, v in stats.items() if k != "per_epoch"},
            "per_epoch": stats["per_epoch"],
            "generated_tokens_so_far": gen_tokens,
            "elapsed_s": round(elapsed(), 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0,
        }
        append_jsonl(log_path, rec)
        for it, resp, sc in zip(items, ro["g"]["responses"], ro["scores"].tolist()):
            rollouts.append({"update": u + 1, "id": it["id"], "prompt": it["messages"][-1]["content"],
                             "response": resp, "reward": sc})
        print(f"[{run_name}] update {u + 1}/{n_updates} reward={rec['reward']:.3f} kl={rec['kl_ref_token_mean']:.4f} "
              f"pg={rec['policy_loss']:.4f} vf={rec['value_loss']:.4f} H={rec['entropy']:.3f} "
              f"clip={rec['clip_fraction']:.3f} len={rec['response_length']:.0f} t={rec['elapsed_s']:.0f}s", flush=True)
        del ro

    b["policy"].save_pretrained(str(out))
    b["value_model"].save_pretrained(str(out.parent / f"{out.name}_value"))
    write_jsonl(res / "train_rollouts.jsonl", rollouts)
    meta = {
        "run_name": run_name, "updates": n_updates, "prompts_per_update": ppu, "ppo_epochs": int(cfg["ppo_epochs"]),
        "clip_epsilon": eps, "kl_beta": beta, "policy_learning_rate": float(cfg["policy_learning_rate"]),
        "value_lora_learning_rate": float(cfg["value_lora_learning_rate"]),
        "value_head_learning_rate": float(cfg["value_head_learning_rate"]), "value_train_mode": cfg.get("value_train_mode"),
        "gamma": float(cfg["gamma"]), "gae_lambda": float(cfg["gae_lambda"]), "value_coef": float(cfg["value_coef"]),
        "missing_eos_penalty": float(cfg.get("missing_eos_penalty", 0.0)), "max_response_length": int(cfg["max_response_length"]),
        "seed": int(cfg["seed"]), "prompt_offset": prompt_offset, "prompt_ids": used_ids,
        "generated_tokens": gen_tokens, "wall_clock_s": round(elapsed(), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "start_policy": cfg["paths"]["ppo_midpoint_policy"], "start_value": cfg["paths"]["ppo_midpoint_value"],
        "adapter_path": str(out),
    }
    save_json(res / "train_meta.json", meta)
    print(f"[{run_name}] done in {meta['wall_clock_s']:.0f}s, peak VRAM {meta['peak_vram_gb']} GB, saved {out}", flush=True)
    clear_gpu()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--prompt-offset", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = args.output or load_yaml(args.config)["output"]
    if adapter_done(out) and not args.force:
        print(f"[{args.run_name}] adapter already at {out} - skipping (use --force to rerun)")
        return
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name, args.prompt_offset)


if __name__ == "__main__":
    main()
