from __future__ import annotations

import argparse
import math
import shutil

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_tokenizer, reference_mode, trainable_parameters
from task1_dpo.dpo import dpo_loss


def row_id(row: dict, index: int):
    for key in ("prompt_id", "pair_id", "id", "example_id", "source_index"):
        if key in row and row[key] is not None:
            return row[key]
    return index


def adapter_done(path) -> bool:
    # done = adapter saved and no leftover resume state
    path = repo_path(path)
    return (path / "adapter_config.json").exists() and not (path / "_resume").exists()


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def filter_encodable(rows, tokenizer, max_length):
    # skip pairs where the prompt alone is already too long (encode_prompt_response would throw)
    kept, dropped = [], []
    for i, row in enumerate(rows):
        n = len(tokenizer.apply_chat_template(
            prompt_messages_from_preference(row), tokenize=True, add_generation_prompt=True
        ))
        (kept if n < max_length - 1 else dropped).append(i)
    return kept, dropped


def concat_pair_batches(chosen: dict, rejected: dict, pad_id: int) -> dict:
    # stack chosen + rejected into one batch so each model call is a single forward pass
    width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])

    def lpad(t, value):
        pad = width - t.shape[1]
        if pad == 0:
            return t
        filler = torch.full((t.shape[0], pad), value, dtype=t.dtype)
        return torch.cat([filler, t], dim=1)

    return {
        "input_ids": torch.cat([lpad(chosen["input_ids"], pad_id), lpad(rejected["input_ids"], pad_id)]),
        "attention_mask": torch.cat([lpad(chosen["attention_mask"], 0), lpad(rejected["attention_mask"], 0)]),
        "response_mask": torch.cat([lpad(chosen["response_mask"], 0), lpad(rejected["response_mask"], 0)]),
    }


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    seed = int(cfg["seed"])
    set_seed(seed)
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    all_rows = read_jsonl(path)
    if max_examples is not None:
        all_rows = all_rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    max_len = int(cfg["max_sequence_length"])
    kept_idx, dropped_idx = filter_encodable(all_rows, tokenizer, max_len)
    rows = [all_rows[i] for i in kept_idx]

    model = load_policy(cfg, trainable=True, fresh_lora=True)
    # GradScaler needs the LoRA weights in fp32
    for p in trainable_parameters(model):
        if p.dtype != torch.float32:
            p.data = p.data.float()

    # own generator so the shuffle order only depends on the seed (needed for resuming)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        generator=generator,
        collate_fn=make_collate(tokenizer, max_len),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "row_ids": [row_id(all_rows[i], i) for i in kept_idx],
        "dropped_ids": [row_id(all_rows[i], i) for i in dropped_idx],
        "dataset_path": str(path),
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "generator": generator,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def _save_resume_state(resume_dir, model, optimizer, scaler, opt_step, micro, epoch):
    resume_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "opt_step": opt_step,
        "micro": micro,
        "epoch": epoch,
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    tmp = resume_dir / "state.pt.tmp"
    torch.save(state, tmp)
    tmp.replace(resume_dir / "state.pt")


def run_training(
    config_path: str,
    run_name: str,
    dataset_path: str | None = None,
    output_path: str | None = None,
    beta: float | None = None,
    max_examples: int | None = None,
    resume: bool = True,
    save_every: int = 20,
    log_every: int = 1,
):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    model, optimizer, loader, tokenizer = bundle["model"], bundle["optimizer"], bundle["loader"], bundle["tokenizer"]
    beta = bundle["beta"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo")) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "train_log.jsonl"
    resume_dir = output / "_resume"

    device = next(model.parameters()).device
    accum = int(cfg["grad_accum_steps"])
    epochs = int(cfg.get("epochs", 1))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
    micro_per_epoch = len(loader)
    total_micro = micro_per_epoch * epochs
    total_steps = math.ceil(total_micro / accum)

    use_scaler = device.type == "cuda" and next(model.parameters()).dtype == torch.float16
    # base model is fp16 so scale the loss; growth is off, the scale only drops on overflow
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler, init_scale=1024.0, growth_interval=10**9)

    opt_step, start_micro, resumed_step = 0, 0, None
    if resume and (resume_dir / "state.pt").exists():
        state = torch.load(resume_dir / "state.pt", map_location="cpu", weights_only=False)
        named = dict(model.named_parameters())
        with torch.no_grad():
            for n, t in state["trainable"].items():
                named[n].copy_(t.to(named[n].device, dtype=named[n].dtype))
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        opt_step, start_micro = int(state["opt_step"]), int(state["micro"])
        resumed_step = opt_step
        del named
        if log_path.exists():  # throw away log lines from after the last save
            import json
            keep = [l for l in log_path.read_text(encoding="utf-8").splitlines()
                    if l.strip() and json.loads(l)["step"] <= opt_step]
            log_path.write_text("".join(l + "\n" for l in keep), encoding="utf-8")
        print(f"Resuming {run_name} from optimizer step {opt_step} (micro-batch {start_micro}/{total_micro}).")
    elif log_path.exists():
        log_path.unlink()  # fresh run, don't append to an old log

    print(f"[{run_name}] pairs={len(bundle['rows'])} (dropped {len(bundle['dropped_ids'])} over-long prompts), "
          f"beta={beta}, micro-batches={total_micro}, optimizer steps={total_steps}")

    save_json(results_dir / "train_data_ids.json", {
        "dataset": bundle["dataset_path"],
        "used_ids": bundle["row_ids"],
        "dropped_ids_overlong_prompt": bundle["dropped_ids"],
    })

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    group = []
    micro = 0
    restored_rng = start_micro == 0

    for epoch in range(epochs):
        bundle["generator"].manual_seed(int(cfg["seed"]) + epoch)
        for chosen, rejected in loader:
            if micro < start_micro:
                micro += 1
                continue
            if not restored_rng:
                torch.set_rng_state(state["cpu_rng"])
                if state.get("cuda_rng") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(state["cuda_rng"])
                restored_rng = True

            n = chosen["input_ids"].shape[0]
            batch = concat_pair_batches(chosen, rejected, tokenizer.pad_token_id)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad(), reference_mode(model):
                ref_logp, _, _ = response_sequence_logprobs(model, batch)
            policy_logp, _, _ = response_sequence_logprobs(model, batch)

            loss, stats = dpo_loss(policy_logp[:n], policy_logp[n:], ref_logp[:n], ref_logp[n:], beta)

            group_start = (micro // accum) * accum
            group_size = min(accum, total_micro - group_start)
            scaler.scale(loss / group_size).backward()
            group.append({"loss": float(loss.detach()), **{k: float(v) for k, v in stats.items()},
                          "resp_tokens_chosen": float(batch["response_mask"][:n].sum(-1).mean()),
                          "resp_tokens_rejected": float(batch["response_mask"][n:].sum(-1).mean())})
            micro += 1

            if micro - group_start == group_size:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = use_scaler and scaler.get_scale() < scale_before
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                rec = {k: sum(g[k] for g in group) / len(group) for k in group[0]}
                rec.update({
                    "step": opt_step,
                    "epoch": epoch,
                    "pairs_seen": micro * int(cfg["batch_size"]),
                    "grad_norm": float(grad_norm),
                    "skipped_overflow": bool(skipped),
                    "lr": optimizer.param_groups[0]["lr"],
                    "beta": beta,
                    "elapsed_s": round(elapsed(), 1),
                    "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0,
                })
                append_jsonl(log_path, rec)
                group = []
                if opt_step % log_every == 0 or opt_step == total_steps:
                    print(f"[{run_name}] step {opt_step}/{total_steps} loss={rec['loss']:.4f} "
                          f"acc={rec['preference_accuracy']:.3f} margin={rec['logit_mean']:.3f} "
                          f"gnorm={rec['grad_norm']:.3f} t={rec['elapsed_s']:.0f}s", flush=True)
                if save_every and opt_step % save_every == 0 and opt_step < total_steps:
                    _save_resume_state(resume_dir, model, optimizer, scaler, opt_step, micro, epoch)

    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    meta = {
        "run_name": run_name,
        "dataset": bundle["dataset_path"],
        "num_pairs": len(bundle["rows"]),
        "num_dropped_overlong": len(bundle["dropped_ids"]),
        "epochs": epochs,
        "beta": beta,
        "batch_size": int(cfg["batch_size"]),
        "grad_accum_steps": accum,
        "effective_batch_pairs": int(cfg["batch_size"]) * accum,
        "optimizer_steps": opt_step,
        "learning_rate": float(cfg["learning_rate"]),
        "lr_schedule": "constant",
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "max_sequence_length": int(cfg["max_sequence_length"]),
        "max_grad_norm": max_grad_norm,
        "lora": cfg["lora"],
        "seed": int(cfg["seed"]),
        "dtype": cfg.get("dtype"),
        "fp16_grad_scaler": use_scaler,
        "wall_clock_s_this_session": round(elapsed(), 1),
        "peak_vram_gb_this_session": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else 0.0,
        "resumed_from_step": resumed_step,
        "adapter_path": str(output),
    }
    save_json(results_dir / "train_meta.json", meta)
    if resume_dir.exists():
        shutil.rmtree(resume_dir, ignore_errors=True)
    print(f"[{run_name}] saved adapter to {output}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--force", action="store_true", help="retrain even if a finished adapter exists")
    args = ap.parse_args()
    out = args.output or load_yaml(args.config)["standard_output"]
    if adapter_done(out) and not args.force:
        print(f"[{args.run_name}] finished adapter already at {out} - skipping (use --force to retrain)")
        return
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples,
                 resume=not args.no_resume, save_every=args.save_every, log_every=args.log_every)


if __name__ == "__main__":
    main()
