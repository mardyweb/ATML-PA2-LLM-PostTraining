from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import batch_generate
from common.logging_utils import load_json, save_json
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    specs = policy_specs(cfg)
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=specs[name], trainable=False)


def out_dir(cfg):
    return repo_path(cfg["results_dir"]) / "task5_feedback"


def problem_text(row):
    if row.get("question"):
        return str(row["question"])
    return prompt_messages(row)[-1]["content"]


def gold_of(row):
    if row.get("gold_final") is not None:
        return str(row["gold_final"])
    return extract_designated_final(row.get("answer", ""))


def generate(cfg, name, rows, tok, batch_size):
    model = load_frozen_policy(cfg, name)
    max_new = int(cfg["math_max_new_tokens"])
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        g = batch_generate(model, tok, [prompt_messages(r) for r in chunk], max_prompt_length=1024,
                           max_new_tokens=max_new, do_sample=False)
        for k, r in enumerate(chunk):
            resp = g["responses"][k]
            pred = extract_designated_final(resp)
            out.append({
                "id": r.get("prompt_id", r.get("source_index", i + k)), "index": i + k,
                "question": problem_text(r), "gold_final": gold_of(r), "policy": name,
                "response": resp, "response_tokens": g["response_lengths"][k], "truncated": g["truncated"][k],
                "pred_final": pred, "format_ok": pred is not None, "correct": exact_reward(resp, gold_of(r)),
                **({"svamp_type": r["svamp_type"]} if "svamp_type" in r else {}),
            })
        print(f"  [{name}] {min(i + batch_size, len(rows))}/{len(rows)}", flush=True)
    clear_gpu(model)
    del model
    clear_gpu()
    return out


def failure_type(r):
    if r["correct"]:
        return "correct"
    if not r["format_ok"]:
        return "no_final_truncated" if r["truncated"] else "no_final"
    return "wrong_final"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg, rows, tok = load_math_evaluation(args.config, args.dataset)
    print("Rows:", len(rows))
    print("Policies:", list(policy_specs(cfg)))
    od = out_dir(cfg)
    od.mkdir(parents=True, exist_ok=True)

    # 1) one greedy response per policy and problem, scored by the exact verifier
    gens = {}
    for name in policy_specs(cfg):
        path = od / f"{args.dataset}_{name}.jsonl"
        if path.exists() and not args.force:
            gens[name] = read_jsonl(path)
            print(f"[{name}] responses already saved - skipping generation")
            continue
        print(f"[{name}] generating {len(rows)} greedy responses (max {cfg['math_max_new_tokens']} new tokens)", flush=True)
        gens[name] = generate(cfg, name, rows, tok, args.batch_size)
        write_jsonl(path, gens[name])

    # 2) fixed pairwise AI judge: each RL policy against SFT on the same problem
    judge = PairwiseAIJudge(cfg, od / "pairwise_judge_cache.json")
    sft = gens["sft"]
    pair_rows, timings = [], []
    for name in ("rlvr", "rlaif"):
        for a, b in zip(gens[name], sft):
            if a["response"] == b["response"]:
                pref = "TIE"  # identical texts tie by definition; counted separately below
            else:
                t0 = time.perf_counter()
                pref = judge.compare(a["question"], a["response"], b["response"])
                timings.append(time.perf_counter() - t0)
            # verifier's view of the same pair: +1 policy better, -1 SFT better, 0 tie
            v = int(a["correct"] > b["correct"]) - int(a["correct"] < b["correct"])
            j = {"A": 1, "B": -1, "TIE": 0}[pref]
            pair_rows.append({"dataset": args.dataset, "policy": name, "id": a["id"], "judge": pref,
                              "win_score": {"A": 1.0, "TIE": 0.5, "B": 0.0}[pref],
                              "policy_correct": a["correct"], "sft_correct": b["correct"],
                              "verifier_pref": v, "judge_pref": j, "identical_responses": a["response"] == b["response"]})
        print(f"[{name}] judged against SFT", flush=True)
    write_jsonl(od / f"{args.dataset}_pairwise_vs_sft.jsonl", pair_rows)

    # 3) summary
    pr = pd.DataFrame(pair_rows)
    new_calls = [t for t in timings if t > 1e-3]
    old = od / f"{args.dataset}_summary.json"
    if new_calls:
        judge_cost = float(np.mean(new_calls))
    else:  # everything came from the cache this time; keep the earlier measurement
        judge_cost = load_json(old).get("judge_mean_seconds_per_new_comparison", float("nan")) if old.exists() else float("nan")
    summary = {"dataset": args.dataset, "num_problems": len(rows), "decoding": "greedy",
               "max_new_tokens": int(cfg["math_max_new_tokens"]), "generation_batch_size": args.batch_size,
               "policies": {}, "judge_mean_seconds_per_new_comparison": judge_cost}
    for name, g in gens.items():
        d = pd.DataFrame(g)
        s = {"exact_accuracy": float(d.correct.mean()), "format_compliance": float(d.format_ok.mean()),
             "mean_tokens": float(d.response_tokens.mean()), "std_tokens": float(d.response_tokens.std()),
             "median_tokens": float(d.response_tokens.median()), "truncation_rate": float(d.truncated.mean()),
             "failure_types": {k: int(v) for k, v in d.apply(failure_type, axis=1).value_counts().items()}}
        if name != "sft":
            p = pr[pr.policy == name]
            decisive = p[p.verifier_pref != 0]
            s.update({
                "win_rate_vs_sft": float(p.win_score.mean()),
                "wins": int((p.judge == "A").sum()), "losses": int((p.judge == "B").sum()),
                "judge_ties": int(((p.judge == "TIE") & ~p.identical_responses).sum()),
                "identical_to_sft_counted_as_tie": int(p.identical_responses.sum()),
                "verifier_judge_agreement": float((p.verifier_pref == p.judge_pref).mean()),
                "judge_picks_correct_when_verifier_decisive": float((decisive.judge_pref == decisive.verifier_pref).mean()) if len(decisive) else float("nan"),
                "num_verifier_decisive_pairs": int(len(decisive)),
            })
        summary["policies"][name] = s
        print(f"  {name}: accuracy={s['exact_accuracy']:.3f} format={s['format_compliance']:.3f} "
              f"len={s['mean_tokens']:.0f}" + (f" win_vs_sft={s['win_rate_vs_sft']:.3f} agree={s['verifier_judge_agreement']:.3f}" if name != "sft" else ""))
    save_json(od / f"{args.dataset}_summary.json", summary)
    print("  wrote", od / f"{args.dataset}_summary.json")


if __name__ == "__main__":
    main()
