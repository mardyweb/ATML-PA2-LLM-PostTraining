"""What does the fixed pairwise judge actually write before it is parsed?

The supplied judge turns anything it can't read into TIE, so a TIE can be a real tie or a format
failure. This re-runs the exact prompt and A/B orientation of PairwiseAIJudge.compare (greedy, so the
result should match the cached one) and keeps the raw text. The judge and its cache are not changed.
"""
from __future__ import annotations

import argparse
import itertools
import re
from collections import Counter

import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from task5_feedback.rlaif import PAIRWISE_RUBRIC, PairwiseAIJudge
from task5_feedback.score_perturbations import ORDER, load_diagnostic_groups, problem_of, response_of

PARSE = re.compile(r"\b(A|B|TIE)\b")


@torch.no_grad()
def raw_compare(judge, problem, a, b):
    key = judge._key(problem, a, b)
    swap = int(key[:8], 16) % 2 == 1
    aa, bb = (b, a) if swap else (a, b)
    text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
    ids = judge.tokenizer.apply_chat_template([{"role": "user", "content": text}], return_tensors="pt",
                                              add_generation_prompt=True).to(next(judge.model.parameters()).device)
    out = judge.model.generate(ids, max_new_tokens=4, do_sample=False, pad_token_id=judge.tokenizer.eos_token_id,
                               eos_token_id=judge.tokenizer.eos_token_id)
    return key, swap, judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def comparisons(cfg, od, sources):
    if "diagnostics" in sources:
        for pid, variants in sorted(load_diagnostic_groups(cfg["paths"]["task5_diagnostics"]).items()):
            problem = problem_of(variants["clean_correct"])
            # the same 10 pairs the group reward uses (first response of each pair shown as A before balancing)
            for i, j in itertools.combinations(range(len(ORDER)), 2):
                yield {"source": "diagnostics", "item": pid, "a_name": ORDER[i], "b_name": ORDER[j],
                       "problem": problem, "a": response_of(variants[ORDER[i]]), "b": response_of(variants[ORDER[j]])}
    for ds in ("gsm", "transfer"):
        if ds not in sources or not (od / f"{ds}_sft.jsonl").exists():
            continue
        sft = read_jsonl(od / f"{ds}_sft.jsonl")
        for name in ("rlvr", "rlaif"):
            path = od / f"{ds}_{name}.jsonl"
            if not path.exists():
                continue
            for a, b in zip(read_jsonl(path), sft):
                if a["response"] != b["response"]:  # identical pairs never reach the judge
                    yield {"source": f"{ds}_{name}_vs_sft", "item": a["id"], "a_name": name, "b_name": "sft",
                           "problem": a["question"], "a": a["response"], "b": b["response"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--sources", nargs="*", default=["diagnostics", "gsm", "transfer"])
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    od = repo_path(cfg["results_dir"]) / "task5_feedback"
    judge = PairwiseAIJudge(cfg, od / "pairwise_judge_cache.json")

    rows = []
    for n, c in enumerate(comparisons(cfg, od, args.sources), 1):
        key, swap, raw = raw_compare(judge, c["problem"], c["a"], c["b"])
        m = PARSE.search(raw.strip().upper())
        shown = m.group(1) if m else None  # label in the order the judge saw the two responses
        result = shown or "TIE"
        if swap:
            result = {"A": "B", "B": "A", "TIE": "TIE"}[result]
        cached = judge.cache.get(key)
        rows.append({"source": c["source"], "item": c["item"], "a_name": c["a_name"], "b_name": c["b_name"],
                     "raw_output": raw, "exact_single_label": raw.strip().upper() in ("A", "B", "TIE"),
                     "parse_failure": m is None, "shown_label": shown, "swapped": swap, "result": result,
                     "cached_result": cached, "matches_cache": cached is None or cached == result,
                     "picked_first_shown": None if shown not in ("A", "B") else shown == "A"})
        if n % 100 == 0:
            print(f"  {n} comparisons", flush=True)
    write_jsonl(od / "judge_raw_outputs.jsonl", rows)

    df = pd.DataFrame(rows)
    table = []
    for src, g in df.groupby("source", sort=False):
        ties = g[g.result == "TIE"]
        dec = g[g.picked_first_shown.notna()]
        table.append({"source": src, "comparisons": len(g),
                      "judge_said_A_or_B": float(g.shown_label.isin(["A", "B"]).mean()),
                      "judge_said_TIE": float((g.shown_label == "TIE").mean()),
                      "unparseable_counted_as_TIE": float(g.parse_failure.mean()),
                      "share_of_TIEs_that_are_parse_failures": float(ties.parse_failure.mean()) if len(ties) else float("nan"),
                      "exact_single_label_output": float(g.exact_single_label.mean()),
                      "decisive_picks_first_shown": float(dec.picked_first_shown.astype(bool).mean()) if len(dec) else float("nan"),
                      "matches_cached_result": float(g.matches_cache.mean())})
    t = pd.DataFrame(table)
    t.to_csv(od / "judge_format_check.csv", index=False)
    lines = ["Raw output of the fixed pairwise judge (same prompt, orientation and greedy decoding as "
             "PairwiseAIJudge.compare). The supplied parser maps unreadable output to TIE.\n",
             "| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
    for src, g in df.groupby("source", sort=False):
        common = Counter(repr(x.strip()) for x in g.raw_output).most_common(8)
        lines.append(f"\nMost common raw outputs, {src}: " + ", ".join(f"{k} x{v}" for k, v in common))
    (od / "judge_format_check.md").write_text("\n".join(lines) + "\n")
    print(t.to_string(index=False))
    print("  wrote", od / "judge_format_check.md")


if __name__ == "__main__":
    main()
