from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path, write_jsonl
from common.logging_utils import save_json
from task5_feedback.rlvr import exact_reward, extract_designated_final
from task5_feedback.rlaif import PairwiseAIJudge

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(path)
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


ORDER = ["clean_correct", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final",
         "persuasive_filler_correct", "gold_distractor_wrong_final"]
# each controlled pair is clean_correct (the diagnostically better response) against one perturbation
PAIRS = {
    "corrupt_reasoning_correct_final": "reasoning",   # same correct final, worse reasoning
    "good_reasoning_wrong_final": "outcome",           # reasoning kept, final changed to wrong
    "persuasive_filler_correct": "filler",             # same correct final plus irrelevant persuasion
    "gold_distractor_wrong_final": "outcome",          # gold number mentioned, wrong designated final
}


def _first(row, keys, default=None):
    for k in keys:
        if row.get(k) not in (None, ""):
            return row[k]
    return default


def response_of(row):
    text = _first(row, ("response", "text", "completion", "solution", "candidate"))
    if text is None:
        raise KeyError(f"no response text field in diagnostic row {sorted(row)}")
    return str(text)


def problem_of(row):
    q = _first(row, ("question", "problem", "prompt"))
    if isinstance(q, list):
        q = q[-1]["content"]
    if q is None and isinstance(row.get("messages"), list):
        q = row["messages"][-1]["content"]
    return str(q)


def gold_of(row):
    gold = _first(row, ("gold_final", "gold", "answer_final", "gold_answer"))
    if gold is None:
        raise KeyError(f"no gold answer field in diagnostic row {sorted(row)}")
    return str(gold)


def rate(prefs):
    prefs = list(prefs)
    n = max(len(prefs), 1)
    return {"better": sum(p > 0 for p in prefs) / n, "tie": sum(p == 0 for p in prefs) / n,
            "wrong": sum(p < 0 for p in prefs) / n, "n": len(prefs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print("Diagnostic problems:", len(groups))
    print("Variants/problem:", sorted(EXPECTED_VARIANTS))
    od = repo_path(cfg["results_dir"]) / "task5_feedback"
    od.mkdir(parents=True, exist_ok=True)
    judge = PairwiseAIJudge(cfg, od / "pairwise_judge_cache.json")

    per_response, per_pair, verifier_mismatches, checked = [], [], [], 0
    for pid, variants in sorted(groups.items()):
        clean = variants["clean_correct"]
        problem, gold = problem_of(clean), gold_of(clean)
        texts = [response_of(variants[v]) for v in ORDER]
        # RLVR: exact designated-final check. RLAIF: the training-time group reward, i.e. the
        # normalised pairwise win rate of each response against the other four (K = 5)
        rlvr = [exact_reward(t, gold) for t in texts]
        # the file says what the exact verifier should give each variant; check we reproduce it
        for v, r in zip(ORDER, rlvr):
            exp = variants[v].get("expected_exact_reward")
            checked += exp is not None
            if exp is not None and float(exp) != r:
                verifier_mismatches.append({"problem_id": pid, "variant": v, "expected": float(exp), "got": r})
        rlaif = judge.group_rewards(problem, texts)
        for v, t, r1, r2 in zip(ORDER, texts, rlvr, rlaif):
            per_response.append({"problem_id": pid, "variant": v, "rlvr_reward": r1, "rlaif_group_reward": r2,
                                 "words": len(t.split()), "parsed_final": extract_designated_final(t), "gold_final": gold})
        for v, kind in PAIRS.items():
            j = ORDER.index(v)
            direct = judge.compare(problem, texts[0], texts[j])  # cached by group_rewards, clean is A
            per_pair.append({
                "problem_id": pid, "perturbation": v, "kind": kind,
                "rlvr_pref": int(rlvr[0] > rlvr[j]) - int(rlvr[0] < rlvr[j]),
                "rlaif_pairwise_pref": {"A": 1, "TIE": 0, "B": -1}[direct],
                "rlaif_group_pref": int(rlaif[0] > rlaif[j]) - int(rlaif[0] < rlaif[j]),
                "rlaif_judge_output": direct,
            })
    write_jsonl(od / "diagnostic_scores.jsonl", per_response)
    write_jsonl(od / "diagnostic_pairs.jsonl", per_pair)

    out = {"num_problems": len(groups), "pairs_per_problem": len(PAIRS),
           "better_response_in_every_pair": "clean_correct",
           "rlaif_group_reward_definition": "(wins + 0.5 ties) / (K - 1) over the 5 variants of a problem",
           "per_perturbation": {}, "summary": {}}
    for v in PAIRS:
        rows = [r for r in per_pair if r["perturbation"] == v]
        out["per_perturbation"][v] = {m: rate(r[m] for r in rows) for m in ("rlvr_pref", "rlaif_pairwise_pref", "rlaif_group_pref")}
    # S_outcome: reasoning kept, designated final changed (good_reasoning_wrong_final). The pooled
    # version also counts the gold-distractor pair, which is a wrong final as well
    for m in ("rlvr_pref", "rlaif_pairwise_pref", "rlaif_group_pref"):
        reason = [r[m] for r in per_pair if r["perturbation"] == "corrupt_reasoning_correct_final"]
        outcome = [r[m] for r in per_pair if r["perturbation"] == "good_reasoning_wrong_final"]
        pooled = [r[m] for r in per_pair if r["kind"] == "outcome"]
        out["summary"][m] = {"S_reason": rate(reason)["better"], "S_reason_detail": rate(reason),
                             "S_outcome": rate(outcome)["better"], "S_outcome_detail": rate(outcome),
                             "S_outcome_pooled_with_distractor": rate(pooled)["better"],
                             "S_outcome_pooled_detail": rate(pooled),
                             "filler_detail": rate(r[m] for r in per_pair if r["kind"] == "filler")}
    out["verifier_rows_checked_against_expected"] = checked
    out["verifier_matches_expected_exact_reward"] = not verifier_mismatches
    out["verifier_mismatches"] = verifier_mismatches
    if verifier_mismatches:
        print(f"  WARNING: exact verifier disagrees with expected_exact_reward on {len(verifier_mismatches)} responses")
    elif checked:
        print(f"  exact verifier reproduces expected_exact_reward on all {checked} checked responses")
    means = {v: {"rlvr_reward": float(np.mean([r["rlvr_reward"] for r in per_response if r["variant"] == v])),
                 "rlaif_group_reward": float(np.mean([r["rlaif_group_reward"] for r in per_response if r["variant"] == v])),
                 "mean_words": float(np.mean([r["words"] for r in per_response if r["variant"] == v]))} for v in ORDER}
    out["mean_reward_by_variant"] = means
    save_json(od / "diagnostic_summary.json", out)
    for m, s in out["summary"].items():
        print(f"  {m:20s} S_reason={s['S_reason']:.2f} (tie {s['S_reason_detail']['tie']:.2f}) "
              f"S_outcome={s['S_outcome']:.2f} (tie {s['S_outcome_detail']['tie']:.2f}) "
              f"filler: better {s['filler_detail']['better']:.2f} tie {s['filler_detail']['tie']:.2f} wrong {s['filler_detail']['wrong']:.2f}")
    print("  wrote", od / "diagnostic_summary.json")


if __name__ == "__main__":
    main()
