"""Puts the in-domain, diagnostic and transfer results for Task 5 into tables, figures and a list of
candidate qualitative examples (results/task5_feedback/)."""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.data import load_yaml, read_jsonl, repo_path  # noqa: E402

POLICIES = ["sft", "rlvr", "rlaif"]
COLORS = {"sft": "#8a8985", "rlvr": "#2a78d6", "rlaif": "#eb6834"}
PREF_COLORS = {"better": "#1baf7a", "tie": "#c9c8c3", "wrong": "#e34948"}
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "axes.axisbelow": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "savefig.dpi": 200,
})


def _load(p):
    return json.loads(p.read_text()) if p.exists() else None


def _md(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                                      (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def _save(fig, d, stem):
    fig.tight_layout()
    fig.savefig(d / f"{stem}.png", bbox_inches="tight")
    fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", d / f"{stem}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    od = repo_path(cfg["results_dir"]) / "task5_feedback"
    figs = od / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    gsm, tr, diag = _load(od / "gsm_summary.json"), _load(od / "transfer_summary.json"), _load(od / "diagnostic_summary.json")

    # in-domain and transfer
    rows = []
    for name in POLICIES:
        r = {"policy": name.upper()}
        for tag, s, fname in (("gsm", gsm, "gsm"), ("svamp", tr, "transfer")):
            if not s:
                continue
            p = s["policies"][name]
            r.update({f"{tag}_accuracy": p["exact_accuracy"], f"{tag}_format": p["format_compliance"],
                      f"{tag}_mean_tokens": p["mean_tokens"], f"{tag}_std_tokens": p["std_tokens"],
                      f"{tag}_truncation": p["truncation_rate"],
                      f"{tag}_win_vs_sft": p.get("win_rate_vs_sft", np.nan),
                      f"{tag}_judge_ties": p.get("judge_ties", np.nan),
                      f"{tag}_identical_to_sft": p.get("identical_to_sft_counted_as_tie", np.nan),
                      f"{tag}_verifier_judge_agree": p.get("verifier_judge_agreement", np.nan)})
            pr = [x for x in read_jsonl(od / f"{fname}_pairwise_vs_sft.jsonl") if x["policy"] == name] \
                if (od / f"{fname}_pairwise_vs_sft.jsonl").exists() else []
            ne = [x for x in pr if not x["identical_responses"]]
            if ne:
                # identical responses tie automatically, so also give the numbers without them
                r[f"{tag}_win_vs_sft_nonidentical"] = float(np.mean([x["win_score"] for x in ne]))
                r[f"{tag}_agree_nonidentical"] = float(np.mean([x["verifier_pref"] == x["judge_pref"] for x in ne]))
        if gsm and tr:
            r["accuracy_drop"] = r["gsm_accuracy"] - r["svamp_accuracy"]
            r["win_rate_change"] = r["svamp_win_vs_sft"] - r["gsm_win_vs_sft"] if name != "sft" else np.nan
        rows.append(r)
    t = pd.DataFrame(rows)
    if len(t.columns) > 1:
        t.to_csv(od / "table_task5_policies.csv", index=False)
        head = []
        if gsm:
            head.append(f"GSM8K eval: {gsm['num_problems']} problems; SVAMP transfer: {tr['num_problems'] if tr else '-'} problems. "
                        f"Greedy decoding, max {gsm['max_new_tokens']} new tokens. Win rate = AI judge vs the SFT response on "
                        "the same problem (win 1, tie 0.5, loss 0); identical responses count as ties. Agreement = verifier and "
                        "judge give the same preference sign for the policy-vs-SFT pair. "
                        f"Judge cost: about {gsm['judge_mean_seconds_per_new_comparison']:.2f} s per new comparison on the GPU.")
        fail = []
        for tag, s in (("gsm", gsm), ("svamp", tr)):
            if s:
                for name in POLICIES:
                    fail.append({"dataset": tag, "policy": name.upper(), **s["policies"][name]["failure_types"]})
        ft = pd.DataFrame(fail).fillna(0)
        for c in ft.columns:
            if c not in ("dataset", "policy"):
                ft[c] = ft[c].astype(int)
        # verifier preference vs judge preference on the policy-vs-SFT pairs
        cross = []
        for tag, fname in (("gsm", "gsm"), ("svamp", "transfer")):
            path = od / f"{fname}_pairwise_vs_sft.jsonl"
            if not path.exists():
                continue
            for name in ("rlvr", "rlaif"):
                pr = [x for x in read_jsonl(path) if x["policy"] == name and not x["identical_responses"]]
                for v in (1, 0, -1):
                    sub = [x for x in pr if x["verifier_pref"] == v]
                    cross.append({"dataset": tag, "policy": name.upper(),
                                  "verifier": {1: "policy correct, SFT wrong", 0: "same correctness", -1: "SFT correct, policy wrong"}[v],
                                  "n": len(sub), "judge_prefers_policy": sum(x["judge_pref"] == 1 for x in sub),
                                  "judge_tie": sum(x["judge_pref"] == 0 for x in sub),
                                  "judge_prefers_sft": sum(x["judge_pref"] == -1 for x in sub)})
        cr = pd.DataFrame(cross)
        (od / "table_task5_policies.md").write_text(
            "\n".join(head) + " Columns ending _nonidentical drop the pairs where the policy and SFT wrote the same text."
            "\n\n" + _md(t) + "\n\nFailure types (counts):\n\n" + _md(ft) +
            "\n\nVerifier vs judge on non-identical policy-vs-SFT pairs (counts):\n\n" + (_md(cr) if len(cr) else "") + "\n")
        print("  wrote", od / "table_task5_policies.md")

        fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
        x = np.arange(2)
        w = 0.26
        for i, name in enumerate(POLICIES):
            r = t[t.policy == name.upper()].iloc[0]
            acc = [r.get("gsm_accuracy", np.nan), r.get("svamp_accuracy", np.nan)]
            fmt = [r.get("gsm_format", np.nan), r.get("svamp_format", np.nan)]
            for ax, vals in ((axes[0], acc), (axes[1], fmt)):
                bars = ax.bar(x + (i - 1) * w, vals, w * 0.92, color=COLORS[name], label=name.upper())
                for b_, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                                fontsize=7, color=INK2)
            if name != "sft":
                win = [r.get("gsm_win_vs_sft", np.nan), r.get("svamp_win_vs_sft", np.nan)]
                axes[2].bar(x + (i - 1.5) * w, win, w * 0.92, color=COLORS[name], label=name.upper())
        axes[2].axhline(0.5, color=INK2, linestyle="--", linewidth=1)
        for ax, title in zip(axes, ("Exact final-answer accuracy", "Format compliance (#### parsed)", "AI-judge win rate vs SFT")):
            ax.set_xticks(x)
            ax.set_xticklabels(["GSM8K (in-domain)", "SVAMP (transfer)"])
            ax.set_ylim(0, 1.05)
            ax.set_title(title)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
        _save(fig, figs, "fig_task5_policies")

    # controlled diagnostics
    if diag:
        names = {"rlvr_pref": "RLVR (exact verifier)", "rlaif_pairwise_pref": "RLAIF (direct pairwise)",
                 "rlaif_group_pref": "RLAIF (group win-rate reward)"}
        rows = []
        for v, d in diag["per_perturbation"].items():
            for m, lab in names.items():
                rows.append({"perturbation": v, "mechanism": lab, "better": d[m]["better"], "tie": d[m]["tie"],
                             "wrong": d[m]["wrong"], "n": d[m]["n"]})
        dt = pd.DataFrame(rows)
        dt.to_csv(od / "table_task5_diagnostics.csv", index=False)
        srows = [{"mechanism": lab, "S_reason": diag["summary"][m]["S_reason"],
                  "reason_tie": diag["summary"][m]["S_reason_detail"]["tie"],
                  "reason_wrong": diag["summary"][m]["S_reason_detail"]["wrong"],
                  "S_outcome": diag["summary"][m]["S_outcome"],
                  "outcome_tie": diag["summary"][m]["S_outcome_detail"]["tie"],
                  "outcome_wrong": diag["summary"][m]["S_outcome_detail"]["wrong"],
                  "S_outcome_incl_distractor": diag["summary"][m].get("S_outcome_pooled_with_distractor", np.nan)}
                 for m, lab in names.items()]
        st = pd.DataFrame(srows)
        vrows = []
        for v, vals in diag["mean_reward_by_variant"].items():
            r = {"variant": v, "mean_rlvr_reward": vals["rlvr_reward"], "mean_rlaif_group_reward": vals["rlaif_group_reward"],
                 "mean_words": vals["mean_words"]}
            if v in diag["per_perturbation"]:
                for m, short in (("rlvr_pref", "rlvr"), ("rlaif_pairwise_pref", "rlaif_pairwise")):
                    d = diag["per_perturbation"][v][m]
                    r[f"{short}_better/tie/wrong"] = f"{d['better']:.2f} / {d['tie']:.2f} / {d['wrong']:.2f}"
            else:
                r["rlvr_better/tie/wrong"] = r["rlaif_pairwise_better/tie/wrong"] = "reference (the better response)"
            vrows.append(r)
        mv = pd.DataFrame(vrows)
        (od / "table_task5_diagnostics.md").write_text(
            f"{diag['num_problems']} problems x 5 variants. Every controlled pair is clean_correct (the better response) "
            "against one perturbation; better / tie / wrong = how often each mechanism prefers clean_correct, ties, or "
            "prefers the perturbed response. S_reason uses the corrupted-reasoning pair; S_outcome uses the "
            "good-reasoning/wrong-final pair (reasoning kept, designated final changed), and S_outcome_incl_distractor also "
            "counts the gold-distractor pair. RLAIF (direct pairwise) is the supplied judge's decision on each controlled "
            "pair; RLAIF (group) ranks by the training-time group win rate over the 5 variants (K = 5)."
            + ("" if diag.get("verifier_matches_expected_exact_reward", True) else
               " WARNING: the verifier did not reproduce expected_exact_reward on some rows.")
            + "\n\n" + _md(st) + "\n\nAll five variants (clean_correct is the reference in every pair):\n\n" + _md(mv)
            + "\n\nPer perturbation, long form:\n\n" + _md(dt) + "\n")
        print("  wrote", od / "table_task5_diagnostics.md")

        perts = list(diag["per_perturbation"])
        mechs = list(names)
        fig, axes = plt.subplots(1, len(perts), figsize=(2.8 * len(perts), 2.8), sharey=True)
        for ax, v in zip(np.atleast_1d(axes), perts):
            bottom = np.zeros(len(mechs))
            for k in ("better", "tie", "wrong"):
                vals = np.array([diag["per_perturbation"][v][m][k] for m in mechs])
                ax.bar(range(len(mechs)), vals, bottom=bottom, color=PREF_COLORS[k], label=k, width=0.6,
                       edgecolor="white", linewidth=1)
                bottom += vals
            ax.set_xticks(range(len(mechs)))
            ax.set_xticklabels(["RLVR", "RLAIF\npairwise", "RLAIF\ngroup"], fontsize=8)
            ax.set_title(v.replace("_", " "), fontsize=8)
            ax.set_ylim(0, 1)
        h, l = np.atleast_1d(axes)[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
        fig.suptitle("Does each feedback source prefer clean_correct over the perturbed response?", y=1.02)
        _save(fig, figs, "fig_task5_diagnostics")

    # qualitative candidates
    out = ["# Task 5 qualitative candidates (auto-selected, read and pick yourself)\n"]
    pairs_path = od / "diagnostic_pairs.jsonl"
    if pairs_path.exists():
        pairs = read_jsonl(pairs_path)
        diag_rows = read_jsonl(cfg["paths"]["task5_diagnostics"])
        by = {(str(r["problem_id"]), r["variant_type"]): r for r in diag_rows}

        def txt(r):
            return str(r.get("response") or r.get("text") or r.get("completion") or r.get("solution") or "")

        for v, title in (("corrupt_reasoning_correct_final", "Reasoning-only corruption"),
                         ("good_reasoning_wrong_final", "Good reasoning, wrong designated final"),
                         ("persuasive_filler_correct", "Persuasive filler"),
                         ("gold_distractor_wrong_final", "Gold-number distractor with a wrong final")):
            dis = [p for p in pairs if p["perturbation"] == v and p["rlvr_pref"] != p["rlaif_pairwise_pref"]]
            out.append(f"## {title}: verifier and judge disagree on {len(dis)} of "
                       f"{sum(p['perturbation'] == v for p in pairs)} problems\n")
            for p in dis[:2]:
                c, x = by.get((p["problem_id"], "clean_correct")), by.get((p["problem_id"], v))
                if c is None or x is None:
                    continue
                out.append(f"- problem {p['problem_id']}: verifier pref {p['rlvr_pref']:+d}, judge pref "
                           f"{p['rlaif_pairwise_pref']:+d} (+1 = prefers clean)\n\n  clean: > {txt(c)[:350]}\n\n  {v}: > {txt(x)[:350]}\n")
    for tag in ("gsm", "transfer"):
        pv = od / f"{tag}_pairwise_vs_sft.jsonl"
        if pv.exists():
            pr = read_jsonl(pv)
            odd = [r for r in pr if r["verifier_pref"] != 0 and r["judge_pref"] == -r["verifier_pref"]]
            out.append(f"## {tag}: judge prefers the wrong answer over the right one in {len(odd)} policy-vs-SFT pairs\n")
            if odd:
                gens = {n: {r["id"]: r for r in read_jsonl(od / f"{tag}_{n}.jsonl")} for n in POLICIES}
                for r in odd[:2]:
                    a, b = gens[r["policy"]][r["id"]], gens["sft"][r["id"]]
                    out.append(f"- {r['policy'].upper()} vs SFT, problem {r['id']} (gold {a['gold_final']}): {r['policy'].upper()} "
                               f"final {a['pred_final']}, SFT final {b['pred_final']}, judge said {r['judge']}\n\n  "
                               f"{r['policy'].upper()}: > {a['response'][:300]}\n\n  SFT: > {b['response'][:300]}\n")
    (od / "qualitative_candidates.md").write_text("\n".join(out) + "\n")
    print("  wrote", od / "qualitative_candidates.md")


if __name__ == "__main__":
    main()
