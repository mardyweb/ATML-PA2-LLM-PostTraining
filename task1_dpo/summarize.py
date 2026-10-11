"""Builds the Task 1 tables, figures and a list of candidate qualitative examples from results/task1_dpo/."""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.data import load_yaml, repo_path  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#8a8985"
INK, INK2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "lines.linewidth": 2, "savefig.dpi": 200,
})


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def _jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def _save(fig, figdir, stem):
    fig.tight_layout()
    fig.savefig(figdir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(figdir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", figdir / f"{stem}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    figdir = res / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    betas = [float(b) for b in cfg["betas"]]

    runs = [("sft", "SFT / reference (no DPO)"), ("standard", "Standard DPO (1 epoch)")]
    runs += [(f"beta_{b:g}", f"Short fork beta={b:g}") for b in betas]
    runs += [("length_balanced", "Length-balanced DPO (1 epoch)")]
    S = {k: _load(res / k / "summary.json") for k, _ in runs}
    M = {k: _load(res / k / "train_meta.json") for k, _ in runs}
    missing = [k for k, _ in runs if S[k] is None]
    if missing:
        print("Not evaluated yet (skipped in tables):", ", ".join(missing))

    # main summary table
    rows = []
    for key, label in runs:
        s, m = S[key], M[key]
        if s is None:
            continue
        h, g = s["heldout_standard"], s.get("generation", {})
        b = s["beta"]
        rows.append({
            "condition": label,
            "train_pairs": m["num_pairs"] if m else 0,
            "optimizer_steps": m["optimizer_steps"] if m else 0,
            "beta": b if m else np.nan,
            "heldout_loss_own_beta": h.get(f"dpo_loss_beta{b:g}") if m else np.nan,
            f"heldout_loss_beta{float(cfg['beta']):g}": h.get(f"dpo_loss_beta{float(cfg['beta']):g}"),
            "heldout_pref_acc": h["preference_accuracy"] if m else np.nan,
            "heldout_margin_mean": h["margin_mean"],
            "kl_per_token": g.get("kl_token_mean"),
            "kl_per_seq": g.get("kl_seq_sum_mean"),
            "rm_mean": g.get("reward_mean"),
            "rm_std": g.get("reward_std"),
            "len_tok_mean": g.get("length_tokens", {}).get("mean"),
            "len_tok_std": g.get("length_tokens", {}).get("std"),
            "len_tok_median": g.get("length_tokens", {}).get("median"),
            "len_tok_iqr": g.get("length_tokens", {}).get("iqr"),
            "truncation_rate": g.get("truncation_rate"),
            "word_limit_compliance": s.get("word_limit", {}).get("compliance_rate"),
            "train_wall_clock_s": m.get("wall_clock_s_this_session") if m else np.nan,
            "train_peak_vram_gb": m.get("peak_vram_gb_this_session") if m else np.nan,
        })
    t1 = pd.DataFrame(rows)
    if len(t1):
        t1.to_csv(res / "table_dpo_summary.csv", index=False)
        (res / "table_dpo_summary.md").write_text(
            "Standard and length-balanced runs train for one full epoch; the beta forks train on the first "
            f"{cfg['short_ablation_examples']} pairs only (different budget). Preference accuracy is undefined "
            "for the SFT reference (margin is identically 0).\n\n" + _md(t1) + "\n")
        print("  wrote", res / "table_dpo_summary.csv")

    # per-stratum table
    strata_rows = []
    for key in ("sft", "standard", "length_balanced"):
        s = S.get(key)
        if s is None or key == "sft":
            continue
        for st, v in s["heldout_length_stratified"].get("strata", {}).items():
            strata_rows.append({"model": key, "stratum": st, "n": v["n"], "pref_acc": v["preference_accuracy"],
                                "margin_mean": v["margin_mean"], "mean_chosen_tokens": v["mean_chosen_tokens"],
                                "mean_rejected_tokens": v["mean_rejected_tokens"]})
    t2 = pd.DataFrame(strata_rows)
    if len(t2):
        t2.to_csv(res / "table_length_strata.csv", index=False)
        (res / "table_length_strata.md").write_text(_md(t2) + "\n")
        print("  wrote", res / "table_length_strata.csv")

    # beta sweep
    have_b = [b for b in betas if S.get(f"beta_{b:g}") and "generation" in S[f"beta_{b:g}"]]
    if have_b:
        panels = [
            ("Held-out preference accuracy", lambda s: s["heldout_standard"]["preference_accuracy"], None),
            ("KL from reference (per token)", lambda s: s["generation"]["kl_token_mean"], 0.0),
            ("Reward-model score (mean)", lambda s: s["generation"]["reward_mean"], "sft"),
            ("Generated length (tokens, mean)", lambda s: s["generation"]["length_tokens"]["mean"], "sft"),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(10, 2.4))
        for ax, (title, f, ref) in zip(axes, panels):
            ys = [f(S[f"beta_{b:g}"]) for b in have_b]
            ax.plot(have_b, ys, color=BLUE, marker="o", markersize=5, label=f"short forks ({cfg['short_ablation_examples']} pairs)")
            if S.get("standard") and "generation" in S["standard"]:
                ax.plot([float(S["standard"]["beta"])], [f(S["standard"])], marker="D", markersize=6,
                        color=ORANGE, linestyle="none", label="standard (1 epoch)")
            if ref == "sft" and S.get("sft") and "generation" in S["sft"]:
                ax.axhline(f(S["sft"]), color=GRAY, linestyle="--", linewidth=1, label="SFT reference")
            elif ref == 0.0:
                ax.axhline(0.0, color=GRAY, linestyle="--", linewidth=1)
            ax.set_xscale("log")
            ax.set_xticks(have_b)
            ax.set_xticklabels([f"{b:g}" for b in have_b])
            ax.minorticks_off()
            ax.set_xlabel(r"$\beta$")
            ax.set_title(title)
        h, l = axes[2].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.08))
        _save(fig, figdir, "fig_beta_tradeoff")

    # training curves
    curves = [(k, lab, c) for (k, lab), c in zip(
        [("standard", "standard (1 epoch)")] + [(f"beta_{b:g}", f"beta={b:g} fork") for b in betas],
        [ORANGE, BLUE, AQUA, YELLOW])]
    curves = [(k, lab, c, _jsonl(res / k / "train_log.jsonl")) for k, lab, c in curves]
    curves = [x for x in curves if x[3]]
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.6))
        def smooth(v, w=10):
            v = np.asarray(v, float)
            if len(v) < 3:
                return v
            w = max(1, min(w, len(v) // 3))
            return np.array([v[max(0, i - w + 1): i + 1].mean() for i in range(len(v))])

        for k, lab, c, log in curves:
            x = [r["pairs_seen"] for r in log]
            for ax, key in ((axes[0], "loss"), (axes[1], "preference_accuracy")):
                y = [r[key] for r in log]
                ax.plot(x, y, color=c, linewidth=0.8, alpha=0.25)
                ax.plot(x, smooth(y), color=c, label=lab, linewidth=1.8)
        axes[0].axhline(np.log(2), color=GRAY, linestyle="--", linewidth=1)
        axes[0].set_title("Training DPO loss (raw + running mean)")
        axes[1].set_title("Training preference accuracy (raw + running mean)")
        for ax in axes:
            ax.set_xlabel("training pairs seen")
        axes[0].legend(frameon=False)
        _save(fig, figdir, "fig_train_curves")

    # length study
    if len(t2):
        strata = list(dict.fromkeys(t2["stratum"]))
        models = [m for m in ("standard", "length_balanced") if m in set(t2["model"])]
        colors = {"standard": BLUE, "length_balanced": ORANGE}
        labels = {"standard": "standard DPO", "length_balanced": "length-balanced DPO"}
        fig, axes = plt.subplots(1, 3, figsize=(10, 2.7), gridspec_kw={"width_ratios": [1.6, 1.2, 0.8]})
        x = np.arange(len(strata))
        w = 0.38
        for i, m in enumerate(models):
            vals = [t2[(t2.model == m) & (t2.stratum == s)]["pref_acc"].mean() for s in strata]
            bars = axes[0].bar(x + (i - (len(models) - 1) / 2) * w, vals, w * 0.92, color=colors[m], label=labels[m])
            for b_, v in zip(bars, vals):
                axes[0].text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(strata, fontsize=8)
        axes[0].set_ylim(0, 1.18)
        axes[0].axhline(0.5, color=GRAY, linestyle="--", linewidth=1)
        axes[0].set_title("Held-out preference accuracy by length stratum")
        axes[0].legend(frameon=False, loc="upper left", ncol=2)

        box_models = [m for m in ("sft", "standard", "length_balanced") if S.get(m) and "generation" in S[m]]
        data = [[r["response_tokens"] for r in _jsonl(res / m / "generations.jsonl")] for m in box_models]
        bp = axes[1].boxplot(data, widths=0.5, patch_artist=True, showfliers=False)
        for patch, m in zip(bp["boxes"], box_models):
            patch.set_facecolor({"sft": GRAY, **colors}[m])
            patch.set_alpha(0.85)
        for med in bp["medians"]:
            med.set_color(INK)
        axes[1].set_xticks(range(1, len(box_models) + 1))
        axes[1].set_xticklabels(["SFT", "standard", "balanced"][: len(box_models)] if box_models == ["sft", "standard", "length_balanced"] else box_models, fontsize=8)
        axes[1].set_title("Generated length (tokens)")

        wl = [S[m]["word_limit"]["compliance_rate"] for m in box_models]
        bars = axes[2].bar(range(len(box_models)), wl, 0.6, color=[{"sft": GRAY, **colors}[m] for m in box_models])
        for b_, v in zip(bars, wl):
            axes[2].text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK2)
        axes[2].set_xticks(range(len(box_models)))
        axes[2].set_xticklabels(["SFT", "std", "bal"][: len(box_models)] if len(box_models) == 3 else box_models, fontsize=8)
        axes[2].set_ylim(0, 1.05)
        axes[2].set_title("Word-limit compliance")
        _save(fig, figdir, "fig_length_study")

    # examples worth reading for the qualitative section
    out = ["# Task 1 qualitative candidates (auto-selected; read and pick yourself)\n"]
    sft_g = {r["id"]: r for r in _jsonl(res / "sft" / "generations.jsonl")}
    std_g = {r["id"]: r for r in _jsonl(res / "standard" / "generations.jsonl")}
    common = [i for i in std_g if i in sft_g]
    if common:
        cand = sorted(common, key=lambda i: (std_g[i]["reward"] - sft_g[i]["reward"]) *
                      (std_g[i]["response_tokens"] / max(sft_g[i]["response_tokens"], 1)), reverse=True)[:6]
        out.append("## A. Standard DPO gets a higher reward than SFT on the same prompt, with a much longer answer\n"
                   "Check whether the longer, higher-reward answer is actually better (correct, concise, on task).\n")
        for i in cand:
            a, b = sft_g[i], std_g[i]
            out.append(f"### prompt id {i}\n**Prompt:** {a['prompt'][:500]}\n\n"
                       f"**SFT** (reward {a['reward']:.2f}, {a['response_tokens']} tok, truncated={a['truncated']}):\n\n> "
                       + a["response"][:900].replace("\n", "\n> ") +
                       f"\n\n**DPO standard** (reward {b['reward']:.2f}, {b['response_tokens']} tok, truncated={b['truncated']}):\n\n> "
                       + b["response"][:900].replace("\n", "\n> ") + "\n")
    wl_std = _jsonl(res / "standard" / "word_limit_generations.jsonl")
    viol = sorted([r for r in wl_std if r.get("compliant") == 0.0], key=lambda r: -r["reward"])[:5]
    if viol:
        out.append("## B. Word-limit violations by standard DPO (highest reward first)\n")
        for r in viol:
            out.append(f"- **{r['prompt']}** -> {r['response_words']} words (limit {r['word_limit']}), reward {r['reward']:.2f}\n\n  > "
                       + r["response"][:600].replace("\n", " ") + "\n")
    pairs = _jsonl(res / "standard" / "pairs_standard_eval.jsonl")
    if pairs:
        longest = sorted(pairs, key=lambda r: -(r["chosen_tokens_full"] - r["rejected_tokens_full"]))
        out.append("## C. Held-out pairs: margin vs length difference (standard DPO)\n"
                   "| id | margin | chosen tok | rejected tok | correct |\n|---|---|---|---|---|")
        for r in longest[:5] + longest[-5:]:
            out.append(f"| {r['id']} | {r['margin']:.2f} | {r['chosen_tokens_full']} | {r['rejected_tokens_full']} | {r['correct']} |")
    (res / "qualitative_candidates.md").write_text("\n".join(out) + "\n")
    print("  wrote", res / "qualitative_candidates.md")


if __name__ == "__main__":
    main()
