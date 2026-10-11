"""Builds the Task 3 tables, figures and candidate qualitative examples from results/task3_grpo/."""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.data import load_yaml, repo_path  # noqa: E402
from common.metrics import safe_corr  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRAY, INK, INK2 = "#8a8985", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "axes.axisbelow": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "savefig.dpi": 200,
})
LABEL = {"fork_grpo": "GRPO (1/T)", "fork_dr_grpo": "Dr. GRPO (1/L)"}


def _load(p):
    return json.loads(p.read_text()) if p.exists() else None


def _jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _md(df):
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else
                                      (f"{v:.4g}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)


def _smooth(v, w=5):
    v = np.asarray(v, float)
    return np.array([v[max(0, i - w + 1): i + 1].mean() for i in range(len(v))])


def _save(fig, d, stem):
    fig.tight_layout()
    fig.savefig(d / f"{stem}.png", bbox_inches="tight")
    fig.savefig(d / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", d / f"{stem}.png")


def _eval_cols(ev):
    return {"heldout_reward": ev["reward_mean"], "reward_sem": ev["reward_sem"], "kl_per_token": ev["kl_token_mean"],
            "entropy": ev["entropy_token_mean"], "len_mean": ev["length_tokens"]["mean"],
            "len_median": ev["length_tokens"]["median"], "truncation_rate": ev["truncation_rate"]}


def length_grad_stats(log):
    """Length-conditioned gradient statistics over all unmasked completions of a run."""
    comps = [c for r in log for c in r["completions"] if not c["masked"] and c.get("grad_norm") is not None]
    if not comps:
        return {}, []
    L = np.array([c["length"] for c in comps], float)
    G = np.array([c["grad_norm"] for c in comps], float)
    A = np.array([abs(c["advantage"]) for c in comps], float)
    keep = A > 1e-6
    per_adv = G[keep] / A[keep]
    med = np.median(L[keep]) if keep.any() else 0
    short, long_ = L[keep] <= med, L[keep] > med
    total = G.sum() if G.sum() > 0 else 1.0
    return {
        "completions_used": int(keep.sum()),
        "corr_length_vs_gradnorm_per_adv": safe_corr(L[keep], per_adv),
        "median_length": float(med),
        "gradnorm_per_adv_short": float(per_adv[short].mean()) if short.any() else float("nan"),
        "gradnorm_per_adv_long": float(per_adv[long_].mean()) if long_.any() else float("nan"),
        "share_of_grad_from_long_half": float(G[keep][long_].sum() / total),
    }, list(zip(L[keep], per_adv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # standard continuation
    log = _jsonl(res / "standard" / "train_log.jsonl")
    meta = _load(res / "standard" / "train_meta.json")
    if log:
        panels = [("reward", "Reward (mean over K)"), ("kl_ref_token_mean", "KL from reference (per token)"),
                  ("group_reward_std", "Within-group reward std"), ("uninformative_fraction", "Uninformative-group fraction"),
                  ("policy_loss", "Policy loss"), ("grad_norm", "Grad norm"), ("entropy", "Entropy"),
                  ("response_length", "Response length (tokens)")]
        fig, axes = plt.subplots(2, 4, figsize=(11, 4.4))
        x = [r["update"] for r in log]
        for ax, (k, t) in zip(axes.flat, panels):
            y = [r[k] for r in log]
            ax.plot(x, y, color=COLORS[0], linewidth=0.9, alpha=0.35, marker="o", markersize=2.5)
            ax.plot(x, _smooth(y), color=COLORS[0], linewidth=1.8)
            ax.set_title(t)
            ax.set_xlabel("update")
        _save(fig, figs, "fig_grpo_standard_trajectories")
        rows = []
        for label, name in (("GRPO midpoint (start)", "midpoint"), (f"GRPO standard (+{len(log)} updates)", "standard")):
            ev = _load(res / name / "eval_summary.json")
            if ev:
                rows.append({"condition": label, **_eval_cols(ev)})
        head = ""
        if meta:
            head = (f"Standard continuation: {meta['updates']} updates x {meta['prompts_per_update']} prompt x "
                    f"K={meta['num_generations']}, {meta['generated_tokens']} generated tokens, wall-clock "
                    f"{meta['wall_clock_s']:.0f} s, peak VRAM {meta['peak_vram_gb']} GB on {meta['gpu']}. "
                    f"Mean within-group reward std {np.mean([r['group_reward_std'] for r in log]):.4g}, "
                    f"uninformative groups {np.mean([r['uninformative_fraction'] for r in log]):.3g}, "
                    f"truncated (masked) completions {np.mean([r['truncated_fraction'] for r in log]):.3g}.\n\n")
        t = pd.DataFrame(rows)
        if len(t):
            t.to_csv(res / "table_grpo_standard.csv", index=False)
        (res / "table_grpo_standard.md").write_text(head + (_md(t) if len(t) else "") + "\n")
        print("  wrote", res / "table_grpo_standard.md")

    # group-size study
    gs = _load(res / "group_size_study.json")
    if gs:
        ks = sorted(gs["per_k"], key=int)
        rows = []
        for k in ks:
            m = gs["per_k"][k]
            rows.append({"K": int(k), "bin": "all", "groups": m["num_groups"], "informative_rate": m["informative_rate"],
                         "informative@0.1sd": m["informative_rate_by_threshold"]["0.1"],
                         "mean_group_std": m["mean_within_group_std"], "var_norm_adv": m["var_normalized_advantage"],
                         "var_centred_reward": m["var_centred_reward"]})
            for bname, bm in gs["per_k_per_bin"][k].items():
                rows.append({"K": int(k), "bin": bname, "groups": bm["num_groups"], "informative_rate": bm["informative_rate"],
                             "informative@0.1sd": bm["informative_rate_by_threshold"]["0.1"],
                             "mean_group_std": bm["mean_within_group_std"], "var_norm_adv": bm["var_normalized_advantage"],
                             "var_centred_reward": bm["var_centred_reward"]})
        t = pd.DataFrame(rows)
        t.to_csv(res / "table_grpo_group_size.csv", index=False)
        (res / "table_grpo_group_size.md").write_text(
            f"{gs['num_prompts']} prompts x {gs['completions_per_prompt']} cached completions = {gs['total_completions']} "
            f"completions for every K. Informative = within-group std > {gs['std_tolerance']:g}; "
            f"informative@0.1sd uses 0.1 x overall reward std ({gs['reward_std']:.4g}). Bins: {gs['binning_rule']} "
            f"(edges {', '.join(f'{e:.3g}' for e in gs['bin_edges_mean_reward'])}).\n\n" + _md(t) + "\n")
        print("  wrote", res / "table_grpo_group_size.md")

        bins = ["hard", "medium", "easy"]
        fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.7))
        kx = [int(k) for k in ks]
        metrics = [("informative_rate", "Informative-group rate (tol)"),
                   ("@0.1", "Informative rate (std > 0.1 sd)"),
                   ("mean_within_group_std", "Mean within-group reward std"),
                   ("var_centred_reward", "Var of centred reward (r - group mean)")]
        for ax, (key, title) in zip(axes, metrics):
            get = (lambda m: m["informative_rate_by_threshold"]["0.1"]) if key == "@0.1" else (lambda m, k_=key: m[k_])
            ax.plot(kx, [get(gs["per_k"][k]) for k in ks], color=INK, marker="o", markersize=4, linewidth=1.8, label="all")
            for i, bname in enumerate(bins):
                ax.plot(kx, [get(gs["per_k_per_bin"][k][bname]) for k in ks], color=COLORS[i], marker="o",
                        markersize=3.5, linewidth=1.4, label=bname)
            ax.set_xticks(kx)
            ax.set_xlabel("group size K")
            ax.set_title(title)
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
        _save(fig, figs, "fig_grpo_group_size")

    # normalization study
    rows, scatter, logs = [], {}, {}
    for name in ("fork_grpo", "fork_dr_grpo"):
        lg = _jsonl(res / name / "train_log.jsonl")
        ev = _load(res / name / "eval_summary.json")
        if not lg and not ev:
            continue
        logs[name] = lg
        stats, scatter[name] = length_grad_stats(lg)
        r = {"condition": LABEL[name]}
        if ev:
            r.update(_eval_cols(ev))
        if lg:
            r["train_len_first_update"] = lg[0]["response_length"]
            r["train_len_last_update"] = lg[-1]["response_length"]
            r["train_reward_mean"] = float(np.mean([x["reward"] for x in lg]))
        r.update(stats)
        rows.append(r)
    t = pd.DataFrame(rows)
    if len(t):
        t.to_csv(res / "table_grpo_normalization.csv", index=False)
        (res / "table_grpo_normalization.md").write_text(
            f"Short forks ({cfg['fork_updates']} updates) from the GRPO midpoint; only the sequence normalization differs. "
            "gradnorm_per_adv = norm of a completion's policy-term gradient divided by |advantage|; short/long split at "
            "the median completion length; share_of_grad_from_long_half = fraction of the summed per-completion "
            "gradient norm coming from the longer half.\n\n" + _md(t) + "\n")
        print("  wrote", res / "table_grpo_normalization.md")

        fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.8))
        for i, name in enumerate(n for n in ("fork_grpo", "fork_dr_grpo") if n in logs):
            pts = scatter.get(name) or []
            if pts:
                axes[0].scatter([p[0] for p in pts], [p[1] for p in pts], s=10, color=COLORS[i], alpha=0.7, label=LABEL[name])
            lg = logs[name]
            if lg:
                axes[1].plot([r["update"] for r in lg], [r["response_length"] for r in lg], color=COLORS[i],
                             marker="o", markersize=3, label=LABEL[name])
            gens = _jsonl(res / name / "eval_generations.jsonl")
            if gens:
                axes[2].hist([g["response_tokens"] for g in gens], bins=25, color=COLORS[i], alpha=0.55, label=LABEL[name])
        axes[0].set_title("Per-completion grad norm / |A| vs length")
        axes[0].set_xlabel("completion length (tokens)")
        axes[1].set_title("Train response length per update")
        axes[1].set_xlabel("update")
        axes[2].set_title("Held-out response length")
        axes[2].set_xlabel("tokens")
        if "heldout_reward" in t:
            axes[3].bar(range(len(t)), t["heldout_reward"], yerr=t["reward_sem"], color=COLORS[: len(t)], width=0.6,
                        capsize=3, error_kw={"linewidth": 1})
            mid = _load(res / "midpoint" / "eval_summary.json")
            if mid:
                axes[3].axhline(mid["reward_mean"], color=GRAY, linestyle="--", linewidth=1, label="midpoint")
                axes[3].legend(frameon=False)
            axes[3].set_xticks(range(len(t)))
            axes[3].set_xticklabels(t["condition"], fontsize=8)
            axes[3].set_title("Held-out reward")
        h, l = axes[1].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
        _save(fig, figs, "fig_grpo_normalization")

    # qualitative candidates
    out = ["# Task 3 qualitative candidates (auto-selected, read and pick yourself)\n"]
    if gs:
        for ex in gs.get("examples", []):
            out.append(f"## Cached group with the {ex['kind']} (prompt {ex['source_index']}, {ex['difficulty_bin']})\n"
                       f"**Prompt:** {ex['prompt']}\n")
            for c in ex["completions"]:
                out.append(f"- reward {c['reward']:.3f}: {c['text'][:200].replace(chr(10), ' ')}")
            out.append("")
    a = {r["id"]: r for r in _jsonl(res / "fork_grpo" / "eval_generations.jsonl")}
    b = {r["id"]: r for r in _jsonl(res / "fork_dr_grpo" / "eval_generations.jsonl")}
    common = [i for i in a if i in b]
    if common:
        out.append("## Same held-out prompt, biggest length difference between GRPO and Dr. GRPO forks\n")
        for i in sorted(common, key=lambda i: -abs(a[i]["response_tokens"] - b[i]["response_tokens"]))[:4]:
            out.append(f"### prompt {i}\n**Prompt:** {a[i]['prompt'][:400]}\n\n"
                       f"**GRPO** (reward {a[i]['reward']:.2f}, {a[i]['response_tokens']} tok):\n\n> "
                       + a[i]["response"][:600].replace("\n", "\n> ") +
                       f"\n\n**Dr. GRPO** (reward {b[i]['reward']:.2f}, {b[i]['response_tokens']} tok):\n\n> "
                       + b[i]["response"][:600].replace("\n", "\n> ") + "\n")
    (res / "qualitative_candidates.md").write_text("\n".join(out) + "\n")
    print("  wrote", res / "qualitative_candidates.md")


if __name__ == "__main__":
    main()
