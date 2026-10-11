"""Builds the Task 2 tables, figures and candidate qualitative examples from results/task2_ppo/."""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.data import load_yaml, repo_path  # noqa: E402
from task2_ppo.analyze_clipping import fork_name  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
GRAY, INK, INK2 = "#8a8985", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "axes.axisbelow": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "savefig.dpi": 200,
})


def _load(p):
    return json.loads(p.read_text()) if p.exists() else None


def _jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def _md(df):
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
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


def _eval_row(label, ev, extra=None):
    if ev is None:
        return None
    return {"condition": label, **(extra or {}),
            "heldout_reward": ev["reward_mean"], "reward_sem": ev["reward_sem"],
            "kl_per_token": ev["kl_token_mean"], "kl_per_seq": ev["kl_seq_sum_mean"],
            "entropy": ev["entropy_token_mean"], "len_mean": ev["length_tokens"]["mean"],
            "len_std": ev["length_tokens"]["std"], "truncation_rate": ev["truncation_rate"]}


def _train_stats(log):
    if not log:
        return {}
    return {"train_reward_mean": float(np.mean([r["reward"] for r in log])),
            "final_kl_ref": log[-1]["kl_ref_token_mean"],
            "mean_post_update_kl": float(np.mean([r["post_update_approx_kl"] for r in log])),
            "max_post_update_kl": float(np.max([r["post_update_approx_kl"] for r in log])),
            "mean_clip_fraction": float(np.mean([r["clip_fraction"] for r in log])),
            "reward_std_over_updates": float(np.std([r["reward"] for r in log])),
            "generated_tokens": log[-1]["generated_tokens_so_far"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    res = repo_path(cfg["results_dir"])
    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    # standard continuation
    log = _jsonl(res / "standard" / "train_log.jsonl")
    meta = _load(res / "standard" / "train_meta.json")
    if log:
        panels = [("reward", "Learned reward"), ("kl_ref_token_mean", "KL from reference (per token)"),
                  ("policy_loss", "Policy loss"), ("value_loss", "Value loss"), ("entropy", "Entropy"),
                  ("clip_fraction", "Clip fraction"), ("grad_norm_policy", "Policy grad norm"),
                  ("response_length", "Response length (tokens)")]
        fig, axes = plt.subplots(2, 4, figsize=(11, 4.4))
        x = [r["update"] for r in log]
        for ax, (k, t) in zip(axes.flat, panels):
            y = [r[k] for r in log]
            ax.plot(x, y, color=COLORS[0], linewidth=0.9, alpha=0.35, marker="o", markersize=2.5)
            ax.plot(x, _smooth(y), color=COLORS[0], linewidth=1.8)
            ax.set_title(t)
            ax.set_xlabel("update")
        _save(fig, figs, "fig_ppo_standard_trajectories")

        rows = []
        n_up = meta["updates"] if meta else len(log)
        for label, name in (("PPO midpoint (start)", "midpoint"), (f"PPO standard (+{n_up} updates)", "standard")):
            r = _eval_row(label, _load(res / name / "eval_summary.json"))
            if r:
                rows.append(r)
        t = pd.DataFrame(rows)
        head = ""
        if meta:
            head = (f"Standard continuation: {meta['updates']} updates x {meta['prompts_per_update']} prompt, "
                    f"{meta['generated_tokens']} generated tokens, wall-clock {meta['wall_clock_s']:.0f} s, "
                    f"peak VRAM {meta['peak_vram_gb']} GB on {meta['gpu']}. Mean critic explained variance "
                    f"{np.mean([r['explained_variance'] for r in log]):.3f}.\n\n")
        if len(t):
            t.to_csv(res / "table_ppo_standard.csv", index=False)
        (res / "table_ppo_standard.md").write_text(head + (_md(t) if len(t) else "") + "\n")
        print("  wrote", res / "table_ppo_standard.md")

    # clipping study
    cached = _load(res / "clipping_cached.json")
    beta0 = float(cfg["kl_beta"])
    eps_list = [float(e) for e in cfg["clip_values"]]
    if cached:
        rows = []
        for e in eps_list:
            c = cached["per_epsilon"][f"{e:g}"]
            rows.append({"epsilon": e, "clip_frac_midpoint": c[0]["clip_fraction"],
                         "affected_frac_midpoint": c[0]["affected_fraction"],
                         "surrogate_gap_midpoint": c[0]["surrogate_gap"],
                         f"clip_frac_after_{c[-1]['epoch']}_steps": c[-1]["clip_fraction"],
                         f"affected_frac_after_{c[-1]['epoch']}_steps": c[-1]["affected_fraction"],
                         f"surrogate_gap_after_{c[-1]['epoch']}_steps": c[-1]["surrogate_gap"]})
        t = pd.DataFrame(rows)
        t.to_csv(res / "table_ppo_clipping_cached.csv", index=False)
        (res / "table_ppo_clipping_cached.md").write_text(
            f"Cached batch: {cached['num_rollouts']} rollouts, {cached['num_tokens']} response tokens. Clip fraction = "
            "tokens with ratio outside [1-eps, 1+eps]; affected = tokens where the clipped term is the one used "
            "(gradient zeroed); surrogate gap = unclipped minus clipped surrogate, per token.\n\n" + _md(t) + "\n")
        print("  wrote", res / "table_ppo_clipping_cached.md")

    rows = []
    for e in eps_list:
        name = fork_name(e, beta0)
        ev = _eval_row(name, _load(res / name / "eval_summary.json"))
        st = _train_stats(_jsonl(res / name / "train_log.jsonl"))
        if not ev and not st:
            continue
        r = {"epsilon": e}
        if ev:
            r.update({k: ev[k] for k in ("heldout_reward", "reward_sem", "kl_per_token", "entropy", "len_mean", "truncation_rate")})
        r.update({k: st.get(k) for k in ("mean_post_update_kl", "max_post_update_kl", "mean_clip_fraction",
                                         "reward_std_over_updates", "generated_tokens")})
        rows.append(r)
    tc = pd.DataFrame(rows)
    if len(tc):
        tc.to_csv(res / "table_ppo_clipping_forks.csv", index=False)
        (res / "table_ppo_clipping_forks.md").write_text(
            f"Short forks ({cfg['fork_updates']} updates, kl_beta={beta0:g}) from the midpoint. Stability statistic: "
            "post_update_kl = k3 estimate of KL(old || new) on each update's rollout after the PPO epochs "
            "(bigger = larger policy jump per update).\n\n" + _md(tc) + "\n")
        print("  wrote", res / "table_ppo_clipping_forks.md")

    if cached:
        fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.7))
        for i, e in enumerate(eps_list):
            c = cached["per_epsilon"][f"{e:g}"]
            ep = [s["epoch"] for s in c]
            axes[0].plot(ep, [s["clip_fraction"] for s in c], color=COLORS[i], marker="o", markersize=4, label=f"eps={e:g}")
            axes[1].plot(ep, [s["affected_fraction"] for s in c], color=COLORS[i], marker="o", markersize=4, label=f"eps={e:g}")
        axes[0].set_title("Cached batch: clip fraction")
        axes[1].set_title("Cached batch: affected-token fraction")
        for ax in axes[:2]:
            ax.set_xlabel("PPO steps on the cached batch")
            ax.set_xticks(range(int(cached["ppo_epochs"]) + 1))
        lr = np.asarray(cached.get("log_ratio_at_midpoint", []))
        if lr.size:
            lim = max(0.6, float(np.percentile(np.abs(lr), 99.5)))
            axes[2].hist(np.clip(lr, -lim, lim), bins=60, color=GRAY)
            for i, e in enumerate(eps_list):
                for v in (np.log(1 - e), np.log(1 + e)):
                    axes[2].axvline(v, color=COLORS[i], linewidth=1.2, linestyle="--")
            axes[2].set_title("log ratio at midpoint (dashed: eps bands)")
            axes[2].set_xlabel("log pi_mid - log pi_old")
        for i, e in enumerate(eps_list):
            lg = _jsonl(res / fork_name(e, beta0) / "train_log.jsonl")
            if lg:
                axes[3].plot([r["update"] for r in lg], [r["post_update_approx_kl"] for r in lg], color=COLORS[i],
                             marker="o", markersize=3, label=f"eps={e:g}")
        axes[3].set_title("Forks: KL(old||new) per update")
        axes[3].set_xlabel("update")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(eps_list), frameon=False, bbox_to_anchor=(0.5, -0.07))
        _save(fig, figs, "fig_ppo_clipping")

    # KL pressure study
    eps0 = float(cfg["clip_epsilon"])
    betas = [float(b) for b in cfg["kl_values"]]
    rows, logs = [], {}
    for b in betas:
        name = fork_name(eps0, b)
        logs[b] = _jsonl(res / name / "train_log.jsonl")
        ev = _eval_row(name, _load(res / name / "eval_summary.json"))
        st = _train_stats(logs[b])
        if not ev and not st:
            continue
        r = {"kl_beta": b}
        if ev:
            r.update({k: ev[k] for k in ("heldout_reward", "reward_sem", "kl_per_token", "kl_per_seq", "entropy",
                                         "len_mean", "len_std", "truncation_rate")})
        r.update({k: st.get(k) for k in ("train_reward_mean", "final_kl_ref", "generated_tokens")})
        rows.append(r)
    mid = _load(res / "midpoint" / "eval_summary.json")
    tk = pd.DataFrame(rows)
    if len(tk):
        tk.to_csv(res / "table_ppo_kl.csv", index=False)
        pre = ""
        if mid:
            pre = (f"Midpoint (no further updates): held-out reward {mid['reward_mean']:.4g}, KL/token "
                   f"{mid['kl_token_mean']:.4g}, entropy {mid['entropy_token_mean']:.4g}, length {mid['length_tokens']['mean']:.4g}.\n\n")
        (res / "table_ppo_kl.md").write_text(pre + _md(tk) + "\n")
        print("  wrote", res / "table_ppo_kl.md")
    if any(logs.values()):
        fig, axes = plt.subplots(2, 4, figsize=(11.5, 4.8))
        traj = [("reward", "Train reward"), ("kl_ref_token_mean", "KL from reference"),
                ("entropy", "Entropy"), ("response_length", "Response length")]
        for i, b in enumerate(betas):
            lg = logs[b]
            if not lg:
                continue
            for ax, (k, t) in zip(axes[0], traj):
                ax.plot([r["update"] for r in lg], [r[k] for r in lg], color=COLORS[i], marker="o",
                        markersize=3, linewidth=1.5, label=f"beta={b:g}")
                ax.set_title(t + " (per update)")
                ax.set_xlabel("update")
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=len(betas), frameon=False, bbox_to_anchor=(0.5, 1.03))
        held = [("heldout_reward", "Held-out reward"), ("kl_per_token", "Held-out KL / token"),
                ("entropy", "Held-out entropy"), ("len_mean", "Held-out length")]
        mid_vals = {"heldout_reward": mid and mid["reward_mean"], "kl_per_token": mid and mid["kl_token_mean"],
                    "entropy": mid and mid["entropy_token_mean"], "len_mean": mid and mid["length_tokens"]["mean"]}
        if "heldout_reward" in tk:
            for ax, (k, t) in zip(axes[1], held):
                ax.bar(range(len(tk)), tk[k], color=COLORS[: len(tk)], width=0.6)
                if k == "heldout_reward":
                    ax.errorbar(range(len(tk)), tk[k], yerr=tk["reward_sem"], fmt="none", ecolor=INK, capsize=3, linewidth=1)
                if mid_vals[k] is not None:
                    ax.axhline(mid_vals[k], color=GRAY, linestyle="--", linewidth=1, label="midpoint")
                ax.set_xticks(range(len(tk)))
                ax.set_xticklabels([f"beta={b:g}" for b in tk["kl_beta"]])
                ax.set_title(t)
            axes[1][0].legend(frameon=False)
        _save(fig, figs, "fig_ppo_kl")

    # qualitative candidates: same held-out prompt, midpoint vs a fork
    out = ["# Task 2 qualitative candidates (auto-selected, read and pick yourself)\n"]
    mid_g = {r["id"]: r for r in _jsonl(res / "midpoint" / "eval_generations.jsonl")}
    for name in (fork_name(eps0, 0.0), "standard"):
        g = {r["id"]: r for r in _jsonl(res / name / "eval_generations.jsonl")}
        common = [i for i in g if i in mid_g]
        if not common:
            continue
        diff = sorted(common, key=lambda i: g[i]["reward"] - mid_g[i]["reward"], reverse=True)
        out.append(f"## {name} vs midpoint: biggest reward gains (check if quality really improved)\n")
        for i in diff[:4]:
            a, b = mid_g[i], g[i]
            out.append(f"### prompt {i}\n**Prompt:** {a['prompt'][:400]}\n\n"
                       f"**midpoint** (reward {a['reward']:.2f}, {a['response_tokens']} tok, truncated={a['truncated']}):\n\n> "
                       + a["response"][:700].replace("\n", "\n> ") +
                       f"\n\n**{name}** (reward {b['reward']:.2f}, {b['response_tokens']} tok, truncated={b['truncated']}):\n\n> "
                       + b["response"][:700].replace("\n", "\n> ") + "\n")
    (res / "qualitative_candidates.md").write_text("\n".join(out) + "\n")
    print("  wrote", res / "qualitative_candidates.md")


if __name__ == "__main__":
    main()
