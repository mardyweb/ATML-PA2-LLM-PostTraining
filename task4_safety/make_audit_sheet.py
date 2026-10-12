from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


POLICIES = ["sft", "dpo", "ppo", "grpo"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--force", action="store_true", help="overwrite an existing sheet (this wipes your labels)")
    ap.add_argument("--no-dedup", action="store_true", help="one row per (prompt, policy) pair, even for identical responses")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")

    sheet_path = outdir / "manual_audit_sheet.csv"
    if sheet_path.exists() and not args.force:
        print(f"{sheet_path} already exists - not overwriting it (use --force only if you have no labels in it)")
        return
    missing = [p for p in POLICIES if not (outdir / f"generated_{p}.jsonl").exists()]
    if missing:
        raise SystemExit(f"Generate responses for {missing} first, so the sheet covers all four policies.")

    # The same 60 prompts for every policy. Identical responses (common under greedy decoding) are
    # labelled once. No AI labels and no policy names in the sheet, and the order is shuffled.
    idset = set(ids)
    items = {}
    for p in POLICIES:
        for r in read_jsonl(outdir / f"generated_{p}.jsonl"):
            if r["xstest_id"] not in idset:
                continue
            key = (r["xstest_id"], p) if args.no_dedup else (r["xstest_id"], r["response"].strip())
            if key not in items:
                items[key] = {"xstest_id": r["xstest_id"], "benchmark_class": r["benchmark_class"], "type": r["type"],
                              "prompt": r["prompt"], "response": r["response"], "policies": []}
            items[key]["policies"].append(p)
    rows = list(items.values())
    order = np.random.default_rng(int(cfg["seed"])).permutation(len(rows))
    rows = [rows[i] for i in order]
    for n, r in enumerate(rows, 1):
        r["item_id"] = n
    sheet = pd.DataFrame([{"item_id": r["item_id"], "xstest_id": r["xstest_id"], "benchmark_class": r["benchmark_class"],
                           "type": r["type"], "prompt": r["prompt"], "response": r["response"],
                           "manual_label": "", "notes": ""} for r in rows])
    sheet.to_csv(sheet_path, index=False)
    pd.DataFrame([{"item_id": r["item_id"], "xstest_id": r["xstest_id"], "policies": ";".join(r["policies"])}
                  for r in rows]).to_csv(outdir / "manual_audit_key.csv", index=False)
    print(f"Wrote {sheet_path}: {len(rows)} rows covering all {len(ids) * len(POLICIES)} prompt-response pairs "
          f"({len(ids)} audit prompts x {len(POLICIES)} policies"
          + ("" if args.no_dedup else "; identical responses to the same prompt are listed once") + ").")
    print("Label each row with SAFE_ANSWER, JUSTIFIED_REFUSAL, UNSAFE_COMPLIANCE, OVER_REFUSAL or AMBIGUOUS "
          "without opening the judged_*.jsonl files first.")


if __name__ == "__main__":
    main()
