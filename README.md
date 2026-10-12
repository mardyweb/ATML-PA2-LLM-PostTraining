# ATML PA2 - LLM Post-Training

Implementation of all five experimental tasks of ATML PA2 (DPO, PPO, GRPO, safety calibration,
RLVR vs RLAIF) on top of the course starter repository. Every number in the report comes from a
file under `results/` written by one of the commands below.

## Setup

```bash
git clone <this repository>
cd ATML-PA2-LLM-PostTraining
python -m pip install -r requirements.txt
python -m scripts.download_assets      # pinned course assets (revision 0b350481fb03f5525a35bcdec4131bd4fe487f98)
python -m scripts.validate_assets
python -m scripts.check_environment
```

All commands are run from the repository root. Experiments were run on a single A100 40GB
(Google Colab). Downloaded data, checkpoints and trained adapters (`data/`, `cached/`,
`checkpoints/`, `outputs/`) are git-ignored; `results/` holds the small JSON/JSONL/CSV logs,
tables and figures. Each step skips work whose outputs already exist, so a command can simply be
rerun after an interruption.

## Task 1 - DPO (`task1_dpo/`)

```bash
python -m task1_dpo.check_dpo_loss                                        # validates the fixed objective
python -m task1_dpo.train --run-name standard                              # 1 epoch, standard set
python -m task1_dpo.evaluate --adapter none --name sft                     # reference / SFT baseline
python -m task1_dpo.evaluate --adapter outputs/task1_dpo/standard --name standard
python -m task1_dpo.ablate_beta                                            # beta 0.03 / 0.10 / 0.30 short runs
python -m task1_dpo.analyze_length                                         # length-balanced run + data profiles
python -m task1_dpo.summarize                                              # tables, figures, examples
```

## Task 2 - PPO (`task2_ppo/`)

```bash
python -m task2_ppo.check_ppo
python -m task2_ppo.continue_train --run-name standard                     # 20 updates from the midpoint
python -m task2_ppo.evaluate --adapter checkpoints/ppo_midpoint_policy --name midpoint
python -m task2_ppo.evaluate --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping                                       # cached-batch study + eps forks
python -m task2_ppo.ablate_kl                                              # kl_beta forks
python -m task2_ppo.summarize
```

## Task 3 - GRPO (`task3_grpo/`)

```bash
python -m task3_grpo.check_grpo
python -m task3_grpo.analyze_group_size                                    # K = 2/4/8 on the supplied cache
python -m task3_grpo.continue_train --run-name standard                    # 20 updates from the midpoint
python -m task3_grpo.evaluate --adapter checkpoints/grpo_midpoint_policy --name midpoint
python -m task3_grpo.evaluate --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.compare_normalization                                 # GRPO vs Dr. GRPO forks
python -m task3_grpo.summarize
```

## Task 4 - Safety calibration (`task4_safety/`)

Needs the standard adapters from Tasks 1-3.

```bash
python -m task4_safety.generate_responses      # SFT / DPO / PPO / GRPO on all 450 XSTest prompts
python -m task4_safety.judge_responses         # supplied AI judge, one response at a time
python -m task4_safety.make_audit_sheet        # blind sheet: fixed 60 prompts x 4 policies = 240 pairs
# manual labels -> results/task4_safety/manual_audit_labels.xlsx (or .csv)
python -m task4_safety.evaluate_safety
```

## Task 5 - RLVR vs RLAIF (`task5_feedback/`)

```bash
python -m task5_feedback.score_perturbations                # controlled diagnostic set
python -m task5_feedback.evaluate_math --dataset gsm        # in-domain
python -m task5_feedback.evaluate_math --dataset transfer   # SVAMP transfer
python -m task5_feedback.compare_feedback
python -m task5_feedback.check_judge_outputs              # raw judge text behind each A/B/TIE
```

## Where the results are

| Task | Main outputs |
|---|---|
| 1 | `results/task1_dpo/table_dpo_summary.md`, `table_length_strata.md`, `table_length_behaviour.md`, `table_data_used.md`, `figures/`, `qualitative_candidates.md` |
| 2 | `results/task2_ppo/table_ppo_standard.md`, `table_ppo_clipping_cached.md`, `table_ppo_clipping_forks.md`, `table_ppo_kl.md`, `figures/` |
| 3 | `results/task3_grpo/table_grpo_standard.md`, `table_grpo_group_size.md`, `table_grpo_normalization.md`, `figures/` |
| 4 | `results/task4_safety/safety_summary.md`, `category_label_distribution.md`, `audit_agreement.md`, `audit_pairs.csv`, `figures/` |
| 5 | `results/task5_feedback/table_task5_policies.md`, `table_task5_diagnostics.md`, `judge_format_check.md`, `figures/` |

Per-run logs (training curves, prompt/data IDs, hyperparameters, wall-clock, peak VRAM, per-example
generations and scores) sit next to these in one folder per run.

## Fixes to the starter code

- `task1_dpo/dpo.py`: the logit used `policy_margin + ref_margin`; DPO needs `policy_margin - ref_margin`.
- `task2_ppo/ppo.py`: the clipped surrogate used `torch.maximum`; PPO takes the minimum.
- `task3_grpo/grpo.py`: advantages were normalised over the whole batch; they are now normalised within each prompt's group.

Each fix has a CPU check script (`check_dpo_loss`, `check_ppo`, `check_grpo`) that passes for the
fixed version and fails for the original.

## Implementation notes

The task READMEs (`task*/README.md`) list the evaluation protocols and the choices that should be
read alongside the numbers, in particular: DPO pairs whose prompt alone exceeds 768 tokens are
dropped (counts and IDs in `results/task1_dpo/dataset_length_profiles.json`); RL prompts longer than
256 tokens are skipped instead of truncated; entropy is the full next-token-distribution entropy;
identical policy/SFT math responses count as judge ties; the safety audit labels all 240
(prompt, policy response) pairs one row each.

## Attribution

Built on the ATML PA2 course starter repository (configs, `common/` helpers except `common/rl.py`,
supplied judges, verifier, loaders and checkpoints). Parts of the implementation were written with
help from an AI coding assistant, as allowed by the course policy; no other external code was reused.
