"""Sanity checks for the GRPO advantages and loss. Runs on CPU:  python -m task3_grpo.check_grpo"""
from __future__ import annotations

import torch

from common.data import repo_path
from common.logging_utils import save_json
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences


def starter_advantages(rewards, group_ids, eps=1e-6):
    return (rewards - rewards.mean()) / rewards.std(unbiased=False).clamp_min(eps)


def run_checks(adv_fn):
    out = {}
    # two prompts with very different reward scales
    r = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 10.5, 11.0, 11.5])
    gid = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    a = adv_fn(r, gid)
    out["zero_mean_within_each_group"] = bool(all(abs(float(a[gid == g].mean())) < 1e-5 for g in (0, 1)))
    out["unit_std_within_each_group"] = bool(all(abs(float(a[gid == g].std(unbiased=False)) - 1) < 1e-3 for g in (0, 1)))
    # shifting one prompt's rewards shouldn't change any advantage
    r2 = r.clone()
    r2[gid == 1] += 100.0
    out["invariant_to_per_group_shift"] = bool(torch.allclose(adv_fn(r2, gid), a, atol=1e-4))
    # a group where every completion gets the same reward carries no signal
    a3 = adv_fn(torch.tensor([0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 3.0, 4.0]), gid)
    out["constant_group_gives_zero_advantage"] = bool(float(a3[:4].abs().max()) < 1e-4)
    # order of the batch shouldn't matter
    perm = torch.randperm(8, generator=torch.Generator().manual_seed(0))
    out["permutation_equivariant"] = bool(torch.allclose(adv_fn(r[perm], gid[perm]), a[perm], atol=1e-5))
    return out


def loss_checks():
    torch.manual_seed(0)
    out = {}
    T = 6
    old = torch.randn(4, T) * 0.1
    ref = old.clone()
    mask = (torch.arange(T)[None, :] < torch.tensor([6, 4, 2, 5])[:, None]).float()
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    # on-policy, policy == ref: KL term is zero and the loss is -mean over sequences of A
    loss, st = grpo_policy_loss(old.clone(), old, adv, mask, ref, 0.2, 0.1, "grpo")
    out["grpo_on_policy_loss_is_minus_mean_adv"] = bool(abs(float(loss) + float(adv.mean())) < 1e-6)
    out["kl_zero_when_policy_equals_ref"] = bool(abs(float(st["sampled_kl"])) < 1e-7)
    # dr_grpo divides by a constant, so long completions keep more weight
    loss_dr, _ = grpo_policy_loss(old.clone(), old, adv, mask, ref, 0.2, 0.1, "dr_grpo", max_completion_length=T)
    want = -float((adv * mask.sum(-1) / T).mean())
    out["dr_grpo_weights_by_length"] = bool(abs(float(loss_dr) - want) < 1e-6)
    # masked (truncated) completions contribute nothing
    m2 = mask_truncated_sequences(mask, [False, True, False, False])
    out["truncated_rows_masked"] = bool(float(m2[1].sum()) == 0.0 and torch.equal(m2[0], mask[0]))
    return out


def main():
    fixed = {**run_checks(group_relative_advantages), **loss_checks()}
    starter = run_checks(starter_advantages)
    print("Fixed GRPO advantages + loss:")
    for k, v in fixed.items():
        print(f"  {k:40s} {'PASS' if v else 'FAIL'}")
    print("Starter advantages (should fail some of these):")
    for k, v in starter.items():
        print(f"  {k:40s} {'PASS' if v else 'FAIL'}")
    save_json(repo_path("results/task3_grpo/grpo_checks.json"), {"fixed": fixed, "starter": starter})
    if not all(fixed.values()):
        raise SystemExit("GRPO code FAILED validation - do not train.")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
