"""Sanity checks for dpo_loss. Runs on CPU in a couple of seconds:  python -m task1_dpo.check_dpo_loss"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from common.data import repo_path
from common.logging_utils import save_json
from task1_dpo.dpo import dpo_loss


def reference_formula(pc, pr, rc, rr, beta):
    """DPO loss written straight from the formula."""
    return -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()


def buggy_starter(pc, pr, rc, rr, beta):
    """Original starter version, only here to show the checks catch it."""
    return -F.logsigmoid(beta * ((pc - pr) + (rc - rr))).mean()


def _loss(fn, *args):
    out = fn(*args)
    return out[0] if isinstance(out, tuple) else out


def run_checks(fn):
    torch.manual_seed(0)
    results = {}

    # policy == ref should give exactly log 2
    rc, rr = -50 * torch.rand(64), -50 * torch.rand(64)
    for beta in (0.03, 0.1, 0.3):
        loss = _loss(fn, rc.clone(), rr.clone(), rc, rr, beta)
        results[f"identity_log2_beta{beta}"] = bool(abs(float(loss) - math.log(2)) < 1e-5)

    # should agree with the formula on random inputs
    pc, pr = -50 * torch.rand(64), -50 * torch.rand(64)
    for beta in (0.03, 0.1, 0.3):
        got = _loss(fn, pc, pr, rc, rr, beta)
        results[f"matches_equation_beta{beta}"] = bool(torch.allclose(got, reference_formula(pc, pr, rc, rr, beta), atol=1e-6))

    # pushing up y+ should lower the loss, pushing up y- should raise it
    base = _loss(fn, rc.clone(), rr.clone(), rc, rr, 0.1)
    results["chosen_up_lowers_loss"] = bool(_loss(fn, rc + 1.0, rr.clone(), rc, rr, 0.1) < base)
    results["rejected_up_raises_loss"] = bool(_loss(fn, rc.clone(), rr + 1.0, rc, rr, 0.1) > base)

    # moving policy and ref by the same amount shouldn't change anything
    shift = 7.0 * torch.randn(64)
    a = _loss(fn, pc, pr, rc, rr, 0.1)
    b = _loss(fn, pc + shift, pr, rc + shift, rr, 0.1)
    results["invariant_to_matched_shift"] = bool(torch.allclose(a, b, atol=1e-5))
    return results


def main():
    fixed = run_checks(dpo_loss)
    starter = run_checks(buggy_starter)
    print("Fixed dpo_loss:")
    for k, v in fixed.items():
        print(f"  {k:32s} {'PASS' if v else 'FAIL'}")
    print("Starter version (should fail some of these):")
    for k, v in starter.items():
        print(f"  {k:32s} {'PASS' if v else 'FAIL'}")
    save_json(repo_path("results/task1_dpo/dpo_loss_checks.json"), {"fixed": fixed, "starter": starter})
    if not all(fixed.values()):
        raise SystemExit("dpo_loss FAILED validation - do not train.")
    print("\nAll checks passed for the fixed dpo_loss.")


if __name__ == "__main__":
    main()
