"""Sanity checks for the PPO objective and GAE. Runs on CPU:  python -m task2_ppo.check_ppo"""
from __future__ import annotations

import torch

from common.data import repo_path
from common.logging_utils import save_json
from common.metrics import masked_mean
from task2_ppo.ppo import compute_gae, ppo_policy_loss


def clipped_surrogate(new_logp, old_logp, adv, mask, eps):
    ratio = torch.exp(new_logp - old_logp)
    obj = torch.minimum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
    return -masked_mean(obj, mask)


def starter_loss(new_logp, old_logp, adv, mask, eps=0.2):
    ratio = torch.exp(new_logp - old_logp)
    obj = torch.maximum(ratio * adv, ratio.clamp(1 - eps, 1 + eps) * adv)
    return -masked_mean(obj, mask), ratio, None


def naive_gae(rewards, values, mask, gamma, lam):
    adv = torch.zeros_like(rewards)
    for b in range(rewards.shape[0]):
        T = int(mask[b].sum())
        for t in range(T):
            total = 0.0
            for k in range(t, T):
                v_next = values[b, k + 1] if k + 1 < T else 0.0
                delta = rewards[b, k] + gamma * v_next - values[b, k]
                total += (gamma * lam) ** (k - t) * delta
            adv[b, t] = total
    return adv


def grad_wrt_logp(fn, new_logp, old_logp, adv, mask, eps):
    x = new_logp.clone().requires_grad_(True)
    loss = fn(x, old_logp, adv, mask, eps=eps)[0]
    loss.backward()
    return loss.detach(), x.grad


def run_checks(fn):
    torch.manual_seed(0)
    out = {}
    one = torch.ones(1, 1)

    # ratio == 1: clipping does nothing, loss is just -mean(A)
    adv = torch.randn(4, 6)
    old = torch.randn(4, 6)
    mask = torch.ones(4, 6)
    loss = fn(old.clone(), old, adv, mask, eps=0.2)[0]
    out["ratio_one_gives_minus_mean_adv"] = bool(torch.allclose(loss, -adv.mean(), atol=1e-6))

    # A > 0 and ratio above 1+eps: objective capped at (1+eps)A, no gradient
    l, g = grad_wrt_logp(fn, torch.log(torch.tensor([[1.5]])), torch.zeros(1, 1), one, one, 0.2)
    out["pos_adv_high_ratio_capped"] = bool(abs(float(l) + 1.2) < 1e-5 and float(g.abs().sum()) == 0.0)

    # A < 0 and ratio below 1-eps: objective is (1-eps)A, no gradient
    l, g = grad_wrt_logp(fn, torch.log(torch.tensor([[0.5]])), torch.zeros(1, 1), -one, one, 0.2)
    out["neg_adv_low_ratio_capped"] = bool(abs(float(l) - 0.8) < 1e-5 and float(g.abs().sum()) == 0.0)

    # A > 0 and ratio below 1-eps: unclipped side, gradient still pushes the ratio up
    l, g = grad_wrt_logp(fn, torch.log(torch.tensor([[0.5]])), torch.zeros(1, 1), one, one, 0.2)
    out["pos_adv_low_ratio_keeps_gradient"] = bool(abs(float(l) + 0.5) < 1e-5 and float(g.sum()) < 0)

    # random inputs with padding should match the formula
    new, old, adv = 0.3 * torch.randn(5, 7), 0.3 * torch.randn(5, 7), torch.randn(5, 7)
    mask = (torch.arange(7)[None, :] < torch.tensor([7, 5, 3, 6, 1])[:, None]).float()
    for eps in (0.05, 0.2, 0.5):
        got = fn(new, old, adv, mask, eps=eps)[0]
        out[f"matches_formula_eps{eps}"] = bool(torch.allclose(got, clipped_surrogate(new, old, adv, mask, eps), atol=1e-6))
    return out


def gae_check():
    torch.manual_seed(1)
    r, v = torch.randn(3, 6), torch.randn(3, 6)
    mask = (torch.arange(6)[None, :] < torch.tensor([6, 4, 2])[:, None]).float()
    adv, ret = compute_gae(r * mask, v * mask, mask, gamma=1.0, lam=0.95)
    ref = naive_gae(r * mask, v * mask, mask, 1.0, 0.95)
    return {"gae_matches_naive": bool(torch.allclose(adv, ref * mask, atol=1e-5)),
            "returns_equal_adv_plus_value": bool(torch.allclose(ret, adv + v * mask, atol=1e-6))}


def main():
    fixed = {**run_checks(ppo_policy_loss), **gae_check()}
    starter = run_checks(starter_loss)
    print("Fixed ppo_policy_loss + GAE:")
    for k, v in fixed.items():
        print(f"  {k:36s} {'PASS' if v else 'FAIL'}")
    print("Starter version (should fail some of these):")
    for k, v in starter.items():
        print(f"  {k:36s} {'PASS' if v else 'FAIL'}")
    save_json(repo_path("results/task2_ppo/ppo_checks.json"), {"fixed": fixed, "starter": starter})
    if not all(fixed.values()):
        raise SystemExit("PPO objective FAILED validation - do not train.")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
