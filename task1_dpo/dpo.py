from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """DPO loss on summed response log-probs, plus a few numbers for logging."""

    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    # starter had policy_margin + ref_margin here. DPO compares the policy's margin
    # against the reference's, so it has to be a minus (see check_dpo_loss.py)
    logits = beta * (policy_margin - ref_margin)

    loss = -F.logsigmoid(logits).mean()

    # implicit rewards, beta * log(pi/ref)
    chosen_reward = beta * (policy_chosen_logp - ref_chosen_logp)
    rejected_reward = beta * (policy_rejected_logp - ref_rejected_logp)

    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "ref_margin_mean": ref_margin.detach().mean(),
        "chosen_reward_mean": chosen_reward.detach().mean(),
        "rejected_reward_mean": rejected_reward.detach().mean(),
        "preference_accuracy": ((policy_margin - ref_margin) > 0).float().mean().detach(),
    }
