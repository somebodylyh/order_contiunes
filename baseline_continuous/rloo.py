"""
RLOO (REINFORCE Leave-One-Out) loss for LO-ARM training.

Based on LO-ARM paper Equation (11):

  gradient ≈ L/2 * {
    (∇log_q(z1_{<i}|x) - ∇log_q(z2_{<i}|x)) * ΔF.detach()
    + ∇F_θ(z1_{<i}, x)
    + ∇F_θ(z2_{<i}, x)
  }

where F_θ(z_{<i}, x) = log p_gen(x_{z_i} | z_{<i}) + log p_policy(z_i | z_{<i}) - log q(z_i | z_{<i}, x)

We express this as a scalar loss whose .backward() gives the correct gradient:

  loss = -0.5 * L * (F1 + F2)                   ← direct F gradient
       + 0.5 * L * ΔF.detach() * (log_q1 - log_q2)  ← REINFORCE on q; ★ stop-grad on ΔF

Key invariant: ΔF = (F1 - F2) MUST be detached to avoid spurious feedback
               into the generator via the control-variate term.
"""

import torch
import torch.nn.functional as F
import math


def compute_gen_log_lik(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma2: float,
) -> torch.Tensor:
    """
    Gaussian log-likelihood: log N(target; pred, σ²I).

    Args:
        pred:   [B, D] predicted vector
        target: [B, D] ground-truth vector
        sigma2: float, noise variance (key temperature hyperparameter)
    Returns:
        log_lik: [B]  (includes constant terms, but they cancel in RLOO ΔF)
    """
    D = pred.shape[-1]
    mse_per_sample = ((pred - target) ** 2).sum(dim=-1)  # [B]
    log_lik = -0.5 * mse_per_sample / sigma2 \
              - 0.5 * D * math.log(2 * math.pi * sigma2)
    return log_lik


def mask_policy_logprob(
    pol_logits_at_k: torch.Tensor,
    chosen: torch.Tensor,
) -> torch.Tensor:
    """
    log p_policy(z_k = chosen | z_{<k}) from already-masked policy logits.

    Args:
        pol_logits_at_k: [B, L]  logits at step k, with selected positions
                                  already set to -inf by apply_policy_mask
        chosen:          [B]     int64 indices of the chosen position
    Returns:
        log_prob: [B]
    """
    log_probs = F.log_softmax(pol_logits_at_k, dim=-1)  # [B, L]
    return log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)  # [B]


def compute_rloo_loss(
    F1: torch.Tensor,
    F2: torch.Tensor,
    log_q1: torch.Tensor,
    log_q2: torch.Tensor,
    L_len: int,
) -> torch.Tensor:
    """
    Scalar loss whose .backward() gives the RLOO gradient estimate.

    Args:
        F1:     [B]  F_θ(z1_{<i}, x)  — MUST be computed with grad
        F2:     [B]  F_θ(z2_{<i}, x)  — MUST be computed with grad
        log_q1: [B]  log q(z1_i | z1_{<i}, x) — MUST be computed with grad
        log_q2: [B]  log q(z2_i | z2_{<i}, x) — MUST be computed with grad
        L_len:  int  sequence length L (scaling factor from Eq. 11)
    Returns:
        loss: scalar (mean over batch)

    ★ CRITICAL: delta_F is detached. This is the RLOO control-variate trick.
      Without .detach(), gradients would flow from the control-variate term
      back into the generator, breaking the unbiasedness of RLOO.
    """
    delta_F = (F1 - F2).detach()   # ★ stop gradient here

    # Direct F terms: gradient flows to both generator and policy
    direct_term = -0.5 * L_len * (F1 + F2)

    # REINFORCE term: gradient only flows to q_net (log_q1, log_q2)
    reinforce_term = 0.5 * L_len * delta_F * (log_q1 - log_q2)

    loss = direct_term + reinforce_term
    return loss.mean()
