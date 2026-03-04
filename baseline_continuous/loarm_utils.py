"""
Gumbel-top-k sampling and Plackett-Luce log-probability utilities for LO-ARM.

Gumbel-top-k samples from the Plackett-Luce distribution:
  p(z_k = j | z_{<k}, θ) = exp(θ_j) / Σ_{i ∉ z_{<k}} exp(θ_i)

Sampling: z = argsort(θ + Gumbel noise, descending)
Log-prob: log p(z_k | z_{<k}) = θ_{z_k} - logsumexp(θ[remaining])
"""

import torch


def gumbel_top_k(logits: torch.Tensor) -> torch.Tensor:
    """
    Sample a full permutation from Plackett-Luce via Gumbel-top-k trick.

    Args:
        logits: [B, L] raw scores
    Returns:
        z: [B, L] long tensor, each row is a permutation of 0..L-1
    """
    gumbel_noise = -torch.log(-torch.log(
        torch.clamp(torch.rand_like(logits), min=1e-20, max=1.0 - 1e-7)
    ))
    noisy_logits = logits + gumbel_noise
    return torch.argsort(noisy_logits, dim=-1, descending=True)


def plackett_luce_logprob(logits: torch.Tensor, z: torch.Tensor, step_k: int) -> torch.Tensor:
    """
    Compute log p_PL(z_k | z_{<k}, logits) for a specific step k.

    Args:
        logits: [B, L] raw scores for the Plackett-Luce model
        z:      [B, L] full permutation (each row is a perm of 0..L-1)
        step_k: int, which step to compute (0-indexed)
    Returns:
        log_prob: [B] log probability of choosing z[:, step_k] at step step_k
    """
    B, L = logits.shape
    chosen = z[:, step_k]  # [B]

    mask = torch.ones(B, L, dtype=torch.bool, device=logits.device)
    if step_k > 0:
        already_chosen = z[:, :step_k]  # [B, step_k]
        mask.scatter_(1, already_chosen, False)

    masked_logits = logits.masked_fill(~mask, float('-inf'))  # [B, L]
    log_denom = torch.logsumexp(masked_logits, dim=-1)         # [B]
    log_num = logits.gather(1, chosen.unsqueeze(1)).squeeze(1)  # [B]

    return log_num - log_denom


def plackett_luce_prefix_logprob(
    logits: torch.Tensor,
    z: torch.Tensor,
    step_k: int,
) -> torch.Tensor:
    """
    Compute log q(z_{<k} | x) = Σ_{j=0}^{k-1} log PL(z_j | z_{<j}, logits).

    This is the REINFORCE coefficient in LO-ARM Eq.(16): the log probability
    of the prefix (first k elements of z), NOT the log prob at step k itself.

    Args:
        logits:  [B, L] Plackett-Luce scores
        z:       [B, L] full permutation
        step_k:  int in [0, L-1]. k=0 → empty prefix → returns zeros.
    Returns:
        log_prob: [B]  cumulative log prob of z[:, 0:k]
    """
    B, L = logits.shape
    if not (0 <= step_k <= L):
        raise ValueError(f"step_k must be in [0, L], got {step_k} with L={L}")
    if step_k == 0:
        return torch.zeros(B, device=logits.device, dtype=logits.dtype)

    # logits_perm[b, j] = logits[b, z[b, j]]
    logits_perm = logits.gather(1, z)   # [B, L]

    # Denominator at step j = logsumexp(logits_perm[b, j:])
    # via reversed cumulative logsumexp
    flipped   = torch.flip(logits_perm, dims=[1])
    cumlogsum = torch.logcumsumexp(flipped, dim=1)
    log_denom = torch.flip(cumlogsum, dims=[1])

    # Note: at j=L-1, log_probs_per_step[:,L-1] is exactly 0 (last item has prob 1).
    log_probs_per_step = logits_perm - log_denom   # [B, L]
    return log_probs_per_step[:, :step_k].sum(dim=1)  # [B]
