import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from baseline_continuous.loarm_utils import gumbel_top_k, plackett_luce_logprob

def test_gumbel_top_k_shape():
    """gumbel_top_k returns a valid permutation of shape [B, L]."""
    B, L = 4, 31
    logits = torch.zeros(B, L)
    z = gumbel_top_k(logits)
    assert z.shape == (B, L), f"Expected ({B},{L}), got {z.shape}"
    # Each row is a permutation of 0..L-1
    for b in range(B):
        assert set(z[b].tolist()) == set(range(L)), "Not a permutation"


def test_gumbel_top_k_differentiable_logits():
    """gumbel_top_k output is integer (no gradient through z itself)."""
    B, L = 2, 5
    logits = torch.randn(B, L, requires_grad=True)
    z = gumbel_top_k(logits)
    assert z.dtype == torch.long


def test_plackett_luce_logprob_sums_to_one():
    """Sum of exp(log_q) over all possible z_k choices at each step ≈ 1."""
    B, L = 1, 4
    logits = torch.randn(B, L)
    z = gumbel_top_k(logits)
    # For each step k, sum over all candidates not yet chosen should be 1
    for k in range(L):
        log_probs = []
        for candidate in range(L):
            if candidate not in z[0, :k].tolist():
                z_mod = z.clone()
                z_mod[0, k] = candidate
                lp = plackett_luce_logprob(logits, z_mod, step_k=k)
                log_probs.append(lp[0].item())
        total = sum(torch.tensor(lp).exp().item() for lp in log_probs)
        assert abs(total - 1.0) < 1e-4, f"Step {k}: sum={total}"


def test_plackett_luce_logprob_shape():
    """plackett_luce_logprob returns [B] tensor."""
    B, L = 4, 31
    logits = torch.zeros(B, L)
    z = gumbel_top_k(logits)
    for k in range(L):
        lp = plackett_luce_logprob(logits, z, step_k=k)
        assert lp.shape == (B,), f"Step {k}: expected ({B},), got {lp.shape}"
