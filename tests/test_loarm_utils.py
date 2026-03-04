import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from baseline_continuous.loarm_utils import gumbel_top_k, plackett_luce_logprob
from baseline_continuous.loarm_utils import plackett_luce_prefix_logprob

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


def test_prefix_logprob_shape():
    """Returns [B] tensor for any valid k."""
    B, L = 4, 8
    logits = torch.randn(B, L)
    z = gumbel_top_k(logits)
    for k in range(L):
        lp = plackett_luce_prefix_logprob(logits, z, step_k=k)
        assert lp.shape == (B,), f"k={k}: expected ({B},), got {lp.shape}"


def test_prefix_logprob_k0_is_zero():
    """k=0 means empty prefix → log prob = 0 for all samples."""
    B, L = 4, 8
    logits = torch.randn(B, L)
    z = gumbel_top_k(logits)
    lp = plackett_luce_prefix_logprob(logits, z, step_k=0)
    assert torch.allclose(lp, torch.zeros(B)), f"k=0 should be zeros, got {lp}"


def test_prefix_logprob_k1_equals_step0():
    """k=1 prefix has one element → equals plackett_luce_logprob at step 0."""
    B, L = 4, 8
    logits = torch.randn(B, L)
    z = gumbel_top_k(logits)
    prefix_lp = plackett_luce_prefix_logprob(logits, z, step_k=1)
    step0_lp   = plackett_luce_logprob(logits, z, step_k=0)
    assert torch.allclose(prefix_lp, step0_lp, atol=1e-5), \
        f"k=1 prefix should equal step-0 log prob"


def test_prefix_logprob_matches_loop():
    """Vectorized result matches naive loop over plackett_luce_logprob."""
    B, L = 3, 6
    logits = torch.randn(B, L)
    z = gumbel_top_k(logits)
    for k in range(L):
        expected = sum(
            plackett_luce_logprob(logits, z, step_k=j) for j in range(k)
        ) if k > 0 else torch.zeros(B)
        got = plackett_luce_prefix_logprob(logits, z, step_k=k)
        # atol=1e-4: logcumsumexp and sequential logsumexp differ in float32 evaluation order
        assert torch.allclose(got, expected, atol=1e-4), \
            f"k={k}: vectorized={got} vs loop={expected}"


def test_prefix_logprob_gradient_flows():
    """Gradient of prefix log prob is numerically correct (verified with gradcheck)."""
    import torch.autograd
    B, L, k = 2, 5, 3
    logits = torch.randn(B, L, dtype=torch.float64, requires_grad=True)   # float64 for gradcheck
    z = gumbel_top_k(logits.detach().float()).long()   # int permutation, no grad

    def fn(log):
        return plackett_luce_prefix_logprob(log, z, step_k=k)

    assert torch.autograd.gradcheck(fn, (logits,), eps=1e-6, atol=1e-4, rtol=1e-3), \
        "Gradient of plackett_luce_prefix_logprob failed gradcheck"


def test_prefix_logprob_full_permutation():
    """step_k=L returns sum of all L step log-probs (full permutation log prob)."""
    B, L = 3, 5
    logits = torch.randn(B, L)
    z = gumbel_top_k(logits)
    full = plackett_luce_prefix_logprob(logits, z, step_k=L)
    expected = sum(plackett_luce_logprob(logits, z, step_k=j) for j in range(L))
    assert torch.allclose(full, expected, atol=1e-4)
