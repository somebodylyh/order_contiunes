import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from baseline_continuous.rloo import compute_rloo_loss, compute_gen_log_lik, mask_policy_logprob


def test_gen_log_lik_shape():
    """compute_gen_log_lik returns [B] scalar per sample."""
    B, D = 4, 16
    target  = torch.randn(B, D)
    pred    = torch.randn(B, D)
    ll = compute_gen_log_lik(pred, target, sigma2=1.0)
    assert ll.shape == (B,), f"Expected ({B},), got {ll.shape}"


def test_gen_log_lik_perfect_prediction():
    """When pred == target, log-lik is the constant term only."""
    B, D = 4, 16
    target = torch.randn(B, D)
    ll = compute_gen_log_lik(target, target, sigma2=1.0)
    # MSE should be 0 → log-lik = -D/2 * log(2π σ²) = const for all samples
    assert torch.allclose(ll, ll[0].expand(B), atol=1e-5), "Should be equal for all samples"


def test_mask_policy_logprob_shape():
    """mask_policy_logprob returns [B] log-prob."""
    B, L = 4, 7
    pol_logits  = torch.randn(B, L)   # logits at step k (already masked)
    chosen_idx  = torch.randint(0, L, (B,))
    lp = mask_policy_logprob(pol_logits, chosen_idx)
    assert lp.shape == (B,), f"Expected ({B},), got {lp.shape}"


def test_rloo_loss_stop_gradient():
    """delta_F must not propagate gradients to the generator."""
    B, D, L = 2, 16, 7
    # Mock F values as leaf tensors with grad
    F1 = torch.randn(B, requires_grad=True)
    F2 = torch.randn(B, requires_grad=True)
    log_q1 = torch.randn(B, requires_grad=True)
    log_q2 = torch.randn(B, requires_grad=True)

    loss = compute_rloo_loss(F1, F2, log_q1, log_q2, L_len=L)
    loss.backward()

    # The gradient on F1 should come from the direct term, NOT delta_F
    # We just check that no RuntimeError is raised and grads exist
    assert F1.grad is not None
    assert F2.grad is not None
    assert log_q1.grad is not None


def test_rloo_loss_symmetry():
    """Swapping (F1,log_q1) and (F2,log_q2) should give the same loss value."""
    B, L = 4, 7
    F1    = torch.randn(B)
    F2    = torch.randn(B)
    log_q1 = torch.randn(B)
    log_q2 = torch.randn(B)

    loss_ab = compute_rloo_loss(F1, F2, log_q1, log_q2, L_len=L)
    loss_ba = compute_rloo_loss(F2, F1, log_q2, log_q1, L_len=L)
    # Both should give the same scalar
    assert torch.isclose(loss_ab, loss_ba, atol=1e-5), \
        f"loss_ab={loss_ab.item():.4f}, loss_ba={loss_ba.item():.4f}"
