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
    """delta_F detach means F1.grad == -0.5*L/B regardless of log_q_prefix values."""
    B, L = 2, 7
    F1            = torch.randn(B, requires_grad=True)
    F2            = torch.randn(B, requires_grad=True)
    log_q1_prefix = torch.randn(B, requires_grad=True)
    log_q2_prefix = torch.randn(B, requires_grad=True)

    loss = compute_rloo_loss(F1, F2, log_q1_prefix, log_q2_prefix, L_len=L)
    loss.backward()

    expected_grad = torch.full((B,), -0.5 * L / B)
    assert torch.allclose(F1.grad, expected_grad, atol=1e-5), \
        f"F1.grad={F1.grad} — delta_F detach may be broken"
    assert torch.allclose(F2.grad, expected_grad, atol=1e-5), \
        f"F2.grad={F2.grad}"
    assert log_q1_prefix.grad is not None
    assert log_q2_prefix.grad is not None


def test_rloo_loss_symmetry():
    """Swapping (F1,log_q1_prefix) and (F2,log_q2_prefix) gives the same loss."""
    B, L = 4, 7
    F1            = torch.randn(B)
    F2            = torch.randn(B)
    log_q1_prefix = torch.randn(B)
    log_q2_prefix = torch.randn(B)

    loss_ab = compute_rloo_loss(F1, F2, log_q1_prefix, log_q2_prefix, L_len=L)
    loss_ba = compute_rloo_loss(F2, F1, log_q2_prefix, log_q1_prefix, L_len=L)
    assert torch.isclose(loss_ab, loss_ba, atol=1e-5), \
        f"loss_ab={loss_ab.item():.4f}, loss_ba={loss_ba.item():.4f}"


def test_rloo_reinforce_gradient_only_on_prefix():
    """REINFORCE term gradient flows to log_q_prefix, not to F1/F2 directly."""
    B, L = 2, 7
    F1            = torch.randn(B, requires_grad=True)
    F2            = torch.randn(B, requires_grad=True)
    log_q1_prefix = torch.randn(B, requires_grad=True)
    log_q2_prefix = torch.randn(B, requires_grad=True)

    loss = compute_rloo_loss(F1, F2, log_q1_prefix, log_q2_prefix, L_len=L)
    loss.backward()

    # F1 gradient must be exactly -0.5*L/B (no REINFORCE contribution to F)
    expected_f_grad = torch.full((B,), -0.5 * L / B)
    assert torch.allclose(F1.grad, expected_f_grad, atol=1e-5)
    # log_q_prefix gradient must be non-zero (REINFORCE provides signal)
    assert log_q1_prefix.grad.abs().sum() > 0
