import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
from baseline_continuous.continuous_loarm import ContinuousLOARM
from baseline_continuous.loarm_utils import gumbel_top_k


def make_base_model():
    config = ContinuousAOGPTConfig(
        block_size=8, vector_dim=16, n_layer=2, n_head=2, n_embd=32,
        dropout=0.0, bias=True, num_init=1,
    )
    return ContinuousAOGPT(config)


def test_loarm_output_shapes():
    """forward_loarm returns gen_preds [B,L,D] and pol_logits [B,L,L]."""
    base = make_base_model()
    model = ContinuousLOARM(base)
    B, L, D = 4, 7, 16   # 7 main tokens (block_size - num_init = 8-1)
    vectors = torch.randn(B, L, D)
    init_vectors = torch.randn(B, 1, D)
    orders = torch.stack([torch.randperm(L) for _ in range(B)])

    gen_preds, pol_logits = model.forward_loarm(vectors, orders, init_vectors)

    assert gen_preds.shape == (B, L, D), f"gen_preds: expected ({B},{L},{D}), got {gen_preds.shape}"
    assert pol_logits.shape == (B, L, L), f"pol_logits: expected ({B},{L},{L}), got {pol_logits.shape}"


def test_loarm_policy_masking():
    """apply_policy_mask zeroes out already-selected positions."""
    base = make_base_model()
    model = ContinuousLOARM(base)
    B, L = 4, 7
    logits = torch.randn(B, L, L)   # [B, t_steps, L]
    orders = torch.stack([torch.randperm(L) for _ in range(B)])

    masked = model.apply_policy_mask(logits, orders)
    # At step k, positions orders[:, :k] should be -inf
    for b in range(B):
        for k in range(1, L):
            selected = orders[b, :k].tolist()
            for pos in selected:
                assert masked[b, k, pos].item() == float('-inf'), \
                    f"Step {k}, pos {pos} should be -inf"


def test_loarm_warm_start():
    """ContinuousLOARM can load weights from a ContinuousAOGPT checkpoint dict."""
    base = make_base_model()
    loarm = ContinuousLOARM(base)
    # Save base model state dict
    ckpt = {'model_state_dict': base.state_dict(), 'config': base.config}
    # Load into a new instance
    base2 = ContinuousAOGPT(base.config)
    base2.load_state_dict(ckpt['model_state_dict'])
    loarm2 = ContinuousLOARM(base2)
    # Policy head should still be freshly initialized (zeros)
    assert loarm2.policy_head.weight.abs().max().item() == 0.0


def test_loarm_gradient_flows_to_policy_head():
    """Gradient from MSE on gen_preds flows to policy_head too (shared backbone)."""
    base = make_base_model()
    model = ContinuousLOARM(base)
    B, L, D = 2, 7, 16
    vectors = torch.randn(B, L, D)
    init_vectors = torch.randn(B, 1, D)
    orders = torch.stack([torch.randperm(L) for _ in range(B)])

    gen_preds, pol_logits = model.forward_loarm(vectors, orders, init_vectors)
    # Compute a simple loss using gen_preds
    loss = ((gen_preds - vectors) ** 2).mean()
    loss.backward()
    # Backbone parameters should have gradients
    assert base.output_proj.weight.grad is not None
