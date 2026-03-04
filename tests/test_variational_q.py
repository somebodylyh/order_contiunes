# tests/test_variational_q.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from baseline_continuous.variational_q import QNetwork
from baseline_continuous.loarm_utils import gumbel_top_k, plackett_luce_logprob


def test_qnetwork_logits_shape():
    """QNetwork returns [B, L] logits."""
    B, L, D = 4, 31, 1024
    q_net = QNetwork(vector_dim=D, seq_len=L, hidden_dim=256)
    x = torch.randn(B, L, D)
    logits = q_net(x)
    assert logits.shape == (B, L), f"Expected ({B},{L}), got {logits.shape}"


def test_qnetwork_sampling():
    """QNetwork + gumbel_top_k returns valid permutations."""
    B, L, D = 4, 31, 1024
    q_net = QNetwork(vector_dim=D, seq_len=L, hidden_dim=256)
    x = torch.randn(B, L, D)
    logits = q_net(x)
    z = gumbel_top_k(logits)
    assert z.shape == (B, L)
    for b in range(B):
        assert set(z[b].tolist()) == set(range(L))


def test_qnetwork_gradients_flow():
    """Gradient flows from plackett_luce_logprob through QNetwork parameters."""
    B, L, D = 2, 5, 16
    q_net = QNetwork(vector_dim=D, seq_len=L, hidden_dim=32)
    x = torch.randn(B, L, D)
    logits = q_net(x)
    z = gumbel_top_k(logits)
    # Compute log prob at step 0
    lp = plackett_luce_logprob(logits, z, step_k=0)
    loss = -lp.mean()
    loss.backward()
    # Check that at least one parameter has a gradient
    grads = [p.grad for p in q_net.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients flowed to QNetwork"
