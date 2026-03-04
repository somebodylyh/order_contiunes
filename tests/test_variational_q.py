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
    """Gradient flows from plackett_luce_logprob through QNetwork output layer."""
    B, L, D = 2, 5, 16
    q_net = QNetwork(vector_dim=D, seq_len=L, hidden_dim=32)
    x = torch.randn(B, L, D)
    logits = q_net(x)
    z = gumbel_top_k(logits)
    lp = plackett_luce_logprob(logits, z, step_k=0)
    loss = -lp.mean()
    loss.backward()

    # net[2] (output projection) must receive a nonzero gradient — it is
    # directly in the logits→loss path.
    out_grad = q_net.net[2].weight.grad
    assert out_grad is not None and out_grad.abs().max() > 0, \
        "Output layer received no gradient"

    # net[0] (input projection) has zero gradient at step 0 by design:
    # zero-init on net[2].weight means d_loss/d_hidden=0 at initialisation.
    # The grad tensor is still allocated (backward was called).
    in_grad = q_net.net[0].weight.grad
    assert in_grad is not None, "Input layer grad tensor was not allocated"
