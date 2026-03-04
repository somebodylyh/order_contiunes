"""
Amortized variational network q_θ(z | x) for LO-ARM.

Takes the FULL unshuffled x (all L vectors, training-time only "god view") and
outputs Plackett-Luce logits for sampling a generation ordering z.

Architecture:
    mean_pool(x) → Linear(D, 4H) → GELU → Linear(4H, L) → logits [B, L]
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Lightweight MLP that maps the full observed sequence to Plackett-Luce logits.

    Inputs x at training time have the 'god view' - all L h-vectors are present
    because the training data provides them. During inference, q_net is NOT used.

    Args:
        vector_dim:  D, dimension of each h-vector
        seq_len:     L, number of main tokens to order
        hidden_dim:  hidden dimension (default 256, much smaller than D=1024)
    """

    def __init__(self, vector_dim: int, seq_len: int, hidden_dim: int = 256):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(vector_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, seq_len),
        )
        # Zero-init output → starts as uniform Plackett-Luce (like AO-ARM)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, D] full unshuffled main vectors (god view)
        Returns:
            logits: [B, L] Plackett-Luce scores
        """
        # Mean pool over sequence dimension: [B, D]
        x_pooled = x.mean(dim=1)
        # MLP: [B, D] → [B, L]
        return self.net(x_pooled)
