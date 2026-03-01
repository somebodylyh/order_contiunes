"""
Continuous Transformer Model for Dense AR Vector Prediction

Key differences from discrete AOGPT:
- Input: Linear projection from vector space (D) instead of token embedding
- Output: Linear projection to vector space (D) instead of vocab logits
- Loss: MSE instead of CrossEntropy
- Position encoding: KEPT (temporal order matters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class CausalSelfAttention(nn.Module):
    """Causal self-attention with optional causal masking."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Key, query, value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x: torch.Tensor, use_causal_mask: bool = True) -> torch.Tensor:
        B, T, C = x.size()

        # Calculate query, key, values
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        if use_causal_mask:
            att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        # Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm architecture."""

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, use_causal_mask: bool = True) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), use_causal_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class ContinuousTransformerConfig:
    """Configuration for ContinuousTransformer."""

    def __init__(
        self,
        vector_dim: int = 32,
        n_layer: int = 4,
        n_head: int = 4,
        n_embd: int = 256,
        block_size: int = 16,
        dropout: float = 0.0,
        bias: bool = True
    ):
        self.vector_dim = vector_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size
        self.dropout = dropout
        self.bias = bias


class ContinuousTransformer(nn.Module):
    """
    Continuous Transformer for Dense AR vector prediction.

    Architecture:
    - Input projection: Linear(D, n_embd)
    - Positional encoding: Learned embeddings
    - Transformer blocks: Standard causal attention
    - Output projection: Linear(n_embd, D)
    - Loss: MSE
    """

    def __init__(self, config: ContinuousTransformerConfig):
        super().__init__()
        self.config = config

        # Input projection (instead of token embedding)
        self.input_proj = nn.Linear(config.vector_dim, config.n_embd)

        # Positional encoding (KEPT - temporal order matters)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output projection (instead of lm_head to vocab)
        self.output_proj = nn.Linear(config.n_embd, config.vector_dim)

        # Initialize weights
        self.apply(self._init_weights)

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[INFO] ContinuousTransformer initialized with {n_params/1e6:.2f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(
        self,
        vectors: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for teacher-forcing training.

        Args:
            vectors: [B, L, D] input vectors (in generation order)
            targets: [B, L, D] target vectors (shifted by 1)

        Returns:
            predictions: [B, L, D] predicted vectors
            loss: MSE loss if targets provided
        """
        B, L, D = vectors.size()
        device = vectors.device

        # Input projection
        x = self.input_proj(vectors)  # [B, L, n_embd]

        # Add positional encoding
        positions = torch.arange(0, L, dtype=torch.long, device=device)
        pos_emb = self.wpe(positions)  # [L, n_embd]
        x = self.drop(x + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, use_causal_mask=True)

        # Final layer norm
        x = self.ln_f(x)

        # Output projection
        predictions = self.output_proj(x)  # [B, L, D]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.mse_loss(predictions, targets)

        return predictions, loss

    def forward_with_hidden(
        self,
        vectors: torch.Tensor,
        order: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_hidden_states: bool = False,
        loss_type: str = 'mse'
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with explicit ordering and optional hidden states.

        Args:
            vectors: [B, L, D] input vectors (shuffled)
            order: [B, L] generation order (indices into vectors)
            targets: [B, L, D] target vectors (in original order)
            return_hidden_states: Whether to return hidden states

        Returns:
            predictions: [B, L, D] predicted vectors (in generation order)
            loss: MSE loss if targets provided
            hidden_states: [B, L, n_embd] if return_hidden_states=True
        """
        B, L, D = vectors.size()
        device = vectors.device

        # Reorder vectors according to order
        # order[b, t] = index of vector to use at position t
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        ordered_vectors = vectors[batch_idx, order]  # [B, L, D]

        # Input projection
        x = self.input_proj(ordered_vectors)  # [B, L, n_embd]

        # Add positional encoding
        positions = torch.arange(0, L, dtype=torch.long, device=device)
        pos_emb = self.wpe(positions)  # [L, n_embd]
        x = self.drop(x + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, use_causal_mask=True)

        # Final layer norm
        hidden_states = self.ln_f(x)  # [B, L, n_embd]

        # Output projection
        predictions = self.output_proj(hidden_states)  # [B, L, D]

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            # Targets are in original order, reorder to match predictions
            ordered_targets = targets[batch_idx, order]  # [B, L, D]
            # For next-step prediction: predict position t from positions 0..t-1
            # predictions[:, t] should predict targets[:, t+1] (shifted)
            # Or: predictions[:, :-1] predicts targets[:, 1:]
            pred_for_loss = predictions[:, :-1]  # [B, L-1, D]
            target_for_loss = ordered_targets[:, 1:]  # [B, L-1, D]
            if loss_type == 'mse':
                loss = F.mse_loss(pred_for_loss, target_for_loss)
            elif loss_type == 'cosine':
                cos_sim = F.cosine_similarity(pred_for_loss, target_for_loss, dim=-1)  # [B, L-1]
                loss = (1.0 - cos_sim).mean()
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")

        if return_hidden_states:
            return predictions, loss, hidden_states
        else:
            return predictions, loss, None

    def forward_ar(
        self,
        shuffled_vectors: torch.Tensor,
        order: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Autoregressive forward pass for evaluation.

        Given shuffled vectors and an order, generate predictions autoregressively.

        Args:
            shuffled_vectors: [B, L, D] shuffled input vectors
            order: [B, L] generation order

        Returns:
            predictions: [B, L, D] predicted vectors (in generation order)
            mse_per_step: [B, L] MSE at each step
        """
        B, L, D = shuffled_vectors.size()
        device = shuffled_vectors.device

        # Reorder to get ground truth in generation order
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        ordered_vectors = shuffled_vectors[batch_idx, order]  # [B, L, D]

        predictions = torch.zeros(B, L, D, device=device)
        mse_per_step = torch.zeros(B, L, device=device)

        # First vector is given (no prediction)
        predictions[:, 0] = ordered_vectors[:, 0]

        for t in range(1, L):
            # Input: vectors up to position t-1
            input_vectors = predictions[:, :t].clone()

            # Project and add positions
            x = self.input_proj(input_vectors)
            positions = torch.arange(t, dtype=torch.long, device=device)
            pos_emb = self.wpe(positions)
            x = x + pos_emb

            # Transformer blocks (no dropout in eval)
            for block in self.blocks:
                x = block(x, use_causal_mask=True)

            # Get last position output
            x = self.ln_f(x[:, -1:])  # [B, 1, n_embd]
            pred = self.output_proj(x).squeeze(1)  # [B, D]

            # Store prediction (normalized for stability)
            predictions[:, t] = F.normalize(pred, p=2, dim=-1)

            # Compute MSE against ground truth
            mse_per_step[:, t] = F.mse_loss(
                predictions[:, t], ordered_vectors[:, t], reduction='none'
            ).mean(dim=-1)

        return predictions, mse_per_step

    def sample_random_orders(
        self,
        batch_size: int,
        seq_length: int,
        device: torch.device
    ) -> torch.Tensor:
        """Generate random orders for warmup phase."""
        orders = torch.stack([
            torch.randperm(seq_length, device=device)
            for _ in range(batch_size)
        ])
        return orders


def test_model():
    """Test the ContinuousTransformer."""
    print("=" * 60)
    print("Testing ContinuousTransformer")
    print("=" * 60)

    # Create config
    config = ContinuousTransformerConfig(
        vector_dim=32,
        n_layer=4,
        n_head=4,
        n_embd=256,
        block_size=16,
        dropout=0.0,
        bias=True
    )

    # Create model
    model = ContinuousTransformer(config)
    model.eval()

    # Test forward pass
    print("\n1. Testing forward pass...")
    B, L, D = 8, 16, 32
    vectors = torch.randn(B, L, D)
    vectors = F.normalize(vectors, p=2, dim=-1)
    targets = torch.randn(B, L, D)
    targets = F.normalize(targets, p=2, dim=-1)

    predictions, loss = model(vectors, targets)
    print(f"   Input shape: {vectors.shape}")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Loss: {loss.item():.4f}")
    assert predictions.shape == (B, L, D), "Prediction shape mismatch"
    print("   ✓ Forward pass works")

    # Test forward_with_hidden
    print("\n2. Testing forward_with_hidden...")
    order = torch.stack([torch.randperm(L) for _ in range(B)])
    predictions, loss, hidden = model.forward_with_hidden(
        vectors, order, targets, return_hidden_states=True
    )
    print(f"   Order shape: {order.shape}")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Hidden states shape: {hidden.shape}")
    print(f"   Loss: {loss.item():.4f}")
    assert hidden.shape == (B, L, config.n_embd), "Hidden state shape mismatch"
    print("   ✓ forward_with_hidden works")

    # Test forward_ar
    print("\n3. Testing autoregressive forward...")
    predictions_ar, mse_per_step = model.forward_ar(vectors, order)
    print(f"   Predictions shape: {predictions_ar.shape}")
    print(f"   MSE per step shape: {mse_per_step.shape}")
    print(f"   Mean MSE: {mse_per_step.mean().item():.4f}")
    print("   ✓ forward_ar works")

    # Test random order sampling
    print("\n4. Testing random order sampling...")
    random_orders = model.sample_random_orders(B, L, vectors.device)
    print(f"   Random orders shape: {random_orders.shape}")
    # Verify each row is a permutation
    for i in range(B):
        assert set(random_orders[i].tolist()) == set(range(L)), "Invalid permutation"
    print("   ✓ Random order sampling works")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_model()
