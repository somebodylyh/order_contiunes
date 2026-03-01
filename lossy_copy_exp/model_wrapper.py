"""
AOGPT Model Wrapper with Hidden States and Logical Position Embeddings

This wrapper extends the base AOGPT model to:
1. Expose hidden states before the lm_head projection
2. Add logical position embeddings to distinguish token roles (x vs y)
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

# Add parent directory to path to import AOGPT
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPT


class AOGPTWithHiddenStates(AOGPT):
    """
    Extended AOGPT that exposes hidden states and adds logical position embeddings.

    CRITICAL: Logical position embeddings tell the model which token is x vs y.
    Without them, the model sees value '15' but doesn't know its role.

    Additional components:
        - logical_pos_emb: Embedding to distinguish x (pos 0) from y (pos 1)
    """

    def __init__(self, config):
        super().__init__(config)

        # CRITICAL: Add logical position embedding
        # This tells the model "which token is x, which is y"
        # Without this, overlapping value ranges make tokens indistinguishable
        self.logical_pos_emb = nn.Embedding(config.block_size, config.n_embd)

        # Initialize with small weights (matching AOGPT initialization)
        nn.init.trunc_normal_(
            self.logical_pos_emb.weight,
            mean=0.0,
            std=0.02,
            a=-3*0.02,
            b=3*0.02
        )

        print(f"Added logical position embeddings: {config.block_size} x {config.n_embd}")

    def forward_with_hidden(self, idx, orders, logical_ids=None, return_hidden_states=False):
        """
        Forward pass with optional hidden state extraction.

        This method is a copy of forward_fn from the base AOGPT model (lines 271-313),
        with modifications to:
        1. Inject logical position embeddings
        2. Return hidden states before lm_head projection

        Args:
            idx: [B, T] token indices
            orders: [B, T] shuffle orders for autoregressive generation
            logical_ids: [B, T] logical position IDs (0=x, 1=y) - REQUIRED for training
            return_hidden_states: If True, return hidden states [B, T+1, n_embd]

        Returns:
            If return_hidden_states=False: (logits, loss)
            If return_hidden_states=True: (logits, loss, hidden_states)
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t+1, dtype=torch.long, device=device)  # shape (t+1) to include the [None] token

        # Shuffle input ids given orders
        idx = self.shuffle(idx, orders)  # of shape (b, t)
        targets = idx  # of shape (b, t)

        # Prepare token embedding, position embedding, target position embedding and shuffle them
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        none_tok_emb = self.transformer.wnonee(torch.tensor([[0]], device=idx.device))  # [None] token embedding (1, 1, n_embd)
        none_tok_emb = none_tok_emb.expand(idx.shape[0], -1, -1)  # expand to (b, 1, n_embd)
        tok_emb = torch.cat([none_tok_emb, tok_emb], dim=1)  # concat to (b, t+1, n_embd)

        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t+1, n_embd)
        pos_emb = pos_emb.unsqueeze(0).expand(idx.shape[0], -1, -1)  # expand to (b, t+1, n_embd)
        pos_emb_prefix = pos_emb[:, :1]  # position embedding prefix on the [None] token (b, 1, n_embd)
        pos_emb_postfix = self.shuffle(pos_emb[:, 1:], orders)  # position embedding postfix (b, t, n_embd); shuffled

        target_pos_emb = self.transformer.wtpe(pos[:t])  # target position embeddings (t, n_embd)
        target_pos_emb = target_pos_emb.unsqueeze(0).expand(idx.shape[0], -1, -1)  # expand to (b, t, n_embd)
        target_pos_emb_prefix = self.shuffle(target_pos_emb, orders)  # shuffle target position embeddings (b, t, n_embd)
        target_pos_emb_postfix = torch.zeros_like(target_pos_emb[:, :1])  # zeros of shape (b, 1, n_embd)
        target_pos_emb_final = torch.cat([target_pos_emb_prefix, target_pos_emb_postfix], dim=1)

        # Base embeddings (from line 297 in original model)
        x = tok_emb + torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1)

        # CRITICAL MODIFICATION: Add logical position embeddings
        if logical_ids is not None:
            # Get logical embeddings for each token
            logical_emb = self.logical_pos_emb(logical_ids)  # [B, T, n_embd]

            # Shuffle to match physical order
            logical_emb_shuffled = self.shuffle(logical_emb, orders)  # [B, T, n_embd]

            # Concatenate with [None] token (zero logical embedding for [None])
            logical_emb_prefix = torch.zeros_like(logical_emb[:, :1])  # [B, 1, n_embd]
            logical_emb_final = torch.cat([logical_emb_prefix, logical_emb_shuffled], dim=1)  # [B, T+1, n_embd]

            # Add to embeddings
            x = x + logical_emb_final

        # Forward the GPT model itself (lines 300-303 in original model)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, target_pos_emb_final)
        x = self.transformer.final_layer(x, target_pos_emb_final)

        # Store hidden states BEFORE lm_head projection (line 303 output)
        hidden_states = x.clone() if return_hidden_states else None

        # If we are given some desired targets also calculate the loss
        logits = self.lm_head(x)  # line 307
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_targets.view(-1), ignore_index=-1)

        if return_hidden_states:
            return logits, loss, hidden_states
        else:
            return logits, loss

    def forward(self, idx, mode='Random', orders=None, random_ratio=None, logical_ids=None, return_hidden_states=False):
        """
        Forward pass with mode selection (compatible with base AOGPT interface).

        Extends base forward to support logical_ids and return_hidden_states.
        """
        if mode is None:
            assert orders is not None and idx.shape == orders.shape, 'mode is None, order should be given and with the same shape of idx'
        else:
            assert mode in ['AR', 'Random', 'Random_CL'], 'mode should be AR or Random or Random_CL'

        # Generate orders if needed
        if mode == 'AR':
            orders = self.set_ascending_orders(idx)
        elif mode == 'Random':
            orders = self.sample_random_orders(idx)
        elif mode == 'Random_CL':
            assert random_ratio is not None
            orders = self.sample_random_orders_CL(idx, random_ratio)

        # Use forward_with_hidden for all cases
        return self.forward_with_hidden(idx, orders, logical_ids=logical_ids, return_hidden_states=return_hidden_states)


def test_model_wrapper():
    """Test the AOGPTWithHiddenStates implementation."""
    print("Testing AOGPTWithHiddenStates...")

    from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPTConfig

    # Create tiny config for testing
    config = AOGPTConfig(
        vocab_size=64,
        block_size=2,
        n_layer=2,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        bias=True
    )

    # Initialize model
    model = AOGPTWithHiddenStates(config)
    model.eval()
    print(f"✓ Model initialized: {model.get_num_params()/1e6:.2f}M parameters")

    # Test forward pass without hidden states
    B, T = 4, 2
    idx = torch.randint(0, 64, (B, T))
    orders = torch.stack([torch.randperm(T) for _ in range(B)])
    logical_ids = torch.tensor([[0, 1]] * B)

    logits, loss = model.forward_with_hidden(idx, orders, logical_ids=logical_ids, return_hidden_states=False)
    print(f"✓ Forward pass (no hidden): logits shape {logits.shape}, loss {loss.item():.4f}")
    assert logits.shape == (B, T+1, 64), f"Expected logits shape {(B, T+1, 64)}, got {logits.shape}"

    # Test forward pass with hidden states
    logits, loss, hidden = model.forward_with_hidden(idx, orders, logical_ids=logical_ids, return_hidden_states=True)
    print(f"✓ Forward pass (with hidden): hidden shape {hidden.shape}")
    assert hidden.shape == (B, T+1, 128), f"Expected hidden shape {(B, T+1, 128)}, got {hidden.shape}"

    # Test forward pass without logical_ids (should still work)
    logits_no_logical, loss_no_logical = model.forward_with_hidden(idx, orders, logical_ids=None, return_hidden_states=False)
    print(f"✓ Forward pass (no logical_ids): loss {loss_no_logical.item():.4f}")

    # Test with different modes
    logits_ar, loss_ar = model(idx, mode='AR', logical_ids=logical_ids)
    print(f"✓ AR mode: loss {loss_ar.item():.4f}")

    logits_random, loss_random = model(idx, mode='Random', logical_ids=logical_ids)
    print(f"✓ Random mode: loss {loss_random.item():.4f}")

    print("\n✅ All model wrapper tests passed!")


if __name__ == '__main__':
    test_model_wrapper()
