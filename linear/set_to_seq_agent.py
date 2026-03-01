"""
Set-to-Sequence Agent for Continuous Vector Ordering

Architecture:
- SetEncoder: Permutation-invariant encoder (NO positional encoding)
- PointerDecoder: Autoregressive decoder WITH step positional encoding

The agent takes a shuffled set of vectors and outputs a permutation
that reorders them to the correct temporal sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class SetEncoderLayer(nn.Module):
    """
    Single transformer encoder layer without positional encoding.
    Permutation invariant by design.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model]
        Returns:
            [B, L, d_model]
        """
        # Self-attention (no positional encoding - permutation invariant)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class SetEncoder(nn.Module):
    """
    Permutation-invariant encoder - NO positional encoding.

    Takes a set of vectors and encodes them without any position information,
    ensuring the encoder is truly permutation invariant.
    """

    def __init__(
        self,
        vector_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.input_proj = nn.Linear(vector_dim, d_model)
        self.layers = nn.ModuleList([
            SetEncoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        # NO positional embedding!

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vectors: [B, L, D] input vectors (shuffled set)
        Returns:
            [B, L, d_model] content encodings (permutation invariant)
        """
        # Project to model dimension
        x = self.input_proj(vectors)  # [B, L, d_model]

        # Apply transformer layers (no position info added)
        for layer in self.layers:
            x = layer(x)

        return x


class PointerDecoderLayer(nn.Module):
    """
    Decoder layer with self-attention and cross-attention to encoder outputs.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        # Causal self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention to encoder
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model] decoder input
            encoder_output: [B, L, d_model] encoder output
            self_attn_mask: [T, T] causal mask for self-attention
            cross_attn_mask: [B, T, L] mask for cross-attention (filled positions)
        """
        # Causal self-attention
        attn_out, _ = self.self_attn(x, x, x, attn_mask=self_attn_mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Cross-attention to encoder (with mask for filled positions)
        if cross_attn_mask is not None:
            # Convert to attention mask format: True means ignore
            cross_attn_out, _ = self.cross_attn(
                x, encoder_output, encoder_output,
                key_padding_mask=cross_attn_mask
            )
        else:
            cross_attn_out, _ = self.cross_attn(x, encoder_output, encoder_output)
        x = self.norm2(x + self.dropout(cross_attn_out))

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)

        return x


class PointerDecoder(nn.Module):
    """
    Autoregressive decoder WITH step positional encoding.

    For each step t, outputs a pointer distribution over unfilled positions
    in the encoder output.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_len: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        # Step positional embedding (HAS position encoding)
        self.step_embedding = nn.Embedding(max_len, d_model)

        # Start token (learnable)
        self.start_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Decoder layers
        self.layers = nn.ModuleList([
            PointerDecoderLayer(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Pointer projection
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)

        # Register causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        )

    def forward_step(
        self,
        encoder_output: torch.Tensor,
        decoder_input: torch.Tensor,
        filled_mask: torch.Tensor,
        step: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single decoding step.

        Args:
            encoder_output: [B, L, d_model]
            decoder_input: [B, t, d_model] previous decoder states
            filled_mask: [B, L] True for already selected positions
            step: current step index

        Returns:
            pointer_logits: [B, L] logits over positions
            decoder_state: [B, t+1, d_model] updated decoder states
        """
        B, L, _ = encoder_output.size()
        device = encoder_output.device

        # Add step embedding to last decoder input
        if decoder_input is None:
            # Use start token for first step
            x = self.start_token.expand(B, 1, -1)
        else:
            x = decoder_input

        # Add step position embedding
        step_positions = torch.arange(x.size(1), dtype=torch.long, device=device)
        step_emb = self.step_embedding(step_positions)
        x = x + step_emb

        # Apply decoder layers
        t = x.size(1)
        causal_mask = self.causal_mask[:t, :t]
        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask=causal_mask)

        # Get last position output for pointer
        query = self.query_proj(x[:, -1:])  # [B, 1, d_model]
        keys = self.key_proj(encoder_output)  # [B, L, d_model]

        # Compute pointer logits
        pointer_logits = torch.matmul(query, keys.transpose(-2, -1))  # [B, 1, L]
        pointer_logits = pointer_logits.squeeze(1) / math.sqrt(self.d_model)  # [B, L]

        # Mask filled positions
        pointer_logits = pointer_logits.masked_fill(filled_mask, float('-inf'))

        return pointer_logits, x

    def decode(
        self,
        encoder_output: torch.Tensor,
        teacher_forcing_targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full autoregressive decoding.

        Args:
            encoder_output: [B, L, d_model]
            teacher_forcing_targets: [B, L] ground truth order for teacher forcing
            teacher_forcing_ratio: probability of using teacher forcing

        Returns:
            permutation: [B, L] selected indices
            log_probs: [B, L] log probabilities of selected actions
            all_logits: [B, L, L] logits at each step (for BC loss)
            entropy: scalar, mean entropy of the policy across batch and steps
        """
        B, L, _ = encoder_output.size()
        device = encoder_output.device

        permutation = torch.zeros(B, L, dtype=torch.long, device=device)
        log_probs = torch.zeros(B, L, device=device)
        all_logits = torch.zeros(B, L, L, device=device)
        entropies = torch.zeros(B, L, device=device)
        filled_mask = torch.zeros(B, L, dtype=torch.bool, device=device)

        decoder_input = None

        for step in range(L):
            # Get pointer logits
            pointer_logits, decoder_state = self.forward_step(
                encoder_output, decoder_input, filled_mask, step
            )
            all_logits[:, step] = pointer_logits

            # Sample or use teacher forcing
            probs = F.softmax(pointer_logits, dim=-1)

            # Compute entropy of the distribution over valid (unmasked) positions
            # Use Categorical which handles the distribution properly
            dist = torch.distributions.Categorical(probs=probs)
            entropies[:, step] = dist.entropy()  # [B]

            if teacher_forcing_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # Teacher forcing: use ground truth
                action = teacher_forcing_targets[:, step]
            else:
                # Sample from distribution
                action = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Compute log prob
            log_prob = torch.log(probs.gather(1, action.unsqueeze(1)) + 1e-10).squeeze(-1)

            # Store results
            permutation[:, step] = action
            log_probs[:, step] = log_prob

            # Update filled mask
            filled_mask = filled_mask.scatter(1, action.unsqueeze(1), True)

            # Update decoder input with selected encoder output
            selected = encoder_output.gather(
                1, action.unsqueeze(1).unsqueeze(2).expand(-1, -1, encoder_output.size(-1))
            )  # [B, 1, d_model]

            if decoder_input is None:
                decoder_input = selected
            else:
                decoder_input = torch.cat([decoder_input, selected], dim=1)

        # Mean entropy across batch and sequence steps
        entropy = entropies.mean()

        return permutation, log_probs, all_logits, entropy


class SetToSeqAgent(nn.Module):
    """
    Set-to-Sequence Agent combining SetEncoder and PointerDecoder.

    Takes shuffled vectors and outputs a permutation to reorder them.
    """

    def __init__(
        self,
        vector_dim: int = 32,
        d_model: int = 256,
        encoder_layers: int = 2,
        encoder_heads: int = 4,
        decoder_layers: int = 2,
        decoder_heads: int = 4,
        max_len: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.encoder = SetEncoder(
            vector_dim=vector_dim,
            d_model=d_model,
            n_heads=encoder_heads,
            n_layers=encoder_layers,
            dropout=dropout
        )

        self.decoder = PointerDecoder(
            d_model=d_model,
            n_heads=decoder_heads,
            n_layers=decoder_layers,
            max_len=max_len,
            dropout=dropout
        )

        # Initialize weights
        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[INFO] SetToSeqAgent initialized with {n_params/1e6:.2f}M parameters")

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
        teacher_forcing_targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.0,
        return_all_logits: bool = False,
        return_entropy: bool = False
    ):
        """
        Forward pass.

        Args:
            vectors: [B, L, D] shuffled input vectors
            teacher_forcing_targets: [B, L] ground truth order for teacher forcing
            teacher_forcing_ratio: probability of using teacher forcing
            return_all_logits: whether to return logits for BC loss
            return_entropy: whether to return mean policy entropy

        Returns:
            permutation: [B, L] predicted permutation
            log_probs: [B, L] log probabilities of actions
            all_logits: [B, L, L] logits at each step (if return_all_logits, else None)
            entropy: scalar mean entropy (if return_entropy, else not returned)
        """
        # Encode (permutation invariant)
        encoder_output = self.encoder(vectors)  # [B, L, d_model]

        # Decode (sequential with position encoding)
        permutation, log_probs, all_logits, entropy = self.decoder.decode(
            encoder_output,
            teacher_forcing_targets,
            teacher_forcing_ratio
        )

        logits_out = all_logits if return_all_logits else None

        if return_entropy:
            return permutation, log_probs, logits_out, entropy
        else:
            return permutation, log_probs, logits_out

    def get_action_probabilities(
        self,
        vectors: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute action probabilities for given actions (for policy gradient).

        Args:
            vectors: [B, L, D] input vectors
            actions: [B, L] actions taken

        Returns:
            log_probs: [B, L] log probabilities
        """
        encoder_output = self.encoder(vectors)

        B, L, _ = encoder_output.size()
        device = encoder_output.device

        log_probs = torch.zeros(B, L, device=device)
        filled_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        decoder_input = None

        for step in range(L):
            pointer_logits, decoder_state = self.decoder.forward_step(
                encoder_output, decoder_input, filled_mask, step
            )

            probs = F.softmax(pointer_logits, dim=-1)
            action = actions[:, step]
            log_prob = torch.log(probs.gather(1, action.unsqueeze(1)) + 1e-10).squeeze(-1)
            log_probs[:, step] = log_prob

            # Update mask and decoder input
            filled_mask = filled_mask.scatter(1, action.unsqueeze(1), True)
            selected = encoder_output.gather(
                1, action.unsqueeze(1).unsqueeze(2).expand(-1, -1, encoder_output.size(-1))
            )
            if decoder_input is None:
                decoder_input = selected
            else:
                decoder_input = torch.cat([decoder_input, selected], dim=1)

        return log_probs


def test_agent():
    """Test the SetToSeqAgent."""
    print("=" * 60)
    print("Testing SetToSeqAgent")
    print("=" * 60)

    # Create agent
    agent = SetToSeqAgent(
        vector_dim=32,
        d_model=256,
        encoder_layers=2,
        encoder_heads=4,
        decoder_layers=2,
        decoder_heads=4,
        max_len=16,
        dropout=0.0
    )
    agent.eval()

    # Test forward pass
    print("\n1. Testing forward pass...")
    B, L, D = 8, 16, 32
    vectors = torch.randn(B, L, D)
    vectors = F.normalize(vectors, p=2, dim=-1)

    permutation, log_probs, _ = agent(vectors)
    print(f"   Input shape: {vectors.shape}")
    print(f"   Permutation shape: {permutation.shape}")
    print(f"   Log probs shape: {log_probs.shape}")

    # Verify permutation is valid
    for b in range(B):
        assert set(permutation[b].tolist()) == set(range(L)), "Invalid permutation"
    print("   ✓ All permutations are valid")

    # Test with teacher forcing
    print("\n2. Testing teacher forcing...")
    gt_order = torch.stack([torch.randperm(L) for _ in range(B)])
    permutation_tf, log_probs_tf, all_logits = agent(
        vectors,
        teacher_forcing_targets=gt_order,
        teacher_forcing_ratio=1.0,
        return_all_logits=True
    )
    print(f"   Permutation with TF shape: {permutation_tf.shape}")
    print(f"   All logits shape: {all_logits.shape}")
    # With 100% TF, permutation should match gt_order
    assert torch.equal(permutation_tf, gt_order), "Teacher forcing not working"
    print("   ✓ Teacher forcing works correctly")

    # Test permutation invariance of encoder
    print("\n3. Testing encoder permutation invariance...")
    # Create a shuffled version of vectors
    shuffle_idx = torch.randperm(L)
    vectors_shuffled = vectors[:, shuffle_idx]

    # Encode both
    enc_original = agent.encoder(vectors)
    enc_shuffled = agent.encoder(vectors_shuffled)

    # The encodings should be permuted versions of each other
    # enc_shuffled[:, i] should match enc_original[:, shuffle_idx[i]]
    enc_original_reordered = enc_original[:, shuffle_idx]

    diff = (enc_shuffled - enc_original_reordered).abs().max().item()
    print(f"   Max difference after reordering: {diff:.2e}")
    assert diff < 1e-5, "Encoder is not permutation equivariant"
    print("   ✓ Encoder is permutation equivariant")

    # Test get_action_probabilities
    print("\n4. Testing get_action_probabilities...")
    actions = torch.stack([torch.randperm(L) for _ in range(B)])
    log_probs_check = agent.get_action_probabilities(vectors, actions)
    print(f"   Log probs shape: {log_probs_check.shape}")
    print(f"   Mean log prob: {log_probs_check.mean().item():.4f}")
    print("   ✓ get_action_probabilities works")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_agent()
