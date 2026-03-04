"""
ContinuousLOARM: ContinuousAOGPT backbone + learnable order-policy head.

Shared-torso design (LO-ARM paper Section 3.3):
  - Backbone transformer computes hidden states
  - Generator head (existing output_proj) predicts continuous h-vectors
  - Policy head (new Linear(n_embd, L)) predicts generation order logits

During training: forward_loarm() returns both heads' outputs for RLOO.
During inference: apply_policy_mask() + softmax gives next-position distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_continuous.continuous_aogpt import ContinuousAOGPT


class ContinuousLOARM(nn.Module):
    """
    Wraps ContinuousAOGPT and adds an order-policy head.

    Args:
        base_model: a ContinuousAOGPT instance (will be modified in-place via
                    joint parameter updates)
    """

    def __init__(self, base_model: ContinuousAOGPT):
        super().__init__()
        self.base = base_model
        L = base_model.config.block_size - base_model.config.num_init  # = 31
        C = base_model.config.n_embd                                    # = 1024

        # Policy head: shared backbone output -> L logits
        # Zero-init: at the start, policy is uniform (approx AO-ARM)
        self.policy_head = nn.Linear(C, L)
        nn.init.zeros_(self.policy_head.weight)
        nn.init.zeros_(self.policy_head.bias)

    def forward_loarm(
        self,
        vectors: torch.Tensor,
        orders: torch.Tensor,
        init_vectors: torch.Tensor,
    ):
        """
        Single forward pass returning both the generator predictions and
        the raw (unmasked) order-policy logits.

        Args:
            vectors:      [B, L, D]  main tokens (unshuffled, original order)
            orders:       [B, L]     generation ordering (each row = permutation)
            init_vectors: [B, 1, D]  always-visible h0 prefix
        Returns:
            gen_preds:  [B, L, D]  predicted vectors at each generation step
                                    gen_preds[:, k, :] predicts vectors[orders[:, k], :]
            pol_logits: [B, L, L]  raw policy logits before masking
                                    pol_logits[:, k, :] = scores for choosing
                                    the (k+1)-th position given context k
        """
        base = self.base
        device = vectors.device
        b, t, d = vectors.size()
        ni = init_vectors.shape[1]
        assert ni + t <= base.config.block_size + 1, \
            f"ni+t={ni+t} exceeds block_size+1={base.config.block_size+1}"

        # Replicate ContinuousAOGPT.forward_fn init-prefix path but
        # intercept hidden states before output_proj to also run policy_head.

        # Shuffle main vectors by the given ordering
        main_shuffled = base.shuffle(vectors, orders)   # [B, t, D]

        # Token embeddings
        init_emb = base.input_proj(init_vectors)        # [B, ni, C]
        main_emb = base.input_proj(main_shuffled)       # [B, t,  C]
        tok_emb  = torch.cat([init_emb, main_emb], dim=1)  # [B, ni+t, C]

        # Positional embeddings (sequence position, NOT shuffled)
        pos_init     = torch.arange(ni, dtype=torch.long, device=device)
        pos_main_all = torch.arange(ni, ni + t, dtype=torch.long, device=device)
        init_pos_emb = base.transformer.wpe(pos_init).unsqueeze(0).expand(b, -1, -1)
        main_pos_emb = base.transformer.wpe(pos_main_all).unsqueeze(0).expand(b, -1, -1)
        x = tok_emb + torch.cat([init_pos_emb, main_pos_emb], dim=1)   # [B, ni+t, C]

        # AdaLN step-index conditioning
        # Layout: [zeros(ni-1) | tpe_main(t) | zeros(1)] = ni+t total
        step_idx = torch.arange(t, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
        tpe_main     = base.transformer.wtpe(step_idx)                  # [B, t, tpe_dim]
        tpe_dim      = tpe_main.shape[-1]
        zeros_early  = torch.zeros(b, ni - 1, tpe_dim, device=device)  # [B, ni-1, tpe_dim]
        zeros_last   = torch.zeros(b, 1,      tpe_dim, device=device)  # [B, 1,    tpe_dim]
        adaLN_cond   = torch.cat([zeros_early, tpe_main, zeros_last], dim=1)  # [B, ni+t, tpe_dim]

        # Transformer forward
        x = base.transformer.drop(x)
        for block in base.transformer.h:
            x = block(x, adaLN_cond)
        x = base.transformer.final_layer(x, adaLN_cond)   # [B, ni+t, C]

        # Extract hidden states at main-step positions: [B, t, C]
        # Position ni-1+k (0-indexed) predicts x[orders[:,k]] at step k.
        h_steps = x[:, ni - 1: ni - 1 + t, :]  # [B, t, C]

        # Generator head (existing)
        gen_preds  = base.output_proj(h_steps)      # [B, t, D]

        # Policy head (new)
        pol_logits = self.policy_head(h_steps)      # [B, t, L]

        return gen_preds, pol_logits

    def apply_policy_mask(
        self,
        pol_logits: torch.Tensor,
        orders: torch.Tensor,
    ) -> torch.Tensor:
        """
        Zero out (set to -inf) already-selected positions in the policy logits.

        At step k, positions orders[:, 0:k] have already been selected and must
        not be chosen again.

        Args:
            pol_logits: [B, t, L]  raw logits
            orders:     [B, t]     generation ordering used in this forward pass
        Returns:
            masked_logits: [B, t, L]  with -inf at already-selected positions
        """
        B, t, L = pol_logits.shape
        masked = pol_logits.clone()
        for k in range(1, t):
            # positions already selected at steps 0..k-1
            selected = orders[:, :k]          # [B, k]
            # Scatter -inf into masked[:, k, selected]
            masked[:, k, :].scatter_(1, selected, float('-inf'))
        return masked

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Configure AdamW with weight decay on 2D+ params."""
        import inspect
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params,   'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
