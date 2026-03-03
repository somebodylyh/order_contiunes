"""
Continuous Vector AOGPT Model

Based on model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm.py,
adapted for continuous vector input/output with MSE loss.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import random


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))

    def forward(self, input):
        return F.rms_norm(input, self.weight.shape, self.weight, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(0.)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.q_norm = RMSNorm(self.n_embd // self.n_head)
        self.k_norm = RMSNorm(self.n_embd // self.n_head)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
                                        .view(1, 1, config.block_size + 1, config.block_size + 1))

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(128, 6 * config.n_embd, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.adaLN(c)).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.ln_1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.ln_2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_f = RMSNorm(config.n_embd)
        self.adaLN_final = nn.Sequential(
            nn.SiLU(),
            nn.Linear(128, 2 * config.n_embd, bias=True)
        )

    def forward(self, x, c):
        shift, scale = (self.adaLN_final(c)).chunk(2, dim=-1)
        x = modulate(self.ln_f(x), shift, scale)
        return x


@dataclass
class ContinuousAOGPTConfig:
    block_size: int = 32
    vector_dim: int = 64       # replaces vocab_size
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = True
    num_init: int = 0          # number of init vectors used as fixed conditioning prefix


class ContinuousAOGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wpe = nn.Embedding(config.block_size + 1, config.n_embd),  # +1 for [None] token
            wtpe = nn.Embedding(config.block_size, 128),               # target position encoding
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            final_layer = FinalLayer(config),
        ))

        # Continuous input projection (replaces wte embedding)
        self.input_proj = nn.Linear(config.vector_dim, config.n_embd)

        # Learnable [None] start token (replaces wnonee embedding)
        self.none_token = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)

        # Continuous output projection (replaces lm_head)
        self.output_proj = nn.Linear(config.n_embd, config.vector_dim, bias=True)

        # Learnable [MASK] vector for MDM
        self.mask_token = nn.Parameter(torch.randn(config.vector_dim) * 0.02)

        # Init all weights
        self.apply(self._init_weights)
        # Special scaled init for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.trunc_normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer),
                                            a=-3*0.02/math.sqrt(2 * config.n_layer),
                                            b=3*0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-3*0.02, b=3*0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=0.02, a=-3*0.02, b=3*0.02)

    def sample_random_orders(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        shuffled_orders = []
        for _ in range(batch_size):
            shuffled_orders.append(torch.randperm(seq_length, device=x.device))
        shuffled_orders = torch.stack(shuffled_orders)
        return shuffled_orders.to(x.device)

    def sample_random_orders_CL(self, x, random_ratio):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        shuffled_orders = []
        for _ in range(batch_size):
            if random.random() < random_ratio:
                shuffled_orders.append(torch.randperm(seq_length, device=x.device))
            else:
                shuffled_orders.append(torch.arange(seq_length, device=x.device))
        shuffled_orders = torch.stack(shuffled_orders)
        return shuffled_orders.to(x.device)

    def set_ascending_orders(self, x):
        batch_size = x.shape[0]
        seq_length = x.shape[1]
        shuffled_orders = []
        for _ in range(batch_size):
            shuffled_orders.append(torch.arange(seq_length, device=x.device))
        shuffled_orders = torch.stack(shuffled_orders)
        return shuffled_orders.to(x.device)

    def shuffle(self, x, orders):
        batch_size, seq_len = x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        shuffled_x = x[batch_indices, orders]
        return shuffled_x

    def unshuffle(self, shuffled_x, orders):
        batch_size, seq_len = shuffled_x.shape[:2]
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, seq_len)
        unshuffled_x = torch.zeros_like(shuffled_x)
        unshuffled_x[batch_indices, orders] = shuffled_x
        return unshuffled_x

    def forward(self, vectors, mode='Random', orders=None, random_ratio=None, init_vectors=None):
        if mode is None:
            assert orders is not None, 'mode is None, orders must be provided'
            return self.forward_fn(vectors, orders, init_vectors)
        elif mode == 'AR':
            orders = self.set_ascending_orders(vectors)
            return self.forward_fn(vectors, orders, init_vectors)
        elif mode == 'Random':
            orders = self.sample_random_orders(vectors)
            return self.forward_fn(vectors, orders, init_vectors)
        elif mode == 'Random_CL':
            assert random_ratio is not None
            orders = self.sample_random_orders_CL(vectors, random_ratio)
            return self.forward_fn(vectors, orders, init_vectors)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def forward_fn(self, vectors, orders, init_vectors=None):
        """
        vectors:      [B, L, D]        main tokens to predict (excl. init prefix)
        orders:       [B, L]           generation order for main tokens
        init_vectors: [B, num_init, D] fixed conditioning prefix (always visible, not predicted)
                                       If None, falls back to legacy [None]-token mode.
        """
        device = vectors.device
        b, t, d = vectors.size()

        if init_vectors is None:
            # ── Legacy mode: prepend learnable [None] token, predict all L tokens ──
            assert t <= self.config.block_size
            pos = torch.arange(0, t + 1, dtype=torch.long, device=device)
            vectors_shuffled = self.shuffle(vectors, orders)
            targets = vectors_shuffled
            tok_emb = self.input_proj(vectors_shuffled)
            none_emb = self.none_token.expand(b, -1, -1)
            tok_emb = torch.cat([none_emb, tok_emb], dim=1)
            pos_emb = self.transformer.wpe(pos).unsqueeze(0).expand(b, -1, -1)
            pos_emb_prefix = pos_emb[:, :1]
            pos_emb_postfix = self.shuffle(pos_emb[:, 1:], orders)
            target_pos_emb = self.transformer.wtpe(pos[:t]).unsqueeze(0).expand(b, -1, -1)
            target_pos_emb_prefix = self.shuffle(target_pos_emb, orders)
            target_pos_emb_postfix = torch.zeros_like(target_pos_emb[:, :1])
            target_pos_emb_final = torch.cat([target_pos_emb_prefix, target_pos_emb_postfix], dim=1)
            x = tok_emb + torch.cat([pos_emb_prefix, pos_emb_postfix], dim=1)
            x = self.transformer.drop(x)
            for block in self.transformer.h:
                x = block(x, target_pos_emb_final)
            x = self.transformer.final_layer(x, target_pos_emb_final)
            predictions = self.output_proj(x)
            shift_preds = predictions[:, :-1, :]
            loss = F.mse_loss(shift_preds, targets)
            return predictions, loss

        # ── Init-prefix mode ──
        # Sequence: [init_0,...,init_{ni-1}, main_shuf[0],...,main_shuf[t-1]]  (total = ni + t)
        # Loss computed on the last t positions (main tokens only).
        # init_vectors are always visible via causal attention; no loss on them.
        ni = init_vectors.shape[1]                        # num_init
        assert ni + t <= self.config.block_size + 1, \
            f"ni+t={ni+t} exceeds block_size+1={self.config.block_size+1}"

        # Shuffle main vectors
        main_shuffled = self.shuffle(vectors, orders)     # [B, t, D]
        targets = main_shuffled

        # Input embeddings: [init_emb | main_emb_shuffled]
        init_emb = self.input_proj(init_vectors)          # [B, ni, C]
        main_emb = self.input_proj(main_shuffled)         # [B, t,  C]
        tok_emb  = torch.cat([init_emb, main_emb], dim=1)  # [B, ni+t, C]

        # Position embeddings: init at positions 0..ni-1; main at original positions ni..ni+t-1
        pos_init = torch.arange(ni, dtype=torch.long, device=device)
        pos_main_all = torch.arange(ni, ni + t, dtype=torch.long, device=device)
        init_pos_emb = self.transformer.wpe(pos_init).unsqueeze(0).expand(b, -1, -1)   # [B, ni, C]
        main_pos_emb = self.transformer.wpe(pos_main_all).unsqueeze(0).expand(b, -1, -1)  # [B, t, C]
        main_pos_emb_shuf = main_pos_emb                                                 # [B, t, C], seq-order pos (no shuffle)
        x = tok_emb + torch.cat([init_pos_emb, main_pos_emb_shuf], dim=1)               # [B, ni+t, C]

        # Target-position embeddings for AdaLN:
        # Tells the model which generation step it is predicting (0-indexed).
        # For init positions 0..ni-2: zeros (not used in loss).
        # For position ni-1 (last init): step 0 (predicts main_shuf[0]).
        # For position ni+i (main_shuf[i], i<t-1): step i+1 (predicts main_shuf[i+1]).
        # For position ni+t-1 (last main): zeros.
        step_idx = torch.arange(t, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)  # [B, t], generation step index
        tpe_main = self.transformer.wtpe(step_idx)        # [B, t, 128]
        zeros_early = torch.zeros(b, ni - 1, 128, device=device)   # [B, ni-1, 128]
        zeros_last  = torch.zeros(b, 1,      128, device=device)   # [B, 1,    128]
        # layout: [zeros(ni-1) | tpe_main(t) | zeros(1)] = ni+t total
        target_pos_emb_final = torch.cat([zeros_early, tpe_main, zeros_last], dim=1)  # [B, ni+t, 128]

        # Transformer forward
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x, target_pos_emb_final)
        x = self.transformer.final_layer(x, target_pos_emb_final)
        predictions = self.output_proj(x)                 # [B, ni+t, D]

        # Loss: prediction at positions [ni-1 .. ni+t-2] predicts main_shuf[0..t-1]
        loss_preds = predictions[:, ni - 1 : ni - 1 + t, :]   # [B, t, D]
        loss = F.mse_loss(loss_preds, targets)

        return predictions, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer


def test_model():
    """Test ContinuousAOGPT forward pass shapes."""
    print("=" * 60)
    print("Testing ContinuousAOGPT")
    print("=" * 60)

    config = ContinuousAOGPTConfig(
        block_size=32,
        vector_dim=64,
        n_layer=4,
        n_head=8,
        n_embd=256,
        dropout=0.0,
        bias=True,
    )
    model = ContinuousAOGPT(config)

    B, L, D = 4, 32, 64
    vectors = torch.randn(B, L, D)

    # Test AR mode
    print("\n1. Testing AR mode...")
    preds, loss = model(vectors, mode='AR')
    print(f"   predictions shape: {preds.shape}")  # [B, L+1, D]
    print(f"   loss: {loss.item():.4f}")
    assert preds.shape == (B, L + 1, D), f"Expected {(B, L+1, D)}, got {preds.shape}"
    print("   OK")

    # Test Random mode
    print("\n2. Testing Random mode...")
    preds, loss = model(vectors, mode='Random')
    print(f"   predictions shape: {preds.shape}")
    print(f"   loss: {loss.item():.4f}")
    assert preds.shape == (B, L + 1, D)
    print("   OK")

    # Test Random_CL mode
    print("\n3. Testing Random_CL mode...")
    preds, loss = model(vectors, mode='Random_CL', random_ratio=0.5)
    print(f"   predictions shape: {preds.shape}")
    print(f"   loss: {loss.item():.4f}")
    assert preds.shape == (B, L + 1, D)
    print("   OK")

    # Test custom orders
    print("\n4. Testing custom orders...")
    orders = torch.stack([torch.randperm(L) for _ in range(B)])
    preds, loss = model(vectors, mode=None, orders=orders)
    print(f"   predictions shape: {preds.shape}")
    print(f"   loss: {loss.item():.4f}")
    assert preds.shape == (B, L + 1, D)
    print("   OK")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_model()
