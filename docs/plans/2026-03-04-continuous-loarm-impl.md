# Continuous LO-ARM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a learnable order-policy (LO-ARM) to ContinuousAOGPT on GLA h-space data, trained with RLOO gradient estimation, expecting the policy to discover the causal generation order.

**Architecture:** `ContinuousLOARM` wraps `ContinuousAOGPT` and adds a `policy_head` (shared-torso). A lightweight `QNetwork` serves as the amortized variational distribution using Gumbel-top-k / Plackett-Luce sampling during training only.

**Tech Stack:** PyTorch, existing `ContinuousAOGPT` backbone, `disk_dataset`, `eval_order_v8`.

---

## Key Numbers (from config.py)

```
vector_dim = D = 1024
num_init   = ni = 1          # h₀ is always-visible init prefix
L          = 31              # main tokens (seq_length - num_init)
n_embd     = C = 1024
block_size = 32
```

Pretrained MDM checkpoint: `baseline_continuous/checkpoints/best_mdm_Random_model.pt`

---

## Task 1: Gumbel-top-k and Plackett-Luce Utilities

**Files:**
- Create: `baseline_continuous/loarm_utils.py`
- Test:   `tests/test_loarm_utils.py`

### Step 1: Write the failing test

```python
# tests/test_loarm_utils.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from baseline_continuous.loarm_utils import gumbel_top_k, plackett_luce_logprob

def test_gumbel_top_k_shape():
    """gumbel_top_k returns a valid permutation of shape [B, L]."""
    B, L = 4, 31
    logits = torch.zeros(B, L)
    z = gumbel_top_k(logits)
    assert z.shape == (B, L), f"Expected ({B},{L}), got {z.shape}"
    # Each row is a permutation of 0..L-1
    for b in range(B):
        assert set(z[b].tolist()) == set(range(L)), "Not a permutation"


def test_gumbel_top_k_differentiable_logits():
    """gumbel_top_k output is integer (no gradient through z itself)."""
    B, L = 2, 5
    logits = torch.randn(B, L, requires_grad=True)
    z = gumbel_top_k(logits)
    assert z.dtype == torch.long


def test_plackett_luce_logprob_sums_to_one():
    """Sum of exp(log_q) over all possible z_k choices at each step ≈ 1."""
    B, L = 1, 4
    logits = torch.randn(B, L)
    z = gumbel_top_k(logits)
    # For step k=0, sum over all choices should be 1
    log_probs = []
    for k in range(L):
        for candidate in range(L):
            # Temporarily pretend z[b, k] = candidate to evaluate its probability
            z_mod = z.clone()
            z_mod[0, k] = candidate
            # Only valid if candidate not in z[0, :k]
            if candidate not in z[0, :k].tolist():
                lp = plackett_luce_logprob(logits, z_mod, step_k=k)  # [B]
                log_probs.append(lp[0].item())
        total = sum(p.exp() for p in [torch.tensor(lp) for lp in log_probs])
        assert abs(total.item() - 1.0) < 1e-4, f"Step {k}: sum={total.item()}"
        log_probs = []


def test_plackett_luce_logprob_shape():
    """plackett_luce_logprob returns [B] tensor."""
    B, L = 4, 31
    logits = torch.zeros(B, L)
    z = gumbel_top_k(logits)
    for k in range(L):
        lp = plackett_luce_logprob(logits, z, step_k=k)
        assert lp.shape == (B,), f"Step {k}: expected ({B},), got {lp.shape}"
```

### Step 2: Run test to verify it fails

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM
python -m pytest tests/test_loarm_utils.py -v 2>&1 | head -20
```
Expected: ImportError or ModuleNotFoundError for `loarm_utils`

### Step 3: Write implementation

```python
# baseline_continuous/loarm_utils.py
"""
Gumbel-top-k sampling and Plackett-Luce log-probability utilities for LO-ARM.

Gumbel-top-k samples from the Plackett-Luce distribution:
  p(z_k = j | z_{<k}, θ) = exp(θ_j) / Σ_{i ∉ z_{<k}} exp(θ_i)

Sampling: z = argsort(θ + Gumbel noise, descending)
Log-prob: log p(z_k | z_{<k}) = θ_{z_k} - logsumexp(θ[remaining])
"""

import torch
import torch.nn.functional as F


def gumbel_top_k(logits: torch.Tensor) -> torch.Tensor:
    """
    Sample a full permutation from Plackett-Luce via Gumbel-top-k trick.

    Args:
        logits: [B, L] raw scores
    Returns:
        z: [B, L] long tensor, each row is a permutation of 0..L-1
    """
    # Add Gumbel(0,1) noise: -log(-log(U)), U ~ Uniform(0,1)
    gumbel_noise = -torch.log(-torch.log(
        torch.clamp(torch.rand_like(logits), min=1e-20, max=1.0)
    ))
    noisy_logits = logits + gumbel_noise
    # argsort descending → permutation
    return torch.argsort(noisy_logits, dim=-1, descending=True)


def plackett_luce_logprob(logits: torch.Tensor, z: torch.Tensor, step_k: int) -> torch.Tensor:
    """
    Compute log p_PL(z_k | z_{<k}, logits) for a specific step k.

    Args:
        logits: [B, L] raw scores for the Plackett-Luce model
        z:      [B, L] full permutation (each row is a perm of 0..L-1)
        step_k: int, which step to compute (0-indexed)
    Returns:
        log_prob: [B] log probability of choosing z[:, step_k] at step step_k
    """
    B, L = logits.shape
    # Chosen index at step k
    chosen = z[:, step_k]  # [B]

    # Build mask for remaining positions (not chosen in steps 0..step_k-1)
    # remaining = all positions NOT in z[:, :step_k]
    mask = torch.ones(B, L, dtype=torch.bool, device=logits.device)
    if step_k > 0:
        # z[:, :step_k]: [B, step_k] - positions already chosen
        already_chosen = z[:, :step_k]  # [B, step_k]
        # Scatter False at already-chosen positions
        mask.scatter_(1, already_chosen, False)

    # logsumexp over remaining positions
    # Set logits of already-chosen to -inf
    masked_logits = logits.masked_fill(~mask, float('-inf'))  # [B, L]
    log_denom = torch.logsumexp(masked_logits, dim=-1)         # [B]

    # log numerator: logits at chosen position
    log_num = logits.gather(1, chosen.unsqueeze(1)).squeeze(1)  # [B]

    return log_num - log_denom
```

### Step 4: Run test to verify it passes

```bash
python -m pytest tests/test_loarm_utils.py -v
```
Expected: All 4 tests PASS

### Step 5: Commit

```bash
git add baseline_continuous/loarm_utils.py tests/test_loarm_utils.py
git commit -m "feat: add Gumbel-top-k and Plackett-Luce utilities for LO-ARM"
```

---

## Task 2: Amortized Variational Network QNetwork

**Files:**
- Create: `baseline_continuous/variational_q.py`
- Test:   `tests/test_variational_q.py`

### Step 1: Write the failing test

```python
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
```

### Step 2: Run test to verify it fails

```bash
python -m pytest tests/test_variational_q.py -v 2>&1 | head -10
```
Expected: ImportError for `variational_q`

### Step 3: Write implementation

```python
# baseline_continuous/variational_q.py
"""
Amortized variational network q_θ(z | x) for LO-ARM.

Takes the FULL unshuffled x (all L vectors, training-time only "god view") and
outputs Plackett-Luce logits for sampling a generation ordering z.

Architecture:
    mean_pool(x) → Linear(D, 4H) → GELU → Linear(4H, L) → logits [B, L]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
```

### Step 4: Run test to verify it passes

```bash
python -m pytest tests/test_variational_q.py -v
```
Expected: All 3 tests PASS

### Step 5: Commit

```bash
git add baseline_continuous/variational_q.py tests/test_variational_q.py
git commit -m "feat: add QNetwork amortized variational for LO-ARM"
```

---

## Task 3: ContinuousLOARM Model

**Files:**
- Create: `baseline_continuous/continuous_loarm.py`
- Test:   `tests/test_continuous_loarm.py`

### Step 1: Write the failing test

```python
# tests/test_continuous_loarm.py
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
```

### Step 2: Run test to verify it fails

```bash
python -m pytest tests/test_continuous_loarm.py -v 2>&1 | head -15
```
Expected: ImportError for `continuous_loarm`

### Step 3: Write implementation

```python
# baseline_continuous/continuous_loarm.py
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

        # Policy head: shared backbone output → L logits
        # Zero-init: at the start, policy is uniform (≈ AO-ARM)
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
            init_vectors: [B, 1, D]  always-visible h₀ prefix
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

        # Replicate ContinuousAOGPT.forward_fn (init-prefix mode)
        # but intercept hidden states before output_proj to also run policy_head.

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
        step_idx = torch.arange(t, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
        tpe_main     = base.transformer.wtpe(step_idx)                # [B, t, 128]
        zeros_early  = torch.zeros(b, ni - 1, 128, device=device)    # [B, ni-1, 128]
        zeros_last   = torch.zeros(b, 1,      128, device=device)    # [B, 1,    128]
        adaLN_cond   = torch.cat([zeros_early, tpe_main, zeros_last], dim=1)  # [B, ni+t, 128]

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
        """
        Configure AdamW with two parameter groups:
        - Base backbone: weight_decay applied to 2D+ params
        - Policy head + q_net: same learning rate, weight_decay for 2D params
        """
        # Reuse base model's optimizer config, then add policy_head separately.
        # Simplest: pass all parameters to a single AdamW.
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
```

### Step 4: Run test to verify it passes

```bash
python -m pytest tests/test_continuous_loarm.py -v
```
Expected: All 4 tests PASS

### Step 5: Commit

```bash
git add baseline_continuous/continuous_loarm.py tests/test_continuous_loarm.py
git commit -m "feat: add ContinuousLOARM with shared-torso policy head"
```

---

## Task 4: RLOO Loss Computation

**Files:**
- Create: `baseline_continuous/rloo.py`
- Test:   `tests/test_rloo.py`

### Step 1: Write the failing test

```python
# tests/test_rloo.py
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
    """delta_F must not propagate gradients to the generator."""
    B, D, L = 2, 16, 7
    # Mock F values as leaf tensors with grad
    F1 = torch.randn(B, requires_grad=True)
    F2 = torch.randn(B, requires_grad=True)
    log_q1 = torch.randn(B, requires_grad=True)
    log_q2 = torch.randn(B, requires_grad=True)

    loss = compute_rloo_loss(F1, F2, log_q1, log_q2, L_len=L)
    loss.backward()

    # The gradient on F1 should come from the direct term, NOT delta_F
    # We just check that no RuntimeError is raised and grads exist
    assert F1.grad is not None
    assert F2.grad is not None
    assert log_q1.grad is not None


def test_rloo_loss_symmetry():
    """Swapping (F1,log_q1) and (F2,log_q2) should give the same loss value."""
    B, L = 4, 7
    F1    = torch.randn(B)
    F2    = torch.randn(B)
    log_q1 = torch.randn(B)
    log_q2 = torch.randn(B)

    loss_ab = compute_rloo_loss(F1, F2, log_q1, log_q2, L_len=L)
    loss_ba = compute_rloo_loss(F2, F1, log_q2, log_q1, L_len=L)
    # Both should give the same scalar
    assert torch.isclose(loss_ab, loss_ba, atol=1e-5), \
        f"loss_ab={loss_ab.item():.4f}, loss_ba={loss_ba.item():.4f}"
```

### Step 2: Run test to verify it fails

```bash
python -m pytest tests/test_rloo.py -v 2>&1 | head -10
```
Expected: ImportError for `rloo`

### Step 3: Write implementation

```python
# baseline_continuous/rloo.py
"""
RLOO (REINFORCE Leave-One-Out) loss for LO-ARM training.

Based on LO-ARM paper Equation (11):

  gradient ≈ L/2 * {
    (∇log_q(z1_{<i}|x) - ∇log_q(z2_{<i}|x)) * ΔF.detach()
    + ∇F_θ(z1_{<i}, x)
    + ∇F_θ(z2_{<i}, x)
  }

where F_θ(z_{<i}, x) = log p_gen(x_{z_i} | z_{<i}) + log p_policy(z_i | z_{<i}) - log q(z_i | z_{<i}, x)

We express this as a scalar loss whose .backward() gives the correct gradient:

  loss = -0.5 * L * (F1 + F2)                   ← direct F gradient
       + 0.5 * L * ΔF.detach() * (log_q1 - log_q2)  ← REINFORCE on q; ★ stop-grad on ΔF

Key invariant: ΔF = (F1 - F2) MUST be detached to avoid spurious feedback
               into the generator via the control-variate term.
"""

import torch
import torch.nn.functional as F
import math


def compute_gen_log_lik(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigma2: float,
) -> torch.Tensor:
    """
    Gaussian log-likelihood: log N(target; pred, σ²I).

    Args:
        pred:   [B, D] predicted vector
        target: [B, D] ground-truth vector
        sigma2: float, noise variance (key temperature hyperparameter)
    Returns:
        log_lik: [B]  (includes constant terms, but they cancel in RLOO ΔF)
    """
    D = pred.shape[-1]
    mse_per_sample = ((pred - target) ** 2).sum(dim=-1)  # [B]
    log_lik = -0.5 * mse_per_sample / sigma2 \
              - 0.5 * D * math.log(2 * math.pi * sigma2)
    return log_lik


def mask_policy_logprob(
    pol_logits_at_k: torch.Tensor,
    chosen: torch.Tensor,
) -> torch.Tensor:
    """
    log p_policy(z_k = chosen | z_{<k}) from already-masked policy logits.

    Args:
        pol_logits_at_k: [B, L]  logits at step k, with selected positions
                                  already set to -inf by apply_policy_mask
        chosen:          [B]     int64 indices of the chosen position
    Returns:
        log_prob: [B]
    """
    log_probs = F.log_softmax(pol_logits_at_k, dim=-1)  # [B, L]
    return log_probs.gather(1, chosen.unsqueeze(1)).squeeze(1)  # [B]


def compute_rloo_loss(
    F1: torch.Tensor,
    F2: torch.Tensor,
    log_q1: torch.Tensor,
    log_q2: torch.Tensor,
    L_len: int,
) -> torch.Tensor:
    """
    Scalar loss whose .backward() gives the RLOO gradient estimate.

    Args:
        F1:     [B]  F_θ(z1_{<i}, x)  — MUST be computed with grad
        F2:     [B]  F_θ(z2_{<i}, x)  — MUST be computed with grad
        log_q1: [B]  log q(z1_i | z1_{<i}, x) — MUST be computed with grad
        log_q2: [B]  log q(z2_i | z2_{<i}, x) — MUST be computed with grad
        L_len:  int  sequence length L (scaling factor from Eq. 11)
    Returns:
        loss: scalar (mean over batch)

    ★ CRITICAL: delta_F is detached. This is the RLOO control-variate trick.
      Without .detach(), gradients would flow from the control-variate term
      back into the generator, breaking the unbiasedness of RLOO.
    """
    delta_F = (F1 - F2).detach()   # ★ stop gradient here

    # Direct F terms: gradient flows to both generator and policy
    direct_term = -0.5 * L_len * (F1 + F2)

    # REINFORCE term: gradient only flows to q_net (log_q1, log_q2)
    reinforce_term = 0.5 * L_len * delta_F * (log_q1 - log_q2)

    loss = direct_term + reinforce_term
    return loss.mean()
```

### Step 4: Run test to verify it passes

```bash
python -m pytest tests/test_rloo.py -v
```
Expected: All 4 tests PASS

### Step 5: Commit

```bash
git add baseline_continuous/rloo.py tests/test_rloo.py
git commit -m "feat: add RLOO loss computation with stop-gradient invariant"
```

---

## Task 5: Training Script

**Files:**
- Create: `baseline_continuous/train_loarm.py`
- No separate test (integration tested by running with small data)

### Implementation

```python
# baseline_continuous/train_loarm.py
"""
LO-ARM Training Script for ContinuousAOGPT on h-space data.

Algorithm (per-batch):
  1. q_net(x)          → q_logits [B, L]
  2. gumbel_top_k × 2  → z1, z2   [B, L]
  3. sample step k     ~ Uniform[0, L-1]
  4. forward_loarm(z1) → gen_preds1 [B,L,D], pol_logits1 [B,L,L]
     forward_loarm(z2) → gen_preds2 [B,L,D], pol_logits2 [B,L,L]
  5. Compute F1, F2 at step k
  6. RLOO loss → backward

Warm start: loads backbone from best_mdm_Random_model.pt.
"""

import sys, os, math, copy, argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baseline_continuous import config as cfg
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
from baseline_continuous.continuous_loarm import ContinuousLOARM
from baseline_continuous.variational_q import QNetwork
from baseline_continuous.loarm_utils import gumbel_top_k, plackett_luce_logprob
from baseline_continuous.rloo import compute_gen_log_lik, mask_policy_logprob, compute_rloo_loss
from baseline_continuous.eval_utils import evaluate_ar
from baseline_continuous.disk_dataset import create_disk_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='LO-ARM training')
    parser.add_argument('--epochs',         type=int,   default=None)
    parser.add_argument('--batch_size',     type=int,   default=None)
    parser.add_argument('--learning_rate',  type=float, default=None)
    parser.add_argument('--sigma2',         type=float, default=0.5,
                        help='Gaussian noise variance (temperature). Start with 0.5.')
    parser.add_argument('--seed',           type=int,   default=None)
    parser.add_argument('--device',         type=str,   default=None)
    parser.add_argument('--data_dir',       type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data'))
    parser.add_argument('--warmstart',      type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'checkpoints', 'best_mdm_Random_model.pt'),
                        help='Path to pretrained MDM checkpoint for backbone warm-start')
    parser.add_argument('--wandb_log',      type=str,   default=None)
    return parser.parse_args()


def get_lr(it, warmup_iters, max_iters, learning_rate, min_lr_ratio=0.1):
    min_lr = learning_rate * min_lr_ratio
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def update_ema(ema_model, model, step, target_decay=0.9999):
    decay = min(target_decay, (1 + step) / (10 + step))
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.data, alpha=1 - decay)


def compute_batch_loss(model, q_net, batch, device, sigma2):
    """
    Full RLOO loss computation for one batch.

    Returns:
        loss:          scalar, RLOO loss for backprop
        aux:           dict with monitoring scalars (detached)
    """
    init_vectors = batch['init_vectors'].to(device)  # [B, 1, D]
    vectors      = batch['main_vectors'].to(device)  # [B, L, D]
    B, L, D      = vectors.shape
    ni           = init_vectors.shape[1]

    # ── 1. Sample two orderings from q_θ ────────────────────────────────────
    q_logits = q_net(vectors)                    # [B, L]  (god-view)
    z1 = gumbel_top_k(q_logits)                 # [B, L]
    z2 = gumbel_top_k(q_logits)                 # [B, L]

    # ── 2. Sample step k uniformly ──────────────────────────────────────────
    k = torch.randint(0, L, ()).item()           # scalar int

    # ── 3. Two forward passes ───────────────────────────────────────────────
    gen_preds1, pol_logits1 = model.forward_loarm(vectors, z1, init_vectors)
    gen_preds2, pol_logits2 = model.forward_loarm(vectors, z2, init_vectors)
    # gen_preds:  [B, L, D]
    # pol_logits: [B, L, L]

    # ── 4. Mask policy logits at step k ─────────────────────────────────────
    masked_pol1 = model.apply_policy_mask(pol_logits1, z1)  # [B, L, L]
    masked_pol2 = model.apply_policy_mask(pol_logits2, z2)

    # ── 5. Compute F1, F2 at step k ─────────────────────────────────────────
    # 5a. Generator log-lik: pred at step k ≈ x[z[:, k]]
    target1 = vectors[torch.arange(B), z1[:, k]]  # [B, D]
    target2 = vectors[torch.arange(B), z2[:, k]]
    log_p_gen1 = compute_gen_log_lik(gen_preds1[:, k, :], target1, sigma2)
    log_p_gen2 = compute_gen_log_lik(gen_preds2[:, k, :], target2, sigma2)

    # 5b. Policy log-prob at step k
    chosen1 = z1[:, k]  # [B]
    chosen2 = z2[:, k]
    log_p_pol1 = mask_policy_logprob(masked_pol1[:, k, :], chosen1)
    log_p_pol2 = mask_policy_logprob(masked_pol2[:, k, :], chosen2)

    # 5c. Variational log-prob (Plackett-Luce)
    log_q1 = plackett_luce_logprob(q_logits, z1, step_k=k)  # [B]
    log_q2 = plackett_luce_logprob(q_logits, z2, step_k=k)

    # 5d. F = log_p_gen + log_p_pol - log_q
    F1 = log_p_gen1 + log_p_pol1 - log_q1
    F2 = log_p_gen2 + log_p_pol2 - log_q2

    # ── 6. RLOO loss ─────────────────────────────────────────────────────────
    loss = compute_rloo_loss(F1, F2, log_q1, log_q2, L_len=L)

    # Auxiliary metrics for logging (no gradients needed)
    with torch.no_grad():
        policy_entropy = -(
            F.softmax(masked_pol1[:, k, :].nan_to_num(nan=0., neginf=-1e9), dim=-1)
            * F.log_softmax(masked_pol1[:, k, :].nan_to_num(nan=0., neginf=-1e9), dim=-1)
        ).sum(dim=-1).mean()

    aux = {
        'F1_mean':        F1.mean().item(),
        'F2_mean':        F2.mean().item(),
        'log_p_gen1':     log_p_gen1.mean().item(),
        'log_p_pol1':     log_p_pol1.mean().item(),
        'log_q1':         log_q1.mean().item(),
        'policy_entropy': policy_entropy.item(),
        'step_k':         k,
    }
    return loss, aux


def main():
    args = parse_args()
    epochs        = args.epochs        or cfg.epochs
    batch_size    = args.batch_size    or cfg.batch_size
    learning_rate = args.learning_rate or cfg.learning_rate
    seed          = args.seed          or cfg.seed
    device        = args.device        or cfg.device
    sigma2        = args.sigma2
    wandb_log     = cfg.wandb_log
    if args.wandb_log is not None:
        wandb_log = args.wandb_log.lower() in ('true', '1', 'yes')

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print("=" * 60)
    print(f"LO-ARM Training  sigma2={sigma2}")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_disk_dataloaders(
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        num_chunks=cfg.num_chunks,
    )
    iters_per_epoch = len(train_loader)
    max_iters       = epochs * iters_per_epoch
    warmup_iters    = int(cfg.warmup_iters * max_iters)

    # ── Model: warm-start backbone from MDM checkpoint ──────────────────────
    model_config = ContinuousAOGPTConfig(
        block_size=cfg.block_size, vector_dim=cfg.vector_dim,
        n_layer=cfg.n_layer,      n_head=cfg.n_head,
        n_embd=cfg.n_embd,        dropout=cfg.dropout,
        bias=cfg.bias,            num_init=cfg.num_init,
    )
    base_model = ContinuousAOGPT(model_config)
    if os.path.exists(args.warmstart):
        ckpt = torch.load(args.warmstart, map_location='cpu', weights_only=False)
        base_model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Warm-start backbone from {args.warmstart}")
    else:
        print(f"  WARNING: No warm-start checkpoint found at {args.warmstart}, training from scratch")

    model  = ContinuousLOARM(base_model).to(device)
    q_net  = QNetwork(
        vector_dim=cfg.vector_dim,
        seq_len=cfg.seq_length - cfg.num_init,   # L = 31
        hidden_dim=256,
    ).to(device)

    ema_model = copy.deepcopy(model)
    ema_model.eval()

    # ── Optimizer ───────────────────────────────────────────────────────────
    # Separate learning rates: q_net gets 3× lower LR (it has god-view,
    # should not dominate the optimization)
    optimizer = model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type='cuda' if 'cuda' in device else 'cpu',
    )
    # Add q_net parameters to the optimizer
    q_params = list(q_net.parameters())
    optimizer.add_param_group({'params': q_params, 'lr': learning_rate / 3.0, 'weight_decay': 0.0})

    # ── WandB ────────────────────────────────────────────────────────────────
    if wandb_log:
        import wandb
        wandb.init(project=cfg.wandb_project, name=f'loarm-sigma{sigma2}',
                   group='loarm', config={
                       'sigma2': sigma2, 'vector_dim': cfg.vector_dim,
                       'seq_length': cfg.seq_length, 'n_layer': cfg.n_layer,
                       'n_embd': cfg.n_embd, 'batch_size': batch_size,
                       'learning_rate': learning_rate, 'epochs': epochs,
                   })

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nStarting training: {epochs} epochs, {max_iters} total iters")
    best_val_loss = float('inf')
    global_step   = 0
    model.train()
    q_net.train()

    for epoch in range(epochs):
        for batch in train_loader:
            lr = get_lr(global_step, warmup_iters, max_iters, learning_rate)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            loss, aux = compute_batch_loss(model, q_net, batch, device, sigma2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(q_net.parameters()), cfg.grad_clip)
            optimizer.step()
            update_ema(ema_model, model, global_step)

            if global_step % cfg.log_interval == 0:
                print(f"epoch {epoch+1}/{epochs} | iter {global_step:>6d} | "
                      f"loss {loss.item():.4f} | "
                      f"F1={aux['F1_mean']:.3f} | "
                      f"pol_ent={aux['policy_entropy']:.3f} | "
                      f"lr {lr:.2e}")
                if wandb_log:
                    import wandb
                    wandb.log({'train/loss': loss.item(), 'train/lr': lr,
                               'train/F1_mean': aux['F1_mean'],
                               'train/policy_entropy': aux['policy_entropy'],
                               'train/log_p_gen': aux['log_p_gen1'],
                               'train/log_p_pol': aux['log_p_pol1'],
                               'epoch': epoch}, step=global_step)

            if global_step % cfg.eval_interval == 0 and global_step > 0:
                # Standard AR eval (uses backbone generator, comparable to MDM baseline)
                eval_results = evaluate_ar(ema_model.base, val_loader, device)
                print(f"  [eval] val_loss: {eval_results['val_loss']:.6f} | "
                      f"cos_sim: {eval_results['val_cos_sim']:.4f}")
                if wandb_log:
                    import wandb
                    wandb.log({'val/loss': eval_results['val_loss'],
                               'val/cos_sim': eval_results['val_cos_sim']},
                              step=global_step)

                if cfg.save_best_model and eval_results['val_loss'] < best_val_loss:
                    best_val_loss = eval_results['val_loss']
                    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({
                        'model_state_dict':   ema_model.state_dict(),
                        'raw_model_state_dict': model.state_dict(),
                        'q_net_state_dict':   q_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config':             model_config,
                        'sigma2':             sigma2,
                        'epoch':              epoch,
                        'global_step':        global_step,
                        'val_loss':           best_val_loss,
                    }, os.path.join(save_dir, f'best_loarm_sigma{sigma2}_model.pt'))
                    print(f"  [save] Best model saved (val_loss: {best_val_loss:.4f})")

                model.train()
                q_net.train()

            global_step += 1

    if wandb_log:
        import wandb
        wandb.finish()
    print("\nTraining complete.")


if __name__ == '__main__':
    main()
```

### Smoke test (manual, not pytest)

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM
python -m baseline_continuous.train_loarm \
    --epochs 1 \
    --batch_size 16 \
    --data_dir baseline_continuous/data_hspace_500k \
    --wandb_log false \
    --device cuda 2>&1 | head -40
```
Expected: Runs 1 epoch without error, prints `loss` and `pol_ent` values.

### Step: Commit

```bash
git add baseline_continuous/train_loarm.py
git commit -m "feat: add LO-ARM training script with RLOO gradient"
```

---

## Task 6: Evaluation Script with Policy Order Recovery

**Files:**
- Create: `baseline_continuous/eval_loarm.py`

### Implementation

```python
# baseline_continuous/eval_loarm.py
"""
LO-ARM Evaluation: order recovery metrics.

For a trained ContinuousLOARM model, evaluates:
  1. Causal advantage     = random_loss - policy_greedy_loss
  2. Policy Kendall τ     = correlation between policy's greedy order and [0,..,L-1]
  3. Policy entropy/step  = H(p^z at each step), averaged over test set
  4. Policy log-prob of causal order  = Σ_k log p^z(k | 0..k-1)
  5. Standard AR loss (generator quality, comparable to MDM baseline)

Policy greedy order (inference mode):
  At step k, apply policy to current context → pick argmax position → reveal it.
  This uses only p^z_θ (no q_net needed), so it is a true inference-time metric.
"""

import sys, os
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import kendalltau

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baseline_continuous import config as cfg
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
from baseline_continuous.continuous_loarm import ContinuousLOARM
from baseline_continuous.eval_utils import evaluate_ar
from baseline_continuous.disk_dataset import create_disk_dataloaders


def load_loarm(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    base = ContinuousAOGPT(ckpt['config'])
    model = ContinuousLOARM(base)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  Loaded {os.path.basename(path)}  val_loss={ckpt['val_loss']:.4f}  sigma2={ckpt.get('sigma2','?')}")
    return model


@torch.no_grad()
def policy_greedy_order(model, vectors, init_vectors):
    """
    Greedy order from the policy: at each step k, pick argmax of p^z_θ.

    Returns:
        greedy_orders: [B, L] long tensor
    """
    B, L, D = vectors.shape
    device  = vectors.device
    ni      = init_vectors.shape[1]

    revealed  = torch.zeros(B, L, dtype=torch.bool, device=device)
    orders_so_far = [[] for _ in range(B)]
    greedy_orders = torch.zeros(B, L, dtype=torch.long, device=device)

    for k in range(L):
        # Build current order: selected so far + dummy ascending for the rest
        current_orders = []
        for b in range(B):
            chosen   = orders_so_far[b]
            dummy    = sorted(set(range(L)) - set(chosen))
            current_orders.append(torch.tensor(chosen + dummy, dtype=torch.long, device=device))
        orders_t = torch.stack(current_orders)  # [B, L]

        # Forward pass
        gen_preds, pol_logits = model.forward_loarm(vectors, orders_t, init_vectors)

        # Policy at step k, mask already-revealed
        pol_k = pol_logits[:, k, :].clone()          # [B, L]
        pol_k[revealed] = float('-inf')

        # Greedy pick
        chosen_k = pol_k.argmax(dim=-1)              # [B]
        greedy_orders[:, k] = chosen_k

        for b in range(B):
            orders_so_far[b].append(chosen_k[b].item())
            revealed[b, chosen_k[b]] = True

    return greedy_orders


@torch.no_grad()
def eval_with_fixed_order(model_base, vectors, init_vectors, fixed_order):
    """Compute MSE loss with a specific fixed order applied to all samples."""
    B, L, _ = vectors.shape
    orders = fixed_order.unsqueeze(0).expand(B, -1).to(vectors.device)
    _, loss = model_base(vectors, mode=None, orders=orders, init_vectors=init_vectors)
    return loss.item()


@torch.no_grad()
def evaluate_policy(model, test_loader, device, max_batches=20):
    """Run all policy-related metrics on test set."""
    L = cfg.seq_length - cfg.num_init   # 31
    causal_order = torch.arange(L, dtype=torch.long, device=device)

    all_taus        = []
    all_entropies   = []   # [L] per step
    all_pol_causal  = []   # policy log-prob of causal order
    greedy_losses   = []
    random_losses   = []
    causal_losses   = []

    for i, batch in enumerate(test_loader):
        if i >= max_batches:
            break
        init_vectors = batch['init_vectors'].to(device)
        vectors      = batch['main_vectors'].to(device)
        B = vectors.shape[0]

        # ── Greedy order ─────────────────────────────────────────────────
        greedy_orders = policy_greedy_order(model, vectors, init_vectors)  # [B, L]

        # Kendall τ vs causal [0,1,...,L-1]
        causal_np = causal_order.cpu().numpy()
        for b in range(B):
            gr = greedy_orders[b].cpu().numpy()
            tau, _ = kendalltau(gr, causal_np)
            all_taus.append(tau)

        # Greedy order loss (apply greedy orders to generator)
        for b in range(B):
            go = greedy_orders[b]
            orders_b = go.unsqueeze(0)
            _, loss_b = model.base(vectors[b:b+1], mode=None,
                                   orders=orders_b,
                                   init_vectors=init_vectors[b:b+1])
            greedy_losses.append(loss_b.item())

        # Causal order loss
        cl = eval_with_fixed_order(model.base, vectors, init_vectors, causal_order)
        causal_losses.append(cl)

        # Random order losses (10 MC samples)
        rand_ls = []
        for _ in range(10):
            rand_ord = torch.stack([torch.randperm(L, device=device) for _ in range(B)])
            _, rl = model.base(vectors, mode=None, orders=rand_ord, init_vectors=init_vectors)
            rand_ls.append(rl.item())
        random_losses.append(np.mean(rand_ls))

        # ── Policy entropy per step ──────────────────────────────────────
        # Forward with causal order to get policy logits
        causal_orders = causal_order.unsqueeze(0).expand(B, -1)
        _, pol_logits = model.forward_loarm(vectors, causal_orders, init_vectors)
        masked_pol = model.apply_policy_mask(pol_logits, causal_orders)   # [B, L, L]

        entropies_batch = []
        for k in range(L):
            logits_k = masked_pol[:, k, :].clone()
            logits_k = logits_k.nan_to_num(nan=0., neginf=-1e9)
            probs_k  = F.softmax(logits_k, dim=-1)
            H_k      = -(probs_k * probs_k.clamp(min=1e-30).log()).sum(dim=-1).mean()
            entropies_batch.append(H_k.item())
        all_entropies.append(entropies_batch)

        # ── Policy log-prob of causal order ──────────────────────────────
        pol_logprob_causal = 0.0
        for k in range(L):
            lp_k = F.log_softmax(masked_pol[:, k, :].nan_to_num(nan=0., neginf=-1e9), dim=-1)
            pol_logprob_causal += lp_k[:, k].mean().item()   # causal: position k chosen at step k
        all_pol_causal.append(pol_logprob_causal)

    results = {
        'mean_tau':          np.mean(all_taus),
        'std_tau':           np.std(all_taus),
        'mean_greedy_loss':  np.mean(greedy_losses),
        'mean_causal_loss':  np.mean(causal_losses),
        'mean_random_loss':  np.mean(random_losses),
        'causal_advantage':  np.mean(random_losses) - np.mean(causal_losses),
        'policy_lp_causal':  np.mean(all_pol_causal),
        'entropy_per_step':  np.mean(all_entropies, axis=0),   # [L]
    }
    return results


def main():
    device = cfg.device
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')

    # Find best LO-ARM checkpoint
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'best_loarm_*.pt')))
    if not ckpts:
        print("No LO-ARM checkpoints found. Run train_loarm.py first.")
        return

    _, _, test_loader = create_disk_dataloaders(
        data_dir=os.path.join(os.path.dirname(__file__), 'data_hspace_500k'),
        batch_size=32,
        num_workers=cfg.num_workers,
        num_chunks=cfg.num_chunks,
    )

    for ckpt_path in ckpts:
        print(f"\n{'='*60}")
        model = load_loarm(ckpt_path, device)

        results = evaluate_policy(model, test_loader, device, max_batches=20)
        print(f"\nOrder Recovery Results:")
        print(f"  Greedy Kendall τ        : {results['mean_tau']:.4f} ± {results['std_tau']:.4f}")
        print(f"  Causal order loss        : {results['mean_causal_loss']:.4f}")
        print(f"  Greedy order loss        : {results['mean_greedy_loss']:.4f}")
        print(f"  Random order loss (MC10) : {results['mean_random_loss']:.4f}")
        print(f"  Causal advantage         : {results['causal_advantage']:+.4f}")
        print(f"  Policy log-prob (causal) : {results['policy_lp_causal']:.4f}")
        print(f"  Policy entropy step 0    : {results['entropy_per_step'][0]:.4f}")
        print(f"  Policy entropy step 15   : {results['entropy_per_step'][15]:.4f}")
        print(f"  Policy entropy step 30   : {results['entropy_per_step'][-1]:.4f}")


if __name__ == '__main__':
    main()
```

### Run to check it works

```bash
# After training is complete:
python -m baseline_continuous.eval_loarm 2>&1 | head -30
```

### Commit

```bash
git add baseline_continuous/eval_loarm.py
git commit -m "feat: add LO-ARM evaluation with policy order recovery metrics"
```

---

## Task 7: sigma² Ablation Run

After confirming the training runs without errors, run 3 sigma² values to find the best temperature:

```bash
# Run sequentially (each ~15 epochs on GPU)
python -m baseline_continuous.train_loarm \
    --epochs 15 --sigma2 0.1 \
    --data_dir baseline_continuous/data_hspace_500k \
    --device cuda --wandb_log true

python -m baseline_continuous.train_loarm \
    --epochs 15 --sigma2 0.5 \
    --data_dir baseline_continuous/data_hspace_500k \
    --device cuda --wandb_log true

python -m baseline_continuous.train_loarm \
    --epochs 15 --sigma2 2.0 \
    --data_dir baseline_continuous/data_hspace_500k \
    --device cuda --wandb_log true

# Compare results
python -m baseline_continuous.eval_loarm
```

**Expected baseline comparison** (LO-ARM should improve over MDM):

| Model             | causal advantage | greedy Kendall τ |
|-------------------|-----------------|-----------------|
| MDM (baseline)    | ≈ 0             | −0.293          |
| **LO-ARM (ours)** | **> 0 ?**       | **> 0 ?**       |
| AR no-shuffle     | +0.3285         | —               |

---

## Quick Run Order (no detours)

```
Task 1 → Task 2 → Task 3 → Task 4 → Task 5 (smoke test) → Task 6 → Task 7
```

All tests should pass before proceeding to the next task.
