# Configurable Teacher-Student Experiment Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend the existing MDM/AR baseline with a YAML config system that switches between 3 randomly-initialized teacher types (`linear_attention`, `sparse_causal_mask`, `standard_causal`) and 2 student types (`ar`, `mdm`), with data caching, via a single `python train.py --config configs/exp001.yaml` entry point.

**Architecture:** 5 new files, 0 existing files modified. Teacher models are randomly initialized transformers (no pretrained weights) that autoregressively generate `[B, L, D]` continuous sequences. Existing `ContinuousAOGPT` is reused as the student. Data cached to `data/teacher_cache/{hash}/` keyed by teacher config.

**Tech Stack:** PyTorch 2.4, PyYAML, numpy memmap, existing `ContinuousAOGPT`

---

### Task 1: YAML Config System

**Files:**
- Create: `baseline_continuous/exp_config.py`
- Create: `configs/exp001_linear_mdm.yaml`
- Create: `configs/exp002_sparse_mdm.yaml`
- Create: `configs/exp003_standard_ar.yaml`

**Step 1: Write `baseline_continuous/exp_config.py`**

```python
"""YAML config loading for teacher-student experiments."""
import hashlib, json
from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


@dataclass
class TeacherConfig:
    type: str = 'standard_causal'  # linear_attention | sparse_causal_mask | standard_causal
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    seq_len: int = 32
    sparsity: float = 0.5          # only used for sparse_causal_mask
    noise_std: float = 0.1
    seed: int = 42


@dataclass
class StudentConfig:
    type: str = 'mdm'              # ar | mdm
    shuffle_input: bool = True     # feed shuffled tokens to student
    use_gt_order: bool = False     # ar only: train in original causal order (lower bound)
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 64
    n_train: int = 10000
    n_val: int = 1000
    n_test: int = 1000
    lr: float = 1e-3
    epochs: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    warmup_ratio: float = 0.05
    device: str = 'cuda'
    seed: int = 42
    log_interval: int = 100
    eval_interval: int = 500
    num_workers: int = 4
    wandb_log: bool = False
    wandb_project: str = 'teacher-student-exp'
    data_root: str = 'data/teacher_cache'
    checkpoint_dir: str = 'checkpoints/new_exp'


@dataclass
class ExpConfig:
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def load_config(path: str) -> ExpConfig:
    with open(path) as f:
        d = yaml.safe_load(f)
    teacher = TeacherConfig(**d.get('teacher', {}))
    student = StudentConfig(**d.get('student', {}))
    training = TrainingConfig(**d.get('training', {}))
    return ExpConfig(teacher=teacher, student=student, training=training)


def teacher_cache_key(cfg: TeacherConfig) -> str:
    """8-char MD5 hash of teacher hyperparams. Same teacher config → same cache dir."""
    d = asdict(cfg)
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]
```

**Step 2: Write `configs/exp001_linear_mdm.yaml`**

```yaml
# Linear Attention Teacher + MDM Student
teacher:
  type: linear_attention
  n_layers: 4
  d_model: 128
  n_heads: 4
  seq_len: 32
  noise_std: 0.1
  seed: 42

student:
  type: mdm
  shuffle_input: true
  use_gt_order: false
  n_layer: 4
  n_head: 4
  n_embd: 256

training:
  batch_size: 64
  n_train: 10000
  n_val: 1000
  n_test: 1000
  lr: 1.0e-3
  epochs: 50
  device: cuda
  wandb_log: false
  data_root: data/teacher_cache
  checkpoint_dir: checkpoints/new_exp
```

**Step 3: Write `configs/exp002_sparse_mdm.yaml`**

```yaml
# Sparse Causal Mask Teacher + MDM Student
teacher:
  type: sparse_causal_mask
  n_layers: 4
  d_model: 128
  n_heads: 4
  seq_len: 32
  sparsity: 0.5
  noise_std: 0.1
  seed: 42

student:
  type: mdm
  shuffle_input: true
  use_gt_order: false
  n_layer: 4
  n_head: 4
  n_embd: 256

training:
  batch_size: 64
  n_train: 10000
  n_val: 1000
  n_test: 1000
  lr: 1.0e-3
  epochs: 50
  device: cuda
  wandb_log: false
  data_root: data/teacher_cache
  checkpoint_dir: checkpoints/new_exp
```

**Step 4: Write `configs/exp003_standard_ar.yaml`**

```yaml
# Standard Causal Teacher + AR Student (GT order = theoretical lower bound)
teacher:
  type: standard_causal
  n_layers: 4
  d_model: 128
  n_heads: 4
  seq_len: 32
  noise_std: 0.1
  seed: 42

student:
  type: ar
  shuffle_input: false
  use_gt_order: true
  n_layer: 4
  n_head: 4
  n_embd: 256

training:
  batch_size: 64
  n_train: 10000
  n_val: 1000
  n_test: 1000
  lr: 1.0e-3
  epochs: 50
  device: cuda
  wandb_log: false
  data_root: data/teacher_cache
  checkpoint_dir: checkpoints/new_exp
```

**Step 5: Verify config loading**

Run:
```bash
python -c "
import sys; sys.path.insert(0,'.')
from baseline_continuous.exp_config import load_config, teacher_cache_key
cfg = load_config('configs/exp001_linear_mdm.yaml')
print('teacher type :', cfg.teacher.type)
print('d_model      :', cfg.teacher.d_model)
print('student type :', cfg.student.type)
print('batch_size   :', cfg.training.batch_size)
print('cache key    :', teacher_cache_key(cfg.teacher))
"
```

Expected: prints 5 lines with no errors; cache key is an 8-char hex string.

**Step 6: Commit**

```bash
git add baseline_continuous/exp_config.py configs/
git commit -m "feat: add YAML config system with TeacherConfig/StudentConfig/TrainingConfig dataclasses"
```

---

### Task 2: Teacher Model Implementations

**Files:**
- Create: `baseline_continuous/teachers.py`

**Step 1: Write `baseline_continuous/teachers.py`**

```python
"""Randomly-initialized teacher models for continuous sequence generation.

Three attention variants (all without positional encoding):
  1. linear_attention   — causal linear attention with ELU+1 feature map
  2. sparse_causal_mask — softmax attention with fixed random sparse causal mask
  3. standard_causal    — standard softmax causal attention (triangular mask)

Architecture per teacher:
  N stacked blocks: LayerNorm → Attention → residual → LayerNorm → FFN → residual
  Randomly initialized (seed-fixed), all parameters frozen, no gradients.

Generation:
  x_0       ~ N(0, I)
  x_t       = teacher(x_{0:t-1})[-1] + N(0, noise_std²·I),  t = 1, ..., L-1
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass

from baseline_continuous.exp_config import TeacherConfig


# ─── Shared block base ────────────────────────────────────────────────────────

class _TeacherBlock(nn.Module):
    """Shared structure: LN → Attention → residual → LN → FFN → residual.
    Subclasses implement _attend() which returns [B, T, D] pre-out-proj."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model={d_model} must divide n_heads={n_heads}"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.d_model  = d_model
        self.ln1      = nn.LayerNorm(d_model)
        self.ln2      = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ff       = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] → [B, H, T, head_dim]"""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, H, T, head_dim] → [B, T, D]"""
        B, H, T, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * d)

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        """Attention computation (excluding out_proj). Returns [B, T, D]."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.out_proj(self._attend(self.ln1(x)))
        x = x + self.ff(self.ln2(x))
        return x


# ─── Attention variants ───────────────────────────────────────────────────────

class _LinearAttentionBlock(_TeacherBlock):
    """Causal linear attention with ELU+1 feature map (no positional encoding).

    Uses the cumsum trick for O(T·head_dim²) causal attention:
        KV_cum[i] = Σ_{j≤i}  k[j]ᵀ ⊗ v[j]   shape [B, H, T, d, d]
        K_cum[i]  = Σ_{j≤i}  k[j]            shape [B, H, T, d]
        out[i]    = q[i] @ KV_cum[i]  /  (q[i] · K_cum[i] + ε)
    """

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = self._split_heads(q)   # [B, H, T, d]
        k = self._split_heads(k)
        v = self._split_heads(v)

        # Feature map: φ(x) = ELU(x) + 1  (non-negative, so denominator > 0)
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        kv     = torch.einsum('bhtd,bhte->bhtde', k, v)     # [B, H, T, d, d]
        kv_cum = kv.cumsum(dim=2)                            # causal prefix sum
        k_cum  = k.cumsum(dim=2)                             # [B, H, T, d]

        out   = torch.einsum('bhtd,bhtde->bhte', q, kv_cum) # [B, H, T, d]
        denom = (q * k_cum).sum(dim=-1, keepdim=True) + 1e-6
        out   = out / denom                                  # [B, H, T, d]
        return self._merge_heads(out)                        # [B, T, D]


class _SparseCausalMaskBlock(_TeacherBlock):
    """Softmax attention with a fixed random sparse causal mask (no positional encoding).

    At init time, for each layer a random causal mask is generated:
      - Standard triangular causal mask as baseline
      - Off-diagonal entries kept with prob (1 - sparsity)
      - Diagonal always kept (position i always attends to itself)
    Mask is deterministic given (seed, layer_idx).
    """

    def __init__(self, d_model: int, n_heads: int, seq_len: int,
                 sparsity: float, seed: int, layer_idx: int):
        super().__init__(d_model, n_heads)
        rng = torch.Generator()
        rng.manual_seed(seed + layer_idx * 997)
        causal   = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        keep     = torch.rand(seq_len, seq_len, generator=rng) > sparsity
        diagonal = torch.eye(seq_len, dtype=torch.bool)
        mask     = causal & (keep | diagonal)           # [L, L] bool
        self.register_buffer('attn_mask', mask)

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scale  = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale   # [B, H, T, T]
        mask   = self.attn_mask[:T, :T]                          # slice to actual T
        scores = scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn   = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        return self._merge_heads(torch.matmul(attn, v))          # [B, T, D]


class _StandardCausalBlock(_TeacherBlock):
    """Standard softmax causal attention with triangular mask (no positional encoding)."""

    def _attend(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            B, H, T, d = q.shape
            scale  = math.sqrt(d)
            scores = torch.matmul(q, k.transpose(-2, -1)) / scale
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal, float('-inf'))
            out    = torch.matmul(F.softmax(scores, dim=-1), v)

        return self._merge_heads(out)                            # [B, T, D]


# ─── N-layer teacher model ────────────────────────────────────────────────────

class Teacher(nn.Module):
    """N-layer transformer teacher. Randomly initialized, all weights frozen."""

    def __init__(self, cfg: TeacherConfig):
        super().__init__()
        self.cfg = cfg
        D, H, L = cfg.d_model, cfg.n_heads, cfg.seq_len

        # Initialize weights with a fixed seed WITHOUT polluting the global RNG.
        with torch.random.fork_rng():
            torch.manual_seed(cfg.seed)
            blocks = []
            for i in range(cfg.n_layers):
                if cfg.type == 'linear_attention':
                    blocks.append(_LinearAttentionBlock(D, H))
                elif cfg.type == 'sparse_causal_mask':
                    blocks.append(_SparseCausalMaskBlock(D, H, L, cfg.sparsity, cfg.seed, i))
                elif cfg.type == 'standard_causal':
                    blocks.append(_StandardCausalBlock(D, H))
                else:
                    raise ValueError(f"Unknown teacher type: {cfg.type!r}")
            self.blocks = nn.ModuleList(blocks)
            self.ln_f   = nn.LayerNorm(D)

        # Freeze everything — teacher never trains
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward_seq(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D] → [B, T, D]"""
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    @torch.no_grad()
    def generate_sequence(self, batch_size: int, length: int,
                          device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Autoregressively generate [batch_size, length, d_model].

        Caller sets torch.manual_seed() before this call for reproducible data.

        Returns:
            'vectors':      [B, L, D]  full sequence in causal order
            'init_vectors': [B, 1, D]  x_0 (used as conditioning prefix for student)
        """
        if device is None:
            device = next(self.parameters()).device
        D    = self.cfg.d_model
        seqs = torch.zeros(batch_size, length, D, device=device)
        seqs[:, 0] = torch.randn(batch_size, D, device=device)   # x_0 ~ N(0, I)

        for t in range(1, length):
            pred       = self.forward_seq(seqs[:, :t])[:, -1]    # [B, D]
            seqs[:, t] = pred + torch.randn_like(pred) * self.cfg.noise_std

        return {'vectors': seqs, 'init_vectors': seqs[:, :1]}


def build_teacher(cfg: TeacherConfig) -> Teacher:
    """Factory: build a Teacher from a TeacherConfig."""
    return Teacher(cfg)
```

**Step 2: Verify all 3 teacher types produce correct shapes and have no gradients**

Run:
```bash
python -c "
import sys; sys.path.insert(0,'.')
import torch
from baseline_continuous.exp_config import TeacherConfig
from baseline_continuous.teachers import build_teacher

for ttype in ['linear_attention', 'sparse_causal_mask', 'standard_causal']:
    cfg = TeacherConfig(type=ttype, n_layers=2, d_model=64, n_heads=4, seq_len=16,
                        noise_std=0.1, seed=42)
    teacher = build_teacher(cfg).cuda()
    torch.manual_seed(0)
    result = teacher.generate_sequence(batch_size=4, length=16, device='cuda')
    vecs = result['vectors']
    assert vecs.shape == (4, 16, 64), f'Shape mismatch: {vecs.shape}'
    assert not vecs.requires_grad,    'Vectors should not require grad'
    assert result['init_vectors'].shape == (4, 1, 64)
    print(f'{ttype}: shape={vecs.shape}, mean_norm={vecs.norm(dim=-1).mean():.2f}  OK')

print('ALL PASS')
"
```

Expected output (3 lines + ALL PASS, no errors):
```
linear_attention: shape=torch.Size([4, 16, 64]), mean_norm=X.XX  OK
sparse_causal_mask: shape=torch.Size([4, 16, 64]), mean_norm=X.XX  OK
standard_causal: shape=torch.Size([4, 16, 64]), mean_norm=X.XX  OK
ALL PASS
```

**Step 3: Verify teacher weight reproducibility (same seed → same weights and same sequences)**

Run:
```bash
python -c "
import sys; sys.path.insert(0,'.')
import torch
from baseline_continuous.exp_config import TeacherConfig
from baseline_continuous.teachers import build_teacher

cfg = TeacherConfig(type='standard_causal', n_layers=2, d_model=64, n_heads=4, seq_len=16, seed=42)
t1 = build_teacher(cfg).cuda()
t2 = build_teacher(cfg).cuda()

# Same seed → same weights
p1 = next(t1.parameters())
p2 = next(t2.parameters())
assert torch.allclose(p1, p2), 'Weights differ!'

# Same data seed → same sequences
torch.manual_seed(99)
r1 = t1.generate_sequence(4, 16, 'cuda')
torch.manual_seed(99)
r2 = t2.generate_sequence(4, 16, 'cuda')
assert torch.allclose(r1['vectors'], r2['vectors']), 'Sequences differ!'
print('Reproducibility: PASS')
"
```

Expected: `Reproducibility: PASS`

**Step 4: Commit**

```bash
git add baseline_continuous/teachers.py
git commit -m "feat: add 3 teacher types (linear_attention, sparse_causal_mask, standard_causal) with fixed-seed init and AR generation"
```

---

### Task 3: Data Generation with Caching

**Files:**
- Create: `baseline_continuous/generate_data.py`

**Step 1: Write `baseline_continuous/generate_data.py`**

```python
"""
generate_data.py — Generate teacher sequences and cache to disk.

Cache is keyed by teacher config hash so the same teacher data is shared across
multiple experiments with different student configs.

Usage:
    python baseline_continuous/generate_data.py --config configs/exp001_linear_mdm.yaml
    python baseline_continuous/generate_data.py --config configs/exp001_linear_mdm.yaml --force
"""
import sys, os, argparse
import torch
import numpy as np
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from baseline_continuous.exp_config import load_config, teacher_cache_key, ExpConfig
from baseline_continuous.teachers import build_teacher


def get_cache_dir(cfg: ExpConfig) -> str:
    key = teacher_cache_key(cfg.teacher)
    return os.path.join(cfg.training.data_root, key)


def cache_exists(cache_dir: str) -> bool:
    required = [
        'train_vectors.npy', 'val_vectors.npy', 'test_vectors.npy',
        'train_init_vectors.npy', 'val_init_vectors.npy', 'test_init_vectors.npy',
        'data_config.pt',
    ]
    return all(os.path.exists(os.path.join(cache_dir, f)) for f in required)


def generate_split(teacher, n_samples: int, seq_len: int, cache_dir: str,
                   split: str, seed: int, chunk_size: int = 2000, device: str = 'cuda'):
    """Generate n_samples sequences and save as numpy memmap files."""
    D         = teacher.cfg.d_model
    vecs_path = os.path.join(cache_dir, f'{split}_vectors.npy')
    init_path = os.path.join(cache_dir, f'{split}_init_vectors.npy')

    vecs_mmap = np.memmap(vecs_path, dtype='float32', mode='w+', shape=(n_samples, seq_len, D))
    init_mmap = np.memmap(init_path, dtype='float32', mode='w+', shape=(n_samples, 1, D))

    torch.manual_seed(seed)
    generated = 0
    while generated < n_samples:
        bs     = min(chunk_size, n_samples - generated)
        result = teacher.generate_sequence(batch_size=bs, length=seq_len, device=device)
        vecs_mmap[generated:generated + bs] = result['vectors'].cpu().numpy()
        init_mmap[generated:generated + bs] = result['init_vectors'].cpu().numpy()
        generated += bs
        print(f"  [{split}] {generated}/{n_samples}", flush=True)

    vecs_mmap.flush()
    init_mmap.flush()
    del vecs_mmap, init_mmap
    print(f"  [{split}] Saved.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--force', action='store_true', help='Regenerate even if cache exists')
    args = parser.parse_args()

    cfg       = load_config(args.config)
    cache_dir = get_cache_dir(cfg)
    key       = teacher_cache_key(cfg.teacher)

    print("=" * 60)
    print(f"Teacher type : {cfg.teacher.type}")
    print(f"d_model      : {cfg.teacher.d_model}  seq_len: {cfg.teacher.seq_len}")
    print(f"Cache key    : {key}")
    print(f"Cache dir    : {cache_dir}")
    print("=" * 60)

    if not args.force and cache_exists(cache_dir):
        print("Cache exists — skipping. Use --force to regenerate.")
        return

    os.makedirs(cache_dir, exist_ok=True)
    tc     = cfg.teacher
    tr     = cfg.training
    device = tr.device
    teacher = build_teacher(tc).to(device)
    print(f"Teacher params: {sum(p.numel() for p in teacher.parameters()):,}\n")

    print(f"Generating train ({tr.n_train} samples)...")
    generate_split(teacher, tr.n_train, tc.seq_len, cache_dir, 'train',
                   seed=tc.seed,          device=device)
    print(f"Generating val ({tr.n_val} samples)...")
    generate_split(teacher, tr.n_val,   tc.seq_len, cache_dir, 'val',
                   seed=tc.seed + 1000,  device=device)
    print(f"Generating test ({tr.n_test} samples)...")
    generate_split(teacher, tr.n_test,  tc.seq_len, cache_dir, 'test',
                   seed=tc.seed + 2000,  device=device)

    torch.save({
        'teacher':    asdict(tc),
        'n_train':    tr.n_train,
        'n_val':      tr.n_val,
        'n_test':     tr.n_test,
        'seq_len':    tc.seq_len,
        'vector_dim': tc.d_model,
        'num_init':   1,
    }, os.path.join(cache_dir, 'data_config.pt'))
    print("\nDone! Config saved to data_config.pt")


if __name__ == '__main__':
    main()
```

**Step 2: Run smoke test (small dataset)**

Temporarily edit `configs/exp001_linear_mdm.yaml` to use small sizes for this test:
```yaml
training:
  n_train: 200
  n_val: 50
  n_test: 50
```

Then run:
```bash
python baseline_continuous/generate_data.py --config configs/exp001_linear_mdm.yaml
```

Expected: completes in < 30 seconds, no errors.

**Step 3: Verify cached data shapes and values**

Run:
```bash
python -c "
import sys; sys.path.insert(0,'.')
import numpy as np, torch
from baseline_continuous.exp_config import load_config, teacher_cache_key

cfg = load_config('configs/exp001_linear_mdm.yaml')
key = teacher_cache_key(cfg.teacher)
d   = f'{cfg.training.data_root}/{key}'

vecs = np.memmap(f'{d}/train_vectors.npy', dtype='float32', mode='r',
                 shape=(cfg.training.n_train, cfg.teacher.seq_len, cfg.teacher.d_model))
v = torch.from_numpy(vecs[:10].copy())
print('shape     :', v.shape)
print('mean_norm :', v.norm(dim=-1).mean().item())  # non-zero, finite

meta = torch.load(f'{d}/data_config.pt', weights_only=False)
print('meta      :', meta['vector_dim'], meta['seq_len'])
print('PASS')
"
```

Expected: shape `[10, 32, 128]`, finite mean_norm, no errors.

**Step 4: Test cache reuse (run again, should skip)**

```bash
python baseline_continuous/generate_data.py --config configs/exp001_linear_mdm.yaml
```

Expected: prints `Cache exists — skipping.`

**Step 5: Commit**

```bash
git add baseline_continuous/generate_data.py
git commit -m "feat: add generate_data.py with teacher-config-keyed disk caching (memmap)"
```

---

### Task 4: Dataset and DataLoader

**Files:**
- Create: `baseline_continuous/exp_dataset.py`

**Step 1: Write `baseline_continuous/exp_dataset.py`**

```python
"""
exp_dataset.py — Dataset and DataLoader for new teacher-student experiments.

Each sample returns:
    init_vectors    [1, D]    x_0 conditioning prefix
    main_vectors    [L-1, D]  tokens x_1..x_{L-1} in original causal order
    shuffled_main   [L-1, D]  main_vectors with a random token permutation
    shuffle_indices [L-1]     permutation applied: shuffled[i] = main[shuffle_indices[i]]
    order           [L-1]     inverse permutation
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple

from baseline_continuous.exp_config import ExpConfig, teacher_cache_key


class TeacherDataset(Dataset):
    def __init__(self, cache_dir: str, split: str, seq_len: int,
                 d_model: int, n_samples: int):
        vecs_path = os.path.join(cache_dir, f'{split}_vectors.npy')
        init_path = os.path.join(cache_dir, f'{split}_init_vectors.npy')
        if not os.path.exists(vecs_path):
            raise FileNotFoundError(
                f"Data not found: {vecs_path}\n"
                f"Run: python baseline_continuous/generate_data.py --config <your.yaml>"
            )
        self.vectors      = np.memmap(vecs_path, dtype='float32', mode='r',
                                      shape=(n_samples, seq_len, d_model))
        self.init_vectors = np.memmap(init_path, dtype='float32', mode='r',
                                      shape=(n_samples, 1, d_model))
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        vectors      = torch.from_numpy(self.vectors[idx].copy())       # [L, D]
        init_vectors = torch.from_numpy(self.init_vectors[idx].copy())  # [1, D]
        main_vectors = vectors[1:]                                       # [L-1, D]

        shuffle_indices = torch.randperm(len(main_vectors))
        shuffled_main   = main_vectors[shuffle_indices]
        order           = torch.argsort(shuffle_indices)                 # inverse permutation

        return {
            'init_vectors':    init_vectors,    # [1, D]
            'main_vectors':    main_vectors,    # [L-1, D]
            'shuffled_main':   shuffled_main,   # [L-1, D]
            'shuffle_indices': shuffle_indices, # [L-1]
            'order':           order,           # [L-1]
        }


def create_exp_dataloaders(cfg: ExpConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders from the teacher's cached data."""
    key       = teacher_cache_key(cfg.teacher)
    cache_dir = os.path.join(cfg.training.data_root, key)
    L, D, tr  = cfg.teacher.seq_len, cfg.teacher.d_model, cfg.training

    train_ds = TeacherDataset(cache_dir, 'train', L, D, tr.n_train)
    val_ds   = TeacherDataset(cache_dir, 'val',   L, D, tr.n_val)
    test_ds  = TeacherDataset(cache_dir, 'test',  L, D, tr.n_test)
    print(f"[dataset] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=tr.batch_size, shuffle=True,
                              num_workers=tr.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=tr.batch_size, shuffle=False,
                              num_workers=tr.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=tr.batch_size, shuffle=False,
                              num_workers=tr.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
```

**Step 2: Verify dataset output shapes**

Run:
```bash
python -c "
import sys; sys.path.insert(0,'.')
from baseline_continuous.exp_config import load_config
from baseline_continuous.exp_dataset import create_exp_dataloaders

cfg = load_config('configs/exp001_linear_mdm.yaml')
train_loader, val_loader, _ = create_exp_dataloaders(cfg)
batch = next(iter(train_loader))
for k, v in batch.items():
    print(f'  {k:20s}: {tuple(v.shape)}')
print('PASS')
"
```

Expected (batch_size=64, seq_len=32, d_model=128):
```
  init_vectors        : (64, 1, 128)
  main_vectors        : (64, 31, 128)
  shuffled_main       : (64, 31, 128)
  shuffle_indices     : (64, 31)
  order               : (64, 31)
```

**Step 3: Commit**

```bash
git add baseline_continuous/exp_dataset.py
git commit -m "feat: add TeacherDataset (memmap) and create_exp_dataloaders with random token shuffle"
```

---

### Task 5: Unified Training Script

**Files:**
- Create: `train.py`

**Step 1: Write `train.py`**

```python
"""
train.py — Unified entry point for teacher-student experiments.

Usage:
    python train.py --config configs/exp001_linear_mdm.yaml
    python train.py --config configs/exp001_linear_mdm.yaml --epochs 5 --device cuda

Teacher data is auto-generated and cached on first run (keyed by teacher config hash).
Student model: existing ContinuousAOGPT with vector_dim = teacher.d_model.

Training input selection:
  student.type=mdm,  shuffle_input=True   → shuffled tokens, Random order training
  student.type=ar,   shuffle_input=False, use_gt_order=True  → original tokens, AR
  student.type=ar,   shuffle_input=True,  use_gt_order=False → shuffled tokens, AR

Evaluation metrics per eval_interval:
  val_loss        — MSE on original causal order (AR mode)
  val_cos_sim     — cosine similarity of predictions vs targets
  token_cos_sim   — mean off-diagonal cos_sim of token representations (collapse check)
"""
import sys, os, math, copy, argparse
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_continuous.exp_config import load_config
from baseline_continuous.exp_dataset import create_exp_dataloaders
from baseline_continuous.generate_data import get_cache_dir, cache_exists, generate_split
from baseline_continuous.teachers import build_teacher
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig


# ─── Helpers ─────────────────────────────────────────────────────────────────

def maybe_generate_data(cfg):
    """Generate and cache teacher data if not already present."""
    cache_dir = get_cache_dir(cfg)
    if cache_exists(cache_dir):
        print(f"[data] Cache found: {cache_dir}")
        return
    print(f"[data] Cache not found — generating now...")
    os.makedirs(cache_dir, exist_ok=True)
    tc, tr = cfg.teacher, cfg.training
    teacher = build_teacher(tc).to(tr.device)
    for split, n, seed_off in [('train', tr.n_train, 0),
                                ('val',   tr.n_val,   1000),
                                ('test',  tr.n_test,  2000)]:
        print(f"  Generating {split} ({n} samples)...")
        generate_split(teacher, n, tc.seq_len, cache_dir, split,
                       seed=tc.seed + seed_off, device=tr.device)
    torch.save({
        'teacher': asdict(tc),
        'n_train': tr.n_train, 'n_val': tr.n_val, 'n_test': tr.n_test,
        'seq_len': tc.seq_len, 'vector_dim': tc.d_model, 'num_init': 1,
    }, os.path.join(cache_dir, 'data_config.pt'))
    print(f"[data] Done.\n")


def get_lr(step: int, warmup: int, total: int, base_lr: float, min_ratio: float = 0.1):
    min_lr = base_lr * min_ratio
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    if step >= total:
        return min_lr
    decay = (step - warmup) / (total - warmup)
    return min_lr + 0.5 * (1 + math.cos(math.pi * decay)) * (base_lr - min_lr)


@torch.no_grad()
def update_ema(ema, model, step, target: float = 0.9999):
    decay = min(target, (1 + step) / (10 + step))
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.data, alpha=1 - decay)


@torch.no_grad()
def evaluate(model, val_loader, device):
    """Evaluate val loss and cos_sim using AR mode on original causal order."""
    model.eval()
    total_loss, total_cos, n = 0.0, 0.0, 0
    for batch in val_loader:
        init_vecs = batch['init_vectors'].to(device)
        main_vecs = batch['main_vectors'].to(device)
        ni, t     = init_vecs.shape[1], main_vecs.shape[1]
        preds, loss = model(main_vecs, mode='AR', init_vectors=init_vecs)
        shift_preds = preds[:, ni - 1: ni - 1 + t]
        cos_sim     = F.cosine_similarity(shift_preds, main_vecs, dim=-1).mean()
        total_loss += loss.item()
        total_cos  += cos_sim.item()
        n          += 1
    model.train()
    return {'val_loss': total_loss / max(n, 1), 'val_cos_sim': total_cos / max(n, 1)}


@torch.no_grad()
def token_cos_sim_offdiag(model, val_loader, device, n_batches: int = 5) -> float:
    """Mean off-diagonal cos_sim between token positions (data collapse check).
    Uses the raw data vectors (not model predictions) to diagnose data quality."""
    model.eval()
    all_mean_vecs = []
    for i, batch in enumerate(val_loader):
        if i >= n_batches:
            break
        vecs = batch['main_vectors'].to(device)          # [B, L-1, D]
        all_mean_vecs.append(vecs.mean(dim=0))           # [L-1, D] batch mean
    mean_vecs  = torch.stack(all_mean_vecs).mean(0)      # [L-1, D]
    norm_vecs  = F.normalize(mean_vecs, dim=-1)
    cos_mat    = norm_vecs @ norm_vecs.T                 # [L-1, L-1]
    L          = cos_mat.shape[0]
    off_diag   = cos_mat[~torch.eye(L, dtype=torch.bool, device=device)]
    model.train()
    return off_diag.abs().mean().item()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Teacher-Student experiment trainer')
    parser.add_argument('--config',    required=True, help='Path to YAML config')
    parser.add_argument('--epochs',    type=int,   default=None)
    parser.add_argument('--device',    type=str,   default=None)
    parser.add_argument('--wandb_log', type=str,   default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.epochs    is not None: cfg.training.epochs    = args.epochs
    if args.device    is not None: cfg.training.device    = args.device
    if args.wandb_log is not None:
        cfg.training.wandb_log = args.wandb_log.lower() in ('true', '1', 'yes')

    tc, sc, tr = cfg.teacher, cfg.student, cfg.training
    device     = tr.device

    torch.manual_seed(tr.seed)
    np.random.seed(tr.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(tr.seed)

    print("=" * 60)
    print(f"Teacher : {tc.type}  d_model={tc.d_model}  seq_len={tc.seq_len}")
    print(f"Student : {sc.type}  shuffle_input={sc.shuffle_input}"
          f"  use_gt_order={sc.use_gt_order}")
    print(f"Training: epochs={tr.epochs}  bs={tr.batch_size}  lr={tr.lr}")
    print("=" * 60)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    maybe_generate_data(cfg)
    train_loader, val_loader, test_loader = create_exp_dataloaders(cfg)

    # ── 2. Student model (ContinuousAOGPT, vector_dim = teacher d_model) ─────
    seq_main  = tc.seq_len - 1
    model_cfg = ContinuousAOGPTConfig(
        block_size = seq_main + 1,   # +1 for the init prefix slot
        vector_dim = tc.d_model,
        n_layer    = sc.n_layer,
        n_head     = sc.n_head,
        n_embd     = sc.n_embd,
        dropout    = sc.dropout,
        bias       = True,
        num_init   = 1,
    )
    model     = ContinuousAOGPT(model_cfg).to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    optimizer = model.configure_optimizers(
        weight_decay=tr.weight_decay,
        learning_rate=tr.lr,
        betas=(0.9, 0.95),
        device_type='cuda' if 'cuda' in device else 'cpu',
    )

    # ── 3. Training mode ─────────────────────────────────────────────────────
    # MDM → Random order;  AR → ascending (AR) order
    train_mode = 'Random' if sc.type == 'mdm' else 'AR'

    iters_per_epoch = len(train_loader)
    max_iters       = tr.epochs * iters_per_epoch
    warmup_iters    = int(tr.warmup_ratio * max_iters)
    print(f"{iters_per_epoch} iters/epoch × {tr.epochs} = {max_iters} total iters\n")

    if tr.wandb_log:
        import wandb
        wandb.init(project=tr.wandb_project,
                   config={'teacher': asdict(tc), 'student': asdict(sc), 'training': asdict(tr)})

    best_val_loss = float('inf')
    global_step   = 0
    model.train()

    for epoch in range(tr.epochs):
        for batch in train_loader:
            init_vecs = batch['init_vectors'].to(device)

            # Select input sequence based on student config
            if sc.shuffle_input:
                vectors = batch['shuffled_main'].to(device)
            else:
                vectors = batch['main_vectors'].to(device)   # original causal order

            lr = get_lr(global_step, warmup_iters, max_iters, tr.lr)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            _, loss = model(vectors, mode=train_mode, init_vectors=init_vecs)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if tr.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr.grad_clip)
            optimizer.step()
            update_ema(ema_model, model, global_step)

            if global_step % tr.log_interval == 0:
                print(f"epoch {epoch+1}/{tr.epochs} | step {global_step:>6d} | "
                      f"loss {loss.item():.4f} | lr {lr:.2e}")
                if tr.wandb_log:
                    import wandb
                    wandb.log({'train/loss': loss.item(), 'train/lr': lr,
                               'epoch': epoch}, step=global_step)

            if global_step % tr.eval_interval == 0 and global_step > 0:
                eval_res   = evaluate(ema_model, val_loader, device)
                tok_cos    = token_cos_sim_offdiag(ema_model, val_loader, device)
                print(f"  [eval] val_loss={eval_res['val_loss']:.4f}"
                      f"  val_cos_sim={eval_res['val_cos_sim']:.4f}"
                      f"  token_cos_sim={tok_cos:.4f}")

                if tr.wandb_log:
                    import wandb
                    wandb.log({'val/loss': eval_res['val_loss'],
                               'val/cos_sim': eval_res['val_cos_sim'],
                               'val/token_cos_sim': tok_cos}, step=global_step)

                if eval_res['val_loss'] < best_val_loss:
                    best_val_loss = eval_res['val_loss']
                    os.makedirs(tr.checkpoint_dir, exist_ok=True)
                    ckpt = os.path.join(tr.checkpoint_dir,
                                        f"best_{tc.type}_{sc.type}.pt")
                    torch.save({
                        'ema_state_dict':   ema_model.state_dict(),
                        'model_state_dict': model.state_dict(),
                        'model_cfg':        model_cfg,
                        'teacher_cfg':      asdict(tc),
                        'student_cfg':      asdict(sc),
                        'epoch':            epoch,
                        'global_step':      global_step,
                        'val_loss':         best_val_loss,
                    }, ckpt)
                    print(f"  [save] {ckpt}  (val_loss={best_val_loss:.4f})")

                model.train()

            global_step += 1

    # ── 4. Final evaluation ──────────────────────────────────────────────────
    print("\n" + "=" * 60 + "\nFinal Evaluation")
    for name, loader in [('val', val_loader), ('test', test_loader)]:
        res     = evaluate(ema_model, loader, device)
        tok_cos = token_cos_sim_offdiag(ema_model, loader, device)
        print(f"  [{name}] loss={res['val_loss']:.6f}"
              f"  cos_sim={res['val_cos_sim']:.4f}"
              f"  token_cos_sim={tok_cos:.4f}")
        if tr.wandb_log:
            import wandb
            wandb.log({f'final/{name}_loss': res['val_loss'],
                       f'final/{name}_cos_sim': res['val_cos_sim']})

    if tr.wandb_log:
        import wandb
        wandb.finish()

    print("\nTraining complete.")


if __name__ == '__main__':
    main()
```

**Step 2: Dry-run smoke test (2 epochs, tiny dataset)**

First, edit `configs/exp001_linear_mdm.yaml` training section for quick test:
```yaml
training:
  n_train: 500
  n_val: 100
  n_test: 100
  epochs: 2
  eval_interval: 50
  log_interval: 10
  num_workers: 0
```

Run:
```bash
python train.py --config configs/exp001_linear_mdm.yaml
```

Expected: Runs 2 epochs with loss printing and at least one eval checkpoint. No errors.

**Step 3: Test all 3 teacher types for 1 epoch**

```bash
python train.py --config configs/exp001_linear_mdm.yaml --epochs 1
python train.py --config configs/exp002_sparse_mdm.yaml --epochs 1
python train.py --config configs/exp003_standard_ar.yaml --epochs 1
```

Each should complete without errors. Confirms teacher type switching works.

**Step 4: Restore production config values in exp001_linear_mdm.yaml**

Revert to `n_train: 10000`, `n_val: 1000`, `epochs: 50`, etc.

**Step 5: Commit**

```bash
git add train.py
git commit -m "feat: add unified train.py — auto data gen, YAML config, AR/MDM student, token collapse eval"
```

---

### Task 6: Full Experiment Run (Optional — production scale)

**Step 1: Generate full datasets for all 3 teacher types**

```bash
# Each teacher has its own cache
python baseline_continuous/generate_data.py --config configs/exp001_linear_mdm.yaml
python baseline_continuous/generate_data.py --config configs/exp002_sparse_mdm.yaml
python baseline_continuous/generate_data.py --config configs/exp003_standard_ar.yaml
```

**Step 2: Run all 3 experiments**

```bash
python train.py --config configs/exp001_linear_mdm.yaml  > logs/exp001.txt 2>&1 &
python train.py --config configs/exp002_sparse_mdm.yaml  > logs/exp002.txt 2>&1 &
python train.py --config configs/exp003_standard_ar.yaml > logs/exp003.txt 2>&1 &
```

**Step 3: Monitor**

```bash
tail -f logs/exp001.txt
grep "\[eval\]" logs/exp001.txt
```

**Step 4: Commit results**

```bash
mkdir -p logs
git add logs/
git commit -m "results: linear_attention / sparse_causal_mask / standard_causal teacher experiments"
```

---

## Summary of New Files

| File | Purpose |
|------|---------|
| `baseline_continuous/exp_config.py` | YAML config dataclasses + `load_config()` + `teacher_cache_key()` |
| `baseline_continuous/teachers.py` | 3 teacher types + `build_teacher()` factory |
| `baseline_continuous/generate_data.py` | Data generation with teacher-config-keyed caching |
| `baseline_continuous/exp_dataset.py` | `TeacherDataset` (memmap) + `create_exp_dataloaders()` |
| `train.py` | Unified entry point: auto-gen data → train student → log metrics |
| `configs/exp001_linear_mdm.yaml` | Linear attention teacher + MDM student |
| `configs/exp002_sparse_mdm.yaml` | Sparse causal mask teacher + MDM student |
| `configs/exp003_standard_ar.yaml` | Standard causal teacher + AR student (lower bound) |

**Zero existing files modified.** All backward-compatible.

## Key Design Decisions

1. **Teacher weight seed isolation**: `torch.random.fork_rng()` ensures teacher weight init doesn't affect the global RNG used for data generation.
2. **Data seed discipline**: `torch.manual_seed(tc.seed + split_offset)` in `generate_split()` makes each split reproducible independently of batch size.
3. **Cache key**: MD5 hash of all teacher hyperparams (type, d_model, n_heads, n_layers, sparsity, noise_std, seed). Changing any param forces regeneration.
4. **Student model**: Existing `ContinuousAOGPT` reused as-is, with `vector_dim=tc.d_model` and `num_init=1`.
5. **Eval mode**: Validation always uses AR mode on original causal order — a consistent reference across all experiments. Training loss is the primary metric of interest.
