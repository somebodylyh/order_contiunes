# Continuous h-space AR Teacher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace wikitext-based data generation with step-by-step h-space AR generation so that the theoretical lower bound σ²=0.09 is truly achievable by the student.

**Architecture:** `ContinuousHSpaceTeacher` generates sequences by iteratively feeding normalized h_t back into GLA (via `inputs_embeds`, bypassing the embedding layer). Noise ε_t ~ N(0, σ²I) is baked into each step. Dataset returns separate normalized `input` and unnormalized `target` tensors; `StudentWithProjection.forward` is extended to accept the target separately.

**Tech Stack:** PyTorch, FLA library (`fla.models.GLAForCausalLM`), existing `gla_exp` framework.

---

### Background: Shape Semantics of `ContinuousAOGPT`

`ContinuousAOGPT.forward_fn` in legacy mode (num_init=0):
- Prepends learnable `[None]` token → input is `[None, x_0, ..., x_{L-1}]` (L+1 tokens)
- Returns `(predictions [B, L+1, D], loss_scalar)`
- `predictions[:, t, :]` = prediction for target `x_t` using context `[None, x_0, ..., x_{t-1}]`
- `shift_preds = predictions[:, :-1, :]` → shape `[B, L, D]` → compared to targets `[B, L, D]`

`StudentWithProjection.forward` currently:
```python
preds_p, _ = self.gpt(xp, mode=mode)   # preds_p: [B, L+1, D], _: internal loss (ignored)
preds      = self.out_proj(preds_p)     # [B, L+1, D]
loss       = F.mse_loss(preds[:, :-1], x)  # [B, L, D] vs [B, L, D]
```
Target `x` is the noisy input sequence. For Method B, we decouple: input is normalized, target is unnormalized.

**Lower bound note:** Position 0 (h_0 random Gaussian, variance 1 per dim) is predicted from `[None]` token → contributes ≈ 1 per dim. Positions 1..L-1 → lower bound = σ². Avg lower bound = (1 + 31 × 0.09) / 32 ≈ **0.118** (vs wikitext's 1.91 — still a strong validation).

---

### Task 1: Extend `TeacherConfig` in `exp_config.py`

**Files:**
- Modify: `gla_exp/exp_config.py`

**Step 1: Add two fields to `TeacherConfig` dataclass**

In `exp_config.py`, change:
```python
@dataclass
class TeacherConfig:
    model_name: str = "fla-hub/gla-340M-15B"
    layer_idx: int = 3
    seq_len: int = 32
    n_train: int = 100000
    n_test: int = 10000
    extract_batch_size: int = 64
    dataset_name: str = "wikitext"
    cache_path: str = "data/teacher_cache"
```
to:
```python
@dataclass
class TeacherConfig:
    model_name: str = "fla-hub/gla-340M-15B"
    layer_idx: int = 3
    seq_len: int = 32
    n_train: int = 100000
    n_test: int = 10000
    extract_batch_size: int = 64
    dataset_name: str = "wikitext"
    cache_path: str = "data/teacher_cache"
    generation_mode: str = "wikitext"   # "wikitext" | "continuous_h"
    sigma: float = 0.3                  # noise std for continuous_h generation
```

**Step 2: Update `teacher_cache_key` to include new fields**

In `teacher_cache_key()`, change `key_fields` dict:
```python
key_fields = {
    "model_name":       cfg.model_name,
    "layer_idx":        cfg.layer_idx,
    "seq_len":          cfg.seq_len,
    "n_train":          cfg.n_train,
    "n_test":           cfg.n_test,
    "dataset_name":     cfg.dataset_name,
    "generation_mode":  cfg.generation_mode,
    "sigma":            cfg.sigma,
}
```

**Step 3: Verify existing configs still load correctly**

Run:
```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM
python -c "
from gla_exp.exp_config import load_config, teacher_cache_key
tc, sc, tr = load_config('gla_exp/configs/exp001_ar_noshuffle.yaml')
print('generation_mode:', tc.generation_mode)
print('sigma:', tc.sigma)
print('cache_key:', teacher_cache_key(tc))
"
```
Expected: `generation_mode: wikitext`, `sigma: 0.3`, cache key is a different 8-char hash from before (since we added new fields — existing wikitext cache dirs will need to be regenerated or force-copied, but this is acceptable).

**Step 4: Commit**
```bash
git add gla_exp/exp_config.py
git commit -m "feat: add generation_mode and sigma fields to TeacherConfig"
```

---

### Task 2: Add `ContinuousHSpaceTeacher` to `teachers.py`

**Files:**
- Modify: `gla_exp/teachers.py`

**Step 1: Add imports at top of file**

After `import torch.nn as nn`, add:
```python
import math
import torch.nn.functional as F
```

**Step 2: Add the new class after `FLATeacher`**

```python
class ContinuousHSpaceTeacher(nn.Module):
    """
    连续 h-space AR Teacher: 用预训练 GLA 前 (layer_idx+1) 层在连续空间自回归生成序列。

    生成过程：
      h_0 = Normalize(N(0,I_D)) × √D
      x_{t-1} = Normalize(h_{t-1}) × √D   (归一化后作为 GLA inputs_embeds 输入)
      μ_t = GLA_4L(inputs_embeds=[x_0,...,x_{t-1}])[:, -1, :]
      ε_t ~ N(0, σ² I_D)
      h_t = μ_t + ε_t                      (存入 hidden.pt，学生的预测目标)

    存入 hidden.pt 的是 h_0..h_{L-1}（h_0 本身已归一化，norm ≡ √D）。
    理论下界（位置 1..L-1）= σ²。

    注：必须在 CUDA 上运行（FLA Triton kernel 要求）。
    """

    def __init__(self, cfg):
        super().__init__()
        from fla.models import GLAForCausalLM

        self.cfg = cfg
        print(f"[ContinuousHSpaceTeacher] Loading {cfg.model_name} ...")
        model = GLAForCausalLM.from_pretrained(cfg.model_name)

        n_keep = cfg.layer_idx + 1
        model.model.layers = nn.ModuleList(
            list(model.model.layers)[:n_keep]
        )
        print(f"[ContinuousHSpaceTeacher] Truncated to {n_keep} layers "
              f"(layer_idx={cfg.layer_idx}, d_model={model.config.hidden_size})")

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model    = model
        self.d_hidden = model.config.hidden_size
        self._hidden_buf = None

        def _hook(module, inp, output):
            h = output[0] if isinstance(output, tuple) else output
            self._hidden_buf = h.detach()

        self.model.model.layers[-1].register_forward_hook(_hook)

    @torch.no_grad()
    def generate_sequence(self, B: int, L: int, sigma: float,
                          device: torch.device) -> torch.Tensor:
        """
        生成 B 条长度为 L 的 h-space 序列。

        返回: [B, L, D] tensor
          - 位置 0: h_0 = Normalize(N(0,I)) × √D
          - 位置 t (t≥1): h_t = GLA_4L(x_{0:t-1})[-1] + ε_t, ε_t ~ N(0, σ²I)
        """
        D = self.d_hidden

        # 初始随机状态，归一化到 √D 尺度
        h0 = torch.randn(B, D, device=device)
        h0 = F.normalize(h0, dim=-1) * math.sqrt(D)

        seq = [h0]  # 存储 h_0 .. h_{L-1}

        for t in range(1, L):
            # 构建归一化历史序列作为 inputs_embeds
            x_hist = torch.stack(
                [F.normalize(h, dim=-1) * math.sqrt(D) for h in seq],
                dim=1
            )  # [B, t, D]

            # GLA forward：bypass embed_tokens，直接输入连续向量
            self._hidden_buf = None
            self.model(inputs_embeds=x_hist)
            assert self._hidden_buf is not None, "Hook 未触发"
            mu_t = self._hidden_buf[:, -1, :]  # [B, D]

            # 加高斯噪声
            eps = torch.randn_like(mu_t) * sigma
            h_t = mu_t + eps
            seq.append(h_t)

        return torch.stack(seq, dim=1)  # [B, L, D]
```

**Step 3: Verify the class instantiates without error**

```bash
python -c "
import torch, sys
sys.path.insert(0, '.')
from gla_exp.exp_config import TeacherConfig
from gla_exp.teachers import ContinuousHSpaceTeacher
cfg = TeacherConfig(generation_mode='continuous_h', sigma=0.3)
teacher = ContinuousHSpaceTeacher(cfg).cuda()
print('d_hidden:', teacher.d_hidden)
print('Teacher loaded OK')
"
```
Expected: `d_hidden: 1024`, `Teacher loaded OK`

**Step 4: Verify `generate_sequence` output shape and norms**

```bash
python -c "
import torch, math, sys
sys.path.insert(0, '.')
from gla_exp.exp_config import TeacherConfig
from gla_exp.teachers import ContinuousHSpaceTeacher
cfg = TeacherConfig(generation_mode='continuous_h', sigma=0.3)
teacher = ContinuousHSpaceTeacher(cfg).cuda()
seq = teacher.generate_sequence(B=4, L=8, sigma=0.3, device=torch.device('cuda'))
print('shape:', seq.shape)           # [4, 8, 1024]
print('h0 norms:', seq[:, 0, :].norm(dim=-1))   # all ≈ √1024 = 32
print('h1 norms:', seq[:, 1, :].norm(dim=-1))   # ≈ 32 ± small noise
print('h1-h0 norm diff:', (seq[:, 1, :] - seq[:, 0, :]).norm(dim=-1))  # > 0
"
```
Expected: shape `[4, 8, 1024]`, h0 norms all ≈ 32.0, h1 norms ≈ 32.

**Step 5: Commit**
```bash
git add gla_exp/teachers.py
git commit -m "feat: add ContinuousHSpaceTeacher for h-space AR generation"
```

---

### Task 3: Extend `generate_data.py` with `continuous_h` branch

**Files:**
- Modify: `gla_exp/generate_data.py`

**Step 1: Add `math` import**

At top of file, `import math` is already present. No change needed.

**Step 2: Add `_generate_continuous_h_samples` function**

After `_fill_samples`, add:

```python
def _generate_continuous_h_samples(teacher, hidden_buf, batch_size, device, desc):
    """用 ContinuousHSpaceTeacher 批量生成 h-space 序列，填入预分配 buffer。"""
    import math
    n_samples = hidden_buf.shape[0]
    L         = hidden_buf.shape[1]
    sigma     = teacher.cfg.sigma
    n_batches = math.ceil(n_samples / batch_size)

    print(f"[generate] 生成 {desc} {n_samples} 样本 ({n_batches} batches × ~{batch_size}, L={L})...")
    offset = 0
    for i in range(n_batches):
        B   = min(batch_size, n_samples - offset)
        seq = teacher.generate_sequence(B=B, L=L, sigma=sigma, device=device)
        hidden_buf[offset:offset + B].copy_(seq.cpu())
        offset += B
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_batches}", flush=True)
```

**Step 3: Add `_generate_continuous_h` function**

After the new function above, add:

```python
def _generate_continuous_h(teacher_cfg: TeacherConfig) -> str:
    """生成并缓存连续 h-space AR 序列。返回 cache_dir 路径。"""
    from gla_exp.teachers import ContinuousHSpaceTeacher

    cache_dir = get_cache_dir(teacher_cfg)
    os.makedirs(cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher = ContinuousHSpaceTeacher(teacher_cfg).to(device)
    D = teacher.d_hidden
    L = teacher_cfg.seq_len
    B = teacher_cfg.extract_batch_size
    n_train, n_test = teacher_cfg.n_train, teacher_cfg.n_test
    N = n_train + n_test

    print(f"[generate] 预分配 hidden [{N}, {L}, {D}] ({N*L*D*4/1e9:.1f} GB)...")
    hidden_all = torch.zeros(N, L, D, dtype=torch.float32)
    perm_all   = torch.zeros(N, L, dtype=torch.long)   # placeholder，暂不使用

    _generate_continuous_h_samples(teacher, hidden_all[:n_train], B, device, "train")
    _generate_continuous_h_samples(teacher, hidden_all[n_train:], B, device, "test")

    print("[generate] Saving to disk...")
    torch.save(hidden_all, os.path.join(cache_dir, "hidden.pt"))
    torch.save(perm_all,   os.path.join(cache_dir, "perm.pt"))
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump({
            "D":               D,
            "L":               L,
            "n_train":         n_train,
            "n_test":          n_test,
            "model_name":      teacher_cfg.model_name,
            "layer_idx":       teacher_cfg.layer_idx,
            "generation_mode": teacher_cfg.generation_mode,
            "sigma":           teacher_cfg.sigma,
        }, f, indent=2)

    print(f"[generate] Done: {n_train} train + {n_test} test  shape=[{N}, {L}, {D}]")
    return cache_dir
```

**Step 4: Modify `generate_and_cache` to dispatch on `generation_mode`**

Change `generate_and_cache`:
```python
def generate_and_cache(teacher_cfg: TeacherConfig, force: bool = False) -> str:
    """生成并缓存 hidden states。返回 cache_dir 路径。"""
    cache_dir = get_cache_dir(teacher_cfg)

    if not force and cache_exists(cache_dir):
        print(f"[cache] Found: {cache_dir}")
        return cache_dir

    if teacher_cfg.generation_mode == "continuous_h":
        return _generate_continuous_h(teacher_cfg)
    else:
        return _generate_wikitext(teacher_cfg)
```

And rename the existing body of `generate_and_cache` into a new `_generate_wikitext` function:

```python
def _generate_wikitext(teacher_cfg: TeacherConfig) -> str:
    """生成并缓存 wikitext hidden states（原有逻辑）。"""
    from gla_exp.teachers import FLATeacher

    cache_dir = get_cache_dir(teacher_cfg)
    os.makedirs(cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[generate] Loading teacher: {teacher_cfg.model_name}")
    teacher = FLATeacher(teacher_cfg).to(device)
    D = teacher.d_hidden

    L, B       = teacher_cfg.seq_len, teacher_cfg.extract_batch_size
    n_train, n_test = teacher_cfg.n_train, teacher_cfg.n_test
    N = n_train + n_test

    train_pool, test_pool = _build_token_pools(teacher_cfg, teacher)

    print(f"[generate] 预分配 hidden [{N}, {L}, {D}] ({N*L*D*4/1e9:.1f} GB)...")
    hidden_all = torch.zeros(N, L, D,  dtype=torch.float32)
    perm_all   = torch.zeros(N, L,     dtype=torch.long)

    _fill_samples(teacher, train_pool, hidden_all[:n_train], perm_all[:n_train], B, device, "train")
    _fill_samples(teacher, test_pool,  hidden_all[n_train:], perm_all[n_train:], B, device, "test")

    print(f"[generate] Saving to disk...")
    torch.save(hidden_all, os.path.join(cache_dir, "hidden.pt"))
    torch.save(perm_all,   os.path.join(cache_dir, "perm.pt"))
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump({
            "D":          D,
            "L":          L,
            "n_train":    n_train,
            "n_test":     n_test,
            "model_name": teacher_cfg.model_name,
            "layer_idx":  teacher_cfg.layer_idx,
        }, f, indent=2)

    print(f"[generate] Done: {n_train} train + {n_test} test  shape=[{N}, {L}, {D}]")
    return cache_dir
```

**Step 5: Smoke-test with tiny dataset**

```bash
python -c "
import sys
sys.path.insert(0, '.')
from gla_exp.exp_config import TeacherConfig
from gla_exp.generate_data import generate_and_cache
cfg = TeacherConfig(
    generation_mode='continuous_h',
    sigma=0.3,
    n_train=32,
    n_test=8,
    seq_len=8,
    extract_batch_size=16,
    cache_path='data/test_cache',
)
cache_dir = generate_and_cache(cfg, force=True)
import torch, json, os
h = torch.load(os.path.join(cache_dir, 'hidden.pt'), weights_only=True)
print('hidden shape:', h.shape)   # [40, 8, 1024]
print('h0 norm mean:', h[:, 0, :].norm(dim=-1).mean().item())  # ≈ 32
with open(os.path.join(cache_dir, 'meta.json')) as f:
    print('meta:', json.load(f))
"
```
Expected: shape `[40, 8, 1024]`, h0 norm ≈ 32.0.

**Step 6: Commit**
```bash
git add gla_exp/generate_data.py
git commit -m "feat: add continuous_h generation branch to generate_data"
```

---

### Task 4: Add `continuous_ar` mode to `exp_dataset.py`

**Files:**
- Modify: `gla_exp/exp_dataset.py`

**Step 1: Add `math` import at top**

Add after `import os, json`:
```python
import math
import torch.nn.functional as F
```

**Step 2: Extend `HiddenStateDataset.__init__` to allow `continuous_ar`**

Change the assertion:
```python
assert student_type in ("ar_noshuffle", "ar_shuffled", "mdm_shuffled"), \
    f"Unknown student_type: {student_type!r}"
```
to:
```python
assert student_type in ("ar_noshuffle", "ar_shuffled", "mdm_shuffled", "continuous_ar"), \
    f"Unknown student_type: {student_type!r}"
```

Also store `D` from the hidden tensor:
In `__init__`, after `self.hidden = torch.load(...)`, add:
```python
self._D = self.hidden.shape[-1]
```

**Step 3: Add `continuous_ar` branch in `__getitem__`**

Add at the start of `__getitem__`, before the `ar_noshuffle` block:
```python
if self.student_type == "continuous_ar":
    hidden = self.hidden[idx].clone().float()  # [L, D]
    D      = self._D
    sqrt_D = math.sqrt(D)
    # input:  normalized h_t → x_t = Normalize(h_t) × √D
    x_in  = F.normalize(hidden, dim=-1) * sqrt_D
    # target: unnormalized h_t (noise baked in during generation)
    x_tgt = hidden
    return {"input": x_in, "target": x_tgt}
```

**Step 4: Verify dataset output shapes**

```bash
python -c "
import sys
sys.path.insert(0, '.')
from gla_exp.exp_config import TeacherConfig
from gla_exp.generate_data import generate_and_cache, get_cache_dir
from gla_exp.exp_dataset import HiddenStateDataset
import math

cfg = TeacherConfig(
    generation_mode='continuous_h', sigma=0.3,
    n_train=32, n_test=8, seq_len=8,
    extract_batch_size=16, cache_path='data/test_cache',
)
cache_dir = get_cache_dir(cfg)
ds = HiddenStateDataset(cache_dir, 'continuous_ar', 'train')
sample = ds[0]
print('input shape:', sample['input'].shape)   # [8, 1024]
print('target shape:', sample['target'].shape) # [8, 1024]
print('input norms:', sample['input'].norm(dim=-1))   # all ≈ 32
print('h0 input vs target match:', (sample['input'][0] - sample['target'][0]).norm().item())  # ≈ 0 (h0 is already normalized)
"
```
Expected: `input shape: [8, 1024]`, `target shape: [8, 1024]`, input norms all ≈ 32.

**Step 5: Commit**
```bash
git add gla_exp/exp_dataset.py
git commit -m "feat: add continuous_ar mode to HiddenStateDataset"
```

---

### Task 5: Modify `train.py` for separate input/target

**Files:**
- Modify: `gla_exp/train.py`

**Step 1: Modify `StudentWithProjection.forward` to accept `target`**

Change:
```python
def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
    xp         = self.in_proj(x)
    preds_p, _ = self.gpt(xp, mode=mode)
    preds      = self.out_proj(preds_p)
    loss       = F.mse_loss(preds[:, :-1], x)
    return loss
```
to:
```python
def forward(self, x: torch.Tensor, mode: str,
            target: torch.Tensor = None) -> torch.Tensor:
    """
    x:      [B, L, D]  student 输入（continuous_ar 模式下为归一化 h）
    target: [B, L, D]  loss 对比目标（continuous_ar 下为未归一化 h；None → 使用 x 本身）
    """
    if target is None:
        target = x
    xp         = self.in_proj(x)
    preds_p, _ = self.gpt(xp, mode=mode)
    preds      = self.out_proj(preds_p)
    loss       = F.mse_loss(preds[:, :-1], target)
    return loss
```

**Step 2: Modify `evaluate` to handle `target` in batch**

Change:
```python
@torch.no_grad()
def evaluate(model, loader, device, mode, noise_scale=0.0):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x = batch["input"].to(device)
        if noise_scale > 0:
            x = x + noise_scale * torch.randn_like(x)
        loss = model(x, mode=mode)
        total += loss.item()
        n     += 1
    model.train()
    return total / max(n, 1)
```
to:
```python
@torch.no_grad()
def evaluate(model, loader, device, mode, noise_scale=0.0):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x_in  = batch["input"].to(device)
        x_tgt = batch["target"].to(device) if "target" in batch else None
        if x_tgt is None:
            if noise_scale > 0:
                x_in = x_in + noise_scale * torch.randn_like(x_in)
            x_tgt = x_in
        loss = model(x_in, mode=mode, target=x_tgt)
        total += loss.item()
        n     += 1
    model.train()
    return total / max(n, 1)
```

**Step 3: Modify the training loop to handle `target`**

Change the training loop body:
```python
for batch in train_loader:
    x = batch["input"].to(device)

    # 动态加噪（每 batch 独立采样 → 防止记忆，提供 σ² 下界）
    if tr.noise_scale > 0:
        x = x + tr.noise_scale * torch.randn_like(x)

    ...
    loss = model(x, mode=mode)
```
to:
```python
for batch in train_loader:
    x_in  = batch["input"].to(device)
    x_tgt = batch["target"].to(device) if "target" in batch else None
    if x_tgt is None:
        # wikitext 模式：动态加噪，target = noisy input
        if tr.noise_scale > 0:
            x_in = x_in + tr.noise_scale * torch.randn_like(x_in)
        x_tgt = x_in

    ...
    loss = model(x_in, mode=mode, target=x_tgt)
```

**Step 4: Quick sanity check (no YAML yet)**

```bash
python -c "
import torch
import sys
sys.path.insert(0, '.')
from gla_exp.train import StudentWithProjection
from gla_exp.exp_config import StudentConfig, TeacherConfig

sc = StudentConfig(type='ar_noshuffle', d_model=32, n_layers=2, n_heads=2)
model = StudentWithProjection(teacher_D=64, sc=sc, seq_len=8)
x_in  = torch.randn(2, 8, 64)
x_tgt = torch.randn(2, 8, 64)

# Old-style call (target=None, falls back to x_in)
loss1 = model(x_in, mode='AR')
print('old-style loss:', loss1.item())

# New-style call (separate target)
loss2 = model(x_in, mode='AR', target=x_tgt)
print('new-style loss:', loss2.item())
"
```
Expected: both losses are positive floats, no errors.

**Step 5: Commit**
```bash
git add gla_exp/train.py
git commit -m "feat: decouple input/target in StudentWithProjection and training loop"
```

---

### Task 6: Write YAML config and run smoke test

**Files:**
- Create: `gla_exp/configs/exp004_continuous_h_ar.yaml`

**Step 1: Create the config file**

```yaml
teacher:
  model_name: "fla-hub/gla-340M-15B"
  layer_idx: 3
  seq_len: 32
  n_train: 100000
  n_test: 10000
  extract_batch_size: 64
  generation_mode: "continuous_h"
  sigma: 0.3
  cache_path: "data/teacher_cache"

student:
  type: "continuous_ar"
  d_model: 256
  n_layers: 5
  n_heads: 4
  chunk_size: 1

training:
  lr: 3e-4
  epochs: 50
  batch_size: 256
  log_interval: 5
  warmup_ratio: 0.05
  ema_decay: 0.9999
  grad_clip: 1.0
  noise_scale: 0.0    # noise already baked into h_t during generation
```

**Step 2: Run a tiny smoke test end-to-end**

First generate a small dataset:
```bash
python -c "
import sys
sys.path.insert(0, '.')
from gla_exp.exp_config import load_config
from gla_exp.generate_data import generate_and_cache

tc, sc, tr = load_config('gla_exp/configs/exp004_continuous_h_ar.yaml')
# Override to tiny size
tc.n_train = 512
tc.n_test  = 64
tc.seq_len = 32
tc.cache_path = 'data/smoke_test_cache'
generate_and_cache(tc, force=True)
print('Data generation OK')
"
```

Then verify training loop runs for 2 steps:
```bash
python -c "
import sys, torch, copy
sys.path.insert(0, '.')
from gla_exp.exp_config import load_config
from gla_exp.generate_data import get_cache_dir
from gla_exp.exp_dataset import create_dataloaders
from gla_exp.train import StudentWithProjection, update_ema
import torch.optim as optim

tc, sc, tr = load_config('gla_exp/configs/exp004_continuous_h_ar.yaml')
tc.n_train = 512; tc.n_test = 64; tc.cache_path = 'data/smoke_test_cache'

cache_dir = get_cache_dir(tc)
train_loader, test_loader = create_dataloaders(
    cache_dir, 'continuous_ar', batch_size=32, chunk_size=1, num_workers=0
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = StudentWithProjection(teacher_D=1024, sc=sc, seq_len=32).to(device)
ema   = copy.deepcopy(model)
opt   = optim.AdamW(model.parameters(), lr=1e-3)

for i, batch in enumerate(train_loader):
    x_in  = batch['input'].to(device)
    x_tgt = batch['target'].to(device)
    opt.zero_grad()
    loss = model(x_in, mode='AR', target=x_tgt)
    loss.backward()
    opt.step()
    update_ema(ema, model, i)
    print(f'step {i}: loss={loss.item():.4f}')
    if i >= 1:
        break
print('Training loop OK')
"
```
Expected: two loss values printed, both positive floats, no errors.

**Step 3: Commit**
```bash
git add gla_exp/configs/exp004_continuous_h_ar.yaml
git commit -m "feat: add exp004 continuous h-space AR config"
```

---

### Task 7: Verify lower bound is achievable (sanity check)

This task verifies that the mathematical lower bound ≈ 0.118 is indeed the minimum achievable loss, not something higher.

**Step 1: Compute oracle lower bound on test set**

```bash
python -c "
import sys, torch, math
import torch.nn.functional as F
sys.path.insert(0, '.')
from gla_exp.exp_config import TeacherConfig
from gla_exp.generate_data import get_cache_dir
from gla_exp.exp_dataset import HiddenStateDataset
from torch.utils.data import DataLoader

tc = TeacherConfig(
    generation_mode='continuous_h', sigma=0.3,
    n_train=512, n_test=64, seq_len=32,
    cache_path='data/smoke_test_cache'
)
cache_dir = get_cache_dir(tc)
ds = HiddenStateDataset(cache_dir, 'continuous_ar', 'test')
loader = DataLoader(ds, batch_size=64)

# Oracle: predicts the NORMALIZED input as the target (best deterministic predictor minus noise)
# True lower bound = sigma^2 = 0.09 per dim for positions 1..L-1
# Position 0: lower bound ≈ 1.0 (predicting random h_0 from nothing)
# Avg lower bound ≈ (1.0 + 31 * 0.09) / 32 ≈ 0.118
sigma = 0.3
L = 32
theoretical_lb = (1.0 + (L - 1) * sigma**2) / L
print(f'Theoretical lower bound: {theoretical_lb:.4f}')

# Compute mean predictor loss (global mean of targets predicts each target)
all_tgt = []
for batch in loader:
    all_tgt.append(batch['target'])
all_tgt = torch.cat(all_tgt, dim=0)  # [N_test, L, D]
global_mean = all_tgt.mean(dim=(0, 1), keepdim=True)  # [1, 1, D]
mean_loss = F.mse_loss(global_mean.expand_as(all_tgt), all_tgt).item()
print(f'Mean predictor loss: {mean_loss:.4f}')
print(f'Expected mean predictor loss ≈ 1.0 + sigma^2 = {1.0 + sigma**2:.4f} (per dim)')
"
```
Expected:
- `Theoretical lower bound: 0.1184`
- `Mean predictor loss ≈ 1.09` (global mean is 0, so per-dim MSE ≈ Var(h_t) ≈ 1 + σ² = 1.09)

**Step 2: No commit needed** — this is analysis only.

---

## Summary of Changes

| File | Change |
|---|---|
| `gla_exp/exp_config.py` | +2 fields (`generation_mode`, `sigma`) in `TeacherConfig`; cache key updated |
| `gla_exp/teachers.py` | +`ContinuousHSpaceTeacher` class with `generate_sequence()` |
| `gla_exp/generate_data.py` | +`_generate_continuous_h`, +`_generate_continuous_h_samples`, rename old body to `_generate_wikitext`, dispatch in `generate_and_cache` |
| `gla_exp/exp_dataset.py` | +`continuous_ar` mode in `HiddenStateDataset` |
| `gla_exp/train.py` | `StudentWithProjection.forward` +optional `target` param; training loop + `evaluate` handle `batch["target"]` |
| `gla_exp/configs/exp004_continuous_h_ar.yaml` | New config file |
| **`train.py` (main logic)** | **No structural change** — backward compatible |

## Expected Final Results

After full 50-epoch training on 100k samples:

| Metric | Expected value |
|---|---|
| Theoretical lower bound | ≈ 0.118 |
| Student train loss (converged) | → 0.118–0.15 |
| Student test loss | ≈ train loss (generative data, no overfitting) |
| Compare: wikitext exp001 test loss | 1.794 |
| Compare: wikitext nominal lower bound | 0.09 (not achievable) |
