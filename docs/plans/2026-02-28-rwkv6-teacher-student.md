# RWKV6 Teacher-Student 实验框架实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 以冻结的 RWKV6 预训练模型为 Teacher，提取中间层 hidden states，分别用 AR（原始顺序）和 MDM（打乱顺序）两种 Student 拟合，对比 loss 差异，验证因果顺序对学习的影响。

**Architecture:**
- Teacher：`fla-hub/rwkv6-169M`，`register_forward_hook` 提取第 6 层 hidden state，全程 `no_grad`
- Data：wikitext 文本 → tokenize → RWKV6 forward → hidden `[N, L, D]` 缓存到磁盘
- Student：`ContinuousAOGPT`（已有），通过 `nn.Linear` projection 对齐 Teacher 维度
- 三组对比实验：`ar_correct` / `mdm_shuffled` / `mdm_correct`

**Tech Stack:** PyTorch, flash-linear-attention (fla), HuggingFace datasets/transformers, 现有 ContinuousAOGPT

**不修改任何现有文件。**

---

## 接口约定（跨 Agent 共享）

### exp_config.py 导出
```python
@dataclass
class TeacherConfig:
    model_name: str        # "fla-hub/rwkv6-169M"
    layer_idx: int         # 提取第几层 hidden state（从 0 起）
    seq_len: int           # 截取 token 数
    batch_size: int        # Teacher forward 的 batch size
    n_batches: int         # 总共 forward 多少批
    dataset_name: str      # "wikitext"
    cache_path: str        # 缓存根目录

@dataclass
class StudentConfig:
    type: str              # "ar_correct" | "mdm_shuffled" | "mdm_correct"
    d_model: int           # Student 自身维度
    n_layers: int
    n_heads: int
    mask_ratio: float      # 仅 mdm* 使用，默认 0.15

@dataclass
class TrainingConfig:
    lr: float
    epochs: int
    batch_size: int
    log_interval: int      # 每 N epoch 记录一次
    val_ratio: float       # 验证集比例

def load_config(yaml_path: str) -> tuple[TeacherConfig, StudentConfig, TrainingConfig]: ...
def teacher_cache_key(cfg: TeacherConfig) -> str: ...  # 8 位 MD5，key = model_name+layer_idx+seq_len+n_batches+dataset_name
```

### teachers.py 导出
```python
class RWKV6Teacher(nn.Module):
    d_hidden: int                          # Teacher 的 hidden_size
    def __init__(self, cfg: TeacherConfig): ...
    def extract(self, input_ids: Tensor) -> dict:
        # 返回:
        # "hidden":   [B, L, D]  原始顺序 hidden state
        # "order":    [B, L]     arange(L)，GT label
        # "shuffled": [B, L, D]  每条序列独立随机打乱
        # "perm":     [B, L]     shuffled[b] = hidden[b][perm[b]]
```

### 磁盘数据格式（generate_data.py 写 / exp_dataset.py 读）
```
{cache_dir}/
  hidden.pt     # torch.Tensor [N, L, D] float32
  perm.pt       # torch.Tensor [N, L]   int64，每条序列的打乱置换
  meta.json     # {"D": int, "L": int, "N": int, "model_name": str, "layer_idx": int}
```

---

## Task 1 — YAML Config 系统

**Files:**
- Create: `baseline_continuous/exp_config.py`
- Create: `configs/exp001_rwkv6_ar_correct.yaml`
- Create: `configs/exp002_rwkv6_mdm_shuffled.yaml`
- Create: `configs/exp003_rwkv6_mdm_correct.yaml`

### exp_config.py 完整实现

```python
"""YAML config for RWKV6 Teacher-Student experiments."""
import hashlib, json
from dataclasses import dataclass, field, asdict
import yaml


@dataclass
class TeacherConfig:
    model_name: str = "fla-hub/rwkv6-169M"
    layer_idx: int = 6
    seq_len: int = 32
    batch_size: int = 8
    n_batches: int = 100
    dataset_name: str = "wikitext"
    cache_path: str = "data/teacher_cache"


@dataclass
class StudentConfig:
    type: str = "ar_correct"      # ar_correct | mdm_shuffled | mdm_correct
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    mask_ratio: float = 0.15


@dataclass
class TrainingConfig:
    lr: float = 3e-4
    epochs: int = 100
    batch_size: int = 64
    log_interval: int = 5
    val_ratio: float = 0.1


def load_config(yaml_path: str):
    with open(yaml_path) as f:
        d = yaml.safe_load(f)
    teacher  = TeacherConfig(**d.get("teacher", {}))
    student  = StudentConfig(**d.get("student", {}))
    training = TrainingConfig(**d.get("training", {}))
    return teacher, student, training


def teacher_cache_key(cfg: TeacherConfig) -> str:
    """8-char MD5 over the fields that define the cached data."""
    key_fields = {
        "model_name":   cfg.model_name,
        "layer_idx":    cfg.layer_idx,
        "seq_len":      cfg.seq_len,
        "n_batches":    cfg.n_batches,
        "dataset_name": cfg.dataset_name,
    }
    return hashlib.md5(
        json.dumps(key_fields, sort_keys=True).encode()
    ).hexdigest()[:8]
```

### YAML configs（三个共享 teacher，不同 student）

**exp001_rwkv6_ar_correct.yaml:**
```yaml
teacher:
  model_name: "fla-hub/rwkv6-169M"
  layer_idx: 6
  seq_len: 32
  batch_size: 8
  n_batches: 100
  dataset_name: "wikitext"
  cache_path: "data/teacher_cache"

student:
  type: ar_correct
  d_model: 256
  n_layers: 4
  n_heads: 4
  mask_ratio: 0.0

training:
  lr: 3.0e-4
  epochs: 100
  batch_size: 64
  log_interval: 5
  val_ratio: 0.1
```

**exp002_rwkv6_mdm_shuffled.yaml:** teacher 完全相同，student.type = `mdm_shuffled`，mask_ratio = 0.15

**exp003_rwkv6_mdm_correct.yaml:** teacher 完全相同，student.type = `mdm_correct`，mask_ratio = 0.15

### 验证
```bash
python -c "
import sys; sys.path.insert(0,'.')
from baseline_continuous.exp_config import load_config, teacher_cache_key
t, s, tr = load_config('configs/exp001_rwkv6_ar_correct.yaml')
print('teacher:', t.model_name, 'layer:', t.layer_idx)
print('student type:', s.type)
print('cache key:', teacher_cache_key(t))
t2, _, _ = load_config('configs/exp002_rwkv6_mdm_shuffled.yaml')
assert teacher_cache_key(t) == teacher_cache_key(t2), 'exp001 & exp002 must share cache'
print('Cache sharing: OK')
"
```

### Commit
```bash
git add baseline_continuous/exp_config.py configs/
git commit -m "feat: add YAML config system (TeacherConfig/StudentConfig/TrainingConfig)"
```

---

## Task 2 — Teacher 数据提取

**Files:**
- Create: `baseline_continuous/teachers.py`

### 完整实现

```python
"""
RWKV6 Teacher: 加载预训练模型，register_forward_hook 提取中间层 hidden state。

使用方式:
    teacher = RWKV6Teacher(cfg)
    result  = teacher.extract(input_ids)   # input_ids: [B, L]
    # result['hidden']:   [B, L, D]  原始顺序
    # result['shuffled']: [B, L, D]  随机打乱
    # result['perm']:     [B, L]     打乱置换
    # result['order']:    [B, L]     arange(L)，GT label

特性:
  - 全部参数 requires_grad=False
  - extract() 全程 torch.no_grad()
  - hook 取 output[0]（RWKV6 每层返回 tuple，首元素为 hidden state）
"""
import torch
import torch.nn as nn
from baseline_continuous.exp_config import TeacherConfig


class RWKV6Teacher(nn.Module):
    def __init__(self, cfg: TeacherConfig):
        super().__init__()
        from fla.models import RWKV6ForCausalLM
        self.cfg    = cfg
        self.model  = RWKV6ForCausalLM.from_pretrained(cfg.model_name)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.d_hidden: int = self.model.config.hidden_size
        self._hidden_buf  = None

        # Hook: 提取第 cfg.layer_idx 层的 hidden state
        def _hook(module, inp, output):
            # RWKV6 block 返回 tuple，第 0 个是 hidden state [B, L, D]
            self._hidden_buf = output[0].detach()

        self.model.model.layers[cfg.layer_idx].register_forward_hook(_hook)

    @torch.no_grad()
    def extract(self, input_ids: torch.Tensor) -> dict:
        """
        input_ids: [B, L]（来自 tokenizer）
        返回:
          hidden:   [B, L, D]  原始顺序 hidden state
          order:    [B, L]     arange(L)，因果顺序 GT label
          shuffled: [B, L, D]  每条序列独立随机打乱
          perm:     [B, L]     打乱置换 index
        """
        B, L = input_ids.shape
        self._hidden_buf = None
        self.model(input_ids)                     # trigger hook
        hidden = self._hidden_buf                 # [B, L, D]
        assert hidden is not None, "Hook did not fire — check layer_idx"

        # 每条序列独立采样一个随机置换
        perms = torch.stack([torch.randperm(L, device=hidden.device)
                             for _ in range(B)])  # [B, L]
        shuffled = hidden[torch.arange(B, device=hidden.device).unsqueeze(1), perms]
        order    = torch.arange(L, device=hidden.device).unsqueeze(0).expand(B, -1)

        return {
            "hidden":   hidden,
            "order":    order,
            "shuffled": shuffled,
            "perm":     perms,
        }
```

### 验证
```bash
python -c "
import sys; sys.path.insert(0,'.')
import torch
from baseline_continuous.exp_config import TeacherConfig
from baseline_continuous.teachers import RWKV6Teacher
cfg = TeacherConfig()
t   = RWKV6Teacher(cfg).cuda()
print('d_hidden:', t.d_hidden)
ids = torch.randint(0, 1000, (2, 32), device='cuda')
r   = t.extract(ids)
print('hidden shape:', r['hidden'].shape)    # [2, 32, D]
print('perm   shape:', r['perm'].shape)      # [2, 32]
assert not r['hidden'].requires_grad
print('OK')
"
```

### Commit
```bash
git add baseline_continuous/teachers.py
git commit -m "feat: add RWKV6Teacher with forward_hook hidden state extraction"
```

---

## Task 3 — 数据生成、Dataset、训练入口

### Task 3a: generate_data.py

**Files:**
- Create: `baseline_continuous/generate_data.py`

```python
"""
generate_data.py — 用 RWKV6Teacher 提取 wikitext hidden states，缓存到磁盘。

缓存结构（{cache_path}/{hash}/）:
  hidden.pt   Tensor [N, L, D]
  perm.pt     Tensor [N, L]
  meta.json   {"D":..., "L":..., "N":..., "model_name":..., "layer_idx":...}

Usage:
  python -m baseline_continuous.generate_data --config configs/exp001_rwkv6_ar_correct.yaml
  python -m baseline_continuous.generate_data --config configs/exp001_rwkv6_ar_correct.yaml --force
"""
import sys, os, json, argparse
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from baseline_continuous.exp_config import load_config, teacher_cache_key, TeacherConfig
from baseline_continuous.teachers import RWKV6Teacher


def get_cache_dir(cfg: TeacherConfig) -> str:
    return os.path.join(cfg.cache_path, teacher_cache_key(cfg))


def cache_exists(cache_dir: str) -> bool:
    return all(os.path.exists(os.path.join(cache_dir, f))
               for f in ["hidden.pt", "perm.pt", "meta.json"])


def generate_and_cache(teacher_cfg: TeacherConfig, force: bool = False) -> str:
    cache_dir = get_cache_dir(teacher_cfg)
    if not force and cache_exists(cache_dir):
        print(f"[cache] Found: {cache_dir}")
        return cache_dir

    os.makedirs(cache_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[generate] Loading teacher: {teacher_cfg.model_name}")
    teacher = RWKV6Teacher(teacher_cfg).to(device)
    D = teacher.d_hidden

    from transformers import AutoTokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(teacher_cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[generate] Loading dataset: {teacher_cfg.dataset_name}")
    ds = load_dataset(teacher_cfg.dataset_name, "wikitext-2-raw-v1", split="train")
    text = "\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]

    L, B = teacher_cfg.seq_len, teacher_cfg.batch_size
    n_total = teacher_cfg.n_batches * B
    max_start = len(tokens) - L
    if max_start <= 0:
        raise ValueError(f"Text too short: {len(tokens)} tokens < seq_len={L}")

    all_hidden, all_perm = [], []
    print(f"[generate] Extracting {n_total} samples ({teacher_cfg.n_batches} batches × {B})...")
    for i in range(teacher_cfg.n_batches):
        starts     = torch.randint(0, max_start, (B,))
        input_ids  = torch.stack([tokens[s:s+L] for s in starts]).to(device)  # [B, L]
        result     = teacher.extract(input_ids)
        all_hidden.append(result["hidden"].cpu())
        all_perm.append(result["perm"].cpu())
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{teacher_cfg.n_batches}")

    hidden_all = torch.cat(all_hidden, dim=0)  # [N, L, D]
    perm_all   = torch.cat(all_perm,   dim=0)  # [N, L]
    N          = hidden_all.shape[0]

    torch.save(hidden_all, os.path.join(cache_dir, "hidden.pt"))
    torch.save(perm_all,   os.path.join(cache_dir, "perm.pt"))
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump({"D": D, "L": L, "N": N,
                   "model_name": teacher_cfg.model_name,
                   "layer_idx":  teacher_cfg.layer_idx}, f, indent=2)

    # ── 数据质量报告 ──────────────────────────────────────────────────────
    import torch.nn.functional as F
    sample = hidden_all[:200]                       # [200, L, D]
    sample_n = F.normalize(sample, dim=-1)
    # within: 同序列相邻 token 的 cos_sim
    cos_within  = F.cosine_similarity(sample_n[:, :-1], sample_n[:, 1:], dim=-1).mean()
    # between: 随机跨序列 token 的 cos_sim
    flat        = sample_n.view(-1, D)
    idx_a       = torch.randint(0, flat.shape[0], (1000,))
    idx_b       = torch.randint(0, flat.shape[0], (1000,))
    cos_between = F.cosine_similarity(flat[idx_a], flat[idx_b], dim=-1).mean()
    print(f"\n[quality] within-seq cos_sim : {cos_within.item():.4f}  (collapse if > 0.99)")
    print(f"[quality] between-seq cos_sim: {cos_between.item():.4f}")
    print(f"[quality] Saved {N} samples, shape [N={N}, L={L}, D={D}]")

    return cache_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    t_cfg, _, _ = load_config(args.config)
    generate_and_cache(t_cfg, force=args.force)
```

### Task 3b: exp_dataset.py

**Files:**
- Create: `baseline_continuous/exp_dataset.py`

```python
"""
exp_dataset.py — 从缓存的 hidden.pt/perm.pt 构建 Dataset。

student_type == "ar_correct":
    input  = hidden[idx]          [L, D]  原始因果顺序
    target = hidden[idx]          [L, D]  同上（collate_fn 里 shift）

student_type == "mdm_shuffled" | "mdm_correct":
    "mdm_shuffled":
        input  = hidden[idx][perm[idx]]   [L, D]  打乱顺序
        target = hidden[idx][perm[idx]]   [L, D]
        perm   = perm[idx]                [L]
    "mdm_correct":
        input  = hidden[idx]              [L, D]  原始顺序 + MDM mask
        target = hidden[idx]              [L, D]
        perm   = arange(L)               [L]  （dummy，保持接口一致）

perm 在此处固定（从缓存读取），不在 Dataset 层重新采样。
若需每 epoch 重新打乱，在 DataLoader 的 collate_fn 中操作。
"""
import os, json
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class HiddenStateDataset(Dataset):
    def __init__(self, cache_dir: str, student_type: str,
                 split: str = "train", val_ratio: float = 0.1):
        assert os.path.exists(os.path.join(cache_dir, "hidden.pt")), \
            f"Cache not found: {cache_dir}. Run generate_data.py first."
        self.hidden = torch.load(os.path.join(cache_dir, "hidden.pt"),
                                 mmap=True, weights_only=True)   # [N, L, D]
        self.perm   = torch.load(os.path.join(cache_dir, "perm.pt"),
                                 mmap=True, weights_only=True)   # [N, L]
        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)

        self.student_type = student_type
        N = self.hidden.shape[0]

        # train/val split（按 val_ratio）
        n_val   = max(1, int(N * val_ratio))
        n_train = N - n_val
        idx = torch.arange(N)
        if split == "train":
            self._idx = idx[:n_train]
        else:
            self._idx = idx[n_train:]

    def __len__(self):
        return len(self._idx)

    def __getitem__(self, i):
        idx    = self._idx[i].item()
        hidden = self.hidden[idx]   # [L, D]
        perm   = self.perm[idx]     # [L]
        L      = hidden.shape[0]

        if self.student_type == "ar_correct":
            return {"input": hidden.clone(), "target": hidden.clone()}

        elif self.student_type == "mdm_shuffled":
            shuffled = hidden[perm]
            return {"input": shuffled.clone(), "target": shuffled.clone(), "perm": perm.clone()}

        elif self.student_type == "mdm_correct":
            return {"input": hidden.clone(), "target": hidden.clone(),
                    "perm": torch.arange(L)}

        else:
            raise ValueError(f"Unknown student_type: {self.student_type}")


def create_dataloaders(cache_dir, student_type, batch_size,
                       val_ratio=0.1, num_workers=4):
    train_ds = HiddenStateDataset(cache_dir, student_type, "train", val_ratio)
    val_ds   = HiddenStateDataset(cache_dir, student_type, "val",   val_ratio)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    print(f"[dataset] {student_type}: train={len(train_ds)}, val={len(val_ds)}")
    return train_loader, val_loader
```

### Task 3c: train.py

**Files:**
- Create: `train.py`

```python
"""
train.py — 统一训练入口。

Usage:
    python train.py --config configs/exp001_rwkv6_ar_correct.yaml
    python train.py --config configs/exp001_rwkv6_ar_correct.yaml --force_regen

流程:
  1. 解析 config → TeacherConfig, StudentConfig, TrainingConfig
  2. 检测缓存 → 不存在则调用 generate_and_cache()
  3. 构建 HiddenStateDataset (train + val)
  4. 构建 Student: ContinuousAOGPT + 输入/输出 projection（D→d_model→D）
  5. 训练循环
  6. 每 log_interval epoch 记录 train_loss, val_loss,
     cos_sim_within, cos_sim_between
  7. 保存指标到 runs/{config_stem}/metrics.jsonl
"""
import sys, os, math, copy, json, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_continuous.exp_config import load_config, teacher_cache_key
from baseline_continuous.generate_data import get_cache_dir, cache_exists, generate_and_cache
from baseline_continuous.exp_dataset import create_dataloaders
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig


# ─── Student 封装（含 projection）────────────────────────────────────────────

class StudentWithProjection(nn.Module):
    """ContinuousAOGPT + 输入 projection（D→d_model）+ 输出 projection（d_model→D）。
    若 D == d_model 则 projection 为 identity。"""

    def __init__(self, teacher_D: int, sc, seq_len: int):
        super().__init__()
        d = sc.d_model
        self.in_proj  = nn.Linear(teacher_D, d, bias=False) if d != teacher_D else nn.Identity()
        self.out_proj = nn.Linear(d, teacher_D, bias=False) if d != teacher_D else nn.Identity()
        gpt_cfg = ContinuousAOGPTConfig(
            block_size = seq_len,
            vector_dim = d,
            n_layer    = sc.n_layers,
            n_head     = sc.n_heads,
            dropout    = 0.0,
            bias       = True,
            num_init   = 0,
        )
        self.gpt = ContinuousAOGPT(gpt_cfg)

    def forward(self, x, mode='AR'):
        """x: [B, L, D] → preds [B, L+1, D], loss scalar"""
        xp      = self.in_proj(x)             # [B, L, d]
        preds_p, loss = self.gpt(xp, mode=mode)
        preds   = self.out_proj(preds_p)      # [B, L+1, D]
        # 重新算 MSE loss（原始 D 空间）
        targets = x
        shift   = preds[:, :-1, :]           # [B, L, D]
        loss    = F.mse_loss(shift, targets)
        return preds, loss


# ─── 评估 ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, sc):
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x = batch["input"].to(device)
        mode = "AR" if sc.type == "ar_correct" else "Random"
        if sc.type in ("mdm_shuffled", "mdm_correct"):
            # MDM: mask mask_ratio positions, predict masked
            mask     = torch.rand(x.shape[:2], device=device) < sc.mask_ratio
            x_masked = x.clone()
            x_masked[mask] = 0.0
            preds, _ = model(x_masked, mode=mode)
            shift    = preds[:, :-1, :]
            loss     = F.mse_loss(shift[mask], x[mask])
        else:
            _, loss  = model(x, mode=mode)
        total += loss.item(); n += 1
    model.train()
    return total / max(n, 1)


@torch.no_grad()
def compute_cos_metrics(loader, device, n_batches=5):
    """cos_sim_within（同序列相邻 token）和 cos_sim_between（跨序列随机 token pair）。"""
    all_vecs = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        all_vecs.append(batch["input"].to(device))   # [B, L, D]
    vecs_cat = torch.cat(all_vecs, dim=0)             # [N, L, D]
    vecs_n   = F.normalize(vecs_cat, dim=-1)

    # within
    cos_w = F.cosine_similarity(vecs_n[:, :-1], vecs_n[:, 1:], dim=-1).mean().item()

    # between
    flat  = vecs_n.view(-1, vecs_n.shape[-1])
    idx_a = torch.randint(0, flat.shape[0], (1000,), device=device)
    idx_b = torch.randint(0, flat.shape[0], (1000,), device=device)
    cos_b = F.cosine_similarity(flat[idx_a], flat[idx_b], dim=-1).mean().item()
    return cos_w, cos_b


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force_regen", action="store_true",
                        help="Force re-generate teacher data")
    args = parser.parse_args()

    tc, sc, tr = load_config(args.config)
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    config_stem = os.path.splitext(os.path.basename(args.config))[0]

    print("=" * 60)
    print(f"Teacher : {tc.model_name}  layer={tc.layer_idx}  L={tc.seq_len}")
    print(f"Student : {sc.type}  d_model={sc.d_model}  layers={sc.n_layers}")
    print(f"Training: epochs={tr.epochs}  lr={tr.lr}  bs={tr.batch_size}")
    print("=" * 60)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    cache_dir = get_cache_dir(tc)
    if args.force_regen or not cache_exists(cache_dir):
        generate_and_cache(tc, force=args.force_regen)

    import json as _json
    with open(os.path.join(cache_dir, "meta.json")) as f:
        meta = _json.load(f)
    D = meta["D"]

    train_loader, val_loader = create_dataloaders(
        cache_dir, sc.type, tr.batch_size, tr.val_ratio)

    # ── 2. Model ─────────────────────────────────────────────────────────────
    torch.manual_seed(42)
    model = StudentWithProjection(D, sc, tc.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr.lr,
                                  weight_decay=0.1, betas=(0.9, 0.95))

    # ── 3. Training ──────────────────────────────────────────────────────────
    os.makedirs(f"runs/{config_stem}", exist_ok=True)
    metrics_path = f"runs/{config_stem}/metrics.jsonl"
    train_mode   = "AR" if sc.type == "ar_correct" else "Random"

    print(f"\n[train] Starting {tr.epochs} epochs...")
    for epoch in range(1, tr.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0

        for batch in train_loader:
            x    = batch["input"].to(device)          # [B, L, D]
            optimizer.zero_grad(set_to_none=True)

            if sc.type in ("mdm_shuffled", "mdm_correct"):
                mask     = torch.rand(x.shape[:2], device=device) < sc.mask_ratio
                x_masked = x.clone()
                x_masked[mask] = 0.0
                preds, _ = model(x_masked, mode=train_mode)
                shift    = preds[:, :-1, :]
                loss     = F.mse_loss(shift[mask], x[mask])
            else:
                _, loss  = model(x, mode=train_mode)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item(); n_batches += 1

        train_loss = total_loss / max(n_batches, 1)

        if epoch % tr.log_interval == 0:
            val_loss = evaluate(model, val_loader, device, sc)
            cos_w, cos_b = compute_cos_metrics(val_loader, device)
            print(f"epoch {epoch:>4d}/{tr.epochs} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | "
                  f"cos_within={cos_w:.4f} | cos_between={cos_b:.4f}")

            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "train_loss": train_loss,
                    "val_loss": val_loss, "cos_within": cos_w,
                    "cos_between": cos_b,
                }) + "\n")
        else:
            if epoch % 10 == 0:
                print(f"epoch {epoch:>4d}/{tr.epochs} | train={train_loss:.4f}")

    print(f"\n[done] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
```

### Commit
```bash
git add baseline_continuous/generate_data.py baseline_continuous/exp_dataset.py train.py
git commit -m "feat: add generate_data, exp_dataset, train.py — RWKV6 Teacher-Student pipeline"
```

---

## 最终验证

```bash
# 快速端到端测试（用极小 n_batches）
python train.py --config configs/exp001_rwkv6_ar_correct.yaml --force_regen
```

期望：无报错，打印 epoch 进度和 cos_sim 指标，`runs/exp001_rwkv6_ar_correct/metrics.jsonl` 生成。

---

## 文件总览

| 文件 | 作用 |
|------|------|
| `baseline_continuous/exp_config.py` | YAML dataclass + cache key |
| `baseline_continuous/teachers.py` | RWKV6 hook 提取 |
| `baseline_continuous/generate_data.py` | 生成 + 缓存 hidden states |
| `baseline_continuous/exp_dataset.py` | Dataset（ar/mdm_shuffled/mdm_correct）|
| `train.py` | 统一入口，自动缓存检测 |
| `configs/exp00{1,2,3}_*.yaml` | 三组对比实验配置 |
