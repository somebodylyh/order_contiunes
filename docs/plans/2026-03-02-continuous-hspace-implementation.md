# Continuous h-space AR Teacher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 用 GLA 4层预训练权重在 h-space 自回归生成序列，数据直接喂给现有的 `train_ar.py` / `train_mdm.py`，理论下界 σ²=0.09 真正可达。

**Architecture:** 写一个新的数据生成脚本 `gla_exp/generate_hspace_memmap.py`，调用 `ContinuousHSpaceTeacher` 生成序列并保存为 `baseline_continuous/disk_dataset.py` 期望的 memmap 格式。更新 `config.py` 超参数。`train_ar.py`、`train_mdm.py`、`ContinuousAOGPT`、`disk_dataset.py`（除一行死 import）**完全不动**。

**Tech Stack:** PyTorch, FLA (`fla.models.GLAForCausalLM`), numpy memmap, 现有 `baseline_continuous` 训练框架。

---

## 数学回顾

```
h_0 = Normalize(N(0,I_D)) × √D                    # init vector, num_init=1
x_{t-1} = Normalize(h_{t-1}) × √D                 # 归一化后作为 GLA inputs_embeds
μ_t = GLA_4L(inputs_embeds=[x_0,...,x_{t-1}])[:,-1,:]
ε_t ~ N(0, σ²I_D),   σ = 0.3
h_t = μ_t + ε_t                                    # t = 1..L-1
```

- `init_vectors` = `[h_0]` (shape `[N,1,D]`)，始终可见，不参与 loss
- `main_vectors` = `[h_1,...,h_{L-1}]` (shape `[N,L-1,D]`)，学生预测目标
- 每个预测位置的理论下界 = **σ² = 0.09**（精确，无额外偏置项）

---

## 数据格式（disk_dataset.py 期望）

```
{data_dir}/
  train_vectors.npy        float32 memmap  [100000, 32, 1024]   完整序列 h_0..h_31
  train_init_vectors.npy   float32 memmap  [100000,  1, 1024]   h_0
  val_vectors.npy          float32 memmap  [ 10000, 32, 1024]
  val_init_vectors.npy     float32 memmap  [ 10000,  1, 1024]
  test_vectors.npy         float32 memmap  [ 10000, 32, 1024]
  test_init_vectors.npy    float32 memmap  [ 10000,  1, 1024]
  data_config.pt           dict: {seq_length, vector_dim, num_init,
                                  train_samples, val_samples, test_samples, sigma}
```

`MemmapDataset.__getitem__` 内部做 `main_vectors = vectors[num_init:]`，所以存完整序列就行。

---

### Task 1: 在 `teachers.py` 添加 `ContinuousHSpaceTeacher`

**Files:**
- Modify: `gla_exp/teachers.py`

**Step 1: 在文件顶部补充 import**

在 `import torch.nn as nn` 后加：
```python
import math
import torch.nn.functional as F
```

**Step 2: 在 `FLATeacher` 之后添加新类**

```python
class ContinuousHSpaceTeacher(nn.Module):
    """
    连续 h-space AR Teacher：用预训练 GLA 前 (layer_idx+1) 层在连续空间自回归生成序列。

    生成过程：
      h_0 = Normalize(N(0,I_D)) × √D
      x_{t-1} = Normalize(h_{t-1}) × √D      (归一化后作为 inputs_embeds 输入)
      μ_t = GLA_4L(inputs_embeds=[x_0,...,x_{t-1}])[:, -1, :]
      ε_t ~ N(0, σ² I_D)
      h_t = μ_t + ε_t                         (t = 1..L-1)

    注：必须在 CUDA 上运行（FLA Triton kernel 要求）。
    """

    def __init__(self, cfg):
        super().__init__()
        from fla.models import GLAForCausalLM

        self.cfg = cfg
        print(f"[ContinuousHSpaceTeacher] Loading {cfg.model_name} ...")
        model = GLAForCausalLM.from_pretrained(cfg.model_name)

        n_keep = cfg.layer_idx + 1
        model.model.layers = nn.ModuleList(list(model.model.layers)[:n_keep])
        print(f"[ContinuousHSpaceTeacher] Truncated to {n_keep} layers "
              f"(d_model={model.config.hidden_size})")

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
        生成 B 条长度为 L 的序列。

        返回: [B, L, D]
          - 位置 0: h_0 = Normalize(N(0,I)) × √D   (norm ≡ √D)
          - 位置 t: h_t = GLA_4L(x_{0:t-1})[-1] + ε_t,  ε_t ~ N(0, σ²I)
        """
        D = self.d_hidden

        h0 = torch.randn(B, D, device=device)
        h0 = F.normalize(h0, dim=-1) * math.sqrt(D)

        seq = [h0]

        for t in range(1, L):
            x_hist = torch.stack(
                [F.normalize(h, dim=-1) * math.sqrt(D) for h in seq],
                dim=1,
            )  # [B, t, D]

            self._hidden_buf = None
            self.model(inputs_embeds=x_hist)
            assert self._hidden_buf is not None, "Hook 未触发"
            mu_t = self._hidden_buf[:, -1, :]  # [B, D]

            h_t = mu_t + torch.randn_like(mu_t) * sigma
            seq.append(h_t)

        return torch.stack(seq, dim=1)  # [B, L, D]
```

**Step 3: 验证 generate_sequence 输出形状与 h_0 范数**

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM
python -c "
import sys, torch, math
sys.path.insert(0, '.')
from gla_exp.exp_config import TeacherConfig
from gla_exp.teachers import ContinuousHSpaceTeacher
cfg = TeacherConfig()
t = ContinuousHSpaceTeacher(cfg).cuda()
seq = t.generate_sequence(B=4, L=8, sigma=0.3, device=torch.device('cuda'))
print('shape:', seq.shape)                          # [4, 8, 1024]
print('h0 norms:', seq[:,0,:].norm(dim=-1))         # all ≈ 32.0
print('h1 norms:', seq[:,1,:].norm(dim=-1))         # ≈ 32 ± small
"
```

**Step 4: Commit**
```bash
git add gla_exp/teachers.py
git commit -m "feat: add ContinuousHSpaceTeacher for h-space AR generation"
```

---

### Task 2: 写数据生成脚本 `gla_exp/generate_hspace_memmap.py`

**Files:**
- Create: `gla_exp/generate_hspace_memmap.py`

**Step 1: 写脚本**

```python
"""
generate_hspace_memmap.py — 用 ContinuousHSpaceTeacher 生成 h-space AR 序列，
保存为 baseline_continuous/disk_dataset.py 期望的 numpy memmap 格式。

Usage:
    python -m gla_exp.generate_hspace_memmap \\
        --data_dir baseline_continuous/data_hspace \\
        --sigma 0.3 --seq_len 32 \\
        --n_train 100000 --n_val 10000 --n_test 10000 \\
        --batch_size 64
"""
import sys, os, argparse, math
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gla_exp.exp_config import TeacherConfig
from gla_exp.teachers import ContinuousHSpaceTeacher


def fill_split(teacher, n_samples, seq_len, sigma, batch_size, device,
               vec_mmap, init_mmap, split_name):
    """批量生成序列，写入预分配的 memmap。"""
    n_batches = math.ceil(n_samples / batch_size)
    print(f"[{split_name}] {n_samples} samples × L={seq_len}  ({n_batches} batches)")
    offset = 0
    for i in range(n_batches):
        B   = min(batch_size, n_samples - offset)
        seq = teacher.generate_sequence(B=B, L=seq_len, sigma=sigma, device=device)
        # seq: [B, L, D]  (float32 on GPU)
        arr = seq.cpu().numpy()          # [B, L, D]
        vec_mmap [offset:offset + B] = arr          # 完整序列
        init_mmap[offset:offset + B] = arr[:, :1]   # h_0 as init
        offset += B
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_batches}", flush=True)
    vec_mmap.flush()
    init_mmap.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True,
                        help="输出目录，例如 baseline_continuous/data_hspace")
    parser.add_argument("--model_name", default="fla-hub/gla-340M-15B")
    parser.add_argument("--layer_idx",  type=int,   default=3)
    parser.add_argument("--sigma",      type=float, default=0.3)
    parser.add_argument("--seq_len",    type=int,   default=32)
    parser.add_argument("--n_train",    type=int,   default=100000)
    parser.add_argument("--n_val",      type=int,   default=10000)
    parser.add_argument("--n_test",     type=int,   default=10000)
    parser.add_argument("--batch_size", type=int,   default=64)
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = TeacherConfig(model_name=args.model_name, layer_idx=args.layer_idx)
    teacher = ContinuousHSpaceTeacher(cfg).to(device)
    D   = teacher.d_hidden
    L   = args.seq_len
    sig = args.sigma

    print(f"[config] D={D}, L={L}, sigma={sig}")
    print(f"[config] train={args.n_train}, val={args.n_val}, test={args.n_test}")

    splits = [
        ("train", args.n_train),
        ("val",   args.n_val),
        ("test",  args.n_test),
    ]

    for split, n in splits:
        vec_path  = os.path.join(args.data_dir, f"{split}_vectors.npy")
        init_path = os.path.join(args.data_dir, f"{split}_init_vectors.npy")
        vec_mmap  = np.memmap(vec_path,  dtype="float32", mode="w+", shape=(n, L, D))
        init_mmap = np.memmap(init_path, dtype="float32", mode="w+", shape=(n, 1, D))
        fill_split(teacher, n, L, sig, args.batch_size, device,
                   vec_mmap, init_mmap, split)
        print(f"[{split}] saved → {vec_path}")

    # 保存 data_config.pt（disk_dataset.py 需要）
    config = {
        "seq_length":    L,
        "vector_dim":    D,
        "num_init":      1,
        "train_samples": args.n_train,
        "val_samples":   args.n_val,
        "test_samples":  args.n_test,
        "sigma":         sig,
        "model_name":    args.model_name,
        "layer_idx":     args.layer_idx,
    }
    torch.save(config, os.path.join(args.data_dir, "data_config.pt"))
    print(f"[done] data_config.pt saved → {args.data_dir}")


if __name__ == "__main__":
    main()
```

**Step 2: 小规模 smoke test**

```bash
python -m gla_exp.generate_hspace_memmap \
    --data_dir /tmp/hspace_smoke \
    --n_train 128 --n_val 32 --n_test 32 \
    --seq_len 8 --batch_size 32
```

验证输出：
```bash
python -c "
import numpy as np, torch, os
d = '/tmp/hspace_smoke'
v  = np.load(os.path.join(d, 'train_vectors.npy'),      mmap_mode='r')
iv = np.load(os.path.join(d, 'train_init_vectors.npy'), mmap_mode='r')
cfg = torch.load(os.path.join(d, 'data_config.pt'), weights_only=False)
print('vectors shape:',      v.shape)   # (128, 8, 1024)
print('init_vectors shape:', iv.shape)  # (128, 1, 1024)
print('h0 norms:', (v[:,0,:]**2).sum(-1)**.5[:4])  # ≈ 32
print('config:', cfg)
"
```

**Step 3: Commit**
```bash
git add gla_exp/generate_hspace_memmap.py
git commit -m "feat: add generate_hspace_memmap.py for disk-compatible h-space data"
```

---

### Task 3: 更新 `baseline_continuous/config.py`

**Files:**
- Modify: `baseline_continuous/config.py`

**Step 1: 替换配置内容**

将整个文件替换为：

```python
# === Data (GLA h-space AR, D=1024) ===
vector_dim = 1024
seq_length = 32
num_init   = 1           # h_0 作为始终可见的 init prefix，不参与 loss
num_chunks = 31          # = seq_length - num_init，MDM token-level shuffle
sigma      = 0.3         # Teacher 生成时的噪声，理论下界 = sigma^2 = 0.09

# 以下字段保留占位（disk_dataset 不使用，train 脚本中通过 cfg.xxx 引用）
dependency_window = 1
num_matrices      = 1
train_init_mode   = 'random'
val_init_mode     = 'random'
noise_scale       = 0.0
alpha             = 0.0
train_samples     = 100000
val_samples       = 10000
test_samples      = 10000

# === Model ===
n_layer    = 5
n_head     = 4
n_embd     = 256
block_size = 32
dropout    = 0.0
bias       = True

# === Training ===
batch_size    = 256
learning_rate = 3e-4
epochs        = 50
warmup_iters  = 0.05     # 5% of total steps
weight_decay  = 0.1
grad_clip     = 1.0
seed          = 42
device        = 'cuda'
num_workers   = 4

# === Logging ===
log_interval   = 100
eval_interval  = 500
save_best_model = True
wandb_log      = True
wandb_project  = 'ao-gpt-mdm-hspace'
```

**Step 2: 验证 config 能被正常 import**

```bash
python -c "
from baseline_continuous import config as cfg
print('vector_dim:', cfg.vector_dim)   # 1024
print('num_init:',   cfg.num_init)     # 1
print('n_embd:',     cfg.n_embd)       # 256
"
```

**Step 3: Commit**
```bash
git add baseline_continuous/config.py
git commit -m "config: update to GLA h-space AR experiment (D=1024, num_init=1)"
```

---

### Task 4: 移除 `disk_dataset.py` 的死 import

**Files:**
- Modify: `baseline_continuous/disk_dataset.py`

**Step 1: 删除第4行死 import**

删除：
```python
from linear.continuous_data_generator import ContinuousDenseARGenerator
```

**Step 2: 验证 import 正常**

```bash
python -c "from baseline_continuous.disk_dataset import create_disk_dataloaders; print('OK')"
```

**Step 3: Commit**
```bash
git add baseline_continuous/disk_dataset.py
git commit -m "fix: remove unused import in disk_dataset.py"
```

---

### Task 5: 端到端 smoke test

**Step 1: 生成正式规模数据**（耗时较长，可后台运行）

```bash
python -m gla_exp.generate_hspace_memmap \
    --data_dir baseline_continuous/data_hspace \
    --n_train 100000 --n_val 10000 --n_test 10000 \
    --seq_len 32 --batch_size 64
```

**Step 2: 验证 disk_dataset 能读取**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from baseline_continuous.disk_dataset import create_disk_dataloaders
from baseline_continuous import config as cfg
train_loader, val_loader, test_loader = create_disk_dataloaders(
    data_dir='baseline_continuous/data_hspace',
    batch_size=4, num_workers=0, num_chunks=cfg.num_chunks,
)
batch = next(iter(train_loader))
print('init_vectors:', batch['init_vectors'].shape)   # [4, 1, 1024]
print('main_vectors:', batch['main_vectors'].shape)   # [4, 31, 1024]
print('shuffled_main:', batch['shuffled_main'].shape) # [4, 31, 1024]
"
```

**Step 3: AR training smoke test（2 步）**

```bash
python baseline_continuous/train_ar.py \
    --epochs 1 \
    --batch_size 16 \
    --data_dir baseline_continuous/data_hspace \
    --no_shuffle
```

Expected: 打印 loss（数量级应在 0.5–5.0 之间），无报错。

**Step 4: MDM training smoke test（2 步）**

```bash
python baseline_continuous/train_mdm.py \
    --epochs 1 \
    --batch_size 16 \
    --data_dir baseline_continuous/data_hspace
```

Expected: 正常运行，无报错。

---

## 改动汇总

| 文件 | 改动 |
|---|---|
| `gla_exp/teachers.py` | + `ContinuousHSpaceTeacher` 类 |
| `gla_exp/generate_hspace_memmap.py` | **新建**：生成 memmap 格式数据 |
| `baseline_continuous/config.py` | 更新超参数（D=1024, num_init=1, ...）|
| `baseline_continuous/disk_dataset.py` | 删除一行死 import |
| `train_ar.py` / `train_mdm.py` / `ContinuousAOGPT` | **不动** |

## 预期结果

| 指标 | 预期值 |
|---|---|
| 理论下界（所有预测位置） | σ² = **0.09** |
| 均值预测器 loss | ≈ 1 + σ² ≈ 1.09 |
| 训练收敛 loss | → 0.09–0.12 |
| wikitext exp001 对比 | 1.794 |
| 运行命令 | `python baseline_continuous/train_ar.py --data_dir baseline_continuous/data_hspace --no_shuffle` |
