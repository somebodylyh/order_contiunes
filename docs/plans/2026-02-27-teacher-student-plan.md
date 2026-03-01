# Teacher-Student 连续向量学习 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 用冻结的 GPT-2 前 4 层生成结构化连续向量序列，训练 Student（5层 ContinuousAOGPT）复现序列，验证 AR vs MDM 的顺序发现能力。

**Architecture:** Teacher（GPT-2 前 4 层，冻结）自回归生成 768 维向量序列 → memmap 存盘 → Student（ContinuousAOGPT 5层）以 MSE loss 在三种训练模式（AR no-shuffle / AR shuffled / MDM）下训练 → eval_order 评测 Kendall's τ。

**Tech Stack:** PyTorch 2.4, HuggingFace transformers, numpy memmap, matplotlib

---

### Task 1: Teacher Generator

**Files:**
- Create: `baseline_continuous/teacher_generator.py`

**Step 1: 写验证脚本（先跑通小 batch）**

```python
# 在文件顶部写这个 sanity check，跑通后再继续
import torch, math
from transformers import GPT2Model

class GPT2Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        gpt2 = GPT2Model.from_pretrained('gpt2')
        self.h    = torch.nn.ModuleList(gpt2.h[:4])
        self.ln_f = gpt2.ln_f
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):           # x: [B, T, 768]
        for block in self.h:
            x = block(x)[0]
        return self.ln_f(x)         # [B, T, 768]

teacher = GPT2Teacher().cuda()
D = 768
scale = math.sqrt(D)

# 生成 2 条长度 8 的序列
B, L = 2, 8
seqs = torch.zeros(B, L, D, device='cuda')
seqs[:, 0] = torch.nn.functional.normalize(torch.randn(B, D, device='cuda'), dim=-1) * scale

for t in range(1, L):
    out = teacher(seqs[:, :t])[:, -1, :]   # [B, 768]
    noise = torch.randn_like(out) * 0.05
    seqs[:, t] = torch.nn.functional.normalize(out + noise, dim=-1) * scale

print('shape:', seqs.shape)
print('norm per step:', seqs.norm(dim=-1))    # 每行应 ≈ 27.7
```

**Step 2: 运行验证**

```bash
python -c "exec(open('baseline_continuous/teacher_generator.py').read())"
```

期望：输出 shape `[2, 8, 768]`，norm 每列 ≈ 27.7

**Step 3: 写完整 teacher_generator.py**

```python
"""
GPT-2 Teacher: 用冻结的 GPT-2 前 4 层自回归生成连续向量序列。

生成公式:
    x_0   ~ Normalize(N(0,I)) × √D
    x_t   = Normalize( GPT2_4L([x_0...x_{t-1}])[-1] + ε ) × √D
    ε     ~ N(0, noise_scale × I)
"""
import math
import torch
import torch.nn.functional as F
from transformers import GPT2Model


class GPT2Teacher(torch.nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        gpt2 = GPT2Model.from_pretrained('gpt2')
        self.h    = torch.nn.ModuleList(gpt2.h[:num_layers])
        self.ln_f = gpt2.ln_f
        for p in self.parameters():
            p.requires_grad_(False)
        self.D     = gpt2.config.n_embd    # 768
        self.scale = math.sqrt(self.D)

    @torch.no_grad()
    def forward(self, x):
        """x: [B, T, D] → [B, T, D]"""
        for block in self.h:
            x = block(x)[0]
        return self.ln_f(x)

    @torch.no_grad()
    def generate_sequence(self, length, batch_size, noise_scale=0.05,
                          init_mode='positive_first', device='cuda'):
        """
        生成 [batch_size, length, D] 的序列。
        init_mode: 'positive_first' → x_0[0]>0; 'negative_first' → x_0[0]<0
        返回 dict: vectors [B,L,D], init_vectors [B,num_init,D]
        """
        D, scale = self.D, self.scale
        seqs = torch.zeros(batch_size, length, D, device=device)

        # x_0: 归一化随机向量 × √D，满足 init_mode 约束
        x0 = F.normalize(torch.randn(batch_size, D, device=device), dim=-1) * scale
        if init_mode == 'positive_first':
            x0[:, 0] = x0[:, 0].abs()
        else:
            x0[:, 0] = -x0[:, 0].abs()
        seqs[:, 0] = x0

        for t in range(1, length):
            out   = self.forward(seqs[:, :t])[:, -1, :]        # [B, D]
            noise = torch.randn_like(out) * noise_scale
            seqs[:, t] = F.normalize(out + noise, dim=-1) * scale

        # num_init = 1（单个起点向量作为 conditioning prefix）
        num_init = 1
        return {
            'vectors':      seqs,
            'init_vectors': seqs[:, :num_init],
        }
```

**Step 4: 运行最终验证**

```bash
python -c "
import sys; sys.path.insert(0,'.')
from baseline_continuous.teacher_generator import GPT2Teacher
import torch
t = GPT2Teacher().cuda()
r = t.generate_sequence(32, batch_size=4, device='cuda')
print('vectors:', r['vectors'].shape)
print('norm:', r['vectors'].norm(dim=-1).mean().item())   # ≈ 27.7
print('OK')
"
```

**Step 5: Commit**

```bash
git add baseline_continuous/teacher_generator.py
git commit -m "add GPT-2 teacher generator (4 layers, frozen, A+ scaling)"
```

---

### Task 2: 数据质量可视化

**Files:**
- Create: `baseline_continuous/visualize_teacher.py`

**Step 1: 写可视化脚本**

```python
"""
数据质量检查：生成 1000 条序列后做 4 项检查。
通过标准（全部满足后才生成完整数据集）:
  within-sample cos_sim (归一化后) < 0.05
  between-sample cos_sim           < 0.05
  L2 norm 恒≈ √768
  attention entropy 非均匀（entropy < log(t)×0.9）
"""
import sys, math, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '.')
from baseline_continuous.teacher_generator import GPT2Teacher

device = 'cuda'
teacher = GPT2Teacher().to(device)
D, scale = 768, math.sqrt(768)

# 生成 1000 条序列
r = teacher.generate_sequence(32, batch_size=1000, device=device)
vecs = r['vectors']          # [1000, 32, 768]
main = vecs[:, 1:, :]        # [1000, 31, 768]（去掉 init）
main_n = F.normalize(main / scale, dim=-1)   # 归一化到单位球

# ── 1. L2 norm ──────────────────────────────────────────────
norms = vecs.norm(dim=-1).cpu()   # [1000, 32]
print(f"L2 norm: mean={norms.mean():.4f}, std={norms.std():.6f}  (expect {scale:.2f} ± ~0)")

# ── 2. Within-sample cos_sim ────────────────────────────────
N, T = main_n.shape[:2]
cos_w = torch.bmm(main_n, main_n.transpose(1,2))   # [N,T,T]
mask  = ~torch.eye(T, dtype=bool, device=device).unsqueeze(0)
within = cos_w[mask].mean().item()
print(f"within-sample cos_sim : {within:.4f}  (expect < 0.05)")

# ── 3. Between-sample cos_sim ───────────────────────────────
p5 = main_n[:, 5, :]   # [N, 768]
cos_b = (p5 @ p5.T)    # [N, N]
mask_b = ~torch.eye(N, dtype=bool, device=device)
between = cos_b[mask_b].mean().item()
print(f"between-sample cos_sim: {between:.4f}  (expect < 0.05)")

# ── 4. Attention entropy ────────────────────────────────────
# 用 hook 捕获第 1 层 attention weights（对 batch 前 8 条）
attn_weights = []
def hook(module, inp, out):
    # out[1] = attn_weights [B, H, T, T]
    if len(out) > 1 and out[1] is not None:
        attn_weights.append(out[1].detach().cpu())

handle = teacher.h[0].attn.register_forward_hook(hook)
with torch.no_grad():
    small = vecs[:8, :16, :]
    _ = teacher(small)
handle.remove()

if attn_weights:
    w = attn_weights[0]        # [8, 12, 16, 16]
    # entropy of last query position
    w_last = w[:, :, -1, :]    # [8, 12, 16]
    w_last = w_last.clamp(min=1e-9)
    entropy = -(w_last * w_last.log()).sum(-1).mean().item()
    uniform_entropy = math.log(16)
    print(f"attention entropy    : {entropy:.4f}  (uniform={uniform_entropy:.4f}, expect < {uniform_entropy*0.9:.4f})")

# ── 结果汇总 ──
print()
passed = within < 0.05 and between < 0.05
print("DATA QUALITY:", "PASS ✓" if passed else "FAIL ✗")
```

**Step 2: 运行检查**

```bash
python baseline_continuous/visualize_teacher.py
```

期望输出：
```
L2 norm: mean=27.7128, std=0.000000
within-sample cos_sim : 0.0xxx  (< 0.05)
between-sample cos_sim: 0.0xxx  (< 0.05)
attention entropy     : x.xxxx  (< uniform×0.9)
DATA QUALITY: PASS ✓
```

若 within-sample > 0.05，需调大 noise_scale（试 0.1）。

**Step 3: Commit**

```bash
git add baseline_continuous/visualize_teacher.py
git commit -m "add teacher data quality visualization"
```

---

### Task 3: 数据集生成

**Files:**
- Modify: `baseline_continuous/pregenerate_data.py`
- Modify: `baseline_continuous/config.py`

**Step 1: 更新 config.py**

将以下字段修改为：
```python
vector_dim  = 768
n_layer     = 5
n_head      = 12
n_embd      = 768
block_size  = 32
num_init    = 1          # teacher 只有 1 个 init 向量
num_chunks  = 31         # = seq_length - num_init
noise_scale = 0.05
alpha       = 0.0        # teacher 模式不用 alpha
train_samples = 500000
wandb_project = 'order-continuous-v9'
```

**Step 2: 更新 pregenerate_data.py**

在 `main()` 中，将 `ContinuousDenseARGenerator` 替换为 `GPT2Teacher`：

```python
# 删除:
# from linear.continuous_data_generator import ContinuousDenseARGenerator
# generator = ContinuousDenseARGenerator(...)

# 新增:
import math
from baseline_continuous.teacher_generator import GPT2Teacher
generator = GPT2Teacher().to(cfg.device)
scale = math.sqrt(cfg.vector_dim)
```

`pregenerate_split` 函数改为调用 `generator.generate_sequence`：

```python
def pregenerate_split(generator, num_samples, seq_length, init_mode,
                      seed, out_dir, split_name, chunk_size=2000, device='cuda'):
    D        = cfg.vector_dim
    num_init = cfg.num_init
    # ... memmap 创建不变 ...
    torch.manual_seed(seed)
    generated = 0
    while generated < num_samples:
        bs     = min(chunk_size, num_samples - generated)
        result = generator.generate_sequence(
            length=seq_length, batch_size=bs,
            noise_scale=cfg.noise_scale, init_mode=init_mode, device=device)
        vectors_mmap[generated:generated+bs] = result['vectors'].cpu().numpy()
        init_mmap[generated:generated+bs]    = result['init_vectors'].cpu().numpy()
        generated += bs
        print(f"  [{split_name}] {generated}/{num_samples}")
```

`get_current_data_config` 中移除 `dependency_window` / `num_matrices` / `alpha`，新增 `teacher='gpt2-4layers'`。

**Step 3: 生成数据（先用 50k 验证速度）**

```bash
python baseline_continuous/pregenerate_data.py --train_samples 50000 --val_samples 2000 --test_samples 2000 --force
```

期望：约 1-2 分钟内完成，无报错。

**Step 4: 验证生成的数据文件**

```bash
python -c "
import numpy as np, torch
vecs = np.memmap('baseline_continuous/data/train_vectors.npy', dtype='float32', mode='r', shape=(50000,32,768))
print('shape:', vecs.shape)
import torch; v = torch.from_numpy(vecs[:100].copy())
print('norm:', v.norm(dim=-1).mean().item())   # ≈ 27.7
"
```

**Step 5: 生成完整数据集**

```bash
python baseline_continuous/pregenerate_data.py --force
```

**Step 6: Commit**

```bash
git add baseline_continuous/config.py baseline_continuous/pregenerate_data.py
git commit -m "v9: update config (768-dim, 5-layer) and pregenerate with GPT-2 teacher"
```

---

### Task 4: 更新 Loss 函数

**Files:**
- Modify: `baseline_continuous/continuous_aogpt.py`

**Step 1: 定位两处 cosine loss，替换为 MSE**

在 `forward_fn` 中有两处 loss 计算：

**Legacy mode（约第 283 行）：**
```python
# 删除:
# cos_sim = F.cosine_similarity(shift_preds_norm, targets, dim=-1)
# loss = (1.0 - cos_sim).mean()

# 替换为:
loss = F.mse_loss(shift_preds, targets)
```
同时删除 `shift_preds_norm = F.normalize(shift_preds, ...)` 这行。

**Init-prefix mode（约第 335 行）：**
```python
# 删除:
# loss_preds_norm = F.normalize(loss_preds, dim=-1, eps=1e-6)
# cos_sim = F.cosine_similarity(loss_preds_norm, targets, dim=-1)
# loss = (1.0 - cos_sim).mean()

# 替换为:
loss = F.mse_loss(loss_preds, targets)
```
同时删除 `loss_preds_norm` 那行。

**Step 2: 快速验证 forward pass**

```bash
python -c "
import sys; sys.path.insert(0,'.')
import torch
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
cfg = ContinuousAOGPTConfig(block_size=32, vector_dim=768, n_layer=5, n_head=12, n_embd=768, num_init=1)
model = ContinuousAOGPT(cfg).cuda()
vecs  = torch.randn(4, 31, 768).cuda() * 27.7   # main tokens
init  = torch.randn(4,  1, 768).cuda() * 27.7   # init prefix
_, loss = model(vecs, mode='AR', init_vectors=init)
print('loss:', loss.item(), '  (expect positive MSE, typ. 750~1500)')
_, loss2 = model(vecs, mode='Random', init_vectors=init)
print('random loss:', loss2.item())
print('OK')
"
```

**Step 3: Commit**

```bash
git add baseline_continuous/continuous_aogpt.py
git commit -m "v9: replace cosine loss with MSE in forward_fn"
```

---

### Task 5: 更新 eval_utils.py

**Files:**
- Modify: `baseline_continuous/eval_utils.py`

**Step 1: 读取当前 eval_utils.py，找到 cos_sim 相关代码**

当前 `evaluate_ar` 函数使用 `val_cos_sim`，替换为 MSE 指标：

```python
@torch.no_grad()
def evaluate_ar(model, val_loader, device):
    model.eval()
    total_loss, total_pred_norm, n_batches = 0.0, 0.0, 0
    for batch in val_loader:
        init_vectors = batch['init_vectors'].to(device)
        main_vectors = batch['main_vectors'].to(device)
        ni = init_vectors.shape[1]
        t  = main_vectors.shape[1]
        predictions, loss = model(main_vectors, mode='AR', init_vectors=init_vectors)
        shift_preds = predictions[:, ni-1 : ni-1+t, :]
        pred_norm   = shift_preds.norm(dim=-1).mean().item()
        total_loss      += loss.item()
        total_pred_norm += pred_norm
        n_batches += 1
    return {
        'val_loss':      total_loss      / n_batches,
        'val_pred_norm': total_pred_norm / n_batches,   # 期望 ≈ √768 ≈ 27.7
    }
```

**Step 2: 更新 train_ar.py 和 train_mdm.py 的 log 行**

将所有出现 `val_cos_sim` 的地方替换为 `val_pred_norm`：

```python
# train_ar.py / train_mdm.py 中的 log 行替换为:
print(f"  [eval] val_loss: {eval_results['val_loss']:.4f} | pred_norm: {eval_results['val_pred_norm']:.2f}")
```

wandb log 也同步更新（`val/cos_sim` → `val/pred_norm`）。

**Step 3: 验证 eval_utils**

```bash
python -c "
import sys; sys.path.insert(0,'.')
import torch
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
from baseline_continuous.eval_utils import evaluate_ar
from baseline_continuous.disk_dataset import create_disk_dataloaders

cfg_m = ContinuousAOGPTConfig(block_size=32, vector_dim=768, n_layer=5,
                               n_head=12, n_embd=768, num_init=1)
model = ContinuousAOGPT(cfg_m).cuda()
_, val_loader, _ = create_disk_dataloaders('baseline_continuous/data', batch_size=32, num_workers=0, num_chunks=31)
res = evaluate_ar(model, val_loader, 'cuda')
print(res)   # {'val_loss': ..., 'val_pred_norm': ...}
"
```

**Step 4: Commit**

```bash
git add baseline_continuous/eval_utils.py baseline_continuous/train_ar.py baseline_continuous/train_mdm.py
git commit -m "v9: update eval metrics (cos_sim -> MSE + pred_norm)"
```

---

### Task 6: 运行三组实验

**Step 1: 确认 GPU 空闲**

```bash
nvidia-smi | grep -E "MiB|Process"
```

**Step 2: 启动三组实验（AR no-shuffle + AR shuffled 在 GPU 0，MDM 在 GPU 1）**

```bash
CUDA_VISIBLE_DEVICES=0 python baseline_continuous/train_ar.py --no_shuffle --wandb_log false \
  > baseline_continuous/log/v9_ar_noshuffle.txt 2>&1 &

CUDA_VISIBLE_DEVICES=0 python baseline_continuous/train_ar.py --wandb_log false \
  > baseline_continuous/log/v9_ar_shuffled.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 python baseline_continuous/train_mdm.py --wandb_log false \
  > baseline_continuous/log/v9_mdm.txt 2>&1 &
```

**Step 3: 检查前 500 iter 结果**

```bash
sleep 60 && grep "eval" baseline_continuous/log/v9_ar_noshuffle.txt | head -3
grep "eval" baseline_continuous/log/v9_ar_shuffled.txt | head -3
grep "eval" baseline_continuous/log/v9_mdm.txt | head -3
```

期望：AR no-shuffle 的 val_loss 应显著低于 AR shuffled 和 MDM（同 v8 规律）。

**Step 4: 训练完成后 Commit logs**

```bash
git add baseline_continuous/log/v9_*.txt baseline_continuous/config.py
git commit -m "v9: GPT-2 teacher data, MSE loss, 5-layer student — results"
```

---

### Task 7: Eval Order 评测

**Step 1: 更新 eval_order_v8.py 适配 v9**

- `num_chunks` 改为 `cfg.num_chunks`（已在 config 中为 31）
- `causal_order_ref = np.arange(cfg.num_chunks)` 自动适配
- 确认 checkpoint 名称一致（`best_mdm_Random_model.pt` 等）

**Step 2: 运行**

```bash
python baseline_continuous/eval_order_v8.py 2>&1 | tee baseline_continuous/log/v9_eval_order.txt
```

**Step 3: Commit**

```bash
git add baseline_continuous/log/v9_eval_order.txt
git commit -m "v9: eval_order results (Kendall tau, causal advantage)"
```
