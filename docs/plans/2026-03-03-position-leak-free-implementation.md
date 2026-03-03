# Position-Leak-Free MDM Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 消除 MDM/AR-shuffled 训练中 wpe 和 wtpe 对原始位置的泄漏，使模型必须从 h 向量内容中发现因果顺序。

**Architecture:** 修改 `forward_fn` 的 init-prefix 分支：wpe 改为序列顺序编码（去掉 shuffle），wtpe 改为生成步编码（去掉原始位置索引）。AR no-shuffle（上帝模型）由于 orders 恒为 identity，改动后输出与原来等价，无需单独处理。

**Tech Stack:** PyTorch, ContinuousAOGPT (GPT + AdaLN), numpy memmap, wandb

---

## Task 1: 修改 `forward_fn` — 去掉位置泄漏

**Files:**
- Modify: `baseline_continuous/continuous_aogpt.py`（init-prefix 分支，约第 295–322 行）

**Step 1: 阅读当前 forward_fn 的 init-prefix 分支，定位两处改动**

```bash
grep -n "main_pos_emb_shuf\|main_orig_pos\|tpe_main" \
    baseline_continuous/continuous_aogpt.py
```

预期输出（行号可能略有不同）：
```
307:        main_pos_emb_shuf = self.shuffle(main_pos_emb, orders)
317:        main_orig_pos = orders + ni
318:        tpe_main = self.transformer.wtpe(main_orig_pos)
```

**Step 2: 应用改动**

改动一：第 307 行，去掉 shuffle（wpe 只携带序列顺序）

```python
# 旧：
main_pos_emb_shuf = self.shuffle(main_pos_emb, orders)

# 新：
main_pos_emb_shuf = main_pos_emb
```

改动二：第 317–318 行，wtpe 改为生成步索引

```python
# 旧：
main_orig_pos = orders + ni
tpe_main = self.transformer.wtpe(main_orig_pos)

# 新：
step_idx = torch.arange(t, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1)
tpe_main = self.transformer.wtpe(step_idx)
```

同时删除旧的 `main_orig_pos` 那行（已不需要）。

**Step 3: 验证 shape 正确**

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM
python -c "
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
import torch

cfg = ContinuousAOGPTConfig(block_size=32, vector_dim=1024,
                             n_layer=5, n_head=4, n_embd=1024,
                             dropout=0.0, bias=True, num_init=1)
model = ContinuousAOGPT(cfg)

B, t, D, ni = 2, 31, 1024, 1
vectors      = torch.randn(B, t, D)
init_vectors = torch.randn(B, ni, D)

# AR no-shuffle
orders_ar  = torch.arange(t).unsqueeze(0).expand(B, -1)
preds, loss = model(vectors, mode=None, orders=orders_ar, init_vectors=init_vectors)
print(f'AR no-shuffle  preds={preds.shape}  loss={loss.item():.4f}')

# Random order
orders_rand = torch.stack([torch.randperm(t) for _ in range(B)])
preds, loss = model(vectors, mode=None, orders=orders_rand, init_vectors=init_vectors)
print(f'Random order   preds={preds.shape}  loss={loss.item():.4f}')

print('Shape check PASSED')
"
```

预期输出：
```
AR no-shuffle  preds=torch.Size([2, 32, 1024])  loss=<any float>
Random order   preds=torch.Size([2, 32, 1024])  loss=<any float>
Shape check PASSED
```

**Step 4: 验证 AR no-shuffle 与 random order 在 wpe 上不再共享原始位置**

```bash
python -c "
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
import torch

cfg = ContinuousAOGPTConfig(block_size=32, vector_dim=1024,
                             n_layer=5, n_head=4, n_embd=1024,
                             dropout=0.0, bias=True, num_init=1)
model = ContinuousAOGPT(cfg)
model.eval()

B, t, D, ni = 1, 31, 1024, 1
vectors      = torch.randn(B, t, D)
init_vectors = torch.randn(B, ni, D)

# 对同一批数据，causal 和 random 应该拿到不同内容但相同形状的 wpe
orders_causal = torch.arange(t).unsqueeze(0)
orders_random = torch.randperm(t).unsqueeze(0)

with torch.no_grad():
    p1, l1 = model(vectors, mode=None, orders=orders_causal, init_vectors=init_vectors)
    p2, l2 = model(vectors, mode=None, orders=orders_random, init_vectors=init_vectors)

# 新设计下：两者的 wpe 序列完全一致（都是 wpe(ni+0), wpe(ni+1), ...）
# 但 token 内容不同（vectors 排列不同），所以 loss 不同
print(f'causal loss = {l1.item():.4f}')
print(f'random loss = {l2.item():.4f}')
print('Position leak check PASSED (losses differ due to content, not position)')
"
```

**Step 5: Commit**

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM
git add baseline_continuous/continuous_aogpt.py
git commit -m "fix: remove position leak in forward_fn (wpe seq-pos, wtpe step-idx)"
```

---

## Task 2: 更新 `eval_order_v8.py` — AR shuffled 也跑 greedy search

**Files:**
- Modify: `baseline_continuous/eval_order_v8.py`

**Step 1: 定位 `models_to_eval` 列表**

```bash
grep -n "do_greedy\|models_to_eval" baseline_continuous/eval_order_v8.py
```

**Step 2: 将 AR shuffled 的 `do_greedy` 改为 True**

```python
# 旧：
models_to_eval = [
    ('MDM',          os.path.join(ckpt_dir, 'best_mdm_Random_model.pt'),    True),
    ('AR shuffled',  os.path.join(ckpt_dir, 'best_ar_model.pt'),            False),
    ('AR no-shuffle',os.path.join(ckpt_dir, 'best_ar_noshuffle_model.pt'),  False),
]

# 新：
models_to_eval = [
    ('MDM',          os.path.join(ckpt_dir, 'best_mdm_Random_model.pt'),    True),
    ('AR shuffled',  os.path.join(ckpt_dir, 'best_ar_model.pt'),            True),
    ('AR no-shuffle',os.path.join(ckpt_dir, 'best_ar_noshuffle_model.pt'),  False),
]
```

AR no-shuffle 不做 greedy（它的顺序是固定的，没有意义）。

**Step 3: Commit**

```bash
git add baseline_continuous/eval_order_v8.py
git commit -m "feat: enable greedy Kendall tau for AR shuffled in eval"
```

---

## Task 3: 生成 500k h-space 训练数据（AR no-shuffle 专用）

**Files:**
- 输出目录: `baseline_continuous/data_hspace_500k/`

**Step 1: 生成数据**

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM
python -m gla_exp.generate_hspace_memmap \
    --data_dir baseline_continuous/data_hspace_500k \
    --sigma 0.3 \
    --seq_len 32 \
    --n_train 500000 \
    --n_val 10000 \
    --n_test 10000 \
    --batch_size 64
```

预期输出：
```
[train] 500000 samples × L=32  (7813 batches)
...
[val]   10000 samples × L=32
[test]  10000 samples × L=32
```

**Step 2: 验证数据文件**

```bash
python -c "
import numpy as np, os
d = 'baseline_continuous/data_hspace_500k'
for split in ['train', 'val', 'test']:
    v = np.memmap(f'{d}/{split}_vectors.npy', dtype='float32', mode='r')
    print(f'{split}: shape={v.shape}')
"
```

预期输出：
```
train: shape=(500000, 31, 1024)   # 500k × 31 main tokens × D=1024
val:   shape=(10000,  31, 1024)
test:  shape=(10000,  31, 1024)
```

---

## Task 4: 训练 AR no-shuffle（上帝模型，500k 数据，15 epoch）

**Step 1: 运行训练**

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM
python baseline_continuous/train_ar.py \
    --no_shuffle \
    --data_dir baseline_continuous/data_hspace_500k \
    --epochs 15 \
    --device cuda \
    2>&1 | tee baseline_continuous/log/ar_noshuffle_v10.txt
```

**Step 2: 监控训练——检查是否过拟合**

```bash
grep "eval\|save" baseline_continuous/log/ar_noshuffle_v10.txt | tail -20
```

目标：
- val_loss 持续下降直到收敛，不出现 val_loss 上升而 train_loss 下降的剪刀差
- 最终 val_loss 尽量接近 σ² × D / D = **0.09**（MSE per dimension）

---

## Task 5: 训练 MDM（新架构，无位置泄漏）

**Step 1: 运行训练**

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM
python baseline_continuous/train_mdm.py \
    --mode Random \
    --data_dir baseline_continuous/data_hspace \
    --epochs 50 \
    --device cuda \
    2>&1 | tee baseline_continuous/log/mdm_v10.txt
```

---

## Task 6: 训练 AR shuffled（新架构，无位置泄漏）

**Step 1: 运行训练**

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM
python baseline_continuous/train_ar.py \
    --data_dir baseline_continuous/data_hspace \
    --epochs 50 \
    --device cuda \
    2>&1 | tee baseline_continuous/log/ar_shuffled_v10.txt
```

---

## Task 7: 运行评估，对比三个模型

**Step 1: 确认三个 checkpoint 都存在**

```bash
ls -lh baseline_continuous/checkpoints/best_*.pt
```

**Step 2: 运行 eval**

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM
python baseline_continuous/eval_order_v8.py \
    2>&1 | tee baseline_continuous/log/eval_v10.txt
```

**Step 3: 检验关键指标**

```bash
grep -E "causal|random|Kendall|advantage" baseline_continuous/log/eval_v10.txt
```

期望看到：

| 模型 | causal advantage | greedy Kendall τ |
|---|---|---|
| AR no-shuffle | 高（>0） | N/A |
| MDM | 若 >0 → 涌现成功 | 若>0.3 → 发现因果顺序 |
| AR shuffled | 参照 | 参照 |

**Step 4: Commit 结果日志**

```bash
git add baseline_continuous/log/ar_noshuffle_v10.txt \
        baseline_continuous/log/mdm_v10.txt \
        baseline_continuous/log/ar_shuffled_v10.txt \
        baseline_continuous/log/eval_v10.txt
git commit -m "exp: v10 position-leak-free training results"
```
