# Baseline Continuous Experiment Log (v2)

## 1. 实验目标

在连续向量序列（Dense AR process）上对比 **AR（固定升序）** 与 **MDM（随机顺序）** 的训练与推理表现。

核心问题：
- 数据经过 block-wise shuffle 后，AR 和 MDM 各自能学到什么？
- MDM 是否能通过随机顺序训练**隐式发现数据的因果结构**（即最优生成顺序）？

## 2. 数据设计

### Dense AR Process

```
x_0 ~ Normalized Gaussian
x_t = tanh( sum_{i=1}^{t} A_{(i-1)%M} @ x_{t-i} * sqrt(M)/sqrt(t) ) + noise
x_t = normalize(x_t)
```

| 参数 | 值 | 说明 |
|------|-----|------|
| vector_dim | 256 | 每个 token 是 256 维连续向量 |
| seq_length | 128 | 序列长度 |
| dependency_window | -1 | Full history：每步依赖所有前序 token |
| num_matrices | 8 | 8 个正交变换矩阵（缩放 1/sqrt(8)） |
| noise_scale | 0.05 | 生成时加入高斯噪声 |
| 非线性 | tanh + L2 normalize | 打破线性结构 |

### Block-wise Shuffle

- 将长度 128 的序列分为 **8 个 chunk**（每个 chunk 16 个 token）
- **chunk 内部保持原始顺序**，仅打乱 chunk 之间的排列
- 训练输入为 `shuffled_vectors`，原始顺序不泄露给模型

### OOD 设计

| 划分 | init_mode | 说明 |
|------|-----------|------|
| Train | positive_first | 初始向量第一维 > 0 |
| Val/Test | negative_first | 初始向量第一维 < 0（OOD） |

### 数据量

| 划分 | 样本数 |
|------|--------|
| Train | 500,000 |
| Val | 10,000 |
| Test | 10,000 |

## 3. 模型架构

**ContinuousAOGPT** — 基于 Transformer + AdaLN 的任意顺序生成模型。

| 参数 | 值 |
|------|-----|
| n_layer | 4 |
| n_head | 4 |
| n_embd | 256 |
| block_size | 128 |
| dropout | 0.0 |
| bias | True |
| 参数量 | ~4.06M |

关键设计：
- **输入**: `input_proj` 将 256 维向量映射到 256 维 embedding
- **输出**: `output_proj` 将 256 维映射回 256 维预测向量
- **Loss**: Cosine distance = `1 - cosine_similarity(pred, target)`
- **顺序控制**: 通过 shuffle/unshuffle 机制实现任意顺序的 causal attention
- **AdaLN**: 目标位置编码通过 Adaptive LayerNorm 注入，告知模型"下一个要预测哪个位置"

## 4. 训练配置

| 参数 | 值 | 与 v1 对比 |
|------|-----|-----------|
| batch_size | 512 | v1: 128 → **4x** |
| learning_rate | 1e-3 | 不变 |
| epochs | 50 | v1: 用 max_iters=40,000 |
| total iters | ~48,700 | v1: 40,000 (未跑完) |
| warmup_iters | 0 | v1: 0 |
| lr schedule | Cosine decay (min_lr = 0.1 * lr) | 不变 |
| weight_decay | 0.0 | 不变 |
| grad_clip | 1.0 | 不变 |
| optimizer | AdamW (betas=0.9, 0.95) | 不变 |
| seed | 42 | 不变 |
| train_samples | 500,000 | v1: 50,000 → **10x** |
| noise_scale | 0.05 | v1: 0.1 → **减半** |

**主要变化：**
1. **训练数据量 10x**：50,000 → 500,000 样本
2. **Batch size 4x**：128 → 512
3. **噪声减半**：noise_scale 0.1 → 0.05
4. **训练更充分**：50 epochs（~48,700 iters），每个样本被看到约 50 次

## 5. 三组实验

### Exp A: AR (shuffled)
- 输入: `shuffled_vectors`
- 顺序: 固定升序 `(0, 1, 2, ..., 127)`
- 训练时间: 04:23 — 06:18（约 1h55min）

### Exp B: AR (no shuffle) — 上界参考
- 输入: `vectors`（原始因果顺序）
- 顺序: 固定升序（即因果顺序本身）
- 训练时间: 06:31 — 08:26（约 1h55min）

### Exp C: MDM (Random)
- 输入: `shuffled_vectors`
- 顺序: 每个 batch 每个样本独立采样一个随机排列
- 训练时间: 08:27 — 10:27（约 2h00min）

## 6. 评估设计

**统一评估标准：** 所有模型在 `batch['vectors']`（原始因果顺序）上以 AR 模式评估。

```python
evaluate_ar(model, val_loader, device)
# → 用 batch['vectors'] + mode='AR'
# → 返回 val_loss (cosine distance), val_cos_sim
```

## 7. 实验结果

### 7.1 训练进度

| 模型 | 训练 epochs | 总 iters | 状态 |
|------|-----------|----------|------|
| AR (shuffled) | 50/50 | ~48,700 | 已完成 |
| AR (no shuffle) | 50/50 | ~48,700 | 已完成 |
| MDM (Random) | 50/50 | ~48,700 | 已完成 |

### 7.2 最终评估结果（50 epochs）

| 模型 | best val_loss | best val_cos_sim | test_loss | test_cos_sim |
|------|-------------|-----------------|-----------|-------------|
| AR (no shuffle) | 0.1037 | 0.8963 | 0.1038 | 0.8962 |
| AR (shuffled) | 0.1312 | 0.8688 | 0.1315 | 0.8685 |
| MDM (Random) | 0.1866 | 0.8134 | 0.1877 | 0.8123 |

### 7.3 训练收敛轨迹

#### AR (shuffled)
| 阶段 | iter | val_loss | val_cos_sim | 备注 |
|------|------|---------|------------|------|
| Epoch 1 | 500 | 0.1595 | 0.8405 | 快速下降 |
| Epoch 5 | 4,500 | 0.1453 | 0.8547 | |
| Epoch 10 | 9,000 | 0.1394 | 0.8606 | |
| Epoch 20 | 19,500 | 0.1337 | 0.8663 | 下降趋缓 |
| Epoch 30 | 29,000 | 0.1322 | 0.8678 | |
| Epoch 40 | 38,500 | 0.1317 | 0.8683 | 接近收敛 |
| Epoch 48 | 46,000 | 0.1312 | 0.8688 | **Best** |
| Final | 48,700 | 0.1314 | 0.8686 | |

#### AR (no shuffle)
| 阶段 | iter | val_loss | val_cos_sim | 备注 |
|------|------|---------|------------|------|
| Epoch 1 | 500 | 0.1238 | 0.8762 | 起始就比 AR shuffled 低 |
| Epoch 5 | 4,500 | 0.1079 | 0.8921 | |
| Epoch 10 | 9,500 | 0.1059 | 0.8941 | |
| Epoch 20 | 19,500 | 0.1047 | 0.8953 | 下降极为缓慢 |
| Epoch 30 | 29,000 | 0.1042 | 0.8958 | |
| Epoch 40 | 39,000 | 0.1039 | 0.8961 | |
| Epoch 48 | 46,500 | 0.1037 | 0.8963 | **Best** |
| Final | 48,700 | 0.1037 | 0.8963 | |

#### MDM (Random)
| 阶段 | iter | val_loss | val_cos_sim | 备注 |
|------|------|---------|------------|------|
| Epoch 1 | 500 | 0.2624 | 0.7376 | 起步慢，初始 loss 很高 |
| Epoch 5 | 4,500 | 0.2091 | 0.7909 | |
| Epoch 10 | 9,000 | 0.1968 | 0.8032 | |
| Epoch 15 | 14,000 | 0.1910 | 0.8090 | |
| Epoch 20 | 19,500 | 0.1875 | 0.8125 | 仍在下降 |
| Epoch 30 | 29,000 | 0.1881 | 0.8119 | 波动明显 |
| Epoch 36 | 35,000 | 0.1870 | 0.8130 | |
| Epoch 41 | 39,500 | 0.1867 | 0.8133 | |
| Epoch 47 | 45,000 | 0.1866 | 0.8134 | **Best** |
| Final | 48,700 | 0.1876 | 0.8124 | val_loss 有波动 |

### 7.4 与 v1 实验对比

| 模型 | v1 val_loss (~21k iters) | v2 test_loss (50 epochs) | 改善 |
|------|------------------------|------------------------|------|
| AR (no shuffle) | 0.3265 | 0.1038 | **-68.2%** |
| AR (shuffled) | 0.3634 | 0.1315 | **-63.8%** |
| MDM (Random) | 0.4315 | 0.1877 | **-56.5%** |

> **注意：** v2 的巨大改善来自多重因素：训练数据 10x、batch size 4x、noise 减半、训练更充分。

### 7.5 分析

**排序与 v1 一致：** AR (no shuffle) < AR (shuffled) < MDM (Random)

- **AR (no shuffle) 最优 (test=0.1038)**：训练和评估都在原始因果顺序上，信号最强，作为理论上限。在 epoch 10 后即接近收敛（val_loss < 0.106），后续 40 epochs 仅从 0.106 降至 0.104。
- **AR (shuffled) 次之 (test=0.1315)**：训练在打乱数据上用升序预测，但评估在原始顺序上仍表现良好。与 oracle 的差距为 0.0277（cos_sim 差距 2.77%），说明 block-wise shuffle 后 chunk 内的局部依赖仍可被利用。
- **MDM (Random) 较弱 (test=0.1877)**：随机顺序训练增加了优化难度。与 oracle 的差距为 0.0839（cos_sim 差距 8.39%）。

**关键观察：**

1. **收敛速度差异明显**：
   - AR (no shuffle) 在 ~epoch 10 后趋于平稳（val_loss ≈ 0.106）
   - AR (shuffled) 在 ~epoch 20 后趋于平稳（val_loss ≈ 0.134）
   - MDM 在 50 epochs 后 val_loss 仍有波动（0.186-0.191），**未完全收敛**

2. **MDM 的波动性**：MDM 在后期训练中 val_loss 波动较大（epoch 20-30 间多次回弹），best model 出现在 epoch 47（0.1866），但 final eval 为 0.1876。这表明 MDM 的优化景观更不稳定。

3. **数据量和 noise 的影响**：相比 v1（50k 样本、noise=0.1），v2（500k 样本、noise=0.05）所有模型都有巨幅改善（56-68%），说明更多数据和更低噪声对模型学习有决定性帮助。

4. **AR shuffled vs MDM 的差距**：AR shuffled 的 test_cos_sim (0.8685) 比 MDM (0.8123) 高出 5.62%。升序固定顺序在 block-shuffled 数据上仍然比随机顺序训练有显著优势。

### 7.6 结论

1. **AR 方法在原始顺序评估上全面优于 MDM**：即使在更充分训练（50 epochs, 500k 样本）的条件下，MDM 仍与 AR 有较大差距。
2. **MDM 收敛更慢且不稳定**：需要更多训练步数或不同的训练策略才能接近 AR 的表现。
3. **数据量和噪声是关键因素**：v2 相比 v1 的巨幅改善（所有模型 56-68% 的 loss 下降）证明了更大数据集和更低噪声的重要性。
4. **Block-wise shuffle 并不完全破坏因果信号**：AR (shuffled) 与 AR (no shuffle) 的差距仅为 2.77%（cos_sim），说明 chunk 内的局部因果关系仍被有效利用。
5. **MDM 尚未展现"发现因果结构"的能力**：在当前设置下，MDM 的随机顺序训练未能隐式学习到等价于固定升序的生成策略。

---

## 8. 后续实验方向

1. **增大模型容量** — n_layer=6, n_embd=512，给 MDM 更多容量应对随机顺序
2. **增加训练步数** — 100-200 epochs，MDM 可能需要更长训练时间
3. **更极端的 shuffle** — 完全 token-level shuffle（而非 block-wise），放大因果信号差异
4. **Curriculum Learning** — Random_CL 模式，逐步增加随机比例
5. **最优顺序搜索** — 训练后用 beam search / greedy 搜索 MDM 的最优生成顺序
6. **MDM 评估方式探索** — 除了固定 AR 评估外，用 MDM 模型自行选择最优生成顺序进行评估

---

*Updated: 2026-02-15*
*WandB Project: `baseline-continuous`*
*Checkpoint Dir: `baseline_continuous/checkpoints/`*
*Log Dir: `baseline_continuous/log/`*
