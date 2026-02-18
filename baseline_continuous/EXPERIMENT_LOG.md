# Baseline Continuous Experiment Log

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
| noise_scale | 0.1 | 生成时加入高斯噪声 |
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
| Train | 50,000 |
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

| 参数 | 值 |
|------|-----|
| batch_size | 128 |
| learning_rate | 1e-3 |
| max_iters | 40,000 |
| warmup_iters | 0 |
| lr schedule | Cosine decay (min_lr = 0.1 * lr) |
| weight_decay | 0.0 |
| grad_clip | 1.0 |
| optimizer | AdamW (betas=0.9, 0.95) |
| seed | 42 |

## 5. 三组实验

### Exp A: AR (shuffled)
- 输入: `shuffled_vectors`
- 顺序: 固定升序 `(0, 1, 2, ..., 127)`
- 训练 loss: 在升序下预测 shuffled data 的 cosine loss

### Exp B: MDM (Random)
- 输入: `shuffled_vectors`
- 顺序: 每个 batch 每个样本独立采样一个随机排列
- 训练 loss: 在随机顺序下预测 shuffled data 的 cosine loss

### Exp C: AR (no shuffle) — 上界参考
- 输入: `vectors`（原始因果顺序）
- 顺序: 固定升序（即因果顺序本身）
- 训练 loss: 因果顺序下的 cosine loss，代表模型能力的理论上界

## 6. 评估设计

**统一评估标准：** 所有模型在 `batch['vectors']`（原始因果顺序）上以 AR 模式评估。

```python
evaluate_ar(model, val_loader, device)
# → 用 batch['vectors'] + mode='AR'
# → 返回 val_loss (cosine distance), val_cos_sim
```


## 7. 实验结果

### 7.1 训练进度

| 模型 | 训练 iters | 状态 | 日志文件 |
|------|-----------|------|---------|
| AR (shuffled) | ~21,900/40,000 |-| `log/ar.txt` |
| AR (no shuffle) | ~21,600/40,000 |-| `log/ar_no.txt` |
| MDM (Random) | ~21,650/40,000 |-| `log/mdm2.13.txt` |

### 7.2 最新评估结果（~21k iters）

| 模型 | val_loss | val_cos_sim | 收敛状态 |
|------|---------|------------|---------|
| AR (no shuffle) | 0.3265 | 0.6735 | 已收敛（趋于平稳） |
| AR (shuffled) | 0.3634 | 0.6366 | 已收敛（趋于平稳） |
| MDM (Random) | 0.4315 | 0.5685 | 仍在缓慢下降 |

### 7.3 分析

**排序符合预期：** AR (no shuffle) < AR (shuffled) < MDM (Random)

- **AR (no shuffle) 最优 (0.3265)**：训练和评估都在原始因果顺序上，信号最强，作为理论上限
- **AR (shuffled) 次之 (0.3634)**：训练在打乱数据上用升序预测，但评估在原始顺序上仍表现不错。说明 block-wise shuffle 后 chunk 内的局部依赖仍可被利用
- **MDM (Random) 较弱 (0.4315)**：随机顺序训练增加了优化难度。但 MDM 在 ~21k iters 仍未完全收敛，有继续下降的趋势

**关键观察：**
- AR (no shuffle) 在 ~15k iters 后趋于平稳（val_loss ≈ 0.3265）
- AR (shuffled) 在 ~18k iters 后趋于平稳（val_loss ≈ 0.3634）
- MDM 在 21k iters 时 val_loss 仍有波动（0.43-0.44），收敛更慢

### 7.4 结论


初步结论：
1. 在当前训练步数下，AR 方法在原始顺序评估上优于 MDM
2. MDM 收敛更慢，需要更多训练步数才能公平对比
3. AR no-shuffle 作为 oracle 如预期表现最好，验证了数据中确实存在因果结构

---

## 8. 后续实验方向

如果 MDM 未能有效发现因果结构，可尝试：

1. **完成 40k 训练** — 当前三组实验都未跑完，MDM 尤其需要更多步数
2. **增大模型容量** — n_layer=6, n_embd=512，给 MDM 更多容量应对随机顺序
3. **增加训练步数** — 80k-200k iters，MDM 收敛更慢
4. **更极端的 shuffle** — 完全 token-level shuffle（而非 block-wise），放大因果信号差异
5. **Curriculum Learning** — Random_CL 模式，逐步增加随机比例
6. **最优顺序搜索** — 训练后用 beam search / greedy 搜索 MDM 的最优生成顺序

---

*Updated: 2026-02-13*
*WandB Project: `baseline-continuous`*
*WandB Group: `baseline-comparison`*
*Checkpoint Dir: `baseline_continuous/checkpoints/`*
*Log Dir: `baseline_continuous/log/`*
