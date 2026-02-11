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
| vector_dim | 64 | 每个 token 是 64 维连续向量 |
| seq_length | 64 | 序列长度 |
| dependency_window | -1 | Full history：每步依赖所有前序 token |
| num_matrices | 8 | 8 个正交变换矩阵（缩放 1/sqrt(8)） |
| noise_scale | 0.1 | 生成时加入高斯噪声 |
| 非线性 | tanh + L2 normalize | 打破线性结构 |

### Block-wise Shuffle

- 将长度 64 的序列分为 **8 个 chunk**（每个 chunk 8 个 token）
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
| Train | 10,000（online 生成，每次不同） |
| Val | 2,000（预生成，固定） |
| Test | 2,000（预生成，固定） |

## 3. 模型架构

**ContinuousAOGPT** — 基于 Transformer + AdaLN 的任意顺序生成模型。

| 参数 | 值 |
|------|-----|
| n_layer | 4 |
| n_head | 4 |
| n_embd | 256 |
| block_size | 64 |
| dropout | 0.0 |
| bias | True |
| 参数量 | ~0.65M |

关键设计：
- **输入**: `input_proj` 将 64 维向量映射到 256 维 embedding
- **输出**: `output_proj` 将 256 维映射回 64 维预测向量
- **Loss**: Cosine distance = `1 - cosine_similarity(pred, target)`
- **顺序控制**: 通过 shuffle/unshuffle 机制实现任意顺序的 causal attention
- **AdaLN**: 目标位置编码通过 Adaptive LayerNorm 注入，告知模型"下一个要预测哪个位置"

## 4. 训练配置

| 参数 | 值 |
|------|-----|
| batch_size | 64 |
| learning_rate | 1e-3 |
| max_iters | 40,000 |
| warmup_iters | 500 |
| lr schedule | Cosine decay (min_lr = 0.1 * lr) |
| weight_decay | 0.0 |
| grad_clip | 1.0 |
| optimizer | AdamW (betas=0.9, 0.95) |
| seed | 42 |

## 5. 三组实验

### Exp A: AR (shuffled)
- 输入: `shuffled_vectors`
- 顺序: 固定升序 `(0, 1, 2, ..., 63)`
- 训练 loss: 在升序下预测 shuffled data 的 cosine loss

### Exp B: MDM (Random)
- 输入: `shuffled_vectors`
- 顺序: 每个 batch 每个样本独立采样一个随机排列
- 训练 loss: 在随机顺序下预测 shuffled data 的 cosine loss

### Exp C: AR (no shuffle) — 上界参考
- 输入: `vectors`（原始因果顺序）
- 顺序: 固定升序（即因果顺序本身）
- 训练 loss: 因果顺序下的 cosine loss，代表模型能力的理论上界

## 6. 预期与分析

### 训练 loss 预期

```
AR (no shuffle) < AR (shuffled) < MDM (Random)
```

- **AR (no shuffle) 最低**: 顺序和因果结构完全一致，信号最强
- **AR (shuffled) 中间**: 固定升序 + block shuffle 后仍有 chunk 内局部依赖可利用
- **MDM (Random) 最高**: 必须兼顾所有随机顺序，优化更难（这是正常的，不是缺陷）

### 推理顺序评估预期

对训练好的模型，分别用 Ground Truth / Ascending / Random 顺序做 inference：

**MDM 模型**（如果学到了因果结构）：
```
GT order loss < Ascending loss < Random loss
```

**AR (shuffled) 模型**（只会升序）：
```
Ascending loss << GT order loss ≈ Random loss
```

**AR (no shuffle) 模型**（学到了因果依赖）：
```
GT order loss < Ascending loss << Random loss
```

## 7. 实验结果

### 7.1 训练 Loss（Best Val Loss）

| 模型 | Best Val Loss | Best Iter |
|------|--------------|-----------|
| AR (no shuffle) | | |
| AR (shuffled) | | |
| MDM (Random) | | |

### 7.2 推理顺序评估（eval_order.py）

#### Validation Set

| 模型 | GT Order Loss | Ascending Loss | Random Loss (MC=20) |
|------|--------------|----------------|---------------------|
| AR (no shuffle) | | | |
| AR (shuffled) | | | |
| MDM (Random) | | | |

#### Test Set

| 模型 | GT Order Loss | Ascending Loss | Random Loss (MC=20) |
|------|--------------|----------------|---------------------|
| AR (no shuffle) | | | |
| AR (shuffled) | | | |
| MDM (Random) | | | |

### 7.3 分析指标

| 模型 | GT vs Random 差距 | GT vs Ascending 差距 | 是否发现因果结构 |
|------|-------------------|---------------------|-----------------|
| AR (no shuffle) | | | |
| AR (shuffled) | | | |
| MDM (Random) | | | |

### 7.4 结论

<!-- 实验结束后填写 -->

---

## 8. 后续实验方向

如果 MDM 未能有效发现因果结构，可尝试：

1. **增大模型容量** — n_layer=6, n_embd=512，给 MDM 更多容量应对随机顺序
2. **增加训练步数** — 80k-200k iters，MDM 收敛更慢
3. **更极端的 shuffle** — 完全 token-level shuffle（而非 block-wise），放大因果信号差异
4. **Curriculum Learning** — Random_CL 模式，逐步增加随机比例
5. **最优顺序搜索** — 训练后用 beam search / greedy 搜索 MDM 的最优生成顺序

---

*Created: 2026-02-06*
*WandB Project: `baseline-continuous`*
*Checkpoint Dir: `baseline_continuous/checkpoints/`*
