# Position-Leak-Free MDM Architecture Design

**Date**: 2026-03-03
**Status**: Approved

---

## 背景与问题

当前 `ContinuousAOGPT` 在 MDM/AR-shuffled 训练时，存在两处位置信息泄漏：

1. **wpe 泄漏**：每个 shuffled token 携带其原始位置的 embedding
   `main_pos_emb_shuf = shuffle(wpe(original_positions), orders)`
   → 模型知道"这个 token 来自原始位置 k"

2. **wtpe 泄漏**：AdaLN 条件携带下一个要预测的 token 的原始位置
   `tpe_main = wtpe(orders + ni)`
   → 模型知道"我下一步要预测原始位置 j"

这导致模型学到的是**位置索引检索**，而非 h 向量之间的**因果内容关系**。MDM 无需从内容中发现顺序，因为顺序已经通过位置直接给出。

---

## 研究目标

- **AR no-shuffle（上帝模型）**：有完整位置信息，逼近理论下界 σ²=0.09，作为绝对上界
- **MDM**：无位置信息，只能从 h 向量的内容中学习，通过大量随机 order 训练后，若能自发发现因果顺序（greedy search 恢复 causal order），则证明 MDM 涌现出了对 GLA 因果结构的理解

---

## 设计方案（Option B）

### 核心改动：`forward_fn` 两处修改

```python
# wpe：去掉 shuffle，改为序列顺序编码
# 旧：main_pos_emb_shuf = self.shuffle(main_pos_emb, orders)
main_pos_emb_shuf = main_pos_emb   # wpe(ni+k)，第 k 个被看到的 token

# wtpe：改为生成步编码，去掉原始位置索引
# 旧：tpe_main = self.transformer.wtpe(orders + ni)
step_idx = torch.arange(t, device=device).unsqueeze(0).expand(b, -1)
tpe_main = self.transformer.wtpe(step_idx)   # wtpe(k)，第 k 步预测
```

其余全部不变：`input_proj`、Transformer blocks（含 AdaLN 结构）、`output_proj`、MSE loss、causal mask。

### AR no-shuffle 不受影响

由于 AR no-shuffle 的 `orders = [0, 1, ..., t-1]`（恒等排列）：
- wpe：`shuffle(wpe, identity)` = `wpe`，改前改后结果相同
- wtpe：索引从 `[ni..ni+t-1]` 变为 `[0..t-1]`，仅是 embedding table 的不同行，重新训练后语义等价

**AR no-shuffle 无需任何修改，也不需要单独的代码路径。**

### 信息对比

| | 改前（泄漏） | 改后（干净） |
|---|---|---|
| 每个 token 携带的位置信息 | 原始位置 k | 序列顺序（第几个被看到） |
| AdaLN 条件 | 目标的原始位置 j | 第几步预测 |
| MDM 学到的 | 位置索引检索 | h 向量内容的条件分布 |
| causal advantage 来源 | 位置信息 + 内容 | 纯内容 |

---

## 训练协议

`train_ar.py` 和 `train_mdm.py` 均**不需要改动**，改动完全封装在 `forward_fn` 内。

| 参数 | AR no-shuffle（上帝模型） | MDM |
|---|---|---|
| train_samples | 500k（扩大，减少过拟合） | 100k（保持） |
| epochs | 10~15（减少，原 50 epoch 过拟合） | 50 |
| orders | [0,1,...,30] 固定 | randperm(31) |

> 旧 checkpoint 不可复用，需重新训练（wtpe 索引语义变化）。

---

## 评估协议

### 现有指标（语义变化）

- **causal order loss**：给定 causal order，wpe 和 wtpe 对 causal/random 两种评估完全相同，causal advantage 完全来自 h 向量内容，不含位置信息贡献
- **random order loss**：同上
- **causal advantage** = random_loss - causal_loss：现在是真正意义上的内容因果优势

### Greedy order search

模型预测"下一个 h 值"，从 remaining tokens 中选：

```python
# cos_sim 与 MSE 在 norm 近似一致时等价（h norm ~95.8 较均匀）
# 保持现有 cos_sim 实现即可
```

greedy Kendall τ 现在测试的是：**模型能否仅凭 h 向量内容自发恢复因果顺序**。

### 新增对比

- AR shuffled 也运行 greedy Kendall τ（目前只对 MDM 跑），用于对比

---

## 预期结果

| 模型 | causal order loss | causal advantage | greedy Kendall τ |
|---|---|---|---|
| AR no-shuffle | 逼近 σ²=0.09 | 高（有位置信息） | N/A |
| MDM（改后） | 介于 σ² 和 AR-shuffled 之间 | 若 > 0 则证明涌现 | 若 > 0.3 则证明发现因果顺序 |
| AR shuffled（改后） | 同 MDM 量级 | 参照 | 参照 |

---

## 文件改动清单

| 文件 | 改动 |
|---|---|
| `baseline_continuous/continuous_aogpt.py` | `forward_fn` 两行 |
| `baseline_continuous/config.py` | `train_samples` 500k，`epochs` 10~15（AR only） |
| `baseline_continuous/eval_order_v8.py` | AR shuffled 也加 greedy Kendall τ |
| 数据生成脚本 | 重新生成 500k AR no-shuffle 训练数据 |
