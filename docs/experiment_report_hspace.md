# GLA H-Space Baseline 对比实验报告

---

## 一、实验背景与动机

本实验在 **GLA-340M 连续自回归生成的 h-space**（D=1024）上，比较三种训练范式在因果结构学习能力上的差异：

- **AR no-shuffle**：标准因果 AR，学习 GLA 的真实生成方向
- **AR block-shuffle**：块间随机打乱，块内保留因果顺序，检验部分因果监督的效果
- **MDM（全随机）**：Any-Order GPT 以全随机排列训练，理论上可泛化到任意顺序生成

核心问题：**模型能否从训练数据中内化 GLA h-space 的因果生成顺序？**

---

## 二、数据生成

训练数据由 `ContinuousHSpaceTeacher`（`baseline_continuous/teacher_generator.py`）在**纯连续 h-space 中自回归生成**

### 生成过程

$$h_0 = \text{Normalize}(\mathcal{N}(0, I_D)) \times \sqrt{D}$$

$$x_t = \text{Normalize}(h_t) \times \sqrt{D} \quad \text{（归一化后作为 inputs\_embeds）}$$

$$\mu_t = \text{GLA\_4L}(\text{inputs\_embeds}=[x_0, \ldots, x_{t-1}])_{-1}$$

$$h_t = \mu_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0, \sigma^2 I), \quad t = 1, \ldots, L-1$$

GLA 模型以前一步的归一化 h 向量作为 `inputs_embeds` 输入，在连续向量空间中自回归迭代，其输出作为下一个 h 向量的均值。

### Teacher 配置

| 参数 | 值 |
|------|----|
| Teacher | `fla-hub/gla-340M-15B`（GLA-340M），截取前 4 层（`layer_idx=3`） |
| 向量维度 | D = 1024 |
| 序列长度 | L = 32（`num_init=1`，h₀ 作为固定前缀，h₁..h₃₁ 为主 token） |
| 噪声 | $\sigma = 0.3$，理论噪声下界 $\sigma^2 = 0.09$ |

### 数据集规模

| 分割 | 样本数 |
|------|--------|
| train | 500,000 |
| val | 10,000 |
| test | 10,000 |

数据以 numpy memmap 格式存储在 `baseline_continuous/data_hspace_500k/`，生成脚本为 `baseline_continuous/generate_hspace_memmap.py`。每次读取仅加载所需页面，内存占用接近零。

### 块间打乱（Block Shuffle）

将 31 个主 token 划分为 **4 个块**（每块约 7-8 个 token），随机打乱块的顺序，块内保留因果顺序：

$$\pi = \text{BlockPerm}(t=31,\ \text{num\_chunks}=4)$$

**理论因果信号比例**：块内相邻 pair 均满足因果顺序，因果梯度信号约占 **87%**（对比全随机的 ~3%）。

---

## 三、模型架构

### ContinuousAOGPT（`baseline_continuous/continuous_aogpt.py`）

| 参数 | 值 |
|------|----|
| 参数量 | 69.31M |
| n_layer | 5 |
| n_head | 4 |
| n_embd | 1024（与 h-space 维度一致，无 projection） |
| block_size | 32 |
| num_init | 1（h₀ 始终可见，不参与 loss） |

### 位置编码设计

模型使用**两套独立的位置编码**，解决原始架构的位置泄漏问题：

- **`wpe`**：序列位置编码（第 k 个被生成的位置）
- **`wtpe`**：生成步编码，通过 AdaLN 注入，编码当前预测的是原始序列中第几个 token

`orders` 参数决定 token 的排列方式，`forward_fn` 按该排列重排输入后做因果 attention：

$$\text{main\_shuffled}[b, k, :] = \text{vectors}[b,\ \text{orders}[b,k],\ :]$$

位置 $k$ 的预测只依赖 $\text{main\_shuffled}[0..k-1]$ 和 init，预测目标为 $\text{main\_shuffled}[k]$。

### 三种训练模式

| 模型 | 训练 mode | orders 来源 | 因果信号比例 |
|------|-----------|------------|------------|
| AR no-shuffle | `mode='AR'` | 恒为 [0,1,...,30] | 100% |
| AR block-shuffle | `mode='AR'` | dataset 预计算的 block 排列 | ~87% |
| MDM | `mode='Random'` | 每 batch 在线采样全随机排列 | ~3% |

> **注**：MDM 的 `mode='Random'` 在内部重新生成随机顺序，dataset 侧的 block shuffle 对 MDM 训练无效（被完全覆盖）。

---

## 四、训练过程

### 统一超参数

| 参数 | 值 |
|------|----|
| 数据 | `data_hspace_500k`（500k train） |
| Epochs | 15 |
| Batch size | 256 |
| 优化器 | AdamW（weight_decay=0.1，β=(0.9, 0.95)） |
| 学习率 | 3e-4，linear warmup（5%）+ cosine decay → 3e-5 |
| EMA decay | adaptive：min(0.9999, (1+step)/(10+step)) |
| Grad clip | 1.0 |
| num_chunks | 4 |

### 训练结果

| 模型 | 训练脚本 | Best val loss | Final test loss | test cos_sim |
|------|---------|--------------|-----------------|-------------|
| AR no-shuffle | `train_ar.py --no_shuffle` | 0.1139 | 0.1136 | 0.9927 |
| AR block-shuffle | `train_ar.py` | 0.1716 | 0.1708 | 0.9890 |
| MDM（全随机） | `train_mdm.py` | 0.2735 | 0.2724 | 0.9841 |

AR no-shuffle 收敛最快，前几个 epoch 即大幅下降；AR block-shuffle 稳定收敛；MDM 因全随机训练的因果梯度信号极弱（~3%），约 1000 步后进入平台期，最终停在 0.27 附近。

---

## 五、顺序评估

评估脚本：`baseline_continuous/eval_order_v8.py`

核心思路：对同一个训练好的模型传入**不同的 `orders` 参数**，比较各种生成顺序下的 MSE loss，度量模型对因果结构的内化程度。

### 5.1 评估指标

| 指标 | 说明 |
|------|------|
| **Causal order loss** | orders=[0,1,...,30]，标准前向 AR |
| **Naive reverse loss** | h₀ 作 init，orders=[30,29,...,0]，第一步预测 h₃₁（不对称设计） |
| **True reverse loss** | init=h₃₁，orders=[0,...,30]，从终点向前预测（真正反向 AR） |
| **Random order loss** | MC 平均（N=10 次随机排列） |
| **Causal advantage** | random_loss − causal_loss，正值表示模型偏好因果方向 |
| **Greedy Kendall τ** | 贪心顺序搜索结果与因果顺序的 Kendall 相关系数 |
| **Greedy order loss** | 将贪心找到的顺序喂回模型计算的 loss |

**Greedy 顺序搜索**：每步从 remaining tokens 中选余弦相似度最高的 token 加入 selected，共 t=31 步，每步一次 forward pass，总计 O(t) 次。

### 5.2 最终评估结果（test set）

| 模型 | causal | naive reverse | true reverse | random | causal adv. | greedy τ | greedy loss |
|------|--------|---------------|--------------|--------|------------|----------|-------------|
| **AR no-shuffle** | **0.1131** | 0.3384 | 0.4925 | 0.4416 | **+0.3285 ✓** | — | — |
| **AR block-shuffle** | **0.1694** | 0.3681 | 0.4827 | 0.3574 | **+0.1880 ✓** | −0.341 | 0.2369 |
| **MDM** | 0.2704 | **0.2161** | 0.5823 | 0.2633 | −0.007 ✗ | −0.293 | 0.2281 |

---

## 六、结果分析

### 6.1 AR no-shuffle：最强因果结构

causal advantage = +0.3285，causal loss（0.1131）约为噪声下界 σ²=0.09 的 1.26 倍。reverse 和 random 顺序下 loss 大幅上升，证明模型对因果方向有强偏好，rollout cos_sim 前三步 [0.933, 0.974, 0.986] 到后三步 [0.993, 0.993, 0.993]，误差累积极小。

### 6.2 AR block-shuffle：有因果结构，greedy 探针失效

causal advantage = +0.1880，**模型确实学到了因果结构**。但贪心搜索 τ=−0.341（反向偏），greedy order loss（0.2369）远高于 causal loss（0.1694）。

原因：greedy 搜索构造的 `selected + sorted(remaining)` 序列对 block-shuffle 训练的模型是 **out-of-distribution** 输入，模型在这些 OOD 序列上预测不可靠，导致贪心反复选错。**模型有能力利用因果顺序（causal loss 低），但 greedy 探针无法正确"问出"这个能力**。

### 6.3 MDM：无因果优势，退化为均值预测器

causal advantage ≈ 0（略负），MDM 未内化因果方向。全随机训练的因果梯度信号仅 ~3%，模型快速退化到均匀分布预测。

**Naive reverse loss（0.2161）低于 causal（0.2704）的解读**：
这是评估设计的 artifact，而非模型学到了逆因果结构：
- Naive reverse 第 k 步有 context `{h₀, h₃₁,...,h_{31-k+1}}`，后期步骤可利用邻近 h_{t+1} 估计 h_t（GLA 递推的逆运算）
- True reverse（h₃₁ 作 init）loss = 0.5823，是所有方案中**最高的**，确认 naive reverse 低 loss 是 artifact

### 6.4 Greedy 是弱探针

greedy loss > causal loss 的现象（尤其在 AR block-shuffle 上）揭示：**greedy 顺序搜索不能可靠恢复模型内化的因果顺序**。更好的推理方案（beam search、梯度优化排列）可能能缩小这个差距，是后续推理端工作的主要方向。

---

## 七、Pipeline 总结

```
1. 数据生成
   ContinuousHSpaceTeacher（fla-hub/gla-340M-15B, layer 3）纯连续空间自回归生成
   → baseline_continuous/generate_hspace_memmap.py
   → data_hspace_500k/{train,val,test}_{vectors,init_vectors}.npy + data_config.pt

2. 数据加载（baseline_continuous/disk_dataset.py）
   MemmapDataset（lazy memmap）
     init_vectors  : h₀ 固定前缀 [B, 1, D]
     main_vectors  : h₁..h₃₁ 原始顺序 [B, 31, D]
     shuffled_main : block shuffle 后的主序列 [B, 31, D]
     shuffle_indices: block shuffle 排列索引 [B, 31]

3. 训练
   AR no-shuffle    : python -m baseline_continuous.train_ar --no_shuffle
                      → checkpoints/best_ar_noshuffle_model.pt
   AR block-shuffle : python -m baseline_continuous.train_ar
                      → checkpoints/best_ar_model.pt
   MDM 全随机       : python -m baseline_continuous.train_mdm
                      → checkpoints/best_mdm_Random_model.pt

4. 评估
   python -m baseline_continuous.eval_order_v8
   对每个模型输出：
     causal / naive-reverse / true-reverse / random order loss
     causal advantage
     greedy 顺序搜索 + Kendall τ + greedy order loss（MDM/AR-shuffle）
```

---

## 八、主要结论

1. **因果结构学习能力**：AR no-shuffle >> AR block-shuffle >> MDM（全随机）
2. **Block shuffle 的价值**：将因果梯度信号从 ~3% 提升到 ~87%，AR block-shuffle 的 causal advantage 从 ≈0 提升到 +0.188，显著优于 MDM
3. **Greedy 探针的局限**：对全随机 MDM 部分有效，对 block-shuffle 训练模型完全失效（OOD 问题）
4. **Reverse 评估的陷阱**：naive reverse 低 loss 是评估设计 artifact；true reverse 对所有当前模型均 OOD，loss 最高
5. **后续方向**：训练侧可探索 AO-GPT with block-constrained orders（显式传入 block 排列，不同于 AR block-shuffle）；推理侧可探索 beam search 或梯度优化的顺序搜索
