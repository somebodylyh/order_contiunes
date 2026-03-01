# Teacher-Student 连续向量学习设计文档

## 目标

用 GPT-2 前 4 层作为 teacher 生成结构化连续向量序列，替换手工设计的 Dense AR Process，验证 Student 模型（AR vs MDM）能否发现序列的因果生成顺序。

---

## Teacher 数据生成

**模型**：GPT-2 small 前 4 层（HuggingFace 加载，权重冻结，去掉 embedding 和 lm_head）

**生成公式**：
```
x_0   ~ Normalize(N(0, I_768)) × √768
x_t   = Normalize( GPT2_4L([x_0…x_{t-1}])[-1] + ε ) × √768
ε     ~ N(0, 0.05 × I_768)
```

**关键细节**：
- 缩放因子 √768：匹配 GPT-2 预训练激活尺度，防止 attention logits 趋零（Attention Temperature Collapse）
- 每步先加噪声再 Normalize 再缩放：保证幅值恒为 √768，不发散不坍缩
- 不加 position embedding：causal mask 已提供时序结构

**数据质量验证**（生成 1000 条后先检查，通过再生成完整数据集）：
- within-sample cos_sim（归一化后）< 0.05
- between-sample cos_sim < 0.05
- L2 norm 恒等于 √768
- Teacher attention weight 非均匀（非 1/t 平均注意力）

---

## 数据集配置

| 参数 | 值 |
|------|----|
| vector_dim | 768 |
| seq_length | 32（4 init + 28 main）|
| noise_scale | 0.05 |
| train_samples | 500k |
| val / test | 10k / 10k |
| OOD | train=positive_first，val/test=negative_first |
| 磁盘占用 | ~12.3 GB（memmap）|

---

## Student 模型

**架构**：ContinuousAOGPT（不变），更新参数：

| 参数 | v8 | v9 |
|------|----|----|
| vector_dim | 64 | 768 |
| n_layer | 4 | 5 |
| n_head | 4 | 12 |
| n_embd | 256 | 768 |
| 参数量 | ~4M | ~85M |

---

## Loss 函数

```python
# 替换原 cosine loss
loss = F.mse_loss(predictions, targets)

# 额外监控（仅 log，不纳入 loss）
pred_norm = predictions.norm(dim=-1).mean()  # 期望 ≈ √768 ≈ 27.7
```

---

## 实验框架

复用 v8 全部框架，三组对比：
- AR no-shuffle / AR shuffled / MDM
- init prefix conditioning（4 个 init 向量）
- eval_order_v8.py 评测 Kendall's τ（MDM 顺序发现能力）

---

## 实施步骤

1. `teacher_generator.py`：Teacher 自回归生成 + 数据质量可视化
2. `pregenerate_data.py`：通过验证后生成完整 memmap 数据集
3. `config.py`：更新超参（vector_dim=768，n_layer=5，n_head=12，n_embd=768）
4. `train_ar.py` / `train_mdm.py`：cosine loss → MSE loss，增加 pred_norm log
5. `eval_utils.py`：评估指标改为 MSE + pred_norm
6. 跑三组实验 → eval_order 评测
