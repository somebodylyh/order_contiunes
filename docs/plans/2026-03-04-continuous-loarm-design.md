# Continuous LO-ARM 设计文档

**日期：** 2026-03-04
**状态：** 已确认，待实现
**对应论文：** "Learning-Order Autoregressive Models with Application to Molecular Graph Generation" (2503.05979v2)

---

## 1. 背景与目标

### 现有 Baseline 状态

在 GLA-340M h-space（D=1024, L=31）上训练了三种模型：

| 模型 | causal advantage | 结论 |
|------|-----------------|------|
| AR no-shuffle | +0.3285 | oracle 上界 |
| AR block-shuffle | +0.1880 | 部分因果信号 |
| **MDM（AO-ARM，均匀 order prior）** | **≈ 0** | **未发现因果结构** |

**目标：** 将 MDM（AO-ARM）升级为 LO-ARM，加入可学习的 order-policy，期望 policy 能自动发现 h-space 的因果生成顺序（即 causal advantage 从 ≈0 提升至接近 AR no-shuffle 的水平）。

---

## 2. 架构设计

### 2.1 主模型：ContinuousLOARM（`baseline_continuous/continuous_loarm.py`）

在 `ContinuousAOGPT` backbone 上增加 order-policy 输出头（shared-torso 设计）：

```
ContinuousAOGPT backbone（5层 Transformer, n_embd=1024）
                    ↓
          output embedding [B, L, 1024]
                   / \
    generator_head   order_policy_head
    Linear(1024, D)   Linear(1024, L)
    （已有）           （新增）
         ↓                  ↓
   预测连续向量 x̂_{z_i}    L 个 logits → mask 已选 → softmax
   log p_θ = -||x - x̂||²/(2σ²)      p^z_θ(z_i | z_{<i}, x̄_{z_{<i}})
```

**关键设计细节：**
- `order_policy_head` 的输出是 position i 对应的 embedding 所预测的"下一步选哪个原始位置"
- 推理时对已选位置 logits 置 `-1e9`，再 softmax
- Backbone 权重与现有 `best_mdm_Random_model.pt` 可用于初始化（warm start）

### 2.2 摊销变分网络 $q_\theta$（`baseline_continuous/variational_q.py`）

```
输入：完整无打乱的真实 x [B, L, D]（训练时的上帝视角）
         ↓ mean pooling → [B, D]
         ↓ Linear(D, 4D) → GELU → Linear(4D, L)
         ↓ q_logits [B, L]
         → Gumbel-top-k → 完整排列 z [B, L]
```

**关键设计细节：**
- $q_\theta$ 只在**训练时**使用，推理时完全不需要
- 输入是**未打乱的完整 x**（包含所有位置的真实 h 向量），有上帝视角
- 轻量 MLP，参数量约 4M（相对 backbone 的 69M 可忽略）

---

## 3. 训练算法（Algorithm 1 的连续空间适配）

### 3.1 每个 batch 的计算步骤

```python
# ── 1. 采样两条完整排列 ──────────────────────────────────────────
q_logits = q_net(x)                          # [B, L]  (上帝视角)
z1 = gumbel_top_k(q_logits)                  # [B, L] 完整排列
z2 = gumbel_top_k(q_logits)                  # [B, L] 完整排列（独立采样）

# ── 2. 随机选一个步骤 ───────────────────────────────────────────
i ~ Uniform[1, L]

# ── 3. 两次 forward pass ─────────────────────────────────────────
# 输入：前 i-1 步已揭示，其余位置为零向量
pred1, pol_logits1 = model(x, orders=z1, reveal_steps=i)
pred2, pol_logits2 = model(x, orders=z2, reveal_steps=i)

# ── 4. 计算 F_θ = log p_gen + log p_policy - log q ───────────────
# 4a. 生成器 log-likelihood（Gaussian，σ² 为超参数）
log_p_gen1 = -|| x[z1[:,i]] - pred1 ||² / (2 * sigma²)   # [B]
log_p_gen2 = -|| x[z2[:,i]] - pred2 ||² / (2 * sigma²)   # [B]

# 4b. Policy log-prob（mask 已选位置）
pol_logits1_masked = mask_selected(pol_logits1, z1[:, :i])
log_p_pol1 = log_softmax(pol_logits1_masked)[z1[:, i]]    # [B]

# 4c. Variational log-prob
log_q1 = log_q(q_logits, z1, i)   # 参见 Gumbel-top-k 的解析式

F1 = log_p_gen1 + log_p_pol1 - log_q1   # [B]
F2 = (同上，z2)

# ── 5. RLOO 梯度（公式 11，关键 stop-gradient）──────────────────
delta_F = (F1 - F2).detach()   # ← ★ stop gradient
log_q_diff = log_q1_i - log_q2_i

loss = -0.5 * L * (F1 + F2) \
     + 0.5 * L * delta_F * log_q_diff

loss.mean().backward()
```

### 3.2 超参数约定

| 超参数 | 建议值 | 说明 |
|--------|--------|------|
| `sigma²` | 0.3 ~ 1.0 | 关键 temperature；小 → policy 保守；大 → 多探索 |
| RLOO samples | 2 | 论文默认 |
| `i` 采样 | Uniform[1, L] | 或按 L/(L-i+1) 加权（减小高步方差） |
| q_net lr | 1e-4 ~ 3e-4 | 与 backbone lr 分开设置 |
| Warm start | MDM checkpoint | backbone 用已训练的 MDM 初始化 |

---

## 4. Stop-Gradient 分析（关键工程细节）

```
梯度流向：
  loss 对 θ_gen（生成器）的梯度 ← F1, F2 的 log_p_gen 项（正常流）
  loss 对 θ_pol（order-policy）的梯度 ← F1, F2 的 log_p_pol 项（正常流）
                                      + delta_F.detach() × log_q_diff（RLOO control variate）
  loss 对 θ_q（变分网络）的梯度 ← delta_F.detach() × log_q_diff（REINFORCE 项）
                                  ← F1, F2 的 -log_q 项（ELBO 的 KL 部分）

  ★ delta_F 必须 detach，否则梯度从 control variate 回流到生成器，
    破坏 RLOO 的无偏性，导致训练崩溃
```

---

## 5. 推理流程（Algorithm 2）

```python
x_state = zeros([B, L, D])     # 全 mask 初始状态
revealed = []

for step in range(L):
    _, pol_logits = model(x_state, orders=revealed, reveal_steps=step)
    pol_logits[:, revealed] = -1e9

    # greedy（确定性推理）或 top-p sampling
    z_next = argmax(pol_logits, dim=-1)           # [B]

    # 生成该位置的值
    pred, _ = model(x_state, orders=revealed + [z_next], reveal_steps=step+1)
    x_state[:, z_next] = pred
    revealed.append(z_next)
```

---

## 6. 评估指标

在 `eval_order_v8.py` 框架基础上新增：

| 指标 | 计算方式 | 期望现象 |
|------|----------|----------|
| **Policy greedy Kendall τ** | LO-ARM 贪心顺序 vs [0,...,30] | 正值 → 发现因果顺序 |
| **Policy log-prob（causal）** | $\sum_i \log p^z(i\|z_{<i})$ under causal order | 高 → policy 偏好因果方向 |
| **Policy entropy per step** | H(p^z_θ) at each step | 随 step 增加而降低 → 策略越来越确定 |
| **Causal advantage（LO-ARM）** | random_loss - causal_loss | 对比 MDM（≈0）的提升 |
| **σ² ablation** | sigma=0.1/0.3/1.0/3.0 | 找到最优 temperature |

---

## 7. 新文件清单

```
baseline_continuous/
├── continuous_loarm.py    # 主模型：ContinuousAOGPT + order_policy_head
├── variational_q.py       # q_θ 摊销变分网络（轻量 MLP + Gumbel-top-k）
├── train_loarm.py         # 训练脚本（RLOO gradient）
└── eval_loarm.py          # 评估脚本（order recovery + policy 分析）
```

---

## 8. 风险与缓解

| 风险 | 缓解方案 |
|------|---------|
| RLOO 方差过高 | 调大 sigma²；引入 baseline（running mean of F）；限制 i 的采样范围 |
| Policy collapse（只学一条路径） | 加 entropy regularization：loss += -λ × H(p^z_θ) |
| q_θ 训练不稳 | 对 q_logits 做 temperature scaling；q_net lr 比 backbone 低 10x |
| 与 MDM 相比无改善 | 检查 sigma² 是否合理；检查 stop-gradient 是否正确 |

---

*设计确认：2026-03-04*
*下一步：writing-plans 生成实现计划*
