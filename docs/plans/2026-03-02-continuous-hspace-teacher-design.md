# 连续 h-space AR Teacher 设计文档

**日期**：2026-03-02
**状态**：已批准，待实现

---

## 背景与动机

### 当前问题

现有实验（wikitext-103 + GLA 4层提取 hidden states）的理论下界无法达到：

- 名义下界：σ² = 0.09（加性高斯噪声方差）
- 实际 test loss：1.794（上帝模型）
- 根本原因：`h_{t+1} = f(h_t, x_{t+1})`，其中 `x_{t+1}` 是真实文本的下一个 token，学生完全看不到，导致不可约语言不确定性 ≈ 1.82

### 核心洞察

如果用 Teacher 自身自回归生成数据，则：
- 学生见到 `h_{0:t-1}` → 能完全还原 Teacher 的确定性输出 `μ_t`
- 唯一的不确定性来源是注入的高斯噪声 `ε_t`
- **理论下界 = σ² 变为真正可达的下界**

---

## 数学定义

### 数据生成过程（Teacher）

```
h_0 = Normalize(N(0, I_D)) × √D        # 归一化随机初始状态，norm ≡ √D

对 t = 1, ..., L-1：
  x_{t-1} = Normalize(h_{t-1}) × √D   # 归一化后作为 GLA 输入（保持流形稳定）
  μ_t = GLA_4L(inputs_embeds=[x_0,..,x_{t-1}])[:, -1, :]   # 确定性输出
  ε_t ~ N(0, σ² I_D)
  h_t = μ_t + ε_t                      # 含噪目标（未归一化）
```

**注**：由于 `h_0` 本身已归一化，`x_0 = h_0`，`Normalize(h_0) × √D = h_0`，无需特殊处理。

### 信噪比分析

GLA 输出 `μ_t` 经过内部 LayerNorm，L2 范数 ≈ √D，**每维方差 ≈ 1**（总能量 D 均匀分布在 D 个维度）。

| 量 | 值 |
|---|---|
| 信号（μ_t）每维方差 | ≈ 1 |
| 噪声（ε_t）每维方差 | σ² |
| Noise-to-Signal Ratio | σ² |
| σ = 0.3 时 NSR | 9%（信号主导） |

**关键错误避免**：σ 是绝对值（每维），不应与 √D 成比例。若设 σ = 0.1 × √D ≈ 3.2，则 σ² ≈ 10.24，NSR = 1024%，信号被彻底淹没。

### 理论下界

当 Student 容量足够，可精确逼近确定性分量 `μ_t`：

$$\mathcal{L}_{\min} = \mathbb{E}[||\hat{h}_t - h_t||^2] = \mathbb{E}[||\mu_t - (\mu_t + \varepsilon_t)||^2] = D \sigma^2$$

**每维平均下界**（实验 loss 的实际对比量）：

$$\mathcal{L}_{\min}^{\text{per-dim}} = \sigma^2 = 0.09$$

与原始 wikitext 实验相同的 σ，但现在这个下界是**真正可达的**。

---

## 方案选型依据

| 方案 | 数值稳定 | 下界可达 | 实现复杂度 |
|---|---|---|---|
| A：纯加性噪声，不归一化 | ❌ 尺度发散 | ✅ | 低 |
| B：归一化输入 + 未归一化目标 | ✅ | ✅ | 中 |
| C：真实 embedding 起步 | ❌ 同 A | ❌ 混合分析 | 高 |

**选择方案 B** 的核心设计决策：
- 归一化限制在"特征传递"环节（`x_t = Normalize(h_t) × √D`）
- 预测目标保留未归一化的 `h_t`（含噪，student 的 MSE 直接等于 σ²）
- 两者解耦，稳定性与理论精确性同时满足

---

## 工程设计

### 数据存储格式

```
{cache_path}/{hash}/
  hidden.pt   Tensor [N, L, D]
                hidden[:, 0, :]   = h_0  (归一化初始状态，norm ≡ √D)
                hidden[:, t, :]   = h_t  (含噪未归一化态，t=1..L-1)
  meta.json   {"D", "L", "N_train", "N_test",
               "model_name", "layer_idx",
               "generation_mode": "continuous_h",
               "sigma": 0.3, "sigma_0": 1.0}
  perm.pt     保留（可选，用于未来 shuffled 变体）
```

### 数据流

```
[teachers.py - ContinuousHSpaceTeacher]
  bypass embed_tokens → 直接 inputs_embeds=[x_0..x_{t-1}]
  逐步生成：loop t=1..L-1，每步一次 GLA forward

[generate_data.py]
  generation_mode == "continuous_h" 分支
  不加载 wikitext，直接生成 N 条序列

[exp_dataset.py - HiddenStateDataset]
  模式 "continuous_ar":
    input  = Normalize(hidden[:, :-1, :]) × √D   # 归一化，作为 student 输入
    target = hidden[:, 1:, :]                      # 未归一化，作为 student 预测目标

[train.py]
  不需要改动，loss = MSE(pred, target) 结构不变
```

### 需要修改的文件

| 文件 | 改动摘要 |
|---|---|
| `gla_exp/teachers.py` | 新增 `ContinuousHSpaceTeacher`：bypass embedding，逐步生成 |
| `gla_exp/generate_data.py` | 新增 `continuous_h` 生成分支 |
| `gla_exp/exp_config.py` | 新增 `sigma`、`generation_mode` 字段 |
| `gla_exp/exp_dataset.py` | 新增 `continuous_ar` 模式 |
| `gla_exp/train.py` | **无需改动** |

### ContinuousHSpaceTeacher 伪代码

```python
class ContinuousHSpaceTeacher(nn.Module):
    def __init__(self, cfg):
        # 加载 GLA-340M，截断到前 layer_idx+1 层（同 FLATeacher）
        # 不加载 embed_tokens 以外的任何东西
        # 在最后一层注册 forward hook 获取 μ_t

    @torch.no_grad()
    def generate_sequence(self, B, L, sigma, device):
        D = self.d_hidden
        # h_0: 归一化到 √D
        h0 = torch.randn(B, D, device=device)
        h0 = F.normalize(h0, dim=-1) * math.sqrt(D)

        seq = [h0]  # 存储 h_0..h_{L-1}

        for t in range(1, L):
            # 构建输入：归一化历史序列
            x_hist = torch.stack([
                F.normalize(h, dim=-1) * math.sqrt(D)
                for h in seq
            ], dim=1)  # (B, t, D)

            # GLA forward（bypass embedding）
            self._hidden_buf = None
            self.model(inputs_embeds=x_hist)
            mu_t = self._hidden_buf[:, -1, :]  # (B, D)

            # 加噪
            eps = torch.randn_like(mu_t) * sigma
            h_t = mu_t + eps
            seq.append(h_t)

        return torch.stack(seq, dim=1)  # (B, L, D)
```

**效率说明**：逐步生成是序列依赖的，无法完全并行化。对 L=32，每条序列需 31 次 GLA forward，批处理 B 条序列同时计算。对 N=110k、B=256，约需 430 批 × 31 步 ≈ 13,330 次 forward，可在 GPU 上接受（约 10-30 分钟）。

若需加速，可利用 FLA 的 recurrent 模式（`use_cache=True`）将复杂度从 O(L²) 降至 O(L)。

---

## 关键超参数

| 参数 | 值 | 说明 |
|---|---|---|
| `sigma` | **0.3** | σ² = 0.09，NSR = 9%，与 wikitext 实验相同，可直接对比 |
| `sigma_0` | 1.0（→ Normalize → √D） | 初始 norm ≡ √D，与 GLA 预训练期望一致 |
| `seq_len L` | 32 | 31 个预测步，不变 |
| `layer_idx` | 3 | 使用前 4 层，不变 |
| `N_train` | 100k | 不变 |
| `N_test` | 10k | 不变 |

---

## 预期实验结果

| 预测器 | 预期 MSE |
|---|---|
| 零预测器 | ≈ D（≈ 1024，per-dim ≈ 1） |
| 均值预测器 | ≈ 1（h_t 各维均值 ≈ 0，方差 ≈ 1） |
| Student（理想） | → σ² = 0.09 |
| 理论下界 | **σ² = 0.09（可达）** |

与 wikitext 实验对比：两者下界相同（σ² = 0.09），但 wikitext 实验实际 loss = 1.794（下界不可达），本实验 loss 应收敛到 0.09（可达），直观验证数据生成方案的根本差异。

---

## 与原设计的对比

| 维度 | 原设计（wikitext + GLA） | 本设计（h-space AR） |
|---|---|---|
| 数据来源 | wikitext-103 真实文本 | Teacher 自生成 |
| h_{t+1} 的不确定性来源 | 真实语言（x_{t+1} 未知） | 注入高斯噪声（ε_t） |
| 理论下界 | σ² = 0.09（名义，不可达） | σ² = 0.09（真实，可达）|
| 实际 test loss | 1.794 | 预期 ≈ 0.09 |
| 数值稳定性 | 无问题（离散 token） | 方案 B 通过归一化保证 |
