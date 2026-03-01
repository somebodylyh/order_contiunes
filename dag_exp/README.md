# Diamond DAG Experiment - The Ultimate Test

## 🎯 实验目标

这是 LO-ARMs 验证的**终极实验**，目标是测试 Agent 能否在没有任何图结构输入的情况下，仅凭 Token 数值**反向破解出数据背后的 DAG 拓扑结构**。

### 前置成果回顾

- ✅ **实验 1 (Modular Sum)**: Agent 学会了"多对一"汇聚结构
- ✅ **实验 2 (Causal Chain)**: Agent 学会了 A→B→C 链式结构，收敛到理论最优 Loss

### 本实验目标

验证 Agent 能否处理**混合了"分支"(Fork)和"汇聚"(Join)的复杂 DAG**。

---

## 📊 DAG 结构

### Diamond Graph (菱形图)

```
       x0 (Root)
      /    \
     /      \
   x1        x2
 (Branch A) (Branch B)
     \      /
      \    /
       x3 (Sink)
```

### 数学定义

**4 节点菱形结构**：

- **Node 0 (Root)**: `x0 ~ Uniform(0, vocab_size-1)`
- **Node 1 (Branch A)**: `x1 = x0 // 2`
- **Node 2 (Branch B)**: `x2 = (x0 + 1) // 2`
- **Node 3 (Sink)**: `x3 = (x1 + x2) % 16`

### 因果依赖

- `x0 → x1`（有损，单向）
- `x0 → x2`（有损，单向）
- `x1, x2 → x3`（汇聚）

### 预期拓扑排序

- Agent **必须**先生成 `x0`（唯一的 Root）
- 然后可以自由选择 `x1` 或 `x2`（两个 Branch 地位对等）
- **最后**必须生成 `x3`（Sink，依赖于两个 Branch）

---

## 📁 文件结构

```
dag_exp/
├── README.md              # 本文档
├── dag_dataset.py         # ✅ Diamond DAG 数据集
├── config_dag.py          # ✅ 实验配置
├── train_dag.py           # ✅ 训练脚本
├── test_dag_setup.py      # ✅ 设置测试脚本
├── run_dag.sh             # ✅ 启动脚本
└── checkpoints/           # (训练时自动创建)
```

---

## 🚀 快速开始

### 1. 测试环境设置

```bash
# 使用 order_lando 环境
/home/admin/anaconda3/envs/order_lando/bin/python dag_exp/test_dag_setup.py
```

### 2. 运行训练

```bash
# 直接运行训练
/home/admin/anaconda3/envs/order_lando/bin/python dag_exp/train_dag.py
```

### 3. 监控训练

训练将自动上传到 **WandB**：
- Project: `LO-ARMs-DAG`
- Run Name: `diamond_dag_v64`

---

## 📈 关键指标

### 第一步（必须选 Root）

| 指标 | 初始值 | 目标值 | 说明 |
|------|--------|--------|------|
| `first_step/p_x0` | ~25% | **>95%** | x0 是根节点，必须第一个选 |
| `first_step/p_x1` | ~25% | **<5%** | x1 是分支，不能第一个选 |
| `first_step/p_x2` | ~25% | **<5%** | x2 是分支，不能第一个选 |
| `first_step/p_x3` | ~25% | **<5%** | x3 是汇聚点，绝对不能第一个选 |

### 第二步（应该选某个分支）

| 指标 | 目标值 | 说明 |
|------|--------|------|
| `second_step/p_any_branch` | **>95%** | 应该选 x1 或 x2 中的一个 |
| `second_step/p_x0` | **~0%** | x0 已被选 |
| `second_step/p_x3` | **~0%** | x3 必须最后选 |

### 最后一步（必须选 Sink）

| 指标 | 初始值 | 目标值 | 说明 |
|------|--------|--------|------|
| `last_step/p_x3` | ~25% | **>95%** | x3 是汇聚点，必须最后选 |

### 拓扑正确性（综合指标）

| 指标 | 目标值 | 说明 |
|------|--------|------|
| `topology/correct` | **>90%** | x0 第一 AND x3 最后 |

---

## 🎨 预期曲线演变

### First Step Probabilities

```
P(select)
  100% ┤              x0 逐渐上升 ─────▲──────
       │             ╱
   75% ┤            ╱
       │           ╱
   50% ┤          ╱
       │         ╱ x1, x2, x3 下降
   25% ┼────────●──────────────╲─────
       │                        ╲
       │                         ╲____
    0% ┤──────────────────────────●───▶ Steps
       0      1500    3000    8000  12000
```

### Branch Symmetry (Second Step)

由于 x1 和 x2 在拓扑上地位对等：
- `second_step/p_x1` + `second_step/p_x2` ≈ **100%**
- Agent 可能随机偏好其中一个（但两者之和应接近 100%）

---

## ⚙️ 配置参数

### 数据集
- `vocab_size = 64` - x0 的取值范围 [0, 64)
- `seq_length = 4` - 序列长度 [x0, x1, x2, x3]
- `num_train_samples = 10000` - 训练样本数

### 模型（稍大容量）
- `n_layer = 3` - Transformer 层数（比之前多一层）
- `n_head = 2` - 注意力头数
- `n_embd = 128` - 嵌入维度

### 训练
- `warmup_steps = 1500` - 预热步数
- `max_iters = 12000` - 总训练步数（DAG 更复杂）
- `learning_rate = 1e-3` - 模型学习率
- `agent_learning_rate = 1e-4` - Agent 学习率

---

## 📊 理论最优 Loss 分析

假设 Agent 学会了正确的拓扑排序 `x0 → {x1, x2} → x3`：

1. **Step 0 (Gen x0)**: 盲猜 → Loss ≈ `ln(64) ≈ 4.16`
2. **Step 1 (Gen x1 or x2)**: 已知 x0，确定性 → Loss ≈ 0
3. **Step 2 (Gen 另一个 Branch)**: 已知 x0，确定性 → Loss ≈ 0
4. **Step 3 (Gen x3)**: 已知 x1, x2，确定性 → Loss ≈ 0

**平均 Loss**: `4.16 / 4 ≈ 1.04`

如果模型准确率达到 ~80-90%，实际 Loss 可能在 `1.2-1.5` 之间。

---

## 🏆 成功标准

训练成功的标志：

✅ `first_step/p_x0` > 90%（Root 第一）
✅ `last_step/p_x3` > 90%（Sink 最后）
✅ `second_step/p_any_branch` > 90%（Branch 在中间）
✅ `topology/correct` > 90%（整体拓扑正确）
✅ `accuracy` > 75%（模型准确率）

---

## 🔬 训练阶段

### Phase A: Warmup (Steps 0-1500)
- **目标**：预热模型，使其能够基本理解数据
- **策略**：使用随机顺序训练模型，Agent 冻结
- **监控**：模型损失和准确率

### Phase B: Co-evolution (Steps 1500-12000)
- **目标**：Agent 和模型共同进化，发现最优拓扑
- **策略**：REINFORCE 算法优化 Agent，监督学习优化模型
- **监控**：选择概率、拓扑正确性、奖励、模型性能

---

## 🐛 调试建议

### 如果 Agent 不收敛

1. **增加训练步数**
   ```python
   max_iters = 15000  # 在 config_dag.py 中修改
   warmup_steps = 2000
   ```

2. **调整 Agent 学习率**
   ```python
   agent_learning_rate = 5e-5  # 降低学习率
   ```

3. **增加模型容量**
   ```python
   n_layer = 4
   n_head = 4
   n_embd = 256
   policy_dim = 256
   ```

### 如果所有位置都是 25%（随机）

- 检查奖励信号是否正确
- 增加 warmup 步数，确保模型先学会基本预测
- 检查 Agent 梯度是否正常传播

### 如果曲线不稳定

- 启用 baseline: `use_baseline = True`
- 降低 Agent 学习率
- 增加 batch size

---

## 💡 关键洞察

### 信息层级

- **x0 (Root)**: 完整信息，能确定整个 DAG
- **x1, x2 (Branches)**: 部分信息，能确定 x3，但不能反推 x0
- **x3 (Sink)**: 最少信息，不能确定任何父节点

### 拓扑对称性

x1 和 x2 在拓扑上是对称的（都在深度 1），所以：
- Agent 可能随机偏好其中一个
- 或者学会 50/50 的概率分布
- 两种情况都是正确的！

### DAG 特性

这个实验展示了 Agent 能够：
1. **识别根节点** - 必须先生成 x0
2. **识别汇聚点** - 必须最后生成 x3
3. **处理对称分支** - x1 和 x2 可以任意顺序
4. **学会拓扑排序** - 遵守依赖关系

---

## 🎓 实验意义

这个实验的成功将证明：

1. **Topology Discovery** - Agent 无需任何图结构输入，仅凭 Token 数值就能反向破解出数据背后的生成拓扑图
2. **Mixed Structure Handling** - 能同时处理"分支"(Fork) 和"汇聚"(Join) 结构
3. **Topological Sort** - 自动学会拓扑排序 —— Root 优先，Sink 最后
4. **Symmetry Recognition** - 正确识别对等节点（两个 Branch）的对称性

**这是验证 LO-ARMs 核心 Novelty 的最后一块拼图。**

---

## 📚 相关实验

- `lossy_copy_exp/` - Lossy Copy 实验（两变量因果，实验 1）
- `causal_chain_exp/` - Causal Chain 实验（三变量链式，实验 2）

---

## 🎯 下一步

实验完成后：

1. 分析 WandB 曲线，验证上述预期
2. 对比三个实验的收敛速度和最终性能
3. 撰写论文：重点强调 **自动拓扑发现** 能力
4. 探索更复杂的 DAG（更多节点、更深层级）

---

**作者**: LO-ARMs Project
**日期**: 2026-01-26
**状态**: ✅ 准备就绪，可以开始训练
**难度**: ⭐⭐⭐⭐⭐ (Ultimate Test)
