# LO-ARMs Block Experiments - Complete Summary

## 🎯 总体目标

验证 LO-ARMs (Learning Optimal ARMs) 系统能否在**没有任何图结构输入**的情况下，仅凭 Token 数值自动发现数据背后的**因果拓扑结构**。

---

## 📊 三大实验矩阵

| 实验 | 结构类型 | 节点数 | 核心挑战 | 状态 |
|------|---------|--------|---------|------|
| **1. Lossy Copy** | 有损复制 (X → Y) | 2 | 识别因果方向 | ✅ 完成 |
| **2. Causal Chain** | 链式 (A → B → C) | 3 | 多层级推理 | ✅ 完成 |
| **3. Diamond DAG** | 菱形 (Fork+Join) | 4 | 混合结构 | ✅ 完成 |

---

## 📁 目录结构

```
AO-GPT-MDM/
├── lossy_copy_exp/          # 实验 1: 有损复制
│   ├── lossy_copy_dataset.py
│   ├── config_lossy_copy.py
│   ├── train_loarms.py
│   ├── test_all.py
│   └── model_wrapper.py
│
├── causal_chain_exp/        # 实验 2: 因果链
│   ├── causal_chain_dataset.py
│   ├── config_chain.py
│   ├── train_chain.py
│   ├── test_chain_setup.py
│   ├── run_chain.sh
│   └── README.md
│
├── dag_exp/                 # 实验 3: 菱形 DAG
│   ├── dag_dataset.py
│   ├── config_dag.py
│   ├── train_dag.py
│   ├── test_dag_setup.py
│   ├── run_dag.sh
│   └── README.md
│
└── EXPERIMENTS_SUMMARY.md   # 本文档
```

---

## 🧪 实验 1: Lossy Copy (X → Y)

### 结构
```
X (Root) ──有损──> Y (Lossy Copy)
   Y = X // k
```

### 核心洞察
- **X 具有完整信息**：知道 X 就能确定 Y
- **Y 信息有损**：知道 Y 无法唯一确定 X（k:1 的映射）

### 预期结果
- `P(select_x_first)` → **~100%**（X 是信息源）
- `P(select_y_first)` → **~0%**（Y 是结果）

### 训练命令
```bash
/home/admin/anaconda3/envs/order_lando/bin/python lossy_copy_exp/train_loarms.py
```

### WandB Project
`LO-ARMs-LossyCopy`

---

## 🧪 实验 2: Causal Chain (A → B → C)

### 结构
```
A (Root) ──> B (Middle) ──> C (Leaf)
         B = A // 2    C = B // 2
```

### 核心洞察
- **A 是源头**：完整信息，能确定整条链
- **B 是中间节点**：部分信息，能确定 C，但不能反推 A
- **C 是叶子**：最少信息，什么都推不出

### 预期结果
- `prob_select_root_first (A)` → **~100%**
- `prob_select_mid_first (B)` → **~0%**
- `prob_select_leaf_first (C)` → **~0%**

### 关键观察
**C 应该最先掉队** → B 其次 → A 最后胜出（反映信息层级！）

### 训练命令
```bash
/home/admin/anaconda3/envs/order_lando/bin/python causal_chain_exp/train_chain.py
```

### WandB Project
`LO-ARMs-CausalChain`

---

## 🧪 实验 3: Diamond DAG (Fork + Join)

### 结构
```
       x0 (Root)
      /    \
    x1      x2
  (Br A)  (Br B)
      \    /
       x3 (Sink)
```

### 数学定义
- `x0 = Uniform(0, 63)` (Root)
- `x1 = x0 // 2` (Branch A)
- `x2 = (x0 + 1) // 2` (Branch B)
- `x3 = (x1 + x2) % 16` (Sink)

### 核心洞察
- **x0 是唯一根节点**：必须第一个生成
- **x1 和 x2 是对称分支**：可以任意顺序
- **x3 是汇聚点**：必须最后生成

### 预期结果
- `first_step/p_x0` → **>95%**（Root 第一）
- `last_step/p_x3` → **>95%**（Sink 最后）
- `second_step/p_any_branch` → **>95%**（Branch 在中间）
- `topology/correct` → **>90%**（整体拓扑正确）

### 训练命令
```bash
/home/admin/anaconda3/envs/order_lando/bin/python dag_exp/train_dag.py
```

### WandB Project
`LO-ARMs-DAG`

---

## 🎓 三个实验的递进关系

### 复杂度递增

```
实验 1 (Lossy Copy)
    简单的 X → Y 因果
    ↓
实验 2 (Causal Chain)
    多层级 A → B → C 推理
    ↓
实验 3 (Diamond DAG)
    混合 Fork/Join 结构
```

### 能力验证

| 能力 | 实验 1 | 实验 2 | 实验 3 |
|------|--------|--------|--------|
| 识别因果方向 | ✓ | ✓ | ✓ |
| 多层级推理 | - | ✓ | ✓ |
| 识别根节点 | ✓ | ✓ | ✓ |
| 识别汇聚点 | - | - | ✓ |
| 处理分支 | - | - | ✓ |
| 对称性识别 | - | - | ✓ |

---

## 🚀 快速启动指南

### 环境设置

所有实验使用相同的 conda 环境：
```bash
# 使用 order_lando 环境（已安装 PyTorch 2.10.0）
/home/admin/anaconda3/envs/order_lando/bin/python
```

### 测试实验设置

```bash
# 测试实验 2
/home/admin/anaconda3/envs/order_lando/bin/python causal_chain_exp/test_chain_setup.py

# 测试实验 3
/home/admin/anaconda3/envs/order_lando/bin/python dag_exp/test_dag_setup.py
```

### 运行训练

```bash
# 实验 1（如果还没运行）
/home/admin/anaconda3/envs/order_lando/bin/python lossy_copy_exp/train_loarms.py

# 实验 2
/home/admin/anaconda3/envs/order_lando/bin/python causal_chain_exp/train_chain.py

# 实验 3
/home/admin/anaconda3/envs/order_lando/bin/python dag_exp/train_dag.py
```

### 并行运行（推荐）

如果有多个 GPU，可以同时运行多个实验：
```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python causal_chain_exp/train_chain.py

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python dag_exp/train_dag.py
```

---

## 📈 关键监控指标

### 实验 1 (Lossy Copy)
- `probs/select_x_first` → 1.0
- `accuracy` → >80%

### 实验 2 (Causal Chain)
- `probs/select_root_first` → 1.0
- `probs/select_mid_first` → 0.0
- `probs/select_leaf_first` → 0.0
- `probs/select_b_second|a_first` → 1.0

### 实验 3 (Diamond DAG)
- `first_step/p_x0` → 1.0
- `last_step/p_x3` → 1.0
- `second_step/p_any_branch` → 1.0
- `topology/correct` → 1.0

---

## 📊 理论最优 Loss

### 实验 1 (2 节点)
```
平均 Loss ≈ ln(vocab_size) / 2 ≈ 4.16 / 2 ≈ 2.08
```

### 实验 2 (3 节点)
```
平均 Loss ≈ ln(vocab_size) / 3 ≈ 4.16 / 3 ≈ 1.39
```

### 实验 3 (4 节点)
```
平均 Loss ≈ ln(vocab_size) / 4 ≈ 4.16 / 4 ≈ 1.04
```

**关键洞察**：节点越多，Agent 能学到的节省越多！

---

## 🎯 成功标准

### 实验 2 (Causal Chain)
✅ `prob_select_root_first` > 90%
✅ `prob_select_mid_first` < 10%
✅ `prob_select_leaf_first` < 10%
✅ `accuracy` > 80%

### 实验 3 (Diamond DAG)
✅ `first_step/p_x0` > 90%
✅ `last_step/p_x3` > 90%
✅ `second_step/p_any_branch` > 90%
✅ `topology/correct` > 90%
✅ `accuracy` > 75%

---

## 🔬 论文贡献点

### 核心 Novelty

1. **自动拓扑发现** - Agent 无需图结构输入，仅凭数值就能反推 DAG
2. **因果推理能力** - 识别因果方向、多层级依赖、汇聚/分支结构
3. **理论最优收敛** - 收敛到理论最优 Loss（仅第一个节点有熵）

### 实验支撑

| 论文 Claim | 实验支撑 |
|-----------|----------|
| 识别因果方向 | 实验 1 |
| 多层级推理 | 实验 2 |
| 复杂 DAG 处理 | 实验 3 |
| 拓扑排序学习 | 实验 2, 3 |
| 对称性识别 | 实验 3 |

---

## 📝 实验检查清单

### 实验 2 (Causal Chain)
- [x] Dataset 创建
- [x] Config 配置
- [x] Train 脚本
- [x] Test 脚本
- [x] README 文档
- [x] 环境测试通过
- [ ] 训练运行
- [ ] 结果分析

### 实验 3 (Diamond DAG)
- [x] Dataset 创建
- [x] Config 配置
- [x] Train 脚本
- [x] Test 脚本
- [x] README 文档
- [x] 环境测试通过
- [ ] 训练运行
- [ ] 结果分析

---

## 🐛 调试指南

### 如果曲线不收敛

1. **增加训练步数**
   - Causal Chain: `max_iters = 10000`
   - Diamond DAG: `max_iters = 15000`

2. **调整学习率**
   ```python
   agent_learning_rate = 5e-5  # 降低 Agent 学习率
   ```

3. **增加模型容量**
   ```python
   n_layer = 4
   n_embd = 256
   ```

### 如果随机选择（所有概率相等）

- 增加 warmup 步数
- 检查奖励信号是否正确
- 确认梯度正常传播

---

## 📊 WandB 仪表板

所有实验的结果将上传到 WandB：

- **实验 1**: `LO-ARMs-LossyCopy`
- **实验 2**: `LO-ARMs-CausalChain`
- **实验 3**: `LO-ARMs-DAG`

关键曲线：
- 选择概率随时间的演变
- 模型损失和准确率
- 每个节点的准确率
- 拓扑正确性（实验 3）

---

## 🎉 完成后的工作

1. **结果分析**
   - 对比三个实验的收敛速度
   - 绘制关键曲线图
   - 验证理论最优 Loss

2. **论文撰写**
   - 强调自动拓扑发现能力
   - 展示三个实验的递进关系
   - 讨论失败案例和限制

3. **扩展实验（可选）**
   - 更大的 vocab size
   - 更多节点的 DAG
   - 树形结构
   - 更复杂的汇聚模式

---

## 📚 相关文件

- `lossy_copy_exp/README.md` - 实验 1 详细文档（待创建）
- `causal_chain_exp/README.md` - 实验 2 详细文档
- `dag_exp/README.md` - 实验 3 详细文档

---

**项目**: LO-ARMs (Learning Optimal ARMs)
**作者**: LO-ARMs Project Team
**日期**: 2026-01-26
**状态**: ✅ 所有实验准备完毕，可以开始训练

---

## 🚦 下一步行动

1. **立即执行**：运行实验 2 和 3 的测试脚本（已完成 ✅）
2. **今天完成**：启动实验 2 (Causal Chain) 的训练
3. **明天完成**：启动实验 3 (Diamond DAG) 的训练
4. **本周完成**：收集所有结果，开始论文撰写

**祝实验成功！🎉**
