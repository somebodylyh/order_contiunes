# Causal Chain Experiment (A → B → C)

## 实验目标

测试 LO-ARMs Agent 能否自动发现**多层级因果链条** `A → B → C`。

### 因果结构

```
A (Root)  →  B = A // 2  →  C = B // 2
   ↓             ↓              ↓
源头节点      中间节点        叶子节点
完整信息      部分信息        最少信息
```

**关键洞察**：整数除法的不可逆性建立单向因果链
- 知道 A → 可以确定 B 和 C（完整信息）
- 知道 B → 只能确定 C，无法反推 A（部分信息）
- 知道 C → 什么都推不出来（最少信息）

**预期结果**：Agent 应该学会 **"先 A，再 B，最后 C"** 的生成顺序

---

## 文件结构

```
causal_chain_exp/
├── README.md                  # 本文档
├── causal_chain_dataset.py    # ✅ 因果链数据集
├── config_chain.py            # ✅ 实验配置
├── train_chain.py             # ✅ 训练脚本
├── test_chain_setup.py        # ✅ 设置测试脚本
├── run_chain.sh               # ✅ 启动脚本
└── checkpoints/               # (训练时自动创建)
```

---

## 快速开始

### 1. 测试环境设置

```bash
# 使用 order_lando 环境
/home/admin/anaconda3/envs/order_lando/bin/python causal_chain_exp/test_chain_setup.py
```

✅ 已验证：所有测试通过

### 2. 运行训练

**方法 1：使用启动脚本**

```bash
/home/admin/anaconda3/envs/order_lando/bin/python causal_chain_exp/train_chain.py
```

**方法 2：修改 run_chain.sh 并运行**

```bash
# 编辑 run_chain.sh，将第一行改为：
# #!/home/admin/anaconda3/envs/order_lando/bin/python

bash causal_chain_exp/run_chain.sh
```

### 3. 监控训练

训练将自动上传到 **WandB**：
- Project: `LO-ARMs-CausalChain`
- Run Name: `chain_A_B_C_v64`

---

## 关键指标

### 第一步选择概率（核心指标）

| 指标 | 初始值 | 目标值 | 说明 |
|------|--------|--------|------|
| `prob_select_root_first` (A) | ~33% | **>95%** | A 是根节点，应该最先选择 |
| `prob_select_mid_first` (B) | ~33% | **<5%** | B 是中间节点，不应先选 |
| `prob_select_leaf_first` (C) | ~33% | **<5%** | C 是叶子节点，不应先选 |

### 条件概率（验证完整顺序）

| 指标 | 目标值 | 说明 |
|------|--------|------|
| `prob_select_b_second\|a_first` | **>90%** | 选了 A 之后，第二步应该选 B |

### 预期曲线演变（The "Money Plot"）

```
P(select)
  100% ┤           A 逐渐上升 ──────▲─────
       │          ╱
   67% ┤         ╱
       │        ╱  B 缓慢下降
   33% ┼───────●────────────╲──────
       │                     ╲
       │   C 最先掉队         ╲____
    0% ┤─────────────────────────●──────▶ Steps
       0     500    1000   3000   8000
```

**关键观察点**：
1. **C 应该最先掉队** - 因为 C 的信息量最小
2. **B 比 C 坚持更久** - 因为 B 至少能确定 C
3. **A 最终称王** - 因为 A 能确定整条链

---

## 配置参数

### 数据集
- `vocab_size = 64` - A 的取值范围 [0, 64)
- `seq_length = 3` - 序列长度 [A, B, C]
- `num_train_samples = 10000` - 训练样本数

### 模型（Tiny）
- `n_layer = 2` - Transformer 层数
- `n_head = 2` - 注意力头数
- `n_embd = 128` - 嵌入维度

### 训练
- `warmup_steps = 1000` - 预热步数（随机顺序）
- `max_iters = 8000` - 总训练步数
- `learning_rate = 1e-3` - 模型学习率
- `agent_learning_rate = 1e-4` - Agent 学习率

---

## 训练阶段

### Phase A: Warmup (Steps 0-1000)
- **目标**：预热模型，使其能够基本理解数据
- **策略**：使用随机顺序训练模型，Agent 冻结
- **监控**：模型损失和准确率

### Phase B: Co-evolution (Steps 1000-8000)
- **目标**：Agent 和模型共同进化，发现最优顺序
- **策略**：REINFORCE 算法优化 Agent，监督学习优化模型
- **监控**：选择概率、奖励、模型性能

---

## 成功标准

训练成功的标志：

✅ `prob_select_root_first` > 90%
✅ `prob_select_mid_first` < 10%
✅ `prob_select_leaf_first` < 10%
✅ `prob_select_b_second|a_first` > 90%
✅ `accuracy` > 80%

---

## 调试建议

### 如果 Agent 不收敛

1. **增加训练步数**
   ```python
   max_iters = 12000  # 在 config_chain.py 中修改
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
   ```

### 如果三个位置都是 33%（随机）

- 检查奖励信号是否正确
- 增加 warmup 步数，确保模型先学会基本预测

### 如果曲线不稳定

- 启用 baseline: `use_baseline = True`
- 降低 Agent 学习率

---

## 实验意义

这个实验的成功将证明：

1. **Multi-hop Reasoning** - Agent 能在多层因果网络中找到**信息量最大的根节点**
2. **Hierarchical Structure Discovery** - Agent 自动发现层级结构，为复杂因果图奠定基础
3. **论文 Claim 2** - Agent 具备深度推理能力，能自动发现层级结构

---

## 下一步

实验完成后：

1. 分析 WandB 曲线，验证上述预期
2. 导出关键曲线图用于论文
3. 尝试更复杂的因果结构（4个节点、树形、DAG）

---

## 相关实验

- `lossy_copy_exp/` - Lossy Copy 实验（两变量因果）
- `modular_sum_exp/` - Modular Sum 实验（独立变量）

---

**作者**: LO-ARMs Project
**日期**: 2026-01-26
**状态**: ✅ 准备就绪，可以开始训练
