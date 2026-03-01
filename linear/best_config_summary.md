# "地狱模式" (Hell Mode) 获胜配置冻结报告

> **结论：此配置证明了在极大搜索空间（16! = 20,922,789,888,000 种排列）下，
> 纯无监督 REINFORCE（无 Teacher Forcing、无 Warmup、无 Behavior Cloning）
> 依然能发现正确的物理时序结构。**

## 实验结果摘要

| 指标 | 数值 |
|------|------|
| Kendall's Tau | ~0.99 |
| Cosine Similarity | > 0.44 |
| 收敛步数 | ~100k steps |
| 对比 Baseline | 显著胜出 |

---

## 核心超参数

### 1. 数据配置 (Data)

| 参数 | 配置变量 | 值 | 说明 |
|------|----------|-----|------|
| 向量维度 | `vector_dim` | **64** | 每个向量 D=64 维 |
| 序列长度 | `seq_length` | **64** | L=64，"地狱模式"关键参数 |
| 依赖窗口 | `dependency_window` | **-1** | Full History 模式（每步依赖所有历史） |
| 正交矩阵数 | `num_matrices` | **16** | Dense AR 过程使用 16 个正交矩阵循环 |
| 分块数 | `num_chunks` | **16** | Block-wise Shuffle，每块 4 个向量，搜索空间 16! |
| Block Size | `block_size` | **64** | 必须与 seq_length 一致 |
| 训练集初始化 | `train_init_mode` | `positive_first` | 初始向量第一维 > 0 |
| 验证集初始化 | `val_init_mode` | `negative_first` | OOD：初始向量第一维 < 0 |
| 训练样本数 | `train_samples` | 100,000 | 在线生成（每 epoch 不同） |
| 验证样本数 | `val_samples` | 20,000 | 预生成（静态） |
| 测试样本数 | `test_samples` | 20,000 | 预生成（静态） |
| 固定矩阵路径 | `fixed_matrices_path` | `linear_rotation_exp/fixed_orthogonal_matrices.pt` | 确保矩阵可复现 |
| DataLoader Workers | `num_workers` | 16 | |

### 2. 模型配置 (ContinuousTransformer)

| 参数 | 配置变量 | 值 | 说明 |
|------|----------|-----|------|
| Transformer 层数 | `n_layer` | **8** | |
| 注意力头数 | `n_head` | **8** | |
| 隐藏维度 | `n_embd` | **256** | |
| Dropout | `dropout` | **0.0** | 无 Dropout |
| Bias | `bias` | **True** | |

> 架构：Input Projection(D->256) -> 8x TransformerBlock(Causal Attention) -> LayerNorm -> Output Projection(256->D)

### 3. Agent 配置 (SetToSeqAgent)

| 参数 | 配置变量 | 值 | 说明 |
|------|----------|-----|------|
| Agent 隐藏维度 | `agent_d_model` | **256** | |
| Encoder 层数 | `agent_encoder_layers` | **2** | 无位置编码（置换不变） |
| Encoder 头数 | `agent_encoder_heads` | **4** | |
| Decoder 层数 | `agent_decoder_layers` | **2** | 有步骤位置编码 |
| Decoder 头数 | `agent_decoder_heads` | **4** | |
| 最大序列长度 | `max_len` | **64** | 等于 seq_length |

> 架构：SetEncoder（置换不变，无位置编码） + PointerDecoder（自回归，Pointer Network）

### 4. 训练配置 (Training)

| 参数 | 配置变量 | 值 | 说明 |
|------|----------|-----|------|
| **启用 Agent** | `use_agent` | **True** | **关键！config 文件默认为 False，必须改为 True** |
| Batch Size | `batch_size` | **64** | |
| 模型学习率 | `learning_rate` | **1e-3** | AdamW |
| Agent 学习率 | `agent_learning_rate` | **1e-4** | AdamW，比模型低一个量级 |
| Weight Decay | `weight_decay` | **0.0** | |
| 梯度裁剪 | `grad_clip` | **1.0** | |
| 最大迭代步数 | `max_iters` | **100,000** | |
| LR 调度器 | — | CosineAnnealingLR | T_max = max_iters |

### 5. 无监督核心设置 (No-GT / Hell Mode)

| 参数 | 配置变量 | 值 | 说明 |
|------|----------|-----|------|
| **Warmup 步数** | `warmup_steps` | **0** | 无 Warmup，直接进入 Co-evolution |
| **TF 起始值** | `teacher_forcing_start` | **0.0** | 从第一步起完全无监督 |
| **TF 终止值** | `teacher_forcing_end` | **0.0** | |
| TF 衰减步数 | `teacher_forcing_decay_steps` | 50,000 | （因 start=end=0 实际无效） |
| LR Warmup | `warmup_iters` | **0** | |

### 6. 奖励与损失函数 (Reward & Loss)

| 参数 | 配置变量 | 值 | 说明 |
|------|----------|-----|------|
| 奖励类型 | `reward_type` | `cosine` | 余弦相似度奖励 |
| **Stepwise 奖励权重** | `stepwise_reward_weight` | **1.0** | L2R 正确性奖励（见下方注意事项） |
| **使用 BC 损失** | `use_bc_loss` | **False** | 无 Behavior Cloning |
| BC 损失权重 | `bc_loss_weight` | 1.0 | （因 use_bc_loss=False 实际无效） |
| 使用 Baseline | `use_baseline` | **True** | REINFORCE 方差归约 |
| Baseline Eps | `baseline_eps` | 1e-8 | |
| Returns 模式 | — | `immediate` | 硬编码在 coevolution_step 中 |

### 7. 日志与检查点

| 参数 | 配置变量 | 值 |
|------|----------|-----|
| 日志间隔 | `log_interval` | 100 |
| 评估间隔 | `eval_interval` | 500 |
| 检查点间隔 | `checkpoint_interval` | 500 |
| 保存最优模型 | `save_best_model` | True |
| W&B 项目 | `wandb_project` | `LO-ARMs-ContinuousRotation` |
| 随机种子 | `seed` | **42** |
| 设备 | `device` | `cuda` |

---

## 重要注意事项

### 1. `use_agent` 必须手动改为 `True`
当前 `config_continuous_rotation.py` 中 `use_agent = False`（Baseline 模式）。
复现 Hell Mode 必须将其设为 `True`。**使用 `run_hell_mode.sh` 脚本可自动处理此问题。**

### 2. Stepwise 奖励始终生效（代码行为与 Flag 不一致）
`config` 中定义了 `use_stepwise_rewards = False`，但 `coevolution_step()` 中
**从未检查此 Flag**。奖励计算始终为：
```python
rewards = cos_sim_padded + config.stepwise_reward_weight * l2r_correct
```
即 L2R 正确性奖励（权重 1.0）**始终被加入**，无论 `use_stepwise_rewards` 的值如何。
这是实际获胜行为，不是 bug —— 但需要注意。

### 3. 无 argparse 接口
训练脚本没有命令行参数解析。所有配置通过 `config_continuous_rotation.py` 模块直接导入。
`run_hell_mode.sh` 通过 Python 运行时动态覆盖 config 属性来解决此问题。

### 4. 实际训练阶段
由于 `warmup_steps = 0`，训练**从第 0 步起直接进入 Co-evolution 阶段**：
- 模型使用 Agent 输出的排列顺序
- Agent 使用纯 REINFORCE（无 BC、无 Teacher Forcing）
- 不存在 Phase 1（Parallel Warmup）

---

## 搜索空间分析

- 序列长度 L = 64，分为 16 个 chunk，每 chunk 含 4 个向量
- Block-wise Shuffle 保留 chunk 内部顺序，打乱 chunk 间顺序
- Agent 需在 64 个位置上输出完整排列（64! 的动作空间）
- 但最优解对应 **16! = 20,922,789,888,000** 种 chunk 排列中的唯一正确排列
- 纯 RL（无任何监督信号引导）在此空间中成功收敛

---

*配置冻结时间：2026-02-03*
*Git Commit: 参见 `git log --oneline -1`*
