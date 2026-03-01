# GLA Teacher-Student 实验报告

---

## 一、实验与代码设置

### 数据设置

| 参数 | 值 |
|------|----|
| Teacher 模型 | fla-hub/gla-340M-15B，截取前 4 层（layer_idx=3） |
| Teacher hidden dim D | 1024 |
| 序列长度 L | 32 |
| 训练集 | 100,000 条（wikitext-103，随机有放回采样） |
| 测试集 | 10,000 条 |
| 训练噪声 | x̃ = h + ε，ε ~ N(0, σ²)，σ = 0.30 |
| 噪声下界 | σ² = 0.09 |

### 模型设置（Student）

| 参数 | 值 |
|------|----|
| 架构 | ContinuousAOGPT + in_proj(1024→256) + out_proj(256→1024) |
| n_layers | 5 |
| n_heads | 4 |
| d_model（GPT 内部维度） | 256 |
| 总参数量 | 5.67M（GPT 5.15M + proj 0.52M） |
| num_init | 0（无前缀条件输入） |
| chunk_size | exp001: 1，exp002/003: 4 |

### 训练设置

| 参数 | 值 |
|------|----|
| Batch size | 256 |
| Epochs | 50 |
| 总 iters | 19,500（390 iters/epoch） |
| 优化器 | AdamW（weight_decay=0.1，β=(0.9, 0.95)） |
| 学习率 | 3e-4，linear warmup（5%）+ cosine decay → 3e-5 |
| EMA decay | 0.9999（adaptive: min(0.9999, (1+step)/(10+step))） |
| Grad clip | 1.0 |
| 评估 | 统一用 AR noshuffle 顺序 + AR 模式（三个实验相同） |

---

## 二、核心数学公式

### 数据生成

Teacher 对输入 token 序列 $x_{1:L}$ 运行 GLA 前 4 层，输出 hidden state 序列：

$$h_t = \text{GLA}_{1:4}(x_{1:t}), \quad t = 1, \ldots, L$$

训练时每 batch 动态加噪：

$$\tilde{h}_t = h_t + \varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,\, \sigma^2 I), \quad \sigma = 0.30$$

### 块间打乱（exp002/003，chunk_size=4）

将 L=32 个位置划分为 8 个连续块（每块 4 个 token），随机打乱块的顺序，块内相对顺序不变：

$$\pi = \text{BlockPerm}(L, \text{chunk\_size}=4, \text{seed}=\text{sample\_idx})$$

$$\tilde{h}^{\text{shuffled}}_t = \tilde{h}_{\pi(t)}$$

### Loss 函数

$$\mathcal{L} = \frac{1}{(L-1) \cdot D} \sum_{t=1}^{L-1} \left\| \hat{h}_{t+1} - \tilde{h}_{t+1} \right\|_2^2$$

其中 $\hat{h}_{t+1}$ 为 Student 在位置 $t$ 处的输出，$\tilde{h}_{t+1}$ 为加噪后的 teacher hidden state。

理论下界：$\mathcal{L}_{\min} = \sigma^2 = 0.09$

---

## 三、最终实验结果

| 实验 | 训练模式 | 输入顺序 | Train Loss | Test Loss（AR noshuffle 统一评估） |
|------|---------|---------|-----------|----------------------------------|
| exp001 AR noshuffle | AR | 原始顺序 | 1.598 | **1.794** |
| exp002 AR shuffled | AR | 块间打乱 | 1.679 | 1.839 |
| exp003 MDM shuffled | Random | 块间打乱 | 1.537 | 1.893 |

噪声下界 σ² = 0.09

**结论：** AR noshuffle test loss 最低（1.794），MDM test loss 最高（1.893），三者差距约 0.1。

---

## 四、代码结构

```
AO-GPT-MDM/
├── train.py                          # 统一训练入口
├── configs/
│   ├── exp001_ar_noshuffle.yaml      # AR + 原始顺序
│   ├── exp002_ar_shuffled.yaml       # AR + 块间打乱
│   └── exp003_mdm_shuffled.yaml      # MDM + 块间打乱
└── baseline_continuous/
    ├── exp_config.py                 # 配置数据类 + YAML 解析
    ├── teachers.py                   # FLATeacher：GLA 模型截断 + hidden 提取
    ├── generate_data.py              # 从 wikitext-103 提取并缓存 hidden states
    ├── exp_dataset.py                # HiddenStateDataset + 块间打乱逻辑
    └── continuous_aogpt.py           # ContinuousAOGPT 模型定义
```

### 各文件职责

**`train.py`**
- `get_lr()`: linear warmup + cosine decay 调度器
- `update_ema()`: adaptive EMA 更新
- `StudentWithProjection`: in_proj + ContinuousAOGPT + out_proj，forward 计算 MSE loss
- `evaluate()`: 在指定 loader/mode 上评估 EMA 模型
- `main()`: 加载数据 → 建模 → 训练循环 → 统一 AR noshuffle 评估

**`baseline_continuous/exp_config.py`**
- `TeacherConfig`、`StudentConfig`、`TrainingConfig`：三个 dataclass，覆盖所有超参
- `load_config(yaml_path)`: 读取 YAML，返回 (TeacherConfig, StudentConfig, TrainingConfig)
- `teacher_cache_key()`: 对 TeacherConfig 计算 MD5 哈希，三个实验共享同一 cache

**`baseline_continuous/teachers.py`**
- `FLATeacher(cfg)`: 加载 GLA-340M，截断到 `layer_idx+1` 层，注册 hook 提取 hidden state
- `extract(input_ids)`: 一次 forward，返回 `{hidden, shuffled, perm, order}`

**`baseline_continuous/generate_data.py`**
- `_build_token_pools()`: 逐条 encode wikitext-103（避免 540M 字符一次性 OOM），返回 train/test token chunk 池
- `_fill_samples()`: 从 token pool 随机采样，批量调用 teacher.extract，填入预分配 buffer
- `generate_and_cache()`: 生成 `hidden.pt`（[N,L,D]）、`perm.pt`（[N,L]）、`meta.json` 并缓存

**`baseline_continuous/exp_dataset.py`**
- `_chunk_perm(global_idx, L, chunk_size)`: 以 sample index 为种子生成块间打乱排列，可复现
- `HiddenStateDataset`: 读取缓存，按 student_type 返回 `{"input": tensor[L,D]}`
  - `ar_noshuffle`: 直接返回原始顺序 hidden
  - `ar_shuffled` / `mdm_shuffled`: chunk_size=1 用 perm.pt，chunk_size>1 用 `_chunk_perm`
- `create_dataloaders()`: 返回 train/test DataLoader

**`baseline_continuous/continuous_aogpt.py`**
- `ContinuousAOGPTConfig`: block_size、vector_dim、n_layer、n_head 等
- `CausalSelfAttention`: QK-Norm + 因果 mask（AR 模式）或无 mask（Random 模式）
- `Block`: RMSNorm + Attention + MLP，AdaLN 通过 target position embedding 调制
- `ContinuousAOGPT.forward(x, mode)`: mode='AR' 使用因果 mask；mode='Random' 每次生成新随机排列，通过 AdaLN 注入 target position 信息

### 数据流

```
wikitext-103
    │
    ▼
generate_data.py
  FLATeacher.extract()
    │  hidden.pt  [110000, 32, 1024]
    │  perm.pt    [110000, 32]
    ▼
exp_dataset.py
  HiddenStateDataset
    │  batch["input"]  [B, 32, 1024]（按 student_type 决定顺序）
    ▼
train.py
  加噪: x̃ = x + N(0, 0.09)
  StudentWithProjection.forward(x̃, mode)
    in_proj → ContinuousAOGPT → out_proj
    loss = MSE(preds[:,:-1], x̃)
    ▼
  EMA 模型 → evaluate(canon_loader, mode="AR")
    test_loss（统一 AR noshuffle 评估）
```
