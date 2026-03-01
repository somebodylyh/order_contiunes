#!/bin/bash
# =============================================================================
# Hell Mode Reproduction Script
# =============================================================================
#
# 复现 "地狱模式" 实验: seq_len=64, num_chunks=16, teacher_forcing=0.0
# 搜索空间: 16! = 20,922,789,888,000 种排列
#
# 预期结果 (100k steps):
#   - Kendall's Tau: ~0.99
#   - Cosine Similarity: > 0.44
#   - 显著击败 Baseline (随机排列)
#
# 用法:
#   chmod +x linear_rotation_exp/run_hell_mode.sh
#   ./linear_rotation_exp/run_hell_mode.sh
#
# =============================================================================

set -euo pipefail

# 切换到项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  Hell Mode: seq_len=64, num_chunks=16, TF=0.0"
echo "  搜索空间: 16! ≈ 2×10^13"
echo "============================================================"
echo ""
echo "项目根目录: $PROJECT_ROOT"
echo "启动时间:   $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# -----------------------------------------------------------------------------
# 通过 Python 运行时覆盖 config 属性，确保参数精确
# 注意: 训练脚本没有 argparse，所有配置通过 config 模块导入
# 此处显式覆盖所有 Hell Mode 关键参数
# -----------------------------------------------------------------------------

python -c "
import linear_rotation_exp.config_continuous_rotation as config

# ===== Hell Mode 核心参数覆盖 =====

# [关键] 启用 Agent (config 默认为 False)
config.use_agent = True

# --- 数据 ---
config.vector_dim       = 64
config.seq_length       = 64
config.block_size       = 64
config.dependency_window = -1     # Full History 模式
config.num_matrices     = 16
config.num_chunks       = 16      # 16 chunks -> 搜索空间 16!
config.train_init_mode  = 'positive_first'
config.val_init_mode    = 'negative_first'
config.train_samples    = 100000
config.val_samples      = 20000
config.test_samples     = 20000
config.num_workers      = 16
config.fixed_matrices_path = 'linear_rotation_exp/fixed_orthogonal_matrices.pt'

# --- 模型 (ContinuousTransformer) ---
config.n_layer  = 8
config.n_head   = 8
config.n_embd   = 256
config.dropout  = 0.0
config.bias     = True

# --- Agent (SetToSeqAgent) ---
config.agent_d_model        = 256
config.agent_encoder_layers = 2
config.agent_encoder_heads  = 4
config.agent_decoder_layers = 2
config.agent_decoder_heads  = 4

# --- 训练 ---
config.batch_size       = 64
config.learning_rate    = 1e-3    # Model LR
config.agent_learning_rate = 1e-4 # Agent LR
config.weight_decay     = 0.0
config.grad_clip        = 1.0
config.max_iters        = 100000

# --- Hell Mode 关键: 无监督 ---
config.warmup_steps             = 0     # 无 Warmup，直接 Co-evolution
config.warmup_iters             = 0
config.teacher_forcing_start    = 0.0   # 完全无 Teacher Forcing
config.teacher_forcing_end      = 0.0
config.teacher_forcing_decay_steps = 50000

# --- 奖励 ---
config.reward_type              = 'cosine'
config.stepwise_reward_weight   = 1.0   # L2R 正确性奖励 (始终生效)
config.use_bc_loss              = False  # 无 Behavior Cloning
config.bc_loss_weight           = 1.0
config.use_baseline             = True   # REINFORCE 方差归约
config.baseline_eps             = 1e-8

# --- 日志 ---
config.log_interval        = 100
config.eval_interval       = 500
config.checkpoint_interval = 500
config.save_best_model     = True
config.wandb_log           = True
config.wandb_project       = 'LO-ARMs-ContinuousRotation'
config.wandb_run_name      = 'hell-mode_s64_c16_tf0'
config.exp_dir             = 'linear_rotation_exp/checkpoints_continuous'

# --- 其他 ---
config.device = 'cuda'
config.seed   = 42

# ===== 启动训练 =====
from linear_rotation_exp.train_continuous_rotation import train
train()
"

echo ""
echo "============================================================"
echo "  Hell Mode 实验完成: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
