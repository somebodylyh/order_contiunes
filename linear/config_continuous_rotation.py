"""
Configuration for Continuous Vector Linear Rotation Experiment

This experiment uses:
- Dense AR (AutoRegressive) process with k orthogonal matrices
- Continuous vectors instead of discrete tokens
- Set-to-Sequence Agent with permutation-invariant encoder
- MSE loss instead of CrossEntropy
"""

# ============================================================================
# Data Configuration
# ============================================================================

vector_dim = 64
seq_length = 32
dependency_window = -1
num_matrices = 8
block_size = 32

# Initialization modes for train/val (OOD validation)
train_init_mode = 'positive_first'
val_init_mode = 'negative_first'

# Chunk configuration
num_chunks = 4
noise_scale = 0.1

# Fixed orthogonal matrices path
fixed_matrices_path = 'linear_rotation_exp/fixed_orthogonal_matrices.pt'

# Dataset sizes
train_samples = 20000
val_samples = 2000
test_samples = 2000
num_workers = 16

# ============================================================================
# Model Configuration (ContinuousTransformer)
# ============================================================================

n_layer = 4
n_head = 8
n_embd = 256
dropout = 0.0
bias = True

# ============================================================================
# Agent Configuration (SetToSeqAgent)
# ============================================================================

# Encoder (permutation invariant)
agent_d_model = 256
agent_encoder_layers = 2
agent_encoder_heads = 4

# Decoder (autoregressive with position encoding)
agent_decoder_layers = 2
agent_decoder_heads = 4

# Policy
policy_dim = 256

# ============================================================================
# Training Configuration
# ============================================================================

# General
use_agent = True

# Optimization
batch_size = 64
learning_rate = 0.001
agent_learning_rate = 0.0001
weight_decay = 0.0
grad_clip = 1.0

# Schedule
max_iters = 50000
warmup_steps = 0
warmup_iters = 0

# Teacher forcing
teacher_forcing_start = 0.0
teacher_forcing_end = 0.0
teacher_forcing_decay_steps = 50000

# ============================================================================
# Reward Configuration
# ============================================================================

reward_type = 'cosine'
use_stepwise_rewards = False
stepwise_reward_weight = 0.0

# Behavior cloning
use_bc_loss = False
bc_loss_weight = 0.0

# Warmup configuration
warmup_bc_weight = 0.0
warmup_use_gt_order = False

# Baseline (for variance reduction)
use_baseline = True
baseline_eps = 1e-08

# ============================================================================
# Logging & Checkpointing
# ============================================================================

# Logging intervals
log_interval = 100
eval_interval = 500
checkpoint_interval = 500
save_best_model = True

# W&B
wandb_log = True
wandb_project = 'LO-ARMs-ContinuousRotation'
wandb_run_name = None

# Experiment directory
exp_dir = 'linear_rotation_exp/checkpoints_continuous'

# ============================================================================
# System
# ============================================================================

device = 'cuda'
seed = 42
