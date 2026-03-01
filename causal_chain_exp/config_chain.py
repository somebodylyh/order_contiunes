"""
Configuration for Causal Chain Experiment (A → B → C)

Tests whether Agent can discover multi-level causal hierarchy.
"""

# ===== Task Configuration =====
# Causal structure: A → B = A//2 → C = B//2
vocab_size = 64          # A ∈ [0, 64), B ∈ [0, 32), C ∈ [0, 16)
seq_length = 3           # [A, B, C]
block_size = seq_length  # Must match sequence length

# ===== Model Configuration =====
# Keep TINY for fast iteration
n_layer = 2
n_head = 2
n_embd = 128
model_type = 'AdaLN6_NoRep_cond_128_trunc_qknorm'
dropout = 0.0
bias = True

# ===== Training Configuration =====
batch_size = 64
learning_rate = 1e-3     # Model learning rate
max_iters = 8000         # Chain reasoning needs more time to converge
warmup_steps = 1000      # Warmup phase
grad_clip = 1.0          # Gradient clipping

# Dataset
num_train_samples = 10000
num_val_samples = 1000
dataset_seed = 42

# ===== Agent Configuration =====
agent_learning_rate = 1e-4  # Lower LR for stable policy learning
policy_dim = 128             # Agent hidden dimension

# Reward type: 'log_prob' (continuous) or 'binary' (0/1)
reward_type = 'log_prob'

# REINFORCE baseline (variance reduction)
use_baseline = True

# ===== Logging Configuration =====
log_interval = 10         # Log every N steps
eval_interval = 100       # Evaluate every N steps
checkpoint_interval = 500 # Save checkpoint every N steps

# Weights & Biases
wandb_log = True
wandb_project = 'LO-ARMs-CausalChain'
wandb_run_name = 'chain_A_B_C_v64'

# Output directory
out_dir = 'causal_chain_exp/checkpoints'

# ===== Device Configuration =====
device = 'cuda'  # Use 'cuda' if available, else 'cpu'
dtype = 'bfloat16' if device == 'cuda' else 'float32'
compile = False

# ===== Optimizer Configuration =====
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1

# ===== Evaluation Configuration =====
eval_iters = 100  # Number of batches for evaluation

# ===== Expected Results =====
#
# Agent should discover the causal chain A → B → C
# and learn to generate in that order.
#
# Expected metrics evolution:
#
#   P(select_root_first) [A]:
#     Initial: ~33% (random)
#     Target:  >95% (A is the root, has full information)
#
#   P(select_mid_first) [B]:
#     Initial: ~33% (random)
#     Target:  <5% (B is middle, cannot determine A)
#
#   P(select_leaf_first) [C]:
#     Initial: ~33% (random)
#     Target:  <5% (C is leaf, has minimal information)
#
# Key observation pattern:
#   • C drops FIRST (least information)
#   • B drops SECOND (partial information, can determine C but not A)
#   • A wins LAST (full information, determines entire chain)
#
# This progressive elimination reflects the information hierarchy!

# ===== Experiment Variants =====

# Variant 1: Larger vocab (harder task)
# vocab_size = 128
# wandb_run_name = 'chain_A_B_C_v128'

# Variant 2: Longer training
# max_iters = 12000
# warmup_steps = 1500
# wandb_run_name = 'chain_A_B_C_long'

# Variant 3: Binary reward (for comparison)
# reward_type = 'binary'
# wandb_run_name = 'chain_A_B_C_binary'

# Variant 4: Larger model (if tiny doesn't converge)
# n_layer = 4
# n_head = 4
# n_embd = 256
# wandb_run_name = 'chain_A_B_C_medium'
