"""
Configuration for LO-ARMs Lossy Copy Experiment

Toy experiment settings optimized for fast iteration and clear results.
"""

# ===== Model Configuration =====
# Use TINY model for fast debugging
n_layer = 2
n_head = 2
n_embd = 128
model_type = 'AdaLN6_NoRep_cond_128_trunc_qknorm'
dropout = 0.0
bias = True

# ===== Task Configuration =====
vocab_size = 64          # Small vocab: x ∈ [0, 63], y ∈ [0, 31]
seq_length = 2           # [x, y] pairs
k_divisor = 2            # y = x // k
block_size = seq_length  # Must match sequence length

# ===== Training Configuration =====
batch_size = 64
learning_rate = 1e-3     # Model learning rate
max_iters = 5000         # Total training iterations
warmup_steps = 1000      # Warmup phase (~20% of training)
grad_clip = 1.0          # Gradient clipping

# Dataset
num_train_samples = 10000
num_val_samples = 1000
dataset_seed = 42

# ===== Agent Configuration =====
agent_learning_rate = 1e-4  # Lower LR for stable policy learning
policy_dim = 128             # Agent hidden dimension

# Reward type: 'log_prob' (continuous) or 'binary' (0/1)
# RECOMMENDED: 'log_prob' for smoother gradients in early training
reward_type = 'log_prob'

# REINFORCE baseline (variance reduction)
use_baseline = True      # Subtract mean reward from returns

# ===== Logging Configuration =====
log_interval = 10         # Log every N steps
eval_interval = 100       # Evaluate every N steps
checkpoint_interval = 500 # Save checkpoint every N steps

# Weights & Biases
wandb_log = True
wandb_project = 'LO-ARMs'
wandb_run_name = 'lossy_copy_k2_v64_tiny'

# Output directory
out_dir = 'lossy_copy_exp/checkpoints'

# ===== Device Configuration =====
device = 'cuda'  # Use 'cuda' if available, else 'cpu'
dtype = 'bfloat16' if device == 'cuda' else 'float32'  # Use bfloat16 on GPU
compile = False  # torch.compile (requires PyTorch 2.0+)

# ===== Optimizer Configuration =====
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1

# ===== Evaluation Configuration =====
eval_iters = 100  # Number of batches for evaluation

# ===== Experiment Variants =====
# To test different configurations, uncomment one of the following:

# # Variant 1: Larger model (for comparison)
# n_layer = 4
# n_head = 4
# n_embd = 256
# wandb_run_name = 'lossy_copy_k2_v64_medium'

# # Variant 2: Larger vocab (harder task)
# vocab_size = 128
# k_divisor = 4
# wandb_run_name = 'lossy_copy_k4_v128_tiny'

# # Variant 3: Binary reward (simpler but sparser)
# reward_type = 'binary'
# wandb_run_name = 'lossy_copy_k2_v64_binary'

# # Variant 4: No warmup (test cold start)
# warmup_steps = 0
# wandb_run_name = 'lossy_copy_k2_v64_no_warmup'

# # Variant 5: Longer training
# max_iters = 10000
# warmup_steps = 2000
# wandb_run_name = 'lossy_copy_k2_v64_long'
