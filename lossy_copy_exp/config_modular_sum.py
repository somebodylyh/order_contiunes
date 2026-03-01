"""
Configuration for Modular Sum Experiment

Extended experiment with 3 variables [x1, x2, y] and configurable causal structure.
"""

# ===== Task Configuration =====
# Core switch: determines causal structure
use_lossy = True  # True: y=(x1+x2)//2, False: y=(x1+x2)%P

# Task parameters
vocab_size = 64          # x1, x2, y ∈ [0, vocab_size)
seq_length = 3           # [x1, x2, y]
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
max_iters = 10000        # Increased for 3-variable task
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
wandb_project = 'LO-ARMs-ModularSum'
# Dynamic run name based on mode
wandb_run_name = f"modular_sum_{'lossy' if use_lossy else 'cyclic'}_v{vocab_size}"

# Output directory
out_dir = f'lossy_copy_exp/checkpoints_modular_sum_{"lossy" if use_lossy else "cyclic"}'

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
# These are hypothesis-driven expectations for each mode

# For use_lossy = True (Lossy mode):
#   Expected agent behavior:
#     - P(first=y) should → 0% (never select y first)
#     - P(first=x1) should → ~50% (select x1 or x2 first)
#     - P(first=x2) should → ~50%
#     - P(first=any_x) should → 100% (always select x before y)
#
#   Reason: Strong causality x1,x2 -> y
#   Agent discovers it must generate x1, x2 before y can be determined

# For use_lossy = False (Modular mode):
#   Expected agent behavior:
#     - P(first=y) should stay ~33% (no preference)
#     - P(first=x1) should stay ~33%
#     - P(first=x2) should stay ~33%
#     OR: Mode collapse to any single variable (initialization bias)
#
#   Reason: Complete symmetry, no causal asymmetry
#   Agent recognizes all three variables are equivalent

# ===== Experiment Variants =====

# Variant 1: Modular mode (for comparison)
# use_lossy = False
# wandb_run_name = 'modular_sum_cyclic_v64'
# out_dir = 'lossy_copy_exp/checkpoints_modular_sum_cyclic'

# Variant 2: Larger vocab
# vocab_size = 128
# wandb_run_name = f"modular_sum_{'lossy' if use_lossy else 'cyclic'}_v128"

# Variant 3: Longer training
# max_iters = 20000
# warmup_steps = 2000
# wandb_run_name = f"modular_sum_{'lossy' if use_lossy else 'cyclic'}_long"

# Variant 4: Binary reward (for comparison)
# reward_type = 'binary'
# wandb_run_name = f"modular_sum_{'lossy' if use_lossy else 'cyclic'}_binary"
