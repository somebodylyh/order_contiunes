"""
Configuration for Diamond DAG Experiment

Tests whether Agent can discover complex DAG topology with both
fork (x0 -> x1, x2) and join (x1, x2 -> x3) structures.
"""

# ===== Task Configuration =====
# DAG structure: x0 → (x1, x2) → x3
vocab_size = 128          # x0 ∈ [0, 128), x1 ∈ [0, 64), x2 ∈ [0, 64), x3 ∈ [0, 32), x4 ∈ [0, 32), x5 ∈ [0, 16)
seq_length = 6           # [x0, x1, x2, x3, x4, x5]
block_size = seq_length  # Must match sequence length

# ===== Model Configuration =====
# Slightly larger than previous experiments (more complex structure)
n_layer = 4              # +1 layer to handle DAG complexity
n_head = 4
n_embd = 128
model_type = 'AdaLN6_NoRep_cond_128_trunc_qknorm'
dropout = 0.0
bias = True

# ===== Training Configuration =====
batch_size = 64
learning_rate = 1e-3     # Model learning rate
max_iters = 12000        # DAG discovery needs more iterations
warmup_steps = 3000      # Longer warmup for stability
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
wandb_project = 'LO-ARMs-DAG'
wandb_run_name = 'diamond_dag_v64'

# Output directory
out_dir = 'dag_exp/checkpoints'

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
# Agent should discover the Diamond DAG topology:
#   x0 (root) → x1, x2 (branches) → x3 (sink)
#
# Expected metrics evolution:
#
#   First Step (t=0) - Must select root:
#     P(select_x0_first):  ~25% → >95%  (x0 is the root)
#     P(select_x1_first):  ~25% → <5%   (x1 is not root)
#     P(select_x2_first):  ~25% → <5%   (x2 is not root)
#     P(select_x3_first):  ~25% → <5%   (x3 is sink, definitely not first)
#
#   Second Step (t=1) - Should select one branch:
#     P(select_branch_second): ~50% → ~100%  (x1 or x2)
#     P(select_x0_second): ~33% → ~0%        (already selected)
#     P(select_x3_second): ~33% → ~0%        (sink comes last)
#
#   Last Step (t=3) - Must select sink:
#     P(select_x3_last):  ~25% → >95%  (x3 is the sink)
#
# Branch Symmetry:
#   x1 and x2 are topologically equivalent (both at depth 1)
#   Expected: P(x1_second) ≈ P(x2_second) ≈ 50%
#   OR: Agent may arbitrarily prefer one, but sum should be ~100%
#
# Theoretical Optimal Loss:
#   Step 0 (Gen x0): Blind guess → Loss ≈ ln(64) ≈ 4.16
#   Step 1 (Gen x1 or x2): Known x0 → Loss ≈ 0
#   Step 2 (Gen other branch): Known x0 → Loss ≈ 0
#   Step 3 (Gen x3): Known x1, x2 → Loss ≈ 0
#   Average: 4.16 / 4 ≈ 1.04
#
# This is the ULTIMATE test - proves Agent can:
#   1. Discover DAG topology from data alone
#   2. Handle mixed fork/join structures
#   3. Learn topological sort (root first, sink last)
#   4. Recognize symmetric branches

# ===== Experiment Variants =====

# Variant 1: Larger vocab (harder task)
# vocab_size = 128
# wandb_run_name = 'diamond_dag_v128'

# Variant 2: Longer training
# max_iters = 15000
# warmup_steps = 2000
# wandb_run_name = 'diamond_dag_long'

# Variant 3: Binary reward (for comparison)
# reward_type = 'binary'
# wandb_run_name = 'diamond_dag_binary'

# Variant 4: Larger model (if convergence is slow)
# n_layer = 4
# n_head = 4
# n_embd = 256
# policy_dim = 256
# wandb_run_name = 'diamond_dag_large'
