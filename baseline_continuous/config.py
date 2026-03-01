# === Data ===
vector_dim = 64
seq_length = 32
dependency_window = 4    # fixed window: each step depends on previous k steps
num_init = 4             # = dependency_window; init vectors used as conditioning prefix
num_matrices = 8
train_init_mode = 'positive_first'
val_init_mode = 'negative_first'
num_chunks = 28          # = seq_length - num_init; token-level shuffle of main tokens
noise_scale = 0.05
alpha = 0.0              # no x_0 bias; dep_window=4 provides sufficient structure
train_samples = 500000
val_samples = 20000
test_samples = 20000

# === Model ===1
n_layer = 4
n_head = 4
n_embd = 256
block_size = 32
dropout = 0.0
bias = True

# === Training ===
batch_size = 512
learning_rate = 1e-3
epochs = 50              # 50 epochs × 976 iters ≈ 48800 iters
warmup_iters = 0.05      # 5% of total training steps (scales with dataset size)
weight_decay = 0.1
grad_clip = 1.0
seed = 42
device = 'cuda'
num_workers = 4

# === Logging ===
log_interval = 100
eval_interval = 500
save_best_model = True
wandb_log = True
wandb_project = 'order-continuous-v9'
