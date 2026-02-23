# === Data ===
vector_dim = 64
seq_length = 32
dependency_window = -1  # full history
num_matrices = 8
train_init_mode = 'positive_first'
val_init_mode = 'negative_first'
num_chunks = 4           # 128/4=32 tokens per chunk, preserves within-block causal order
noise_scale = 0.05
alpha = 0.3
train_samples = 500000
val_samples = 20000
test_samples = 20000

# === Model ===
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
wandb_project = 'order-continuous-v4'
