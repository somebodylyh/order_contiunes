# === Data ===
vector_dim = 256
seq_length = 128
dependency_window = -1  # full history
num_matrices = 8
train_init_mode = 'positive_first'
val_init_mode = 'negative_first'
num_chunks = 32
noise_scale = 0.05
train_samples = 2000000
val_samples = 200000
test_samples = 200000

# === Model ===
n_layer = 4
n_head = 4
n_embd = 256
block_size = 128
dropout = 0.0
bias = True

# === Training ===
batch_size = 512
learning_rate = 1e-3
epochs = 20              # 20 epochs × 976 iters ≈ 19520 iters, each sample seen ~20x
warmup_iters = 2000      # linear warmup for first 2000 steps
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
wandb_project = 'baseline-continuous'
