# === Data ===
vector_dim = 64
seq_length = 64
dependency_window = -1  # full history
num_matrices = 8
train_init_mode = 'positive_first'
val_init_mode = 'negative_first'
num_chunks = 8
noise_scale = 0.1
train_samples = 50000
val_samples = 10000
test_samples = 10000

# === Model ===
n_layer = 4
n_head = 4
n_embd = 256
block_size = 64
dropout = 0.0
bias = True

# === Training ===
batch_size = 128
learning_rate = 1e-3
max_iters = 80000
warmup_iters = 0.0
weight_decay = 0.0
grad_clip = 1.0
seed = 42
device = 'cuda'

# === Logging ===
log_interval = 50
eval_interval = 100
save_best_model = True
wandb_log = True
wandb_project = 'baseline-continuous'
