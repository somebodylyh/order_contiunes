# === Data (GLA h-space AR, D=1024) ===
vector_dim = 1024
seq_length = 32
num_init   = 1           # h_0 作为始终可见的 init prefix，不参与 loss
num_chunks = 31          # = seq_length - num_init，MDM token-level shuffle
sigma      = 0.3         # Teacher 生成时的噪声，理论下界 = sigma^2 = 0.09

# 以下字段保留占位（train 脚本通过 cfg.xxx 引用，用于 wandb logging）
dependency_window = 1
num_matrices      = 1
train_init_mode   = 'random'
val_init_mode     = 'random'
noise_scale       = 0.0
alpha             = 0.0
train_samples     = 100000
val_samples       = 10000
test_samples      = 10000

# === Model ===
n_layer    = 5
n_head     = 4
n_embd     = 256
block_size = 32
dropout    = 0.0
bias       = True

# === Training ===
batch_size    = 256
learning_rate = 3e-4
epochs        = 50
warmup_iters  = 0.05     # 5% of total steps
weight_decay  = 0.1
grad_clip     = 1.0
seed          = 42
device        = 'cuda'
num_workers   = 4

# === Logging ===
log_interval    = 100
eval_interval   = 500
save_best_model = True
wandb_log       = True
wandb_project   = 'ao-gpt-mdm-hspace'
