"""
zero_predictor_baseline.py
验证：零预测器的 MSE 是否接近 student 的 test loss（~1.794）。

若 MSE(zeros, h_noisy) ≈ 1.794，说明 student 几乎没学到有效的条件预测信息，
loss 主要来自 h 本身的方差，而非σ² = 0.09 的噪声。

Usage:
    python gla_exp/analysis/zero_predictor_baseline.py
"""
import sys, os, json, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gla_exp.exp_config import load_config
from gla_exp.generate_data import get_cache_dir, cache_exists
from gla_exp.exp_dataset import create_dataloaders

NOISE_SCALE = 0.30
CONFIG_PATH = "gla_exp/configs/exp001_ar_noshuffle.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tc, sc, tr = load_config(CONFIG_PATH)
cache_dir = get_cache_dir(tc)
assert cache_exists(cache_dir), f"Cache not found: {cache_dir}. Run generate_data first."

_, test_loader = create_dataloaders(cache_dir, "ar_noshuffle", batch_size=256, chunk_size=1, num_workers=4)

total_zero_loss = 0.0
total_mean_loss = 0.0
total_h_var     = 0.0
n_batches       = 0
all_means       = []

print(f"Running zero-predictor baseline on test set (noise σ={NOISE_SCALE})...\n")

with torch.no_grad():
    # Pass 1: compute global mean
    for batch in test_loader:
        x = batch["input"].to(DEVICE)  # [B, L, D]
        all_means.append(x.mean(dim=(0, 1)))  # [D]

    global_mean = torch.stack(all_means).mean(0)  # [D]

    # Pass 2: compute losses
    for batch in test_loader:
        x     = batch["input"].to(DEVICE)   # [B, L, D]  (clean hidden states)
        noise = NOISE_SCALE * torch.randn_like(x)
        x_noisy = x + noise                 # 模拟训练时的加噪目标

        # 零预测器: 预测全零
        zero_loss = F.mse_loss(torch.zeros_like(x_noisy), x_noisy).item()
        # 均值预测器: 预测全局均值
        mean_pred = global_mean.unsqueeze(0).unsqueeze(0).expand_as(x_noisy)
        mean_loss = F.mse_loss(mean_pred, x_noisy).item()
        # h 的逐维方差 (无噪声)
        h_var     = x.var(unbiased=False).item()

        total_zero_loss += zero_loss
        total_mean_loss += mean_loss
        total_h_var     += h_var
        n_batches += 1

zero_loss_avg = total_zero_loss / n_batches
mean_loss_avg = total_mean_loss / n_batches
h_var_avg     = total_h_var     / n_batches

print("=" * 55)
print(f"  σ² (noise floor, theoretical lower bound) : {NOISE_SCALE**2:.4f}")
print(f"  Zero predictor MSE(0, h_noisy)            : {zero_loss_avg:.4f}")
print(f"  Mean predictor MSE(μ, h_noisy)            : {mean_loss_avg:.4f}")
print(f"  h variance (per-dim, no noise)            : {h_var_avg:.4f}")
print(f"  Reported AR-noshuffle test loss            : 1.7936")
print("=" * 55)
print()
print("  Zero predictor 理论值 = Var(h) + σ²")
print(f"  = {h_var_avg:.4f} + {NOISE_SCALE**2:.4f} = {h_var_avg + NOISE_SCALE**2:.4f}")
print()
if abs(zero_loss_avg - 1.7936) < 0.05:
    print("  结论: Student test loss ≈ 零预测器 loss → student 几乎未学到有效预测！")
    print("        gap 来自 h 本身的方差（语言不确定性），而非模型能力不足。")
else:
    print(f"  Student loss: 1.7936 vs Zero predictor: {zero_loss_avg:.4f}")
    print("  Student 显著优于零预测器。")
