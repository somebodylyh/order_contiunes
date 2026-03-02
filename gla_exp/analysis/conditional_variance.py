"""
conditional_variance.py
计算 Var(h_{t+1} | h_t)，量化从 h_t 预测 h_{t+1} 的不可约不确定性。

方法：用 k-NN（k=20）将测试集按 h_t 相似度分组，
计算同组内 h_{t+1} 的方差 → 近似 Var(h_{t+1} | h_t)。

若 Var(h_{t+1} | h_t) ≈ Var(h) ≈ 1.7，
说明 h_t 对预测 h_{t+1} 几乎没有帮助，
即 h 序列在 h-空间不是 Markov 链，
理论下界 σ² = 0.09 不可达。

Usage:
    python gla_exp/analysis/conditional_variance.py
"""
import sys, os, math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from gla_exp.exp_config import load_config
from gla_exp.generate_data import get_cache_dir, cache_exists
from gla_exp.exp_dataset import HiddenStateDataset
from torch.utils.data import DataLoader

CONFIG_PATH = "gla_exp/configs/exp001_ar_noshuffle.yaml"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
K           = 20      # k-NN 邻居数
N_SAMPLES   = 5000    # 从测试集取前 N 条（太大会 OOM）
POSITION    = 15      # 分析第几个位置（取中间位置，避免边界效应）

tc, sc, tr = load_config(CONFIG_PATH)
cache_dir = get_cache_dir(tc)
assert cache_exists(cache_dir), f"Cache not found: {cache_dir}"

ds     = HiddenStateDataset(cache_dir, "ar_noshuffle", "test", chunk_size=1)
loader = DataLoader(ds, batch_size=512, shuffle=False, num_workers=4)

print(f"Loading {N_SAMPLES} test samples...")
all_h = []
for batch in loader:
    all_h.append(batch["input"])   # [B, L, D]
    if sum(x.shape[0] for x in all_h) >= N_SAMPLES:
        break

H = torch.cat(all_h, dim=0)[:N_SAMPLES]  # [N, L, D]
print(f"Loaded: {H.shape}  (N={H.shape[0]}, L={H.shape[1]}, D={H.shape[2]})")

# ── 全局统计量 ─────────────────────────────────────────────────────────────────
h_var_global = H.var(unbiased=False).item()
h_mean_global = H.mean().item()
print(f"\n全局统计: mean={h_mean_global:.4f}, per-dim var={h_var_global:.4f}")
print(f"σ² (noise floor)          = {0.30**2:.4f}")
print(f"零预测器理论 loss           = {h_var_global + 0.30**2:.4f}\n")

# ── 在 POSITION 位置做 k-NN 条件方差 ──────────────────────────────────────────
H_t  = H[:, POSITION,     :].to(DEVICE)  # [N, D]  condition on h_t
H_t1 = H[:, POSITION + 1, :].to(DEVICE)  # [N, D]  predict h_{t+1}

# 归一化后计算 cosine 相似度作为距离
H_t_norm = F.normalize(H_t, dim=-1)
sim_mat   = H_t_norm @ H_t_norm.T  # [N, N]

# k-NN：取最相似的 K 个（排除自身）
sim_mat.fill_diagonal_(-1.0)
topk_idx = sim_mat.topk(K, dim=1).indices  # [N, K]

# 每个点的条件组 = 自身 + K 个邻居的 h_{t+1}
cond_vars = []
for i in range(H.shape[0]):
    neighbors = H_t1[topk_idx[i]]      # [K, D]
    group     = torch.cat([H_t1[i:i+1], neighbors], dim=0)  # [K+1, D]
    cond_vars.append(group.var(unbiased=False).item())

cond_var_mean = sum(cond_vars) / len(cond_vars)

print("=" * 55)
print(f"  位置 t = {POSITION},  k-NN k = {K}")
print(f"  全局 Var(h_{{t+1}})              : {H_t1.var().item():.4f}")
print(f"  条件 Var(h_{{t+1}} | h_t) [kNN] : {cond_var_mean:.4f}")
print(f"  σ² (noise floor)               : {0.30**2:.4f}")
print("=" * 55)
reduction = (H_t1.var().item() - cond_var_mean) / H_t1.var().item() * 100
print(f"\n  h_t 对预测 h_{{t+1}} 的方差削减比例: {reduction:.1f}%")
if reduction < 10:
    print("  结论: h_t 对预测 h_{t+1} 几乎无帮助（<10% 方差削减）。")
    print("        h 序列不构成 h-空间 Markov 链，σ²=0.09 理论下界不可达。")
elif reduction < 30:
    print("  结论: h_t 提供少量预测信息，但 h-空间结构性很弱。")
else:
    print("  结论: h_t 对预测 h_{t+1} 有显著帮助，可能存在 h-空间结构性。")
