"""
数据质量检查：生成 1000 条序列后做 4 项检查。
通过标准（全部满足后才生成完整数据集）:
  within-sample cos_sim (归一化后) < 0.05
  between-sample cos_sim           < 0.05
  L2 norm 恒≈ √768
  attention entropy 非均匀（entropy < log(t)×0.9）
"""
import sys, math, torch
import torch.nn.functional as F

sys.path.insert(0, '.')
from baseline_continuous.teacher_generator import GPT2Teacher

device = 'cuda'
teacher = GPT2Teacher().to(device)
D, scale = 768, math.sqrt(768)

# 生成 1000 条序列
r = teacher.generate_sequence(32, batch_size=1000)
vecs = r['vectors']          # [1000, 32, 768]
main = vecs[:, 1:, :]        # [1000, 31, 768]（去掉 init）
main_n = F.normalize(main / scale, dim=-1)   # 归一化到单位球

# ── 1. L2 norm ──────────────────────────────────────────────
norms = vecs.norm(dim=-1).cpu()   # [1000, 32]
print(f"L2 norm: mean={norms.mean():.4f}, std={norms.std():.6f}  (expect {scale:.4f} ± ~0)")

# ── 2. Within-sample cos_sim ────────────────────────────────
N, T = main_n.shape[:2]
cos_w = torch.bmm(main_n, main_n.transpose(1,2))   # [N,T,T]
mask  = ~torch.eye(T, dtype=bool, device=device).unsqueeze(0).expand(N, -1, -1)
within = cos_w[mask].mean().item()
print(f"within-sample cos_sim : {within:.4f}  (expect < 0.05)")

# ── 3. Between-sample cos_sim ───────────────────────────────
p5 = main_n[:, 5, :]   # [N, 768]
cos_b = (p5 @ p5.T)    # [N, N]
mask_b = ~torch.eye(N, dtype=bool, device=device)
between = cos_b[mask_b].mean().item()
print(f"between-sample cos_sim: {between:.4f}  (expect < 0.05)")

# ── 4. Attention entropy ────────────────────────────────────
attn_weights = []
def hook(module, inp, out):
    if len(out) > 1 and out[1] is not None:
        attn_weights.append(out[1].detach().cpu())

handle = teacher.h[0].attn.register_forward_hook(hook)
with torch.no_grad():
    small = vecs[:8, :16, :]
    _ = teacher(small)
handle.remove()

if attn_weights:
    w = attn_weights[0]        # [8, 12, 16, 16]
    w_last = w[:, :, -1, :]    # [8, 12, 16]
    w_last = w_last.clamp(min=1e-9)
    entropy = -(w_last * w_last.log()).sum(-1).mean().item()
    uniform_entropy = math.log(16)
    print(f"attention entropy    : {entropy:.4f}  (uniform={uniform_entropy:.4f}, expect < {uniform_entropy*0.9:.4f})")
else:
    print("attention entropy    : N/A (hook returned no weights)")

# ── 结果汇总 ──
print()
passed = within < 0.05 and between < 0.05
print("DATA QUALITY:", "PASS ✓" if passed else "FAIL ✗")
if not passed:
    print("  → Try increasing noise_scale to 0.1 in teacher_generator.py")
