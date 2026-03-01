"""
train.py — 统一训练入口，GLA Teacher-Student 实验框架。

Usage:
    python train.py --config configs/exp001_ar_noshuffle.yaml
    python train.py --config configs/exp001_ar_noshuffle.yaml --force_regen

三个 baseline:
  exp001 ar_noshuffle → AR student + 原始顺序 → 上帝模型（无噪声下界）
  exp002 ar_shuffled  → AR student + 块间打乱
  exp003 mdm_shuffled → MDM student + 块间打乱 → 关注对象

noise_scale > 0 时：
  - 训练目标加噪: x_noisy = h + N(0, σ²)，loss 下界 = σ²
  - canon_ar_loss：所有实验统一用 AR noshuffle 模式评估，方便横向比较
  - 当 canon_ar_loss → σ²，说明 student 完全习得了 teacher 的 sequential 结构
"""
import sys, os, json, argparse, math, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_continuous.exp_config import load_config
from baseline_continuous.generate_data import get_cache_dir, cache_exists, generate_and_cache
from baseline_continuous.exp_dataset import create_dataloaders, HiddenStateDataset
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig


# ─── LR Schedule ──────────────────────────────────────────────────────────────

def get_lr(it, warmup_iters, max_iters, learning_rate, min_lr_ratio=0.1):
    """Linear warmup + cosine decay."""
    min_lr = learning_rate * min_lr_ratio
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


# ─── EMA ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def update_ema(ema_model, model, step, target_decay=0.9999):
    """Adaptive decay: warm-up 阶段快速跟踪，稳定后趋向 target_decay。"""
    decay = min(target_decay, (1 + step) / (10 + step))
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.data, alpha=1 - decay)


# ─── Student 模型 ─────────────────────────────────────────────────────────────

class StudentWithProjection(nn.Module):
    """ContinuousAOGPT + in_proj（D→d_model）+ out_proj（d_model→D）。"""

    def __init__(self, teacher_D: int, sc, seq_len: int):
        super().__init__()
        d = sc.d_model
        self.in_proj  = nn.Linear(teacher_D, d, bias=False) if d != teacher_D else nn.Identity()
        self.out_proj = nn.Linear(d, teacher_D, bias=False) if d != teacher_D else nn.Identity()
        gpt_cfg = ContinuousAOGPTConfig(
            block_size = seq_len,
            vector_dim = d,
            n_layer    = sc.n_layers,
            n_head     = sc.n_heads,
            dropout    = 0.0,
            bias       = True,
            num_init   = 0,
        )
        self.gpt = ContinuousAOGPT(gpt_cfg)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        x: [B, L, teacher_D]（训练时为加噪后的 x_noisy）
        loss = MSE(shift_preds, x)，在 teacher_D 空间
        噪声已在 train loop 中加入 x，故 loss 下界 = σ²
        """
        xp         = self.in_proj(x)
        preds_p, _ = self.gpt(xp, mode=mode)
        preds      = self.out_proj(preds_p)
        loss       = F.mse_loss(preds[:, :-1], x)
        return loss


# ─── 评估 ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, mode, noise_scale=0.0):
    """用 EMA 模型评估。noise_scale > 0 时加与训练一致的噪声，使 loss 下界 = σ²。"""
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        x = batch["input"].to(device)
        if noise_scale > 0:
            x = x + noise_scale * torch.randn_like(x)
        loss = model(x, mode=mode)
        total += loss.item()
        n     += 1
    model.train()
    return total / max(n, 1)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--force_regen", action="store_true")
    args = parser.parse_args()

    tc, sc, tr  = load_config(args.config)
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    config_stem = os.path.splitext(os.path.basename(args.config))[0]

    mode = "AR" if sc.type in ("ar_noshuffle", "ar_shuffled") else "Random"

    # ── 1. Teacher 数据 ───────────────────────────────────────────────────────
    cache_dir = get_cache_dir(tc)
    if args.force_regen or not cache_exists(cache_dir):
        generate_and_cache(tc, force=args.force_regen)

    with open(os.path.join(cache_dir, "meta.json")) as f:
        D = json.load(f)["D"]
    print(f"[data] Teacher D={D}")

    # 实验自身的 train/test loader
    train_loader, test_loader = create_dataloaders(
        cache_dir, sc.type, tr.batch_size,
        chunk_size=sc.chunk_size, num_workers=4,
    )

    # Canonical AR noshuffle test loader（所有实验统一评估基准）
    if sc.type == "ar_noshuffle":
        canon_loader = test_loader      # exp001 本身就是 AR noshuffle
    else:
        canon_ds     = HiddenStateDataset(cache_dir, "ar_noshuffle", "test", chunk_size=1)
        canon_loader = DataLoader(canon_ds, batch_size=tr.batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True)

    # ── 2. LR schedule 参数 ───────────────────────────────────────────────────
    iters_per_epoch = len(train_loader)
    max_iters       = tr.epochs * iters_per_epoch
    warmup_iters    = int(tr.warmup_ratio * max_iters)

    wandb.init(
        project = "ao-gpt-mdm",
        name    = config_stem,
        config  = {
            "teacher_model":  tc.model_name,
            "teacher_layer":  tc.layer_idx,
            "seq_len":        tc.seq_len,
            "student_type":   sc.type,
            "student_mode":   mode,
            "d_model":        sc.d_model,
            "n_layers":       sc.n_layers,
            "n_heads":        sc.n_heads,
            "chunk_size":     sc.chunk_size,
            "lr":             tr.lr,
            "epochs":         tr.epochs,
            "batch_size":     tr.batch_size,
            "warmup_ratio":   tr.warmup_ratio,
            "ema_decay":      tr.ema_decay,
            "grad_clip":      tr.grad_clip,
            "noise_scale":    tr.noise_scale,
            "noise_floor":    tr.noise_scale ** 2,
            "max_iters":      max_iters,
            "warmup_iters":   warmup_iters,
        },
    )

    print("=" * 60)
    print(f"Teacher : {tc.model_name}  layer={tc.layer_idx}  L={tc.seq_len}")
    print(f"Student : {sc.type}  d_model={sc.d_model}  layers={sc.n_layers}  "
          f"mode={mode}  chunk_size={sc.chunk_size}")
    print(f"Training: epochs={tr.epochs}  lr={tr.lr}  bs={tr.batch_size}  "
          f"warmup={warmup_iters}  ema={tr.ema_decay}  noise_σ={tr.noise_scale}")
    print(f"[lr]    {iters_per_epoch} iters/epoch × {tr.epochs} epochs = {max_iters} total")
    print(f"[noise] floor σ² = {tr.noise_scale**2:.6f}")
    print("=" * 60)

    # ── 3. Student 模型 + EMA ─────────────────────────────────────────────────
    torch.manual_seed(42)
    model     = StudentWithProjection(D, sc, tc.seq_len).to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tr.lr, weight_decay=0.1, betas=(0.9, 0.95)
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Student params: {n_params / 1e6:.2f}M  "
          f"(GPT {sum(p.numel() for p in model.gpt.parameters())/1e6:.2f}M + "
          f"proj {(sum(p.numel() for p in model.in_proj.parameters()) + sum(p.numel() for p in model.out_proj.parameters()))/1e6:.3f}M)")

    # ── 4. 训练循环 ───────────────────────────────────────────────────────────
    runs_dir     = os.path.join("runs", config_stem)
    os.makedirs(runs_dir, exist_ok=True)
    metrics_path = os.path.join(runs_dir, "metrics.jsonl")

    print(f"\n[train] Starting {tr.epochs} epochs")
    global_step = 0
    model.train()

    for epoch in range(1, tr.epochs + 1):
        total_loss, n_batches = 0.0, 0

        for batch in train_loader:
            x = batch["input"].to(device)

            # 动态加噪（每 batch 独立采样 → 防止记忆，提供 σ² 下界）
            if tr.noise_scale > 0:
                x = x + tr.noise_scale * torch.randn_like(x)

            lr = get_lr(global_step, warmup_iters, max_iters, tr.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            loss = model(x, mode=mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr.grad_clip)
            optimizer.step()
            update_ema(ema_model, model, global_step, target_decay=tr.ema_decay)

            total_loss += loss.item()
            n_batches  += 1
            global_step += 1

        train_loss = total_loss / max(n_batches, 1)
        lr_now     = get_lr(global_step, warmup_iters, max_iters, tr.lr)

        if epoch % tr.log_interval == 0:
            # 统一评估：所有实验都用 AR noshuffle 顺序 + AR 模式，公平可比
            test_loss = evaluate(ema_model, canon_loader, device, "AR", tr.noise_scale)

            print(f"epoch {epoch:>4d}/{tr.epochs} | train={train_loss:.4f} | "
                  f"test={test_loss:.4f} | lr={lr_now:.2e}")
            wandb.log({
                "train_loss":  train_loss,
                "test_loss":   test_loss,
                "noise_floor": tr.noise_scale ** 2,
                "lr":          lr_now,
                "epoch":       epoch,
            })
            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch":      epoch,
                    "train_loss": train_loss,
                    "test_loss":  test_loss,
                    "lr":         lr_now,
                }) + "\n")
        else:
            wandb.log({"train_loss": train_loss, "lr": lr_now, "epoch": epoch})

    print(f"\n[done] Metrics → {metrics_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
