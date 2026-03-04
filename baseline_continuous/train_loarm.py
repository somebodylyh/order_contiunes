# baseline_continuous/train_loarm.py
"""
LO-ARM Training Script for ContinuousAOGPT on h-space data.

Algorithm (per-batch):
  1. q_net(x)          → q_logits [B, L]
  2. gumbel_top_k × 2  → z1, z2   [B, L]
  3. sample step k     ~ Uniform[0, L-1]
  4. forward_loarm(z1) → gen_preds1 [B,L,D], pol_logits1 [B,L,L]
     forward_loarm(z2) → gen_preds2 [B,L,D], pol_logits2 [B,L,L]
  5. Compute F1, F2 at step k
  6. RLOO loss → backward

Warm start: loads backbone from best_mdm_Random_model.pt.
"""

import sys, os, math, copy, argparse
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baseline_continuous import config as cfg
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
from baseline_continuous.continuous_loarm import ContinuousLOARM
from baseline_continuous.variational_q import QNetwork
from baseline_continuous.loarm_utils import gumbel_top_k, plackett_luce_logprob
from baseline_continuous.rloo import compute_gen_log_lik, mask_policy_logprob, compute_rloo_loss
from baseline_continuous.eval_utils import evaluate_ar
from baseline_continuous.disk_dataset import create_disk_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='LO-ARM training')
    parser.add_argument('--epochs',         type=int,   default=None)
    parser.add_argument('--batch_size',     type=int,   default=None)
    parser.add_argument('--learning_rate',  type=float, default=None)
    parser.add_argument('--sigma2',         type=float, default=0.5,
                        help='Gaussian noise variance (temperature). Start with 0.5.')
    parser.add_argument('--seed',           type=int,   default=None)
    parser.add_argument('--device',         type=str,   default=None)
    parser.add_argument('--data_dir',       type=str,
                        default=os.path.join(os.path.dirname(__file__), 'data'))
    parser.add_argument('--warmstart',      type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             'checkpoints', 'best_mdm_Random_model.pt'),
                        help='Path to pretrained MDM checkpoint for backbone warm-start')
    parser.add_argument('--wandb_log',      type=str,   default=None)
    return parser.parse_args()


def get_lr(it, warmup_iters, max_iters, learning_rate, min_lr_ratio=0.1):
    min_lr = learning_rate * min_lr_ratio
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def update_ema(ema_model, model, step, target_decay=0.9999):
    decay = min(target_decay, (1 + step) / (10 + step))
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.data, alpha=1 - decay)


def compute_batch_loss(model, q_net, batch, device, sigma2):
    """
    Full RLOO loss computation for one batch.

    Returns:
        loss: scalar, RLOO loss for backprop
        aux:  dict with monitoring scalars (detached)
    """
    init_vectors = batch['init_vectors'].to(device)   # [B, 1, D]
    vectors      = batch['main_vectors'].to(device)   # [B, L, D]
    B, L, D      = vectors.shape

    # ── 1. Sample two orderings from q_θ ────────────────────────────────────
    q_logits = q_net(vectors)                   # [B, L]  (god-view)
    z1 = gumbel_top_k(q_logits)                 # [B, L]
    z2 = gumbel_top_k(q_logits)                 # [B, L]

    # ── 2. Sample step k uniformly ──────────────────────────────────────────
    # Exclude last step (k=L-1): Plackett-Luce log-prob is always 0 there
    # (only one position left), so no gradient flows — skip it.
    k = torch.randint(0, L - 1, ()).item()      # scalar int in [0, L-2]

    # ── 3. Two forward passes ───────────────────────────────────────────────
    gen_preds1, pol_logits1 = model.forward_loarm(vectors, z1, init_vectors)
    gen_preds2, pol_logits2 = model.forward_loarm(vectors, z2, init_vectors)

    # ── 4. Mask policy logits at step k ─────────────────────────────────────
    masked_pol1 = model.apply_policy_mask(pol_logits1, z1)   # [B, L, L]
    masked_pol2 = model.apply_policy_mask(pol_logits2, z2)

    # ── 5. Compute F1, F2 at step k ─────────────────────────────────────────
    # 5a. Generator log-lik: gen_preds[:,k,:] predicts vectors[z[:,k],:]
    target1 = vectors[torch.arange(B), z1[:, k]]   # [B, D]
    target2 = vectors[torch.arange(B), z2[:, k]]
    log_p_gen1 = compute_gen_log_lik(gen_preds1[:, k, :], target1, sigma2)
    log_p_gen2 = compute_gen_log_lik(gen_preds2[:, k, :], target2, sigma2)

    # 5b. Policy log-prob at step k
    chosen1 = z1[:, k]   # [B]
    chosen2 = z2[:, k]
    log_p_pol1 = mask_policy_logprob(masked_pol1[:, k, :], chosen1)
    log_p_pol2 = mask_policy_logprob(masked_pol2[:, k, :], chosen2)

    # 5c. Variational log-prob (Plackett-Luce under q_θ)
    log_q1 = plackett_luce_logprob(q_logits, z1, step_k=k)   # [B]
    log_q2 = plackett_luce_logprob(q_logits, z2, step_k=k)

    # 5d. F = log_p_gen + log_p_pol - log_q
    F1 = log_p_gen1 + log_p_pol1 - log_q1
    F2 = log_p_gen2 + log_p_pol2 - log_q2

    # ── 6. RLOO loss ─────────────────────────────────────────────────────────
    loss = compute_rloo_loss(F1, F2, log_q1, log_q2, L_len=L)

    # Auxiliary metrics (no grad needed)
    with torch.no_grad():
        logits_k = masked_pol1[:, k, :].nan_to_num(nan=0., neginf=-1e9)
        probs_k  = F.softmax(logits_k, dim=-1)
        policy_entropy = -(probs_k * probs_k.clamp(min=1e-30).log()).sum(dim=-1).mean()

    aux = {
        'F1_mean':        F1.mean().item(),
        'log_p_gen1':     log_p_gen1.mean().item(),
        'log_p_pol1':     log_p_pol1.mean().item(),
        'log_q1':         log_q1.mean().item(),
        'policy_entropy': policy_entropy.item(),
        'step_k':         k,
    }
    return loss, aux


def main():
    args = parse_args()
    epochs        = args.epochs        or cfg.epochs
    batch_size    = args.batch_size    or cfg.batch_size
    learning_rate = args.learning_rate or cfg.learning_rate
    seed          = args.seed          or cfg.seed
    device        = args.device        or cfg.device
    sigma2        = args.sigma2
    wandb_log     = cfg.wandb_log
    if args.wandb_log is not None:
        wandb_log = args.wandb_log.lower() in ('true', '1', 'yes')

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print("=" * 60)
    print(f"LO-ARM Training  sigma2={sigma2}")
    print("=" * 60)

    # ── Data ────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_disk_dataloaders(
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        num_chunks=cfg.num_chunks,
    )
    iters_per_epoch = len(train_loader)
    max_iters       = epochs * iters_per_epoch
    warmup_iters    = int(cfg.warmup_iters * max_iters)

    # ── Model: warm-start backbone from MDM checkpoint ──────────────────────
    model_config = ContinuousAOGPTConfig(
        block_size=cfg.block_size, vector_dim=cfg.vector_dim,
        n_layer=cfg.n_layer,       n_head=cfg.n_head,
        n_embd=cfg.n_embd,         dropout=cfg.dropout,
        bias=cfg.bias,             num_init=cfg.num_init,
    )
    base_model = ContinuousAOGPT(model_config)
    if os.path.exists(args.warmstart):
        ckpt = torch.load(args.warmstart, map_location='cpu', weights_only=False)
        base_model.load_state_dict(ckpt['model_state_dict'])
        print(f"  Warm-start backbone from {args.warmstart}")
    else:
        print(f"  WARNING: No warm-start checkpoint found at {args.warmstart}, training from scratch")

    L      = cfg.seq_length - cfg.num_init    # 31
    model  = ContinuousLOARM(base_model).to(device)
    q_net  = QNetwork(vector_dim=cfg.vector_dim, seq_len=L, hidden_dim=256).to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type='cuda' if 'cuda' in device else 'cpu',
    )
    # q_net at 1/3 LR (god-view network, should not dominate)
    q_params = list(q_net.parameters())
    optimizer.add_param_group({'params': q_params, 'lr': learning_rate / 3.0, 'weight_decay': 0.0})

    # ── WandB ────────────────────────────────────────────────────────────────
    if wandb_log:
        import wandb
        wandb.init(project=cfg.wandb_project, name=f'loarm-sigma{sigma2}',
                   group='loarm', config={
                       'sigma2': sigma2, 'vector_dim': cfg.vector_dim,
                       'seq_length': cfg.seq_length, 'n_layer': cfg.n_layer,
                       'n_embd': cfg.n_embd, 'batch_size': batch_size,
                       'learning_rate': learning_rate, 'epochs': epochs,
                   })

    # ── Training loop ────────────────────────────────────────────────────────
    print(f"\nStarting training: {epochs} epochs, {max_iters} total iters")
    best_val_loss = float('inf')
    global_step   = 0
    model.train()
    q_net.train()

    for epoch in range(epochs):
        for batch in train_loader:
            lr = get_lr(global_step, warmup_iters, max_iters, learning_rate)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            loss, aux = compute_batch_loss(model, q_net, batch, device, sigma2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(q_net.parameters()), cfg.grad_clip)
            optimizer.step()
            update_ema(ema_model, model, global_step)

            if global_step % cfg.log_interval == 0:
                print(f"epoch {epoch+1}/{epochs} | iter {global_step:>6d} | "
                      f"loss {loss.item():.4f} | "
                      f"F1={aux['F1_mean']:.3f} | "
                      f"pol_ent={aux['policy_entropy']:.3f} | "
                      f"lr {lr:.2e}")
                if wandb_log:
                    import wandb
                    wandb.log({'train/loss': loss.item(), 'train/lr': lr,
                               'train/F1_mean': aux['F1_mean'],
                               'train/policy_entropy': aux['policy_entropy'],
                               'train/log_p_gen': aux['log_p_gen1'],
                               'train/log_p_pol': aux['log_p_pol1'],
                               'epoch': epoch}, step=global_step)

            if global_step % cfg.eval_interval == 0 and global_step > 0:
                eval_results = evaluate_ar(ema_model.base, val_loader, device)
                print(f"  [eval] val_loss: {eval_results['val_loss']:.6f} | "
                      f"cos_sim: {eval_results['val_cos_sim']:.4f}")
                if wandb_log:
                    import wandb
                    wandb.log({'val/loss': eval_results['val_loss'],
                               'val/cos_sim': eval_results['val_cos_sim']},
                              step=global_step)

                if cfg.save_best_model and eval_results['val_loss'] < best_val_loss:
                    best_val_loss = eval_results['val_loss']
                    save_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({
                        'model_state_dict':     ema_model.state_dict(),
                        'raw_model_state_dict': model.state_dict(),
                        'q_net_state_dict':     q_net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config':               model_config,
                        'sigma2':               sigma2,
                        'epoch':                epoch,
                        'global_step':          global_step,
                        'val_loss':             best_val_loss,
                    }, os.path.join(save_dir, f'best_loarm_sigma{sigma2}_model.pt'))
                    print(f"  [save] Best model saved (val_loss: {best_val_loss:.4f})")

                model.train()
                q_net.train()

            global_step += 1

    if wandb_log:
        import wandb
        wandb.finish()
    print("\nTraining complete.")


if __name__ == '__main__':
    main()
