"""
MDM (Masked Diffusion Model / Random Order) Training Script for ContinuousAOGPT

Lightweight single-GPU training with random order.
Supports curriculum learning via Random_CL mode.
Uses pre-generated disk data (memmap) for memory efficiency.
"""

import sys
import os
import math
import copy
import argparse

import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baseline_continuous import config as cfg
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
from baseline_continuous.eval_utils import evaluate_ar
from baseline_continuous.disk_dataset import create_disk_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='MDM training for ContinuousAOGPT')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--wandb_log', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'data'))
    parser.add_argument('--mode', type=str, default='Random', choices=['Random', 'Random_CL'],
                        help='Training mode: Random or Random_CL (curriculum learning)')
    parser.add_argument('--random_ratio', type=float, default=0.5,
                        help='Ratio of random orders in Random_CL mode')
    return parser.parse_args()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.data, alpha=1 - decay)


def get_lr(it, warmup_iters, max_iters, learning_rate, min_lr_ratio=0.1):
    """Cosine learning rate schedule with warmup."""
    min_lr = learning_rate * min_lr_ratio
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def main():
    args = parse_args()

    epochs = args.epochs if args.epochs is not None else cfg.epochs
    wandb_log = cfg.wandb_log
    if args.wandb_log is not None:
        wandb_log = args.wandb_log.lower() in ('true', '1', 'yes')
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    learning_rate = args.learning_rate if args.learning_rate is not None else cfg.learning_rate
    seed = args.seed if args.seed is not None else cfg.seed
    device = args.device if args.device is not None else cfg.device
    train_mode = args.mode
    random_ratio = args.random_ratio

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print("=" * 60)
    print(f"ContinuousAOGPT — MDM Training (mode={train_mode})")
    print("=" * 60)

    # Load data from disk (memmap, near-zero memory)
    print("\nLoading data from disk...")
    train_loader, val_loader, test_loader = create_disk_dataloaders(
        data_dir=args.data_dir,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        num_chunks=cfg.num_chunks,
    )

    # Compute total iterations for LR schedule
    iters_per_epoch = len(train_loader)
    max_iters = epochs * iters_per_epoch
    warmup_iters = int(cfg.warmup_iters * max_iters) if cfg.warmup_iters < 1 else int(cfg.warmup_iters)
    print(f"  {iters_per_epoch} iters/epoch x {epochs} epochs = {max_iters} total iters")

    # Create model
    print("\nCreating model...")
    model_config = ContinuousAOGPTConfig(
        block_size=cfg.block_size,
        vector_dim=cfg.vector_dim,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    model = ContinuousAOGPT(model_config)
    model = model.to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()

    optimizer = model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type='cuda' if 'cuda' in device else 'cpu',
    )

    if wandb_log:
        import wandb
        wandb.init(project=cfg.wandb_project, name=f'mdm-{train_mode}', group='baseline-comparison', config={
            'mode': train_mode,
            'random_ratio': random_ratio if train_mode == 'Random_CL' else None,
            'vector_dim': cfg.vector_dim,
            'seq_length': cfg.seq_length,
            'n_layer': cfg.n_layer,
            'n_head': cfg.n_head,
            'n_embd': cfg.n_embd,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'max_iters': max_iters,
            'warmup_iters': warmup_iters,
            'weight_decay': cfg.weight_decay,
            'dependency_window': cfg.dependency_window,
            'num_matrices': cfg.num_matrices,
            'train_samples': len(train_loader.dataset),
        })

    # Training loop (epoch-based)
    print(f"\nStarting training for {epochs} epochs...")
    best_val_loss = float('inf')
    global_step = 0
    model.train()

    for epoch in range(epochs):
        for batch in train_loader:
            vectors = batch['shuffled_vectors'].to(device)

            lr = get_lr(global_step, warmup_iters, max_iters, learning_rate)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if train_mode == 'Random':
                _, loss = model(vectors, mode='Random')
            elif train_mode == 'Random_CL':
                _, loss = model(vectors, mode='Random_CL', random_ratio=random_ratio)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            update_ema(ema_model, model)

            if global_step % cfg.log_interval == 0:
                print(f"epoch {epoch+1}/{epochs} | iter {global_step:>6d} | loss {loss.item():.4f} | lr {lr:.2e}")
                if wandb_log:
                    import wandb
                    wandb.log({'train/loss': loss.item(), 'train/lr': lr, 'epoch': epoch}, step=global_step)

            if global_step % cfg.eval_interval == 0 and global_step > 0:
                eval_results = evaluate_ar(ema_model, val_loader, device)
                print(f"  [eval] val_loss: {eval_results['val_loss']:.4f} | val_cos_sim: {eval_results['val_cos_sim']:.4f}")
                if wandb_log:
                    import wandb
                    wandb.log({
                        'val/loss': eval_results['val_loss'],
                        'val/cos_sim': eval_results['val_cos_sim'],
                    }, step=global_step)

                if cfg.save_best_model and eval_results['val_loss'] < best_val_loss:
                    best_val_loss = eval_results['val_loss']
                    save_path = os.path.join(os.path.dirname(__file__), 'checkpoints')
                    os.makedirs(save_path, exist_ok=True)
                    torch.save({
                        'model_state_dict': ema_model.state_dict(),
                        'raw_model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': model_config,
                        'epoch': epoch,
                        'global_step': global_step,
                        'val_loss': best_val_loss,
                    }, os.path.join(save_path, f'best_mdm_{train_mode}_model.pt'))
                    print(f"  [save] Best model saved (val_loss: {best_val_loss:.4f})")

                model.train()

            global_step += 1

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    final_results = evaluate_ar(ema_model, val_loader, device)
    print(f"  val_loss: {final_results['val_loss']:.4f}")
    print(f"  val_cos_sim: {final_results['val_cos_sim']:.4f}")

    test_results = evaluate_ar(ema_model, test_loader, device)
    print(f"  test_loss: {test_results['val_loss']:.4f}")
    print(f"  test_cos_sim: {test_results['val_cos_sim']:.4f}")

    if wandb_log:
        import wandb
        wandb.log({
            'final/val_loss': final_results['val_loss'],
            'final/val_cos_sim': final_results['val_cos_sim'],
            'final/test_loss': test_results['val_loss'],
            'final/test_cos_sim': test_results['val_cos_sim'],
        })
        wandb.finish()

    print("\nTraining complete.")


if __name__ == '__main__':
    main()
