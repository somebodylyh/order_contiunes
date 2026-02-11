"""
AR (Autoregressive) Training Script for ContinuousAOGPT

Lightweight single-GPU training with ascending order.
Input: shuffled vectors (block-wise shuffled, ground truth order NOT leaked).
"""

import sys
import os
import math
import argparse
import time

import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baseline_continuous import config as cfg
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
from linear.continuous_data_generator import ContinuousDenseARGenerator
from linear.continuous_dataset import create_continuous_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='AR training for ContinuousAOGPT')
    parser.add_argument('--max_iters', type=int, default=None)
    parser.add_argument('--wandb_log', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--no_shuffle', action='store_true',
                        help='Use original ordered vectors instead of shuffled (pure AR baseline)')
    return parser.parse_args()


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


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=None, no_shuffle=False):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    total_cos_sim = 0.0
    total_batches = 0

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        vectors = batch['vectors'].to(device) if no_shuffle else batch['shuffled_vectors'].to(device)
        predictions, loss = model(vectors, mode='AR')

        # Compute cosine similarity
        shift_preds = predictions[:, :-1, :]
        cos_sim = F.cosine_similarity(shift_preds, vectors, dim=-1).mean()

        total_loss += loss.item()
        total_cos_sim += cos_sim.item()
        total_batches += 1

    model.train()
    return {
        'val_loss': total_loss / max(total_batches, 1),
        'val_cos_sim': total_cos_sim / max(total_batches, 1),
    }


def main():
    args = parse_args()

    # Override config with command line args
    max_iters = args.max_iters if args.max_iters is not None else cfg.max_iters
    wandb_log = cfg.wandb_log
    if args.wandb_log is not None:
        wandb_log = args.wandb_log.lower() in ('true', '1', 'yes')
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    learning_rate = args.learning_rate if args.learning_rate is not None else cfg.learning_rate
    seed = args.seed if args.seed is not None else cfg.seed
    device = args.device if args.device is not None else cfg.device
    no_shuffle = args.no_shuffle

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print("=" * 60)
    print(f"ContinuousAOGPT — AR Training {'(no shuffle)' if no_shuffle else '(shuffled)'}")
    print("=" * 60)

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader, test_loader, generator = create_continuous_dataloaders(
        vector_dim=cfg.vector_dim,
        seq_length=cfg.seq_length,
        dependency_window=cfg.dependency_window,
        num_matrices=cfg.num_matrices,
        train_samples=cfg.train_samples,
        val_samples=cfg.val_samples,
        test_samples=cfg.test_samples,
        batch_size=batch_size,
        num_workers=16,
        seed=seed,
        train_init_mode=cfg.train_init_mode,
        val_init_mode=cfg.val_init_mode,
        num_chunks=cfg.num_chunks,
        noise_scale=cfg.noise_scale,
    )

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

    # Optimizer
    optimizer = model.configure_optimizers(
        weight_decay=cfg.weight_decay,
        learning_rate=learning_rate,
        betas=(0.9, 0.95),
        device_type='cuda' if 'cuda' in device else 'cpu',
    )

    # Wandb
    if wandb_log:
        import wandb
        run_name = 'ar-no-shuffle' if no_shuffle else 'ar-shuffled'
        wandb.init(project=cfg.wandb_project, name=run_name, config={
            'mode': 'AR',
            'no_shuffle': no_shuffle,
            'vector_dim': cfg.vector_dim,
            'seq_length': cfg.seq_length,
            'n_layer': cfg.n_layer,
            'n_head': cfg.n_head,
            'n_embd': cfg.n_embd,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'max_iters': max_iters,
            'warmup_iters': cfg.warmup_iters,
            'weight_decay': cfg.weight_decay,
            'dependency_window': cfg.dependency_window,
            'num_matrices': cfg.num_matrices,
        })

    # Training loop
    print(f"\nStarting training for {max_iters} iterations...")
    best_val_loss = float('inf')
    train_iter = iter(train_loader)
    model.train()

    for it in range(max_iters):
        # Get batch (cycle through dataloader)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Use original ordered vectors if no_shuffle, else use shuffled
        vectors = batch['vectors'].to(device) if no_shuffle else batch['shuffled_vectors'].to(device)

        # Set learning rate
        lr = get_lr(it, cfg.warmup_iters, max_iters, learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward
        predictions, loss = model(vectors, mode='AR')

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Logging
        if it % cfg.log_interval == 0:
            print(f"iter {it:>6d} | loss {loss.item():.4f} | lr {lr:.2e}")
            if wandb_log:
                import wandb
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': lr,
                    'iter': it,
                })

        # Evaluation
        if it % cfg.eval_interval == 0 and it > 0:
            eval_results = evaluate(model, val_loader, device, no_shuffle=no_shuffle)
            print(f"  [eval] val_loss: {eval_results['val_loss']:.4f} | val_cos_sim: {eval_results['val_cos_sim']:.4f}")

            if wandb_log:
                import wandb
                wandb.log({
                    'val/loss': eval_results['val_loss'],
                    'val/cos_sim': eval_results['val_cos_sim'],
                    'iter': it,
                })

            # Save best model
            if cfg.save_best_model and eval_results['val_loss'] < best_val_loss:
                best_val_loss = eval_results['val_loss']
                save_path = os.path.join(os.path.dirname(__file__), 'checkpoints')
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': model_config,
                    'iter': it,
                    'val_loss': best_val_loss,
                }, os.path.join(save_path, 'best_ar_no_shuffle_model.pt' if no_shuffle else 'best_ar_model.pt'))
                print(f"  [save] Best model saved (val_loss: {best_val_loss:.4f})")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    final_results = evaluate(model, val_loader, device, no_shuffle=no_shuffle)
    print(f"  val_loss: {final_results['val_loss']:.4f}")
    print(f"  val_cos_sim: {final_results['val_cos_sim']:.4f}")

    test_results = evaluate(model, test_loader, device, no_shuffle=no_shuffle)
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
