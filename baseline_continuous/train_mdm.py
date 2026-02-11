"""
MDM (Masked Diffusion Model / Random Order) Training Script for ContinuousAOGPT

Lightweight single-GPU training with random order.
Supports curriculum learning via Random_CL mode.
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
    parser = argparse.ArgumentParser(description='MDM training for ContinuousAOGPT')
    parser.add_argument('--max_iters', type=int, default=None)
    parser.add_argument('--wandb_log', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--mode', type=str, default='Random', choices=['Random', 'Random_CL'],
                        help='Training mode: Random or Random_CL (curriculum learning)')
    parser.add_argument('--random_ratio', type=float, default=0.5,
                        help='Ratio of random orders in Random_CL mode')
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
def evaluate(model, val_loader, device, num_mc_samples=5, max_batches=None):
    """
    Evaluate on validation set with both AR and Random order losses.

    Uses Monte Carlo estimation for random order loss (average over multiple random orders).
    """
    model.eval()
    total_ar_loss = 0.0
    total_ar_cos_sim = 0.0
    total_random_loss = 0.0
    total_random_cos_sim = 0.0
    total_batches = 0

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        vectors = batch['shuffled_vectors'].to(device)

        # AR loss (ascending order on shuffled data)
        predictions_ar, loss_ar = model(vectors, mode='AR')
        shift_preds_ar = predictions_ar[:, :-1, :]
        cos_sim_ar = F.cosine_similarity(shift_preds_ar, vectors, dim=-1).mean()

        # Random loss (Monte Carlo: average over multiple random orders)
        mc_losses = []
        mc_cos_sims = []
        for _ in range(num_mc_samples):
            # Generate orders first, then pass to model to ensure consistency
            orders_rand = model.sample_random_orders(vectors)
            predictions_rand, loss_rand = model(vectors, mode=None, orders=orders_rand)
            shift_preds_rand = predictions_rand[:, :-1, :]
            targets_rand = model.shuffle(vectors, orders_rand)
            cos_sim_rand = F.cosine_similarity(shift_preds_rand, targets_rand, dim=-1).mean()
            mc_losses.append(loss_rand.item())
            mc_cos_sims.append(cos_sim_rand.item())

        total_ar_loss += loss_ar.item()
        total_ar_cos_sim += cos_sim_ar.item()
        total_random_loss += sum(mc_losses) / len(mc_losses)
        total_random_cos_sim += sum(mc_cos_sims) / len(mc_cos_sims)
        total_batches += 1

    n = max(total_batches, 1)
    model.train()
    return {
        'val_ar_loss': total_ar_loss / n,
        'val_ar_cos_sim': total_ar_cos_sim / n,
        'val_random_loss': total_random_loss / n,
        'val_random_cos_sim': total_random_cos_sim / n,
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
    train_mode = args.mode
    random_ratio = args.random_ratio

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    print("=" * 60)
    print(f"ContinuousAOGPT — MDM Training (mode={train_mode})")
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
        wandb.init(project=cfg.wandb_project, name=f'mdm-{train_mode}', config={
            'mode': train_mode,
            'random_ratio': random_ratio if train_mode == 'Random_CL' else None,
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

        vectors = batch['shuffled_vectors'].to(device)

        # Set learning rate
        lr = get_lr(it, cfg.warmup_iters, max_iters, learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward with random/curriculum order
        if train_mode == 'Random':
            predictions, loss = model(vectors, mode='Random')
        elif train_mode == 'Random_CL':
            predictions, loss = model(vectors, mode='Random_CL', random_ratio=random_ratio)

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
            eval_results = evaluate(model, val_loader, device)
            print(f"  [eval] ar_loss: {eval_results['val_ar_loss']:.4f} | "
                  f"ar_cos_sim: {eval_results['val_ar_cos_sim']:.4f} | "
                  f"random_loss: {eval_results['val_random_loss']:.4f} | "
                  f"random_cos_sim: {eval_results['val_random_cos_sim']:.4f}")

            if wandb_log:
                import wandb
                wandb.log({
                    'val/ar_loss': eval_results['val_ar_loss'],
                    'val/ar_cos_sim': eval_results['val_ar_cos_sim'],
                    'val/random_loss': eval_results['val_random_loss'],
                    'val/random_cos_sim': eval_results['val_random_cos_sim'],
                    'iter': it,
                })

            # Save best model (using random loss as metric)
            val_metric = eval_results['val_random_loss']
            if cfg.save_best_model and val_metric < best_val_loss:
                best_val_loss = val_metric
                save_path = os.path.join(os.path.dirname(__file__), 'checkpoints')
                os.makedirs(save_path, exist_ok=True)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': model_config,
                    'iter': it,
                    'val_loss': best_val_loss,
                }, os.path.join(save_path, f'best_mdm_{train_mode}_model.pt'))
                print(f"  [save] Best model saved (val_random_loss: {best_val_loss:.4f})")

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    final_results = evaluate(model, val_loader, device)
    print(f"  val_ar_loss: {final_results['val_ar_loss']:.4f}")
    print(f"  val_ar_cos_sim: {final_results['val_ar_cos_sim']:.4f}")
    print(f"  val_random_loss: {final_results['val_random_loss']:.4f}")
    print(f"  val_random_cos_sim: {final_results['val_random_cos_sim']:.4f}")

    test_results = evaluate(model, test_loader, device)
    print(f"  test_ar_loss: {test_results['val_ar_loss']:.4f}")
    print(f"  test_ar_cos_sim: {test_results['val_ar_cos_sim']:.4f}")
    print(f"  test_random_loss: {test_results['val_random_loss']:.4f}")
    print(f"  test_random_cos_sim: {test_results['val_random_cos_sim']:.4f}")

    if wandb_log:
        import wandb
        wandb.log({
            'final/val_ar_loss': final_results['val_ar_loss'],
            'final/val_ar_cos_sim': final_results['val_ar_cos_sim'],
            'final/val_random_loss': final_results['val_random_loss'],
            'final/val_random_cos_sim': final_results['val_random_cos_sim'],
            'final/test_ar_loss': test_results['val_ar_loss'],
            'final/test_ar_cos_sim': test_results['val_ar_cos_sim'],
            'final/test_random_loss': test_results['val_random_loss'],
            'final/test_random_cos_sim': test_results['val_random_cos_sim'],
        })
        wandb.finish()

    print("\nTraining complete.")


if __name__ == '__main__':
    main()
