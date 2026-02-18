"""
Evaluate MDM and AR models with different generation orders.

Tests whether MDM learned the optimal (causal) order by comparing:
1. Ground truth order (unshuffle back to original causal sequence)
2. Ascending order (fixed left-to-right, same as AR)
3. Random orders (Monte Carlo average)
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baseline_continuous import config as cfg
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
from baseline_continuous.disk_dataset import create_disk_dataloaders


def load_model(checkpoint_path, device):
    """Load a trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt['config']
    model = ContinuousAOGPT(model_config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    step = ckpt.get('iter', ckpt.get('global_step', '?'))
    print(f"Loaded checkpoint from step {step}, val_loss={ckpt['val_loss']:.4f}")
    return model


@torch.no_grad()
def eval_with_order(model, vectors, orders):
    """Evaluate model with specific orders. Returns per-sample loss."""
    predictions, loss = model(vectors, mode=None, orders=orders)
    # Also compute per-sample loss
    vectors_shuffled = model.shuffle(vectors, orders)
    shift_preds = predictions[:, :-1, :]
    cos_sim = F.cosine_similarity(shift_preds, vectors_shuffled, dim=-1)  # [B, L]
    per_sample_loss = (1.0 - cos_sim).mean(dim=-1)  # [B]
    return loss.item(), per_sample_loss


@torch.no_grad()
def evaluate_all_orders(model, dataloader, device, num_mc_samples=20, max_batches=None):
    """
    Evaluate a model with ground truth, ascending, and random orders.
    """
    results = {
        'gt_losses': [],
        'asc_losses': [],
        'random_losses': [],
    }

    for i, batch in enumerate(dataloader):
        if max_batches is not None and i >= max_batches:
            break

        shuffled_vectors = batch['shuffled_vectors'].to(device)
        gt_order = batch['order'].to(device)  # ground truth causal order
        B, L, D = shuffled_vectors.shape

        # 1. Ground truth order (unshuffle -> original causal sequence)
        gt_loss, gt_per_sample = eval_with_order(model, shuffled_vectors, gt_order)
        results['gt_losses'].append(gt_loss)

        # 2. Ascending order (0, 1, 2, ..., L-1)
        asc_orders = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        asc_loss, asc_per_sample = eval_with_order(model, shuffled_vectors, asc_orders)
        results['asc_losses'].append(asc_loss)

        # 3. Random orders (Monte Carlo)
        mc_losses = []
        for _ in range(num_mc_samples):
            rand_orders = model.sample_random_orders(shuffled_vectors)
            rand_loss, _ = eval_with_order(model, shuffled_vectors, rand_orders)
            mc_losses.append(rand_loss)
        results['random_losses'].append(sum(mc_losses) / len(mc_losses))

    return {k: sum(v) / len(v) for k, v in results.items()}


def main():
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    print("Creating dataloaders from disk data...")
    train_loader, val_loader, test_loader = create_disk_dataloaders(
        data_dir=data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        num_chunks=cfg.num_chunks,
    )

    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')

    # Evaluate each model
    models_to_eval = [
        ('MDM (Random)', os.path.join(ckpt_dir, 'best_mdm_Random_model.pt')),
        ('AR (shuffled)', os.path.join(ckpt_dir, 'best_ar_model.pt')),
        ('AR (no shuffle)', os.path.join(ckpt_dir, 'best_ar_no_shuffle_model.pt')),
    ]

    for name, path in models_to_eval:
        if not os.path.exists(path):
            print(f"\nSkipping {name}: checkpoint not found at {path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        model = load_model(path, device)

        print("\n--- Validation Set (20 batches) ---")
        val_results = evaluate_all_orders(model, val_loader, device, num_mc_samples=20, max_batches=20)
        print(f"  Ground truth order loss: {val_results['gt_losses']:.4f}")
        print(f"  Ascending order loss:    {val_results['asc_losses']:.4f}")
        print(f"  Random order loss (MC):  {val_results['random_losses']:.4f}")

        print("\n--- Test Set (20 batches) ---")
        test_results = evaluate_all_orders(model, test_loader, device, num_mc_samples=20, max_batches=20)
        print(f"  Ground truth order loss: {test_results['gt_losses']:.4f}")
        print(f"  Ascending order loss:    {test_results['asc_losses']:.4f}")
        print(f"  Random order loss (MC):  {test_results['random_losses']:.4f}")

        # Analysis
        print("\n--- Analysis ---")
        gt_vs_rand = val_results['random_losses'] - val_results['gt_losses']
        gt_vs_asc = val_results['asc_losses'] - val_results['gt_losses']
        print(f"  GT order advantage over random: {gt_vs_rand:.4f} ({'better' if gt_vs_rand > 0 else 'worse'})")
        print(f"  GT order advantage over ascending: {gt_vs_asc:.4f} ({'better' if gt_vs_asc > 0 else 'worse'})")


if __name__ == '__main__':
    main()
