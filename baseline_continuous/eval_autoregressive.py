"""
Evaluate MDM and AR models with AUTOREGRESSIVE GENERATION.

Unlike teacher-forcing eval (eval_order.py), this generates tokens step-by-step
using the model's own predictions as context. This reveals whether MDM learned
causal structure, because:

- Teacher-forcing: model always sees ground truth context → order doesn't matter
- Autoregressive: model sees its OWN predictions → errors compound
  - Causal order: predict causes first → good context → small errors
  - Random order: predict effects before causes → bad context → errors snowball

If MDM learned causal structure, GT order generation should be MUCH better than
random order generation.
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
def autoregressive_generate(model, batch_size, seq_length, orders, device):
    """
    Generate a full sequence autoregressively using the model's own predictions.

    At each step t:
      - Context = [None token, pred_σ(0), pred_σ(1), ..., pred_σ(t-1)]  (model's own outputs)
      - Target position = σ(t)  (told via AdaLN)
      - Predict x_σ(t)

    Returns: generated [B, L, D] in the ORIGINAL position order (unshuffled).
    """
    B, L = batch_size, seq_length
    D = model.config.vector_dim
    n_embd = model.config.n_embd

    # Storage for generated vectors in shuffled order
    generated_shuffled = torch.zeros(B, L, D, device=device)

    # Build position embeddings
    pos = torch.arange(0, L + 1, dtype=torch.long, device=device)
    pos_emb_all = model.transformer.wpe(pos)  # [L+1, n_embd]
    target_pos_emb_all = model.transformer.wtpe(pos[:L])  # [L, 128]

    # Shuffled position embeddings
    pos_emb_shuffled = model.shuffle(
        pos_emb_all[1:].unsqueeze(0).expand(B, -1, -1), orders
    )  # [B, L, n_embd]
    target_pos_emb_shuffled = model.shuffle(
        target_pos_emb_all.unsqueeze(0).expand(B, -1, -1), orders
    )  # [B, L, 128]

    # Start with just the [None] token
    # We'll build up the KV sequence incrementally
    none_emb = model.none_token.expand(B, -1, -1)  # [B, 1, n_embd]
    prefix_pos = pos_emb_all[0].unsqueeze(0).unsqueeze(0).expand(B, -1, -1)  # [B, 1, n_embd]

    # Collect all token embeddings as we generate
    all_tok_emb = [none_emb]  # start with [None]
    all_pos_emb = [prefix_pos]  # position for [None]

    for step in range(L):
        # Current sequence length (including [None])
        cur_len = step + 1

        # Build input sequence: [None, pred_0, pred_1, ..., pred_{step-1}]
        tok_seq = torch.cat(all_tok_emb, dim=1)  # [B, cur_len, n_embd]
        pos_seq = torch.cat(all_pos_emb, dim=1)  # [B, cur_len, n_embd]
        x = tok_seq + pos_seq

        # Target position embedding for each position in sequence
        # For positions 0..step-1: their target_pos (already generated)
        # For position step (current): target_pos of orders[:, step]
        target_pos_list = [target_pos_emb_shuffled[:, :step]]  # [B, step, 128]
        # Prepend zeros for [None] position
        zeros_for_none = torch.zeros(B, 1, 128, device=device)
        target_pos_seq = torch.cat([zeros_for_none] + [target_pos_list[0]], dim=1)

        # But we need target_pos for the NEXT position (what we're about to predict)
        # The AdaLN at position i conditions on "what position am I predicting next"
        # In forward_fn: target_pos_emb_prefix[i] = target_pos of orders[i]
        # And the prediction at position i predicts orders[i]
        # So we need: [target_of_orders[0], target_of_orders[1], ..., target_of_orders[step], 0]
        # But we only feed cur_len tokens, so:
        next_target = target_pos_emb_shuffled[:, step:step+1]  # [B, 1, 128]
        target_pos_for_forward = torch.cat([
            target_pos_emb_shuffled[:, :step],  # [B, step, 128] for previous positions
            next_target,                          # [B, 1, 128] for current prediction
        ], dim=1)  # [B, step+1, 128]
        # Prepend is not needed — [None] position uses target_pos of orders[0]
        # Actually re-reading forward_fn more carefully:
        # target_pos_emb_final = [shuffled_target_pos (L positions), zeros (1 position)]
        # So position 0 ([None]) gets target_pos of orders[0], position 1 gets target_pos of orders[1], etc.
        # For our step-by-step: we have cur_len positions, the last one predicts orders[step]
        # target_pos_emb_final for our sequence:
        target_pos_final = torch.cat([
            target_pos_emb_shuffled[:, :step+1],  # [B, step+1, 128]
        ], dim=1)  # This covers [None] pos through current pos

        # Wait - in the original forward_fn, target_pos_emb_final has L+1 entries:
        # [target_pos_shuffled[0], ..., target_pos_shuffled[L-1], zeros]
        # And the sequence is [None, vec_shuffled[0], ..., vec_shuffled[L-1]]
        # So position index 0 ([None]) receives target_pos_shuffled[0] via AdaLN
        # and its output predicts vec_shuffled[0] (= vec at orders[0])
        #
        # For step-by-step at step `step`:
        # Input: [None, pred_0, ..., pred_{step-1}]  (length = step+1)
        # target_pos: [tpe_shuffled[0], tpe_shuffled[1], ..., tpe_shuffled[step]]
        # The last position's output predicts vec_shuffled[step]

        # Forward through transformer
        x = model.transformer.drop(x)
        for block in model.transformer.h:
            x = block(x, target_pos_final)
        x = model.transformer.final_layer(x, target_pos_final)

        # Get prediction from the last position
        pred = model.output_proj(x[:, -1:, :])  # [B, 1, D]
        generated_shuffled[:, step] = pred.squeeze(1)

        # Prepare embedding for next step
        pred_emb = model.input_proj(pred)  # [B, 1, n_embd]
        pred_pos = pos_emb_shuffled[:, step:step+1]  # [B, 1, n_embd]
        all_tok_emb.append(pred_emb)
        all_pos_emb.append(pred_pos)

    # Unshuffle back to original order
    generated = model.unshuffle(generated_shuffled, orders)
    return generated


@torch.no_grad()
def eval_autoregressive(model, dataloader, device, num_mc_samples=10, max_batches=20):
    """
    Evaluate autoregressive generation quality with different orders.

    For each batch:
    1. Generate with GT (causal) order
    2. Generate with ascending order
    3. Generate with random orders (MC average)

    Compare generated sequences against ground truth using cosine similarity.
    """
    results = {
        'gt_cos_sim': [], 'gt_loss': [],
        'asc_cos_sim': [], 'asc_loss': [],
        'rand_cos_sim': [], 'rand_loss': [],
    }

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        shuffled_vectors = batch['shuffled_vectors'].to(device)
        vectors = batch['vectors'].to(device)  # ground truth in original order
        gt_order = batch['order'].to(device)
        B, L, D = shuffled_vectors.shape

        print(f"  batch {i+1}/{max_batches}...", end='\r')

        # 1. Generate with GT (causal) order
        gen_gt = autoregressive_generate(model, B, L, gt_order, device)
        cos_gt = F.cosine_similarity(gen_gt, vectors, dim=-1).mean().item()
        loss_gt = 1.0 - cos_gt
        results['gt_cos_sim'].append(cos_gt)
        results['gt_loss'].append(loss_gt)

        # 2. Generate with ascending order
        asc_orders = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        gen_asc = autoregressive_generate(model, B, L, asc_orders, device)
        # Unshuffle is identity for ascending, so compare against shuffled_vectors
        # Actually no — autoregressive_generate already unshuffles.
        # For ascending order on shuffled input, the generated sequence in original order
        # should be compared with vectors (original order ground truth)
        cos_asc = F.cosine_similarity(gen_asc, vectors, dim=-1).mean().item()
        loss_asc = 1.0 - cos_asc
        results['asc_cos_sim'].append(cos_asc)
        results['asc_loss'].append(loss_asc)

        # 3. Random orders (MC)
        mc_cos = []
        for _ in range(num_mc_samples):
            rand_orders = model.sample_random_orders(shuffled_vectors)
            gen_rand = autoregressive_generate(model, B, L, rand_orders, device)
            cos_rand = F.cosine_similarity(gen_rand, vectors, dim=-1).mean().item()
            mc_cos.append(cos_rand)
        avg_cos_rand = sum(mc_cos) / len(mc_cos)
        results['rand_cos_sim'].append(avg_cos_rand)
        results['rand_loss'].append(1.0 - avg_cos_rand)

    # Average across batches
    return {k: sum(v) / len(v) for k, v in results.items()}


def main():
    device = cfg.device
    seed = cfg.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    print("Creating dataloaders from disk data...")
    _, val_loader, test_loader = create_disk_dataloaders(
        data_dir=data_dir,
        batch_size=64,  # smaller batch for AR generation (more memory intensive)
        num_workers=cfg.num_workers,
        num_chunks=cfg.num_chunks,
    )

    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')

    models_to_eval = [
        ('MDM (Random)', os.path.join(ckpt_dir, 'best_mdm_Random_model.pt')),
        ('AR (shuffled)', os.path.join(ckpt_dir, 'best_ar_model.pt')),
    ]

    num_mc = 10
    max_batches = 10

    for name, path in models_to_eval:
        if not os.path.exists(path):
            print(f"\nSkipping {name}: checkpoint not found at {path}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"  Mode: Autoregressive Generation")
        print(f"  MC samples: {num_mc}, Max batches: {max_batches}")
        print(f"{'='*60}")
        model = load_model(path, device)

        print("\n--- Validation Set (AR generation) ---")
        val_results = eval_autoregressive(
            model, val_loader, device,
            num_mc_samples=num_mc, max_batches=max_batches
        )
        print(f"\n  {'Order':<20} {'Loss':>10} {'Cos Sim':>10}")
        print(f"  {'-'*40}")
        print(f"  {'GT (causal)':<20} {val_results['gt_loss']:>10.4f} {val_results['gt_cos_sim']:>10.4f}")
        print(f"  {'Ascending':<20} {val_results['asc_loss']:>10.4f} {val_results['asc_cos_sim']:>10.4f}")
        print(f"  {'Random (MC avg)':<20} {val_results['rand_loss']:>10.4f} {val_results['rand_cos_sim']:>10.4f}")

        print("\n--- Analysis ---")
        gt_vs_rand = val_results['rand_loss'] - val_results['gt_loss']
        gt_vs_asc = val_results['asc_loss'] - val_results['gt_loss']
        print(f"  GT advantage over random:    {gt_vs_rand:+.4f} ({'GT better' if gt_vs_rand > 0 else 'Random better'})")
        print(f"  GT advantage over ascending: {gt_vs_asc:+.4f} ({'GT better' if gt_vs_asc > 0 else 'Asc better'})")

        print("\n--- Test Set (AR generation) ---")
        test_results = eval_autoregressive(
            model, test_loader, device,
            num_mc_samples=num_mc, max_batches=max_batches
        )
        print(f"\n  {'Order':<20} {'Loss':>10} {'Cos Sim':>10}")
        print(f"  {'-'*40}")
        print(f"  {'GT (causal)':<20} {test_results['gt_loss']:>10.4f} {test_results['gt_cos_sim']:>10.4f}")
        print(f"  {'Ascending':<20} {test_results['asc_loss']:>10.4f} {test_results['asc_cos_sim']:>10.4f}")
        print(f"  {'Random (MC avg)':<20} {test_results['rand_loss']:>10.4f} {test_results['rand_cos_sim']:>10.4f}")


if __name__ == '__main__':
    main()
