# baseline_continuous/eval_loarm.py
"""
LO-ARM Evaluation: order recovery metrics.

For a trained ContinuousLOARM model, evaluates:
  1. Causal advantage     = random_loss - policy_greedy_loss
  2. Policy Kendall τ     = correlation between policy's greedy order and [0,..,L-1]
  3. Policy entropy/step  = H(p^z at each step), averaged over test set
  4. Policy log-prob of causal order  = Σ_k log p^z(k | 0..k-1)
  5. Standard AR loss (generator quality, comparable to MDM baseline)

Policy greedy order (inference mode):
  At step k, apply policy to current context → pick argmax position → reveal it.
  This uses only p^z_θ (no q_net needed), so it is a true inference-time metric.
"""

import sys, os
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import kendalltau

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baseline_continuous import config as cfg
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
from baseline_continuous.continuous_loarm import ContinuousLOARM
from baseline_continuous.eval_utils import evaluate_ar
from baseline_continuous.disk_dataset import create_disk_dataloaders


def load_loarm(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    base = ContinuousAOGPT(ckpt['config'])
    model = ContinuousLOARM(base)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  Loaded {os.path.basename(path)}  val_loss={ckpt['val_loss']:.4f}  sigma2={ckpt.get('sigma2','?')}")
    return model


@torch.no_grad()
def policy_greedy_order(model, vectors, init_vectors):
    """
    Greedy order from the policy: at each step k, pick argmax of p^z_θ.

    Returns:
        greedy_orders: [B, L] long tensor
    """
    B, L, D = vectors.shape
    device  = vectors.device
    ni      = init_vectors.shape[1]

    revealed  = torch.zeros(B, L, dtype=torch.bool, device=device)
    orders_so_far = [[] for _ in range(B)]
    greedy_orders = torch.zeros(B, L, dtype=torch.long, device=device)

    for k in range(L):
        # Build current order: selected so far + dummy ascending for the rest
        current_orders = []
        for b in range(B):
            chosen   = orders_so_far[b]
            dummy    = sorted(set(range(L)) - set(chosen))
            current_orders.append(torch.tensor(chosen + dummy, dtype=torch.long, device=device))
        orders_t = torch.stack(current_orders)  # [B, L]

        # Forward pass
        gen_preds, pol_logits = model.forward_loarm(vectors, orders_t, init_vectors)

        # Policy at step k, mask already-revealed
        pol_k = pol_logits[:, k, :].clone()          # [B, L]
        pol_k[revealed] = float('-inf')

        # Greedy pick
        chosen_k = pol_k.argmax(dim=-1)              # [B]
        greedy_orders[:, k] = chosen_k

        for b in range(B):
            orders_so_far[b].append(chosen_k[b].item())
            revealed[b, chosen_k[b]] = True

    return greedy_orders


@torch.no_grad()
def eval_with_fixed_order(model_base, vectors, init_vectors, fixed_order):
    """Compute MSE loss with a specific fixed order applied to all samples."""
    B, L, _ = vectors.shape
    orders = fixed_order.unsqueeze(0).expand(B, -1).to(vectors.device)
    _, loss = model_base(vectors, mode=None, orders=orders, init_vectors=init_vectors)
    return loss.item()


@torch.no_grad()
def evaluate_policy(model, test_loader, device, max_batches=20):
    """Run all policy-related metrics on test set."""
    L = cfg.seq_length - cfg.num_init   # 31
    causal_order = torch.arange(L, dtype=torch.long, device=device)

    all_taus        = []
    all_entropies   = []   # [L] per step
    all_pol_causal  = []   # policy log-prob of causal order
    greedy_losses   = []
    random_losses   = []
    causal_losses   = []

    for i, batch in enumerate(test_loader):
        if i >= max_batches:
            break
        init_vectors = batch['init_vectors'].to(device)
        vectors      = batch['main_vectors'].to(device)
        B = vectors.shape[0]

        # ── Greedy order ─────────────────────────────────────────────────
        greedy_orders = policy_greedy_order(model, vectors, init_vectors)  # [B, L]

        # Kendall τ vs causal [0,1,...,L-1]
        causal_np = causal_order.cpu().numpy()
        for b in range(B):
            gr = greedy_orders[b].cpu().numpy()
            tau, _ = kendalltau(gr, causal_np)
            all_taus.append(tau)

        # Greedy order loss (apply greedy orders to generator)
        for b in range(B):
            go = greedy_orders[b]
            orders_b = go.unsqueeze(0)
            _, loss_b = model.base(vectors[b:b+1], mode=None,
                                   orders=orders_b,
                                   init_vectors=init_vectors[b:b+1])
            greedy_losses.append(loss_b.item())

        # Causal order loss
        cl = eval_with_fixed_order(model.base, vectors, init_vectors, causal_order)
        causal_losses.append(cl)

        # Random order losses (10 MC samples)
        rand_ls = []
        for _ in range(10):
            rand_ord = torch.stack([torch.randperm(L, device=device) for _ in range(B)])
            _, rl = model.base(vectors, mode=None, orders=rand_ord, init_vectors=init_vectors)
            rand_ls.append(rl.item())
        random_losses.append(np.mean(rand_ls))

        # ── Policy entropy per step ──────────────────────────────────────
        # Forward with causal order to get policy logits
        causal_orders = causal_order.unsqueeze(0).expand(B, -1)
        _, pol_logits = model.forward_loarm(vectors, causal_orders, init_vectors)
        masked_pol = model.apply_policy_mask(pol_logits, causal_orders)   # [B, L, L]

        entropies_batch = []
        for k in range(L):
            logits_k = masked_pol[:, k, :].clone()
            logits_k = logits_k.nan_to_num(nan=0., neginf=-1e9)
            probs_k  = F.softmax(logits_k, dim=-1)
            H_k      = -(probs_k * probs_k.clamp(min=1e-30).log()).sum(dim=-1).mean()
            entropies_batch.append(H_k.item())
        all_entropies.append(entropies_batch)

        # ── Policy log-prob of causal order ──────────────────────────────
        pol_logprob_causal = 0.0
        for k in range(L):
            lp_k = F.log_softmax(masked_pol[:, k, :].nan_to_num(nan=0., neginf=-1e9), dim=-1)
            pol_logprob_causal += lp_k[:, k].mean().item()   # causal: position k chosen at step k
        all_pol_causal.append(pol_logprob_causal)

    results = {
        'mean_tau':          np.mean(all_taus),
        'std_tau':           np.std(all_taus),
        'mean_greedy_loss':  np.mean(greedy_losses),
        'mean_causal_loss':  np.mean(causal_losses),
        'mean_random_loss':  np.mean(random_losses),
        'causal_advantage':  np.mean(random_losses) - np.mean(causal_losses),
        'policy_lp_causal':  np.mean(all_pol_causal),
        'entropy_per_step':  np.mean(all_entropies, axis=0),   # [L]
    }
    return results


def main():
    device = cfg.device
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')

    # Find best LO-ARM checkpoint
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, 'best_loarm_*.pt')))
    if not ckpts:
        print("No LO-ARM checkpoints found. Run train_loarm.py first.")
        return

    _, _, test_loader = create_disk_dataloaders(
        data_dir=os.path.join(os.path.dirname(__file__), 'data_hspace_500k'),
        batch_size=32,
        num_workers=cfg.num_workers,
        num_chunks=cfg.num_chunks,
    )

    for ckpt_path in ckpts:
        print(f"\n{'='*60}")
        model = load_loarm(ckpt_path, device)

        results = evaluate_policy(model, test_loader, device, max_batches=20)
        print(f"\nOrder Recovery Results:")
        print(f"  Greedy Kendall τ        : {results['mean_tau']:.4f} ± {results['std_tau']:.4f}")
        print(f"  Causal order loss        : {results['mean_causal_loss']:.4f}")
        print(f"  Greedy order loss        : {results['mean_greedy_loss']:.4f}")
        print(f"  Random order loss (MC10) : {results['mean_random_loss']:.4f}")
        print(f"  Causal advantage         : {results['causal_advantage']:+.4f}")
        print(f"  Policy log-prob (causal) : {results['policy_lp_causal']:.4f}")
        print(f"  Policy entropy step 0    : {results['entropy_per_step'][0]:.4f}")
        print(f"  Policy entropy step 15   : {results['entropy_per_step'][15]:.4f}")
        print(f"  Policy entropy step 30   : {results['entropy_per_step'][-1]:.4f}")


if __name__ == '__main__':
    main()
