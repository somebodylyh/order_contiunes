"""
Order evaluation for v8 (init-prefix architecture).

For each model (MDM, AR shuffled, AR no-shuffle), evaluates:
  1. Causal order loss  -- ascending order (0→27), the ground-truth causal order
  2. Random order loss  -- MC average over K random permutations
  3. Greedy order search (MDM only) -- at each step, pick the next token whose prediction
     is closest to what the model expects; measures whether MDM has internalized order.
  4. Kendall's τ between greedy-found order and causal order

Greedy algorithm (O(L) forward passes):
  At step k with selected=[s0..s_{k-1}]:
    - Build full_order = selected + sorted(remaining)  (dummy suffix doesn't affect step-k prediction)
    - Run one forward pass
    - Read prediction at position (ni-1+k)  [conditioned on init + s0..s_{k-1}]
    - Pick the remaining token j with highest cos_sim to that prediction
"""

import sys, os, math
import torch
import torch.nn.functional as F
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baseline_continuous import config as cfg
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig
from baseline_continuous.disk_dataset import create_disk_dataloaders


# ── helpers ─────────────────────────────────────────────────────────────────

def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = ContinuousAOGPT(ckpt['config'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()
    print(f"  Loaded {os.path.basename(path)}  val_loss={ckpt['val_loss']:.4f}")
    return model


@torch.no_grad()
def eval_with_order(model, main_vectors, init_vectors, orders):
    """Returns scalar mean loss over the batch for the given orders."""
    _, loss = model(main_vectors, mode=None, orders=orders, init_vectors=init_vectors)
    return loss.item()


@torch.no_grad()
def greedy_order_search(model, main_vectors, init_vectors):
    """
    For each sample in the batch, greedily find the ordering that minimises
    the model's cumulative loss.

    Returns:
      greedy_orders : [B, t]  int64 tensor of found orders
    """
    device = main_vectors.device
    B, t, D = main_vectors.shape
    ni = init_vectors.shape[1]

    selected   = [[] for _ in range(B)]     # selected[b] = list of chosen token indices so far
    remaining  = [list(range(t)) for _ in range(B)]

    greedy_orders = torch.zeros(B, t, dtype=torch.long, device=device)

    for step in range(t):
        # Build full order: selected_so_far + sorted(remaining)
        full_orders = []
        for b in range(B):
            order_b = selected[b] + sorted(remaining[b])
            full_orders.append(torch.tensor(order_b, dtype=torch.long, device=device))
        orders_tensor = torch.stack(full_orders)   # [B, t]

        # One forward pass
        preds, _ = model(main_vectors, mode=None, orders=orders_tensor, init_vectors=init_vectors)
        # Prediction at position (ni-1+step) predicts the (step+1)-th token in the ordering
        pred_step = preds[:, ni - 1 + step, :]        # [B, D]
        pred_step_norm = F.normalize(pred_step, dim=-1, eps=1e-6)

        # For each sample, pick the remaining token with highest cos_sim to pred_step
        for b in range(B):
            rem = remaining[b]
            cands = main_vectors[b, rem, :]           # [len(rem), D]
            cands_norm = F.normalize(cands, dim=-1, eps=1e-6)
            sims = (cands_norm @ pred_step_norm[b])   # [len(rem)]
            best_idx = sims.argmax().item()
            chosen = rem[best_idx]
            selected[b].append(chosen)
            remaining[b].remove(chosen)
            greedy_orders[b, step] = chosen

    return greedy_orders   # [B, t]


def kendall_tau(order_a, order_b):
    """Kendall's τ between two permutations (1-D numpy arrays)."""
    n = len(order_a)
    concordant = discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff_a = order_a[i] - order_a[j]
            diff_b = order_b[i] - order_b[j]
            if diff_a * diff_b > 0:
                concordant += 1
            elif diff_a * diff_b < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom > 0 else 0.0


# ── main ────────────────────────────────────────────────────────────────────

def main():
    device  = cfg.device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load data
    print("Loading data...")
    _, val_loader, test_loader = create_disk_dataloaders(
        data_dir=os.path.join(os.path.dirname(__file__), 'data_hspace_500k'),
        batch_size=64,
        num_workers=0,
        num_chunks=cfg.num_chunks,
    )

    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    models_to_eval = [
        ('MDM',          os.path.join(ckpt_dir, 'best_mdm_Random_model.pt'),    True),
        ('AR shuffled',  os.path.join(ckpt_dir, 'best_ar_model.pt'),            True),
        ('AR no-shuffle',os.path.join(ckpt_dir, 'best_ar_noshuffle_model.pt'),  False),
    ]

    n_main = cfg.seq_length - cfg.num_init         # 31 main tokens
    causal_order_ref = np.arange(n_main)           # 0,1,...,30 is the causal order

    for name, path, do_greedy in models_to_eval:
        if not os.path.exists(path):
            print(f"\nSkipping {name}: {path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Model: {name}")
        print(f"{'='*60}")
        model = load_model(path, device)

        for split_name, loader in [('val', val_loader), ('test', test_loader)]:
            causal_losses, rand_losses, greedy_taus, greedy_losses = [], [], [], []
            reverse_losses, true_reverse_losses = [], []
            N_MC = 10     # random MC samples
            N_BATCHES = 10  # enough for stable estimates

            for i, batch in enumerate(loader):
                if i >= N_BATCHES:
                    break
                main_v = batch['main_vectors'].to(device)   # [B, 28, D]
                init_v = batch['init_vectors'].to(device)   # [B, 4,  D]
                B, t, _ = main_v.shape

                # 1. Causal order loss
                causal_orders = torch.arange(t, device=device).unsqueeze(0).expand(B, -1)
                cl = eval_with_order(model, main_v, init_v, causal_orders)
                causal_losses.append(cl)

                # 1b. Naive reverse order loss (h_0 as init, predicts h_31 first — asymmetric)
                reverse_orders = torch.arange(t - 1, -1, -1, device=device).unsqueeze(0).expand(B, -1)
                rl = eval_with_order(model, main_v, init_v, reverse_orders)
                reverse_losses.append(rl)

                # 1c. True reverse: init=h_T, main=[h_{T-1},...,h_0], causal orders
                #   Construct from existing batch on-the-fly:
                #   full_prefix = [h_0, h_1, ..., h_{T-1}]  (init + main[:-1])
                #   rev_init    = main_v[:, -1:, :]           = h_T
                #   rev_main    = flip([h_0,...,h_{T-1}])     = [h_{T-1},...,h_0]
                rev_init = main_v[:, -1:, :]
                full_prefix = torch.cat([init_v, main_v[:, :-1, :]], dim=1)  # [B, t, D]
                rev_main = full_prefix.flip(1)
                causal_orders = torch.arange(t, device=device).unsqueeze(0).expand(B, -1)
                trl = eval_with_order(model, rev_main, rev_init, causal_orders)
                true_reverse_losses.append(trl)

                # 2. Random order loss (MC)
                mc = [eval_with_order(model, main_v, init_v,
                                      model.sample_random_orders(main_v))
                      for _ in range(N_MC)]
                rand_losses.append(sum(mc) / N_MC)

                # 3. Greedy order search + Kendall's τ + greedy loss  (MDM only, first 2 batches)
                if do_greedy and i < 2:
                    greedy_ords = greedy_order_search(model, main_v, init_v)  # [B, t]
                    # 3a. Kendall's τ vs causal order
                    for b in range(B):
                        go = greedy_ords[b].cpu().numpy()
                        tau = kendall_tau(go, causal_order_ref)
                        greedy_taus.append(tau)
                    # 3b. Loss when evaluating WITH the greedy order
                    gl = eval_with_order(model, main_v, init_v, greedy_ords)
                    greedy_losses.append(gl)

            causal_mean       = sum(causal_losses)       / len(causal_losses)
            reverse_mean      = sum(reverse_losses)      / len(reverse_losses)
            true_rev_mean     = sum(true_reverse_losses) / len(true_reverse_losses)
            rand_mean         = sum(rand_losses)         / len(rand_losses)
            advantage         = rand_mean - causal_mean

            print(f"\n  [{split_name}]")
            print(f"    causal  order loss       : {causal_mean:.4f}  (h_0→h_31, forward AR)")
            print(f"    naive reverse loss       : {reverse_mean:.4f}  (h_0 init, predict h_31 first)")
            print(f"    true  reverse loss       : {true_rev_mean:.4f}  (h_T init, predict h_{{T-1}} first)")
            print(f"    random  order loss       : {rand_mean:.4f}  (MC n={N_MC})")
            print(f"    causal advantage         : {advantage:+.4f}  "
                  f"({'causal better ✓' if advantage > 0 else 'no advantage ✗'})")

            if do_greedy and greedy_taus:
                tau_mean = sum(greedy_taus) / len(greedy_taus)
                tau_std  = (sum((x - tau_mean)**2 for x in greedy_taus) / len(greedy_taus))**0.5
                n_samples = len(greedy_taus)
                greedy_loss_mean = sum(greedy_losses) / len(greedy_losses)
                print(f"    greedy Kendall τ  : {tau_mean:.4f} ± {tau_std:.4f}  "
                      f"(n={n_samples} samples)")
                print(f"    τ interpretation  : "
                      f"{'learned order ✓' if tau_mean > 0.3 else ('reverse order ✗' if tau_mean < -0.3 else 'weak/no order signal ✗')}")
                print(f"    greedy order loss : {greedy_loss_mean:.4f}  "
                      f"(vs causal {causal_mean:.4f}, random {rand_mean:.4f})")


if __name__ == '__main__':
    main()
