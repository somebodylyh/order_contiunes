"""Shared evaluation utilities for baseline experiments."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_ar(model, val_loader, device, max_batches=None):
    """
    Evaluate model using AR (ascending) order.

    All models use this same function so metrics are directly comparable.
    Input is batch['vectors'] (original order) -- evaluates whether the model
    can predict the next vector given the true dependency structure.

    Returns:
        dict with keys: val_loss, val_cos_sim
    """
    model.eval()
    total_loss = 0.0
    total_cos_sim = 0.0
    total_pred_norm = 0.0
    total_batches = 0

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        init_vectors = batch['init_vectors'].to(device) if 'init_vectors' in batch else None
        main_vectors = batch['main_vectors'].to(device) if 'main_vectors' in batch else batch['vectors'].to(device)

        predictions, loss = model(main_vectors, mode='AR', init_vectors=init_vectors)

        # cos_sim against the main targets (AR ascending order = identity permutation)
        ni = init_vectors.shape[1] if init_vectors is not None else 0
        if ni > 0:
            # predictions[:, ni-1:ni-1+t, :] predicts main_vectors
            t = main_vectors.shape[1]
            shift_preds = predictions[:, ni - 1: ni - 1 + t, :]
            cos_sim = F.cosine_similarity(shift_preds, main_vectors, dim=-1).mean()
        else:
            shift_preds = predictions[:, :-1, :]
            cos_sim = F.cosine_similarity(shift_preds, main_vectors, dim=-1).mean()

        pred_norm = shift_preds.norm(dim=-1).mean()

        total_loss += loss.item()
        total_cos_sim += cos_sim.item()
        total_pred_norm += pred_norm.item()
        total_batches += 1

    n = max(total_batches, 1)
    model.train()
    return {
        'val_loss': total_loss / n,
        'val_cos_sim': total_cos_sim / n,
        'val_pred_norm': total_pred_norm / n,
    }


import numpy as np


def _kendall_tau(order_a, order_b):
    """Kendall tau between two permutations (numpy arrays)."""
    n = len(order_a)
    concordant = discordant = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            da = order_a[i] - order_a[j]
            db = order_b[i] - order_b[j]
            if da * db > 0:
                concordant += 1
            elif da * db < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom > 0 else 0.0


@torch.no_grad()
def evaluate_per_step_loss(model, loader, device, num_batches=10):
    """
    Per-step causal MSE: for each step k=0..t-1, mean MSE when predicting
    the k-th token in causal order given true prefix.
    Returns: numpy array [t]
    """
    model.eval()
    accum = None
    count = 0
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        main_v = batch['main_vectors'].to(device)   # [B, t, D]
        init_v = batch['init_vectors'].to(device)    # [B, ni, D]
        B, t, D = main_v.shape
        ni = init_v.shape[1]
        orders = torch.arange(t, device=device).unsqueeze(0).expand(B, -1)
        preds, _ = model(main_v, mode=None, orders=orders, init_vectors=init_v)
        loss_preds = preds[:, ni - 1: ni - 1 + t, :]   # [B, t, D]
        per_step = ((loss_preds - main_v) ** 2).mean(dim=(0, 2))  # [t]
        if accum is None:
            accum = per_step.cpu().numpy()
        else:
            accum += per_step.cpu().numpy()
        count += 1
    model.train()
    return accum / max(count, 1)   # [t]


@torch.no_grad()
def evaluate_rollout(model, loader, device, num_batches=5):
    """
    Autoregressive rollout in causal order using model's own predictions.
    Returns: numpy array [t] of per-step cos_sim against ground truth.
    """
    model.eval()
    all_cos = []
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        main_v = batch['main_vectors'].to(device)   # [B, t, D]
        init_v = batch['init_vectors'].to(device)
        B, t, D = main_v.shape
        ni = init_v.shape[1]

        generated = main_v.clone()   # replaced step-by-step with model outputs
        cos_sims = []
        orders = torch.arange(t, device=device).unsqueeze(0).expand(B, -1)

        for step in range(t):
            preds, _ = model(generated, mode=None, orders=orders, init_vectors=init_v)
            pred_step = preds[:, ni - 1 + step, :]          # [B, D]
            generated[:, step, :] = pred_step.detach()      # feed own prediction
            cos = F.cosine_similarity(pred_step, main_v[:, step, :], dim=-1).mean().item()
            cos_sims.append(cos)

        all_cos.append(cos_sims)
    model.train()
    return np.mean(all_cos, axis=0)   # [t]


@torch.no_grad()
def evaluate_order_quality(model, loader, device, num_batches=2):
    """
    Greedy order search: measures whether the model discovers causal order.
    Returns dict with:
      mean_tau       : float, mean Kendall tau vs causal order
      step_mse       : numpy [t], greedy per-step prediction MSE
      pos_correct    : numpy [t], fraction of samples where pos j was chosen at step j
    """
    model.eval()
    all_taus, all_step_mses = [], []
    pos_correct = None
    n_samples = 0

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        main_v = batch['main_vectors'].to(device)
        init_v = batch['init_vectors'].to(device)
        B, t, D = main_v.shape
        ni = init_v.shape[1]

        if pos_correct is None:
            pos_correct = np.zeros(t)
        causal_ref = np.arange(t)

        selected   = [[] for _ in range(B)]
        remaining  = [list(range(t)) for _ in range(B)]
        greedy_ord = torch.zeros(B, t, dtype=torch.long, device=device)
        step_mses  = torch.zeros(B, t)

        for step in range(t):
            full_orders = [
                torch.tensor(selected[b] + sorted(remaining[b]), dtype=torch.long, device=device)
                for b in range(B)
            ]
            orders_t = torch.stack(full_orders)
            preds, _ = model(main_v, mode=None, orders=orders_t, init_vectors=init_v)
            pred_step = preds[:, ni - 1 + step, :]
            pred_norm = F.normalize(pred_step, dim=-1, eps=1e-6)

            for b in range(B):
                rem = remaining[b]
                cands_norm = F.normalize(main_v[b, rem, :], dim=-1, eps=1e-6)
                sims = cands_norm @ pred_norm[b]
                best = sims.argmax().item()
                chosen = rem[best]
                step_mses[b, step] = ((pred_step[b] - main_v[b, chosen]) ** 2).mean().item()
                selected[b].append(chosen)
                remaining[b].remove(chosen)
                greedy_ord[b, step] = chosen

        for b in range(B):
            go = greedy_ord[b].cpu().numpy()
            all_taus.append(_kendall_tau(go, causal_ref))
            for j in range(t):
                if go[j] == causal_ref[j]:
                    pos_correct[j] += 1
            n_samples += 1
        all_step_mses.append(step_mses.mean(dim=0).numpy())

    model.train()
    return {
        'mean_tau':    float(np.mean(all_taus)) if all_taus else 0.0,
        'step_mse':    np.mean(all_step_mses, axis=0) if all_step_mses else np.zeros(t),
        'pos_correct': pos_correct / max(n_samples, 1) if pos_correct is not None else np.zeros(t),
    }
