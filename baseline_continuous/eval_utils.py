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
