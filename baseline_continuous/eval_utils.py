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
    total_batches = 0

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        vectors = batch['vectors'].to(device)
        predictions, loss = model(vectors, mode='AR')

        shift_preds = predictions[:, :-1, :]
        cos_sim = F.cosine_similarity(shift_preds, vectors, dim=-1).mean()

        total_loss += loss.item()
        total_cos_sim += cos_sim.item()
        total_batches += 1

    n = max(total_batches, 1)
    model.train()
    return {
        'val_loss': total_loss / n,
        'val_cos_sim': total_cos_sim / n,
    }
