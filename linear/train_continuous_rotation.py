"""
Training script for Continuous Vector Linear Rotation Experiment

Implements co-evolution training of:
- ContinuousTransformer: Predicts next vector in Dense AR sequence
- SetToSeqAgent: Learns optimal ordering (permutation) from shuffled vectors

Training phases (Curriculum Learning - Mixed Strategy):
1. Phase 1 (Parallel Warmup): 0 - warmup_steps
   - Model: Trained with Ground Truth L2R order (learns correct AR dynamics)
   - Agent: Only BC loss (supervised learning toward L2R pattern)
   - Addresses cold start by ensuring both are "ready" before co-evolution

2. Phase 2 (Co-evolution): warmup_steps+
   - Model: Uses Agent's output order
   - Agent: REINFORCE + BC (policy learning with RL)
   - Teacher forcing decays over time

Key features:
- Dense AR process with k orthogonal matrices
- OOD validation (different initialization region)
- Permutation-invariant encoder for agent
- Cosine similarity + L2R correctness rewards
"""

import os
import sys
import time
import math
import argparse
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
from scipy.stats import kendalltau

# Import experiment modules
from . import config_continuous_rotation as config
from .continuous_data_generator import ContinuousDenseARGenerator
from .continuous_dataset import create_continuous_dataloaders
from .continuous_model import ContinuousTransformer, ContinuousTransformerConfig
from .set_to_seq_agent import SetToSeqAgent

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not available, logging disabled")


# ============================================================================
# Argument Parsing
# ============================================================================

def get_args():
    """Parse command line arguments (overrides config values)."""
    parser = argparse.ArgumentParser(description='Continuous Rotation Experiment')

    # Smoothness reward (Geometric Prior for breaking symmetry)
    parser.add_argument('--smoothness_weight', type=float, default=0.0,
                        help='Weight for smoothness penalty reward (default: 0.0, disabled)')

    # Allow overriding key config values via CLI
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--max_iters', type=int, default=None, help='Max training iterations')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--use_agent', action='store_true', default=None, help='Enable agent mode')
    parser.add_argument('--no_agent', action='store_true', help='Disable agent (baseline mode)')
    parser.add_argument('--model_loss_type', type=str, default='mse',
                        choices=['mse', 'cosine'],
                        help='Model loss type: mse (default) or cosine')
    parser.add_argument('--use_contrastive_reward', action='store_true', default=False,
                        help='Use contrastive gap reward instead of absolute cosine similarity')
    parser.add_argument('--contrastive_model_loss', action='store_true', default=False,
                        help='Use margin-based contrastive loss for model (Option B)')
    parser.add_argument('--contrastive_margin', type=float, default=0.1,
                        help='Margin for contrastive model loss')
    parser.add_argument('--entropy_weight', type=float, default=0.0,
                        help='Weight for entropy regularization (0.0 = disabled, try 0.01-0.05)')
    parser.add_argument('--per_step_reward', action='store_true', default=False,
                        help='Use per-step cosine similarity as Agent reward instead of sequence-level gap')

    args, _ = parser.parse_known_args()

    # Apply CLI overrides to config
    if args.seed is not None:
        config.seed = args.seed
    if args.max_iters is not None:
        config.max_iters = args.max_iters
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.use_agent:
        config.use_agent = True
    if args.no_agent:
        config.use_agent = False

    # Store model_loss_type in config for access in step functions
    config.model_loss_type = args.model_loss_type

    # Store contrastive reward settings in config
    config.use_contrastive_reward = args.use_contrastive_reward
    config.contrastive_model_loss = args.contrastive_model_loss
    config.contrastive_margin = args.contrastive_margin
    config.entropy_weight = args.entropy_weight
    config.per_step_reward = args.per_step_reward

    return args


# ============================================================================
# Utility Functions
# ============================================================================

def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def compute_returns(rewards: torch.Tensor, mode: str = 'immediate', gamma: float = 0.99) -> torch.Tensor:
    """
    Compute returns from rewards.

    Args:
        rewards: [B, L] tensor of rewards
        mode: 'immediate' (r_t), 'cumulative' (discounted sum), 'final' (sum at last step)
        gamma: discount factor

    Returns:
        returns: [B, L] tensor of returns
    """
    B, L = rewards.shape

    if mode == 'immediate':
        return rewards
    elif mode == 'cumulative':
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(B, device=rewards.device)
        for t in reversed(range(L)):
            running_return = rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
        return returns
    elif mode == 'final':
        returns = torch.zeros_like(rewards)
        returns[:, -1] = rewards.sum(dim=-1)
        return returns
    else:
        raise ValueError(f"Unknown return mode: {mode}")


def normalize_returns(returns: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize returns for variance reduction."""
    mean = returns.mean()
    std = returns.std()
    return (returns - mean) / (std + eps)


def compute_contrastive_gap(
    model: ContinuousTransformer,
    shuffled_vectors: torch.Tensor,
    agent_perm: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute contrastive gap between Agent's ordering and random baseline.

    Returns:
        loss_agent: [B] per-sample loss with Agent's ordering
        loss_random: [B] per-sample loss with random ordering
        gap: [B] per-sample gap (loss_random - loss_agent)
    """
    B, L, D = shuffled_vectors.shape

    # Generate random permutation (no gradients)
    random_perm = torch.stack([torch.randperm(L, device=device) for _ in range(B)])

    # Reorder vectors
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
    X_agent = shuffled_vectors[batch_idx, agent_perm]   # [B, L, D]
    X_random = shuffled_vectors[batch_idx, random_perm]  # [B, L, D]

    # Forward pass for Agent's ordering
    pred_agent, _, _ = model.forward_with_hidden(shuffled_vectors, agent_perm, targets=None)

    # Forward pass for random ordering (with no_grad to save memory, model still sees gradients from agent path)
    pred_random, _, _ = model.forward_with_hidden(shuffled_vectors, random_perm, targets=None)

    # Compute cosine loss: 1 - cos_sim
    # pred[:, :-1] predicts target[:, 1:]
    cos_sim_agent = F.cosine_similarity(pred_agent[:, :-1], X_agent[:, 1:], dim=-1)  # [B, L-1]
    cos_sim_random = F.cosine_similarity(pred_random[:, :-1], X_random[:, 1:], dim=-1)  # [B, L-1]

    loss_agent = (1.0 - cos_sim_agent).mean(dim=-1)   # [B]
    loss_random = (1.0 - cos_sim_random).mean(dim=-1)  # [B]

    # Gap: positive means Agent is better than random
    gap = loss_random - loss_agent  # [B]

    return loss_agent, loss_random, gap


def compute_metrics(
    permutation: torch.Tensor,
    ground_truth_order: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    targets: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        permutation: [B, L] predicted permutation from agent
        ground_truth_order: [B, L] ground truth order (indices to recover original L2R sequence)
        predictions: [B, L, D] predicted vectors (optional)
        targets: [B, L, D] target vectors (optional)

    Returns:
        Dictionary of metrics
    """
    B, L = permutation.shape
    device = permutation.device

    metrics = {}

    # L2R order correct: does the agent output match ground_truth_order?
    l2r_correct = (permutation == ground_truth_order).all(dim=-1).float().mean().item()
    metrics['l2r_order_correct'] = l2r_correct

    # Per-position accuracy: agent picks correct position at each step
    for t in range(min(L, 5)):  # First 5 positions
        pos_correct = (permutation[:, t] == ground_truth_order[:, t]).float().mean().item()
        metrics[f'position_{t}_correct'] = pos_correct

    # Kendall Tau correlation: compare agent output with ground truth order
    tau_list = []
    for b in range(B):
        tau, _ = kendalltau(permutation[b].cpu().numpy(), ground_truth_order[b].cpu().numpy())
        if not np.isnan(tau):
            tau_list.append(tau)
    metrics['kendall_tau'] = np.mean(tau_list) if tau_list else 0.0

    # Reconstruction error (L1 distance between predicted and GT orders)
    recon_error = (permutation - ground_truth_order).abs().float().mean().item()
    metrics['reconstruction_error'] = recon_error

    # Prediction metrics (if provided)
    if predictions is not None and targets is not None:
        # MSE
        mse = F.mse_loss(predictions, targets).item()
        metrics['mse'] = mse

        # Cosine similarity
        cos_sim = F.cosine_similarity(predictions, targets, dim=-1).mean().item()
        metrics['cosine_similarity'] = cos_sim

    return metrics


class MetricsTracker:
    """Track and average metrics over intervals."""

    def __init__(self):
        self.metrics = defaultdict(list)

    def add(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self.metrics[k].append(v)

    def get_average(self) -> Dict[str, float]:
        return {k: np.mean(v) for k, v in self.metrics.items()}

    def reset(self):
        self.metrics = defaultdict(list)


# ============================================================================
# Training Functions
# ============================================================================

def get_teacher_forcing_ratio(step: int, config) -> float:
    """Compute teacher forcing ratio with linear decay."""
    if step >= config.teacher_forcing_decay_steps:
        return config.teacher_forcing_end
    ratio = config.teacher_forcing_start - (
        (config.teacher_forcing_start - config.teacher_forcing_end)
        * step / config.teacher_forcing_decay_steps
    )
    return ratio


def baseline_step(
    model: ContinuousTransformer,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, float]:
    """
    Baseline training step: Train model with random orders (no Agent).

    Args:
        model: ContinuousTransformer
        optimizer: Model optimizer
        batch: Batch of data
        device: Device to use

    Returns:
        Dictionary of metrics
    """
    model.train()

    # Move batch to device
    vectors = batch['vectors'].to(device)  # [B, L, D] ground truth order
    shuffled_vectors = batch['shuffled_vectors'].to(device)  # [B, L, D]
    B, L, D = vectors.shape

    # Generate random orders
    random_orders = model.sample_random_orders(B, L, device)

    # Forward pass with random order
    predictions, loss, _ = model.forward_with_hidden(
        shuffled_vectors, random_orders, targets=vectors,
        loss_type=config.model_loss_type
    )

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer.step()

    return {
        'model_loss': loss.item(),
        'mse': loss.item()
    }


@torch.no_grad()
def evaluate_baseline(
    model: ContinuousTransformer,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 50
) -> Dict[str, float]:
    """
    Evaluate model without Agent (random orders).

    Args:
        model: ContinuousTransformer
        dataloader: DataLoader for evaluation
        device: Device to use
        max_batches: Maximum number of batches to evaluate

    Returns:
        Dictionary of averaged metrics
    """
    model.eval()

    tracker = MetricsTracker()

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        vectors = batch['vectors'].to(device)
        shuffled_vectors = batch['shuffled_vectors'].to(device)
        B, L, D = vectors.shape

        # Use random orders (no agent)
        random_orders = model.sample_random_orders(B, L, device)

        # Model predicts with random order
        predictions, model_loss, _ = model.forward_with_hidden(
            shuffled_vectors, random_orders, targets=vectors,
            loss_type=config.model_loss_type
        )

        # Reorder for metrics
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        ordered_targets = vectors[batch_idx, random_orders]

        # Cosine similarity
        cos_sim = F.cosine_similarity(
            predictions[:, :-1], ordered_targets[:, 1:], dim=-1
        ).mean().item()

        metrics = {
            'model_loss': model_loss.item(),
            'mse': model_loss.item(),
            'cosine_similarity': cos_sim
        }
        tracker.add(metrics)

    return tracker.get_average()


def warmup_step_curriculum(
    model: ContinuousTransformer,
    agent: SetToSeqAgent,
    optimizer_model: torch.optim.Optimizer,
    optimizer_agent: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, float]:
    """
    Phase 1: Parallel Warmup - Train both Model and Agent simultaneously.

    - Model: Uses Ground Truth L2R order (learns correct AR dynamics)
    - Agent: Only BC loss (learns L2R pattern via supervised learning)

    This addresses the cold start problem by ensuring both components
    are "ready" before co-evolution begins.

    Args:
        model: ContinuousTransformer
        agent: SetToSeqAgent
        optimizer_model: Model optimizer
        optimizer_agent: Agent optimizer
        batch: Batch of data
        device: Device to use

    Returns:
        Dictionary of metrics
    """
    model.train()
    agent.train()

    # Move batch to device
    vectors = batch['vectors'].to(device)           # [B, L, D] GT ordered vectors
    shuffled_vectors = batch['shuffled_vectors'].to(device)  # [B, L, D]
    gt_order = batch['order'].to(device)            # [B, L] L2R order indices
    B, L, D = vectors.shape

    # ===== Model: Train with Ground Truth order =====
    # Use gt_order to reorder shuffled_vectors back to original L2R sequence
    # This ensures Model learns correct Dense AR dynamics on "pure" data
    # gt_order[t] = position in shuffled_vectors where original vector t is located
    predictions, model_loss, _ = model.forward_with_hidden(
        shuffled_vectors, gt_order, targets=vectors,
        loss_type=config.model_loss_type
    )

    # ===== Agent: Only BC loss (supervised learning toward L2R) =====
    # Agent learns to output L2R order without RL (policy_loss = 0)
    _, _, all_logits = agent(shuffled_vectors, return_all_logits=True)

    # BC loss: cross-entropy between agent logits and GT L2R order
    # Note: all_logits has -inf for masked positions, which is correct for softmax
    # but we need to handle potential numerical issues
    # Replace -inf with a large negative value that works well with softmax
    all_logits_safe = torch.where(
        torch.isinf(all_logits),
        torch.full_like(all_logits, -100.0),  # Large enough to be ~0 after softmax
        all_logits
    )
    bc_loss = F.cross_entropy(all_logits_safe.view(-1, L), gt_order.view(-1))
    agent_loss = config.warmup_bc_weight * bc_loss  # Higher weight for faster convergence

    # ===== Backward Pass: Update Both =====
    # Model update
    optimizer_model.zero_grad()
    model_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer_model.step()

    # Agent update
    optimizer_agent.zero_grad()
    agent_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), config.grad_clip)
    optimizer_agent.step()

    # ===== Compute Metrics =====
    with torch.no_grad():
        # Get agent's current predictions for monitoring
        permutation, _, _ = agent(shuffled_vectors, teacher_forcing_ratio=0.0)
        # L2R correct: agent output matches gt_order (positions to recover original sequence)
        l2r_correct = (permutation == gt_order).all(dim=-1).float().mean().item()

        # Per-position accuracy: does agent pick the correct position at each step?
        pos_correct = {}
        for t in range(min(L, 5)):
            pos_correct[f'position_{t}_correct'] = (permutation[:, t] == gt_order[:, t]).float().mean().item()

    metrics = {
        'model_loss': model_loss.item(),
        'mse': model_loss.item(),
        'agent_loss': agent_loss.item(),
        'bc_loss': bc_loss.item(),
        'l2r_order_correct': l2r_correct,
    }
    metrics.update(pos_correct)

    return metrics


def coevolution_step(
    model: ContinuousTransformer,
    agent: SetToSeqAgent,
    optimizer_model: torch.optim.Optimizer,
    optimizer_agent: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    global_step: int,
    smoothness_weight: float = 0.0,
    use_contrastive_reward: bool = False,
    contrastive_model_loss: bool = False,
    contrastive_margin: float = 0.1,
    entropy_weight: float = 0.0,
    per_step_reward: bool = False
) -> Dict[str, float]:
    """
    Co-evolution training step: Train both model and agent.

    Args:
        model: ContinuousTransformer
        agent: SetToSeqAgent
        optimizer_model: Model optimizer
        optimizer_agent: Agent optimizer
        batch: Batch of data
        device: Device to use
        global_step: Current training step
        smoothness_weight: Weight for smoothness penalty
        use_contrastive_reward: Use contrastive gap reward instead of absolute cosine similarity
        contrastive_model_loss: Use margin-based contrastive loss for model
        contrastive_margin: Margin for contrastive model loss
        entropy_weight: Weight for entropy regularization (0 = disabled)
        per_step_reward: Use per-step cosine similarity as Agent reward (with contrastive Model loss)

    Returns:
        Dictionary of metrics
    """
    model.train()
    agent.train()

    # Move batch to device
    vectors = batch['vectors'].to(device)  # [B, L, D] ground truth time order
    shuffled_vectors = batch['shuffled_vectors'].to(device)  # [B, L, D]
    gt_order = batch['order'].to(device)  # [B, L] ground truth order to recover
    B, L, D = vectors.shape

    # Teacher forcing ratio
    tf_ratio = get_teacher_forcing_ratio(global_step, config)

    # ========== Agent Forward Pass ==========
    # Agent receives shuffled vectors and predicts permutation
    permutation, log_probs, all_logits, agent_entropy = agent(
        shuffled_vectors,
        teacher_forcing_targets=gt_order,
        teacher_forcing_ratio=tf_ratio,
        return_all_logits=True,
        return_entropy=True
    )

    # ========== Model Forward Pass ==========
    # Model predicts vectors given the order from agent
    predictions, model_loss_original, _ = model.forward_with_hidden(
        shuffled_vectors,
        permutation.detach(),  # Detach to prevent gradients to agent through model
        targets=vectors,
        loss_type=config.model_loss_type
    )

    # ========== Compute Rewards ==========
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)

    # Initialize contrastive metrics
    loss_agent_scalar = 0.0
    loss_random_scalar = 0.0
    gap_scalar = 0.0

    if use_contrastive_reward:
        # ===== Contrastive Gap (for Model loss + logging) =====
        loss_agent, loss_random, gap = compute_contrastive_gap(
            model, shuffled_vectors, permutation.detach(), device
        )

        # Store for logging
        loss_agent_scalar = loss_agent.mean().item()
        loss_random_scalar = loss_random.mean().item()
        gap_scalar = gap.mean().item()

        # Per-step cosine similarity (used for per_step_reward and metrics)
        X_agent = shuffled_vectors[batch_idx, permutation.detach()]
        cos_sim = F.cosine_similarity(
            predictions[:, :-1], X_agent[:, 1:], dim=-1
        )  # [B, L-1]

        # L2R correctness bonus (optional)
        l2r_correct = (permutation == gt_order).float()

        if per_step_reward:
            # ===== Per-step reward: immediate cosine similarity =====
            cos_sim_padded = torch.cat([
                torch.zeros(B, 1, device=device),
                cos_sim.detach()
            ], dim=1)  # [B, L]
            rewards = cos_sim_padded + config.stepwise_reward_weight * l2r_correct
        else:
            # ===== Sequence-level gap reward (broadcast) =====
            gap_reward = gap.detach().unsqueeze(1).expand(B, L)  # [B, L]
            rewards = gap_reward + config.stepwise_reward_weight * l2r_correct

        # Model loss (contrastive, independent of reward type)
        if contrastive_model_loss:
            # Option B: Margin-based contrastive
            contrastive_loss = F.relu(loss_agent - loss_random + contrastive_margin).mean()
            model_loss = contrastive_loss
        else:
            # Option A: Just minimize on Agent's ordering
            model_loss = loss_agent.mean()
    else:
        # ===== Original Absolute Cosine Similarity Reward =====
        ordered_targets = vectors[batch_idx, permutation.detach()]
        cos_sim = F.cosine_similarity(
            predictions[:, :-1], ordered_targets[:, 1:], dim=-1
        )  # [B, L-1]

        cos_sim_padded = torch.cat([
            torch.zeros(B, 1, device=device),
            cos_sim
        ], dim=1)  # [B, L]

        l2r_correct = (permutation == gt_order).float()
        rewards = cos_sim_padded + config.stepwise_reward_weight * l2r_correct

        # Use original model loss
        model_loss = model_loss_original

    # ========== Smoothness Penalty (unchanged) ==========
    smoothness_penalty = torch.tensor(0.0, device=device)
    if smoothness_weight > 0.0:
        X_ordered = shuffled_vectors[batch_idx, permutation.detach()]
        diffs = X_ordered[:, 1:] - X_ordered[:, :-1]
        smoothness_penalty = diffs.norm(dim=-1).mean(dim=-1)
        rewards = rewards - smoothness_weight * smoothness_penalty.unsqueeze(1)

    # ========== Agent Loss (REINFORCE + BC) ==========
    # Compute returns
    returns = compute_returns(rewards, mode='immediate')
    if config.use_baseline:
        returns = normalize_returns(returns, config.baseline_eps)

    # Policy gradient loss
    policy_loss = -(log_probs * returns.detach()).mean()

    # Behavior Cloning loss
    bc_loss = torch.tensor(0.0, device=device)
    if config.use_bc_loss:
        # all_logits: [B, L, L], gt_order: [B, L]
        # Replace -inf with a large negative value that works well with softmax
        all_logits_safe = torch.where(
            torch.isinf(all_logits),
            torch.full_like(all_logits, -100.0),
            all_logits
        )
        bc_loss = F.cross_entropy(
            all_logits_safe.view(-1, L),
            gt_order.view(-1)
        )

    # Entropy regularization: maximize entropy by minimizing -entropy
    entropy_bonus = torch.tensor(0.0, device=device)
    if entropy_weight > 0.0:
        entropy_bonus = -entropy_weight * agent_entropy  # negative because we want to maximize entropy

    agent_loss = policy_loss + config.bc_loss_weight * bc_loss + entropy_bonus

    # ========== Backward Pass ==========
    # Agent update
    optimizer_agent.zero_grad()
    agent_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.parameters(), config.grad_clip)
    optimizer_agent.step()

    # Model update
    optimizer_model.zero_grad()
    model_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
    optimizer_model.step()

    # ========== Compute Metrics ==========
    with torch.no_grad():
        ordered_targets_for_metrics = vectors[batch_idx, permutation]
        metrics = compute_metrics(permutation, gt_order, predictions[:, :-1], ordered_targets_for_metrics[:, 1:])

    metrics.update({
        'model_loss': model_loss.item(),
        'agent_loss': agent_loss.item(),
        'policy_loss': policy_loss.item(),
        'bc_loss': bc_loss.item(),
        'mean_reward': rewards.mean().item(),
        'mean_cos_sim': cos_sim.mean().item(),
        'tf_ratio': tf_ratio,
        'smoothness_penalty': smoothness_penalty.mean().item() if smoothness_weight > 0.0 else 0.0,
        'agent_entropy': agent_entropy.item(),
    })

    # Contrastive-specific metrics
    if use_contrastive_reward:
        metrics.update({
            'loss_agent': loss_agent_scalar,
            'loss_random': loss_random_scalar,
            'gap': gap_scalar,
            'reward_std': rewards.std().item(),
        })

    return metrics


@torch.no_grad()
def evaluate(
    model: ContinuousTransformer,
    agent: SetToSeqAgent,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = 50
) -> Dict[str, float]:
    """
    Evaluate model and agent on validation/test set.

    Args:
        model: ContinuousTransformer
        agent: SetToSeqAgent
        dataloader: DataLoader for evaluation
        device: Device to use
        max_batches: Maximum number of batches to evaluate

    Returns:
        Dictionary of averaged metrics
    """
    model.eval()
    agent.eval()

    tracker = MetricsTracker()

    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        vectors = batch['vectors'].to(device)
        shuffled_vectors = batch['shuffled_vectors'].to(device)
        gt_order = batch['order'].to(device)
        B, L, D = vectors.shape

        # Agent predicts order (no teacher forcing)
        permutation, log_probs, _ = agent(shuffled_vectors, teacher_forcing_ratio=0.0)

        # Model predicts with agent's order
        predictions, model_loss, _ = model.forward_with_hidden(
            shuffled_vectors, permutation, targets=vectors,
            loss_type=config.model_loss_type
        )

        # Reorder for metrics
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
        ordered_targets = vectors[batch_idx, permutation]

        # Compute metrics
        metrics = compute_metrics(
            permutation, gt_order,
            predictions[:, :-1], ordered_targets[:, 1:]
        )
        metrics['model_loss'] = model_loss.item()

        tracker.add(metrics)

    return tracker.get_average()


# ============================================================================
# Main Training Loop
# ============================================================================

def train():
    """Main training function."""
    # Parse CLI arguments (overrides config)
    args = get_args()

    use_agent = getattr(config, 'use_agent', True)
    mode_str = "With Agent" if use_agent else "Baseline (No Agent)"

    print("=" * 70)
    print(f"Continuous Vector Linear Rotation Experiment — {mode_str}")
    print("=" * 70)

    # Set device
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Set seed
    set_seed(config.seed)

    # Create experiment directory
    exp_dir = config.exp_dir if use_agent else config.exp_dir + '_baseline'
    os.makedirs(exp_dir, exist_ok=True)

    # ========== Data ==========
    print("\n[INFO] Creating dataloaders...")
    train_loader, val_loader, test_loader, generator = create_continuous_dataloaders(
        vector_dim=config.vector_dim,
        seq_length=config.seq_length,
        dependency_window=config.dependency_window,
        num_matrices=getattr(config, 'num_matrices', None),
        train_samples=config.train_samples,
        val_samples=config.val_samples,
        test_samples=config.test_samples,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        fixed_matrices_path=config.fixed_matrices_path,
        train_init_mode=config.train_init_mode,
        val_init_mode=config.val_init_mode,
        num_chunks=getattr(config, 'num_chunks', 4),
        noise_scale=getattr(config, 'noise_scale', 0.0)
    )

    # ========== Model ==========
    print("\n[INFO] Creating model...")
    model_config = ContinuousTransformerConfig(
        vector_dim=config.vector_dim,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        dropout=config.dropout,
        bias=config.bias
    )
    model = ContinuousTransformer(model_config).to(device)

    # ========== Agent (only if use_agent) ==========
    agent = None
    optimizer_agent = None
    scheduler_agent = None
    if use_agent:
        print("\n[INFO] Creating agent...")
        agent = SetToSeqAgent(
            vector_dim=config.vector_dim,
            d_model=config.agent_d_model,
            encoder_layers=config.agent_encoder_layers,
            encoder_heads=config.agent_encoder_heads,
            decoder_layers=config.agent_decoder_layers,
            decoder_heads=config.agent_decoder_heads,
            max_len=config.seq_length,
            dropout=config.dropout
        ).to(device)
        optimizer_agent = AdamW(
            agent.parameters(),
            lr=config.agent_learning_rate,
            weight_decay=config.weight_decay
        )
        scheduler_agent = CosineAnnealingLR(optimizer_agent, T_max=config.max_iters)
    else:
        print("\n[INFO] Baseline mode — no Agent")

    # ========== Optimizers ==========
    optimizer_model = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate schedulers
    scheduler_model = CosineAnnealingLR(optimizer_model, T_max=config.max_iters)

    # ========== Logging ==========
    wandb_run = None
    if config.wandb_log and WANDB_AVAILABLE:
        run_name = config.wandb_run_name
        if run_name is None:
            run_name = 'baseline_no_agent' if not use_agent else None
        wandb_run = wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={k: v for k, v in vars(config).items() if not k.startswith('_')}
        )

    # ========== Training ==========
    print("\n[INFO] Starting training...")
    if use_agent:
        print(f"       Warmup steps: {config.warmup_steps}")
        print(f"       Teacher forcing decay: {config.teacher_forcing_decay_steps} steps")
        print(f"       Smoothness weight: {args.smoothness_weight}")
        if args.use_contrastive_reward:
            print(f"       Contrastive reward: ENABLED")
            if args.contrastive_model_loss:
                print(f"       Model loss: contrastive-margin (margin={args.contrastive_margin})")
            else:
                print(f"       Model loss: contrastive-cosine (loss_agent)")
            print(f"       Agent reward: {'per-step cos_sim' if args.per_step_reward else 'sequence-level gap'}")
        else:
            print(f"       Model loss type: {config.model_loss_type}")
        if args.entropy_weight > 0.0:
            print(f"       Entropy weight: {args.entropy_weight}")
    print(f"       Max iterations: {config.max_iters}")

    train_tracker = MetricsTracker()
    best_val_mse = float('inf')
    global_step = 0
    train_iter = iter(train_loader)

    start_time = time.time()

    for step in range(config.max_iters):
        # Get batch (cycle through dataset)
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # Training step
        if not use_agent:
            # Baseline: always train model with random orders
            metrics = baseline_step(model, optimizer_model, batch, device)
            phase = 'baseline'
        elif step < config.warmup_steps:
            # Phase 1: Parallel Warmup
            # - Model trained with GT L2R order (learns correct AR dynamics)
            # - Agent trained with BC loss only (learns L2R pattern)
            metrics = warmup_step_curriculum(
                model, agent, optimizer_model, optimizer_agent, batch, device
            )
            phase = 'warmup'
        else:
            # Phase 2: Co-evolution
            # - Model uses Agent's output order
            # - Agent uses REINFORCE + BC
            metrics = coevolution_step(
                model, agent, optimizer_model, optimizer_agent,
                batch, device, step - config.warmup_steps,
                smoothness_weight=args.smoothness_weight,
                use_contrastive_reward=args.use_contrastive_reward,
                contrastive_model_loss=args.contrastive_model_loss,
                contrastive_margin=args.contrastive_margin,
                entropy_weight=args.entropy_weight,
                per_step_reward=args.per_step_reward
            )
            phase = 'coevolution'

        train_tracker.add(metrics)
        global_step = step

        # Logging
        if (step + 1) % config.log_interval == 0:
            avg_metrics = train_tracker.get_average()
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed

            # Print summary
            if phase == 'baseline':
                print(f"[{phase}] Step {step+1}/{config.max_iters} | "
                      f"MSE: {avg_metrics.get('mse', 0):.4f} | "
                      f"{steps_per_sec:.1f} steps/s")
            elif phase == 'warmup':
                print(f"[{phase}] Step {step+1}/{config.max_iters} | "
                      f"Model MSE: {avg_metrics.get('mse', 0):.4f} | "
                      f"Agent BC Loss: {avg_metrics.get('bc_loss', 0):.4f} | "
                      f"L2R: {avg_metrics.get('l2r_order_correct', 0):.2%}")
            else:
                if args.use_contrastive_reward:
                    print(f"[{phase}] Step {step+1}/{config.max_iters} | "
                          f"Loss: {avg_metrics.get('model_loss', 0):.4f} | "
                          f"Gap: {avg_metrics.get('gap', 0):.4f} | "
                          f"L2R: {avg_metrics.get('l2r_order_correct', 0):.2%} | "
                          f"Tau: {avg_metrics.get('kendall_tau', 0):.3f} | "
                          f"Ent: {avg_metrics.get('agent_entropy', 0):.3f}")
                else:
                    print(f"[{phase}] Step {step+1}/{config.max_iters} | "
                          f"MSE: {avg_metrics.get('model_loss', 0):.4f} | "
                          f"L2R: {avg_metrics.get('l2r_order_correct', 0):.2%} | "
                          f"Tau: {avg_metrics.get('kendall_tau', 0):.3f} | "
                          f"CosSim: {avg_metrics.get('mean_cos_sim', 0):.3f} | "
                          f"TF: {avg_metrics.get('tf_ratio', 0):.2f} | "
                          f"Ent: {avg_metrics.get('agent_entropy', 0):.3f}")

            # W&B logging
            if wandb_run:
                wandb.log({f"train/{k}": v for k, v in avg_metrics.items()}, step=step)
                wandb.log({"train/lr_model": scheduler_model.get_last_lr()[0]}, step=step)
                if scheduler_agent:
                    wandb.log({"train/lr_agent": scheduler_agent.get_last_lr()[0]}, step=step)

            train_tracker.reset()

        # Evaluation
        if (step + 1) % config.eval_interval == 0:
            if use_agent:
                val_metrics = evaluate(model, agent, val_loader, device)
                print(f"[EVAL] Step {step+1} | "
                      f"Val MSE: {val_metrics.get('mse', val_metrics.get('model_loss', 0)):.4f} | "
                      f"L2R: {val_metrics.get('l2r_order_correct', 0):.2%} | "
                      f"Tau: {val_metrics.get('kendall_tau', 0):.3f} | "
                      f"CosSim: {val_metrics.get('cosine_similarity', 0):.3f}")
            else:
                val_metrics = evaluate_baseline(model, val_loader, device)
                print(f"[EVAL] Step {step+1} | "
                      f"Val MSE: {val_metrics.get('mse', 0):.4f} | "
                      f"CosSim: {val_metrics.get('cosine_similarity', 0):.3f}")

            if wandb_run:
                wandb.log({f"val/{k}": v for k, v in val_metrics.items()}, step=step)

            # Save best model
            val_mse = val_metrics.get('mse', val_metrics.get('model_loss', float('inf')))
            if val_mse < best_val_mse and config.save_best_model:
                best_val_mse = val_mse
                save_checkpoint(model, agent, optimizer_model, optimizer_agent,
                               step, val_metrics, os.path.join(exp_dir, 'best_model.pt'))
                print(f"       [SAVED] New best model with Val MSE: {val_mse:.4f}")

        # Checkpointing
        if (step + 1) % config.checkpoint_interval == 0:
            save_checkpoint(model, agent, optimizer_model, optimizer_agent,
                           step, {}, os.path.join(exp_dir, f'checkpoint_{step+1}.pt'))

        # Update schedulers (after optimizer.step())
        scheduler_model.step()
        if scheduler_agent:
            scheduler_agent.step()

    # ========== Final Evaluation ==========
    print("\n" + "=" * 70)
    print(f"Final Evaluation — {mode_str}")
    print("=" * 70)

    if use_agent:
        val_metrics = evaluate(model, agent, val_loader, device, max_batches=100)
        test_metrics = evaluate(model, agent, test_loader, device, max_batches=100)
    else:
        val_metrics = evaluate_baseline(model, val_loader, device, max_batches=100)
        test_metrics = evaluate_baseline(model, test_loader, device, max_batches=100)

    print("\nValidation (OOD) Results:")
    for k, v in sorted(val_metrics.items()):
        print(f"  {k}: {v:.4f}")

    print("\nTest (OOD) Results:")
    for k, v in sorted(test_metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Save final model
    save_checkpoint(model, agent, optimizer_model, optimizer_agent,
                   config.max_iters, test_metrics, os.path.join(exp_dir, 'final_model.pt'))

    if wandb_run:
        wandb.log({f"final_val/{k}": v for k, v in val_metrics.items()})
        wandb.log({f"final_test/{k}": v for k, v in test_metrics.items()})
        wandb.finish()

    print("\n[INFO] Training complete!")
    print(f"       Best Val MSE: {best_val_mse:.4f}")
    print(f"       Final Test MSE: {test_metrics.get('mse', test_metrics.get('model_loss', 0)):.4f}")


def save_checkpoint(
    model: ContinuousTransformer,
    agent: Optional[SetToSeqAgent],
    optimizer_model: torch.optim.Optimizer,
    optimizer_agent: Optional[torch.optim.Optimizer],
    step: int,
    metrics: Dict[str, float],
    path: str
):
    """Save training checkpoint."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_model_state_dict': optimizer_model.state_dict(),
        'metrics': metrics,
        'config': {k: v for k, v in vars(config).items() if not k.startswith('_')}
    }
    if agent is not None:
        checkpoint['agent_state_dict'] = agent.state_dict()
    if optimizer_agent is not None:
        checkpoint['optimizer_agent_state_dict'] = optimizer_agent.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: ContinuousTransformer,
    agent: Optional[SetToSeqAgent] = None,
    optimizer_model: Optional[torch.optim.Optimizer] = None,
    optimizer_agent: Optional[torch.optim.Optimizer] = None
) -> int:
    """Load training checkpoint."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if agent is not None and 'agent_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['agent_state_dict'])
    if optimizer_model is not None:
        optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
    if optimizer_agent is not None and 'optimizer_agent_state_dict' in checkpoint:
        optimizer_agent.load_state_dict(checkpoint['optimizer_agent_state_dict'])
    return checkpoint['step']


if __name__ == '__main__':
    train()
