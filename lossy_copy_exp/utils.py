"""
Utility functions for LO-ARMs Lossy Copy Experiment

Includes metrics computation, logging helpers, and checkpointing utilities.
"""

import torch
import numpy as np
import os
import json
from datetime import datetime


def compute_selection_probability(first_actions, x_position=0):
    """
    Compute P(select_x_first) - the probability that Agent selects x position first.

    This is the key metric for verifying convergence to optimal order.

    Args:
        first_actions: [B] tensor of first position selected by Agent
        x_position: Position index of x token (typically 0)

    Returns:
        prob: Scalar probability in [0, 1]
    """
    return (first_actions == x_position).float().mean().item()


def compute_accuracy(logits, targets, position_idx=None):
    """
    Compute prediction accuracy.

    Args:
        logits: [B, T+1, vocab_size] model logits (includes [None] token)
        targets: [B, T] ground truth tokens
        position_idx: If provided, compute accuracy for specific position only

    Returns:
        accuracy: Scalar accuracy in [0, 1]
    """
    # Remove [None] token logits
    logits = logits[:, :-1, :]  # [B, T, vocab_size]

    # Get predictions
    preds = logits.argmax(dim=-1)  # [B, T]

    # Compute accuracy
    if position_idx is not None:
        correct = (preds[:, position_idx] == targets[:, position_idx]).float()
    else:
        correct = (preds == targets).float()

    return correct.mean().item()


def compute_returns(rewards, mode='immediate', gamma=0.99):
    """
    Compute returns for REINFORCE algorithm.

    Args:
        rewards: List of [B] reward tensors (one per step)
        mode: Return computation mode
            - 'immediate': Use immediate rewards (no discounting)
            - 'cumulative': Discounted cumulative sum
            - 'final': Only use final reward
        gamma: Discount factor (only used for 'cumulative' mode)

    Returns:
        returns: List of [B] return tensors (one per step)
    """
    if mode == 'immediate':
        # Use immediate rewards as-is
        return rewards

    elif mode == 'cumulative':
        # Compute discounted cumulative returns
        # G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        returns = []
        G = torch.zeros_like(rewards[-1])  # Initialize with final reward
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)
        return returns

    elif mode == 'final':
        # Use only final reward for all steps
        final_reward = rewards[-1]
        return [final_reward] * len(rewards)

    else:
        raise ValueError(f"Unknown return mode: {mode}")


def normalize_returns(returns, baseline=None, eps=1e-8):
    """
    Normalize returns for variance reduction.

    Args:
        returns: List of [B] return tensors
        baseline: Optional baseline to subtract (e.g., mean return)
        eps: Small constant for numerical stability

    Returns:
        normalized_returns: List of [B] normalized return tensors
    """
    # Stack returns
    returns_stacked = torch.stack(returns)  # [num_steps, B]

    # Compute baseline if not provided
    if baseline is None:
        baseline = returns_stacked.mean()

    # Subtract baseline
    normalized = returns_stacked - baseline

    # Normalize by std (optional, helps with variance)
    std = returns_stacked.std()
    if std > eps:
        normalized = normalized / std

    # Unstack
    return [normalized[i] for i in range(len(returns))]


class MetricsTracker:
    """
    Track and compute running statistics for metrics.
    """

    def __init__(self, window_size=100):
        self.window_size = window_size
        self.metrics = {}
        self.history = {}

    def update(self, **kwargs):
        """
        Update metrics with new values.

        Args:
            **kwargs: metric_name=value pairs
        """
        for name, value in kwargs.items():
            if name not in self.metrics:
                self.metrics[name] = []
                self.history[name] = []

            # Add to recent window
            self.metrics[name].append(value)
            if len(self.metrics[name]) > self.window_size:
                self.metrics[name].pop(0)

            # Add to full history
            self.history[name].append(value)

    def get_mean(self, name):
        """Get mean of recent values."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return np.mean(self.metrics[name])

    def get_std(self, name):
        """Get std of recent values."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return np.std(self.metrics[name])

    def get_all_means(self):
        """Get dictionary of all metric means."""
        return {name: self.get_mean(name) for name in self.metrics.keys()}

    def reset(self):
        """Reset recent metrics (keep history)."""
        self.metrics = {name: [] for name in self.metrics.keys()}


def save_checkpoint(model, agent, optimizer_model, optimizer_agent, iter_num, config, out_dir):
    """
    Save training checkpoint.

    Args:
        model: AOGPT model
        agent: OrderPolicyNet
        optimizer_model: Model optimizer
        optimizer_agent: Agent optimizer
        iter_num: Current iteration number
        config: Experiment configuration
        out_dir: Output directory
    """
    os.makedirs(out_dir, exist_ok=True)

    checkpoint = {
        'model': model.state_dict(),
        'agent': agent.state_dict(),
        'optimizer_model': optimizer_model.state_dict(),
        'optimizer_agent': optimizer_agent.state_dict(),
        'iter_num': iter_num,
        'config': config,
    }

    checkpoint_path = os.path.join(out_dir, f'checkpoint_{iter_num}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    # Also save as latest
    latest_path = os.path.join(out_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)


def load_checkpoint(checkpoint_path, model, agent, optimizer_model=None, optimizer_agent=None):
    """
    Load training checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model: AOGPT model to load into
        agent: OrderPolicyNet to load into
        optimizer_model: Optional model optimizer
        optimizer_agent: Optional agent optimizer

    Returns:
        iter_num: Iteration number from checkpoint
        config: Configuration from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['model'])
    agent.load_state_dict(checkpoint['agent'])

    if optimizer_model is not None and 'optimizer_model' in checkpoint:
        optimizer_model.load_state_dict(checkpoint['optimizer_model'])

    if optimizer_agent is not None and 'optimizer_agent' in checkpoint:
        optimizer_agent.load_state_dict(checkpoint['optimizer_agent'])

    iter_num = checkpoint.get('iter_num', 0)
    config = checkpoint.get('config', {})

    print(f"Loaded checkpoint from {checkpoint_path} (iter {iter_num})")

    return iter_num, config


def log_metrics(metrics, iter_num, phase='train', wandb_run=None):
    """
    Log metrics to console and optionally to wandb.

    Args:
        metrics: Dictionary of metric_name: value pairs
        iter_num: Current iteration number
        phase: 'train' or 'eval'
        wandb_run: Optional wandb run object
    """
    # Console logging
    metrics_str = ' | '.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
    print(f"[{phase.upper()} iter {iter_num}] {metrics_str}")

    # Wandb logging
    if wandb_run is not None:
        wandb_metrics = {f'{phase}/{k}': v for k, v in metrics.items()}
        wandb_metrics['iter'] = iter_num
        wandb_run.log(wandb_metrics)


def create_experiment_dir(base_dir, run_name):
    """
    Create experiment directory with timestamp.

    Args:
        base_dir: Base directory for experiments
        run_name: Name of the run

    Returns:
        exp_dir: Full path to experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'{run_name}_{timestamp}')
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Created experiment directory: {exp_dir}")
    return exp_dir


def save_config(config, exp_dir):
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary or module
        exp_dir: Experiment directory
    """
    # Convert config module to dict if needed
    if not isinstance(config, dict):
        config_dict = {}
        for k, v in vars(config).items():
            if not k.startswith('_'):
                # Try to serialize, convert to string if fails
                try:
                    json.dumps(v)
                    config_dict[k] = v
                except (TypeError, ValueError):
                    # Convert to string for non-serializable objects
                    config_dict[k] = str(v)
    else:
        config_dict = config

    config_path = os.path.join(exp_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2, default=str)

    print(f"Saved config to {config_path}")


def test_utils():
    """Test utility functions."""
    print("Testing utility functions...")

    # Test compute_selection_probability
    first_actions = torch.tensor([0, 0, 1, 0, 1])
    prob = compute_selection_probability(first_actions, x_position=0)
    print(f"✓ Selection probability: {prob:.2f} (expected 0.60)")
    assert abs(prob - 0.6) < 0.01

    # Test compute_accuracy
    logits = torch.randn(2, 3, 10)  # [B=2, T+1=3, vocab=10]
    targets = torch.randint(0, 10, (2, 2))  # [B=2, T=2]
    acc = compute_accuracy(logits, targets)
    print(f"✓ Accuracy: {acc:.2f}")

    # Test compute_returns (immediate)
    rewards = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    returns_imm = compute_returns(rewards, mode='immediate')
    assert len(returns_imm) == 2
    print(f"✓ Immediate returns: {[r.tolist() for r in returns_imm]}")

    # Test compute_returns (cumulative)
    returns_cum = compute_returns(rewards, mode='cumulative', gamma=0.9)
    print(f"✓ Cumulative returns: {[r.tolist() for r in returns_cum]}")

    # Test normalize_returns
    returns_norm = normalize_returns(returns_imm)
    print(f"✓ Normalized returns: {[r.tolist() for r in returns_norm]}")

    # Test MetricsTracker
    tracker = MetricsTracker(window_size=3)
    tracker.update(loss=1.0, acc=0.5)
    tracker.update(loss=2.0, acc=0.6)
    tracker.update(loss=1.5, acc=0.7)
    print(f"✓ Tracker mean loss: {tracker.get_mean('loss'):.2f}")
    print(f"  All means: {tracker.get_all_means()}")

    print("\n✅ All utility tests passed!")


if __name__ == '__main__':
    test_utils()
