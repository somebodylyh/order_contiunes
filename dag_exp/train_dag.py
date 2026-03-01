"""
LO-ARMs Training Script for Diamond DAG Experiment

Tests whether Agent can discover complex DAG topology with both
fork (x0 -> x1, x2) and join (x1, x2 -> x3) structures.

Two-phase training:
1. Warmup: Train model with random orders (frozen agent)
2. Co-evolution: Train both model and agent with REINFORCE

Expected Result:
- Agent should learn to generate x0 (root) first
- Then x1 or x2 (branches, in any order)
- Finally x3 (sink) last
- P(select_root_first) should converge to ~100%
- P(select_sink_last) should converge to ~100%
"""

import os
import sys
import time
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import components
from dag_exp.dag_dataset import DiamondDAGDataset
from lossy_copy_exp.model_wrapper import AOGPTWithHiddenStates
from lossy_copy_exp.order_policy_net import OrderPolicyNet
from lossy_copy_exp.utils import (
    compute_accuracy,
    compute_returns,
    normalize_returns,
    MetricsTracker,
    save_checkpoint,
    load_checkpoint,
    log_metrics,
    create_experiment_dir,
    save_config
)
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPTConfig

# Import config
import dag_exp.config_dag as config


def compute_dag_selection_metrics(actions_list, seq_length=6):
    """
    Compute comprehensive selection metrics for Diamond DAG.

    Args:
        actions_list: List of [B] tensors, one per step
        seq_length: Sequence length (should be 6)

    Returns:
        Dictionary with detailed metrics
    """
    metrics = {}

    # First step (t=0): Should select x0 (root)
    first_actions = actions_list[0]  # [B]
    metrics['first_step/p_x0'] = (first_actions == 0).float().mean().item()  # Root
    metrics['first_step/p_x1'] = (first_actions == 1).float().mean().item()  # Branch A
    metrics['first_step/p_x2'] = (first_actions == 2).float().mean().item()  # Branch B
    metrics['first_step/p_x3'] = (first_actions == 3).float().mean().item()  # Sink
    metrics['first_step/p_x4'] = (first_actions == 4).float().mean().item()  # Branch C
    metrics['first_step/p_x5'] = (first_actions == 5).float().mean().item()  # Sink

    # Second step (t=1): Should select one of the branches (x1 or x2)
    if len(actions_list) > 1:
        second_actions = actions_list[1]  # [B]
        metrics['second_step/p_x0'] = (second_actions == 0).float().mean().item()
        metrics['second_step/p_x1'] = (second_actions == 1).float().mean().item()
        metrics['second_step/p_x2'] = (second_actions == 2).float().mean().item()
        metrics['second_step/p_x3'] = (second_actions == 3).float().mean().item()
        metrics['second_step/p_x4'] = (second_actions == 4).float().mean().item()
        metrics['second_step/p_x5'] = (second_actions == 5).float().mean().item()
        # Combined branch probability
        p_branch = ((second_actions == 1) | (second_actions == 2)).float().mean().item()
        metrics['second_step/p_any_branch'] = p_branch

    # Third step (t=2): Should select the other branch
    if len(actions_list) > 2:
        third_actions = actions_list[2]  # [B]
        metrics['third_step/p_x0'] = (third_actions == 0).float().mean().item()
        metrics['third_step/p_x1'] = (third_actions == 1).float().mean().item()
        metrics['third_step/p_x2'] = (third_actions == 2).float().mean().item()
        metrics['third_step/p_x3'] = (third_actions == 3).float().mean().item()
        metrics['third_step/p_x4'] = (third_actions == 4).float().mean().item()
        metrics['third_step/p_x5'] = (third_actions == 5).float().mean().item()

    # Fourth step (t=3): Should select one of the remaining variables
    if len(actions_list) > 3:
        fourth_actions = actions_list[3]  # [B]
        metrics['fourth_step/p_x0'] = (fourth_actions == 0).float().mean().item()
        metrics['fourth_step/p_x1'] = (fourth_actions == 1).float().mean().item()
        metrics['fourth_step/p_x2'] = (fourth_actions == 2).float().mean().item()
        metrics['fourth_step/p_x3'] = (fourth_actions == 3).float().mean().item()
        metrics['fourth_step/p_x4'] = (fourth_actions == 4).float().mean().item()
        metrics['fourth_step/p_x5'] = (fourth_actions == 5).float().mean().item()

    # Fifth step (t=4): Should select one of the remaining variables
    if len(actions_list) > 4:
        fifth_actions = actions_list[4]  # [B]
        metrics['fifth_step/p_x0'] = (fifth_actions == 0).float().mean().item()
        metrics['fifth_step/p_x1'] = (fifth_actions == 1).float().mean().item()
        metrics['fifth_step/p_x2'] = (fifth_actions == 2).float().mean().item()
        metrics['fifth_step/p_x3'] = (fifth_actions == 3).float().mean().item()
        metrics['fifth_step/p_x4'] = (fifth_actions == 4).float().mean().item()
        metrics['fifth_step/p_x5'] = (fifth_actions == 5).float().mean().item()

    # Last step (t=5): Should select x5 (sink)
    if len(actions_list) > 5:
        last_actions = actions_list[5]  # [B]
        metrics['last_step/p_x0'] = (last_actions == 0).float().mean().item()
        metrics['last_step/p_x1'] = (last_actions == 1).float().mean().item()
        metrics['last_step/p_x2'] = (last_actions == 2).float().mean().item()
        metrics['last_step/p_x3'] = (last_actions == 3).float().mean().item()
        metrics['last_step/p_x4'] = (last_actions == 4).float().mean().item()
        metrics['last_step/p_x5'] = (last_actions == 5).float().mean().item()  # Sink

        # Topological correctness: x0 first AND x5 last
        topo_correct = (first_actions == 0) & (last_actions == 5)
        metrics['topology/correct'] = topo_correct.float().mean().item()

    return metrics


def get_batch(dataloader_iter, dataloader, device):
    """
    Get next batch from dataloader, handling epoch boundaries.

    Args:
        dataloader_iter: Iterator for dataloader
        dataloader: DataLoader object
        device: Device to move data to

    Returns:
        batch: Dictionary with tensors moved to device
        dataloader_iter: Updated iterator
    """
    try:
        batch = next(dataloader_iter)
    except StopIteration:
        # Restart iterator at epoch boundary
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)

    # Move to device
    batch = {k: v.to(device) for k, v in batch.items()}

    return batch, dataloader_iter


def warmup_step(model, optimizer_model, batch, device):
    """
    Warmup training step: train model with random orders.

    Args:
        model: AOGPTWithHiddenStates
        optimizer_model: Model optimizer
        batch: Data batch
        device: Device

    Returns:
        metrics: Dictionary of metrics
    """
    model.train()

    tokens = batch['tokens']  # [B, T=6]
    logical_ids = batch['logical_ids']  # [B, T=6]
    B, T = tokens.shape

    # Random orders
    orders = model.sample_random_orders(tokens)

    # Forward pass
    logits, loss = model.forward_with_hidden(
        tokens, orders, logical_ids=logical_ids, return_hidden_states=False
    )

    # Backward pass
    optimizer_model.zero_grad()
    loss.backward()

    # Gradient clipping
    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    optimizer_model.step()

    # Compute metrics
    acc = compute_accuracy(logits, tokens)
    acc_x0 = compute_accuracy(logits, tokens, position_idx=0)
    acc_x1 = compute_accuracy(logits, tokens, position_idx=1)
    acc_x2 = compute_accuracy(logits, tokens, position_idx=2)
    acc_x3 = compute_accuracy(logits, tokens, position_idx=3)
    acc_x4 = compute_accuracy(logits, tokens, position_idx=4)
    acc_x5 = compute_accuracy(logits, tokens, position_idx=5)

    return {
        'loss': loss.item(),
        'accuracy': acc,
        'accuracy_x0': acc_x0,
        'accuracy_x1': acc_x1,
        'accuracy_x2': acc_x2,
        'accuracy_x3': acc_x3,
        'accuracy_x4': acc_x4,
        'accuracy_x5': acc_x5,
    }


def coevolution_step(model, agent, optimizer_model, optimizer_agent, batch, device):
    """
    Co-evolution training step: train both model and agent.

    Uses REINFORCE algorithm with continuous rewards (log probabilities).

    Args:
        model: AOGPTWithHiddenStates
        agent: OrderPolicyNet
        optimizer_model: Model optimizer
        optimizer_agent: Agent optimizer
        batch: Data batch
        device: Device

    Returns:
        metrics: Dictionary of metrics
    """
    model.train()
    agent.train()

    tokens = batch['tokens']  # [B, T=6] - shuffled/physical order
    logical_ids = batch['logical_ids']  # [B, T=6] - always [0, 1, 2, 3, 4, 5]
    unshuffled_tokens = batch['unshuffled_tokens']  # [B, T=6] - always [x0, x1, x2, x3, x4, x5]
    B, T = tokens.shape

    # ===== Phase 1: Rollout with Agent =====
    # Start with empty sequence and iteratively fill positions

    # Initialize filled mask (0=available, 1=filled)
    filled_mask = torch.zeros(B, T, device=device)

    # Partial tokens (start with placeholders)
    partial_tokens = torch.zeros(B, T, dtype=torch.long, device=device)

    # Track actions and log probs for REINFORCE
    actions_list = []
    log_probs_list = []
    rewards_list = []

    # Generated order (which position to fill at each step)
    generated_order = torch.zeros(B, T, dtype=torch.long, device=device)

    for step in range(T):
        # Get current order so far (for positions already filled)
        current_order = generated_order.clone()

        # Get hidden states from current partial sequence
        with torch.no_grad():
            _, _, hidden_states = model.forward_with_hidden(
                partial_tokens,
                current_order,
                logical_ids=logical_ids,
                return_hidden_states=True
            )

        # Agent selects next position to fill
        action_probs = agent(hidden_states, filled_mask)
        actions, log_probs = agent.sample_action(hidden_states, filled_mask)

        # Store for REINFORCE
        actions_list.append(actions)
        log_probs_list.append(log_probs)

        # Update generated order
        generated_order[torch.arange(B), actions] = step

        # Fill selected positions with true tokens (from unshuffled)
        # action indicates which logical position to fill (0=x0, 1=x1, 2=x2, 3=x3, 4=x4, 5=x5)
        for b in range(B):
            action_b = actions[b].item()
            # Get the token at this logical position
            partial_tokens[b, action_b] = unshuffled_tokens[b, action_b]

        # Mark position as filled
        filled_mask[torch.arange(B), actions] = 1

        # ===== Compute Reward =====
        # Use log probability of correct token as reward
        # This gives continuous feedback even when prediction is wrong

        # Get model predictions with current partial sequence
        logits, _ = model.forward_with_hidden(
            partial_tokens,
            generated_order,
            logical_ids=logical_ids,
            return_hidden_states=False
        )

        # Get log probabilities
        log_probs_model = F.log_softmax(logits, dim=-1)  # [B, T+1, vocab_size]

        # For each filled position, compute log prob of correct token
        batch_indices = torch.arange(B, device=device)

        # Get correct tokens
        correct_tokens = unshuffled_tokens[batch_indices, actions]
        pred_logits = log_probs_model[batch_indices, actions, correct_tokens]

        # Reward is the log probability (range: [-inf, 0])
        if config.reward_type == 'log_prob':
            reward = pred_logits
        elif config.reward_type == 'binary':
            # Binary reward: 1 if correct, 0 if wrong
            preds = logits[batch_indices, actions, :].argmax(dim=-1)
            reward = (preds == correct_tokens).float()
        else:
            raise ValueError(f"Unknown reward type: {config.reward_type}")

        rewards_list.append(reward)

    # ===== Phase 2: REINFORCE Update for Agent =====
    # Compute returns
    returns = compute_returns(rewards_list, mode='immediate')

    # Normalize returns (variance reduction)
    if config.use_baseline:
        returns = normalize_returns(returns)

    # Compute policy loss
    policy_loss = 0
    for log_prob, ret in zip(log_probs_list, returns):
        policy_loss += -(log_prob * ret.detach()).mean()
    policy_loss = policy_loss / T  # Average over steps

    # Update agent
    optimizer_agent.zero_grad()
    policy_loss.backward()

    # Gradient clipping for agent
    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(agent.parameters(), config.grad_clip)

    optimizer_agent.step()

    # ===== Phase 3: Supervised Update for Model =====
    # Train model with true tokens and generated order
    logits_model, loss_model = model.forward_with_hidden(
        unshuffled_tokens,
        generated_order.detach(),  # Detach to avoid backprop through agent
        logical_ids=logical_ids,
        return_hidden_states=False
    )

    # Update model
    optimizer_model.zero_grad()
    loss_model.backward()

    # Gradient clipping
    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    optimizer_model.step()

    # ===== Compute Metrics =====
    # DAG-specific selection metrics
    dag_metrics = compute_dag_selection_metrics(actions_list, seq_length=T)

    # Model accuracy
    acc = compute_accuracy(logits_model, unshuffled_tokens)
    acc_x0 = compute_accuracy(logits_model, unshuffled_tokens, position_idx=0)
    acc_x1 = compute_accuracy(logits_model, unshuffled_tokens, position_idx=1)
    acc_x2 = compute_accuracy(logits_model, unshuffled_tokens, position_idx=2)
    acc_x3 = compute_accuracy(logits_model, unshuffled_tokens, position_idx=3)
    acc_x4 = compute_accuracy(logits_model, unshuffled_tokens, position_idx=4)
    acc_x5 = compute_accuracy(logits_model, unshuffled_tokens, position_idx=5)

    # Average reward
    avg_reward = torch.stack(rewards_list).mean().item()

    metrics = {
        'loss': loss_model.item(),
        'policy_loss': policy_loss.item(),
        'accuracy': acc,
        'accuracy_x0': acc_x0,
        'accuracy_x1': acc_x1,
        'accuracy_x2': acc_x2,
        'accuracy_x3': acc_x3,
        'accuracy_x4': acc_x4,
        'accuracy_x5': acc_x5,
        'avg_reward': avg_reward,
    }
    metrics.update(dag_metrics)

    return metrics


@torch.no_grad()
def evaluate(model, agent, dataloader, device, num_batches=10):
    """
    Evaluate model and agent.

    Args:
        model: AOGPTWithHiddenStates
        agent: OrderPolicyNet
        dataloader: Validation dataloader
        device: Device
        num_batches: Number of batches to evaluate

    Returns:
        metrics: Dictionary of averaged metrics
    """
    model.eval()
    agent.eval()

    metrics_tracker = MetricsTracker(window_size=num_batches)
    dataloader_iter = iter(dataloader)

    for _ in range(num_batches):
        batch, dataloader_iter = get_batch(dataloader_iter, dataloader, device)

        tokens = batch['tokens']
        logical_ids = batch['logical_ids']
        unshuffled_tokens = batch['unshuffled_tokens']
        B, T = tokens.shape

        # Evaluate with agent-selected order
        filled_mask = torch.zeros(B, T, device=device)
        partial_tokens = torch.zeros(B, T, dtype=torch.long, device=device)
        generated_order = torch.zeros(B, T, dtype=torch.long, device=device)
        actions_list = []

        for step in range(T):
            current_order = generated_order.clone()

            _, _, hidden_states = model.forward_with_hidden(
                partial_tokens,
                current_order,
                logical_ids=logical_ids,
                return_hidden_states=True
            )

            action_probs = agent(hidden_states, filled_mask)
            actions = action_probs.argmax(dim=-1)  # Greedy selection for eval

            actions_list.append(actions)

            generated_order[torch.arange(B), actions] = step

            for b in range(B):
                action_b = actions[b].item()
                partial_tokens[b, action_b] = unshuffled_tokens[b, action_b]

            filled_mask[torch.arange(B), actions] = 1

        # Final prediction
        logits, loss = model.forward_with_hidden(
            unshuffled_tokens,
            generated_order,
            logical_ids=logical_ids,
            return_hidden_states=False
        )

        # Metrics
        dag_metrics = compute_dag_selection_metrics(actions_list, seq_length=T)

        acc = compute_accuracy(logits, unshuffled_tokens)
        acc_x0 = compute_accuracy(logits, unshuffled_tokens, position_idx=0)
        acc_x1 = compute_accuracy(logits, unshuffled_tokens, position_idx=1)
        acc_x2 = compute_accuracy(logits, unshuffled_tokens, position_idx=2)
        acc_x3 = compute_accuracy(logits, unshuffled_tokens, position_idx=3)
        acc_x4 = compute_accuracy(logits, unshuffled_tokens, position_idx=4)
        acc_x5 = compute_accuracy(logits, unshuffled_tokens, position_idx=5)

        update_dict = {
            'loss': loss.item(),
            'accuracy': acc,
            'accuracy_x0': acc_x0,
            'accuracy_x1': acc_x1,
            'accuracy_x2': acc_x2,
            'accuracy_x3': acc_x3,
            'accuracy_x4': acc_x4,
            'accuracy_x5': acc_x5,
        }
        update_dict.update(dag_metrics)

        metrics_tracker.update(**update_dict)

    return metrics_tracker.get_all_means()


def main():
    """Main training loop."""
    print("=" * 80)
    print("LO-ARMs: 6-Node DAG Topology Discovery")
    print("=" * 80)
    print("Goal: Agent should discover DAG structure x0 → (x1, x2) → (x3, x4) → x5")
    print("      - x0 (root) must be generated first")
    print("      - x1, x2 (branches) follow x0")
    print("      - x3, x4 (middle layer) follow x1, x2")
    print("      - x5 (sink) must be generated last")
    print("=" * 80)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Device setup
    device = config.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    print(f"Using device: {device}")

    # Create experiment directory
    exp_dir = create_experiment_dir(config.out_dir, config.wandb_run_name)
    save_config(config, exp_dir)

    # Initialize wandb
    wandb_run = None
    if config.wandb_log:
        try:
            import wandb
            # Filter config to only include JSON-serializable values
            config_dict = {}
            for key, value in vars(config).items():
                # Skip private/magic attributes and non-serializable types
                if not key.startswith('_'):
                    # Test if value is JSON serializable
                    try:
                        import json
                        json.dumps(value)
                        config_dict[key] = value
                    except (TypeError, ValueError):
                        # Skip non-serializable values
                        pass

            wandb_run = wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config_dict
            )
            print(f"Initialized wandb: {config.wandb_project}/{config.wandb_run_name}")
        except ImportError:
            print("wandb not installed, skipping logging")
            config.wandb_log = False

    # ===== Create Datasets =====
    print("\nCreating datasets...")
    train_dataset = DiamondDAGDataset(
        vocab_size=config.vocab_size,
        num_samples=config.num_train_samples,
        seed=config.dataset_seed
    )
    val_dataset = DiamondDAGDataset(
        vocab_size=config.vocab_size,
        num_samples=config.num_val_samples,
        seed=config.dataset_seed + 1
    )
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    )

    # ===== Create Model =====
    print("\nInitializing model...")
    model_config = AOGPTConfig(
        vocab_size=config.vocab_size,
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias
    )
    model = AOGPTWithHiddenStates(model_config).to(device)

    # ===== Create Agent =====
    print("\nInitializing agent...")
    agent = OrderPolicyNet(
        d_model=config.n_embd,
        policy_dim=config.policy_dim,
        num_positions=config.seq_length
    ).to(device)

    # ===== Create Optimizers =====
    print("\nCreating optimizers...")
    optimizer_model = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    optimizer_agent = torch.optim.AdamW(
        agent.parameters(),
        lr=config.agent_learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=0.0  # No weight decay for agent
    )

    # ===== Training Loop =====
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    print(f"Warmup steps: {config.warmup_steps}")
    print(f"Co-evolution steps: {config.max_iters - config.warmup_steps}")
    print(f"Total iterations: {config.max_iters}")
    print("=" * 80 + "\n")

    metrics_tracker = MetricsTracker(window_size=config.log_interval)
    train_loader_iter = iter(train_loader)
    best_topo_correct = 0.0

    for iter_num in range(config.max_iters):
        iter_start_time = time.time()

        # Get batch
        batch, train_loader_iter = get_batch(train_loader_iter, train_loader, device)

        # Training step
        if iter_num < config.warmup_steps:
            # Phase A: Warmup (frozen agent, random orders)
            metrics = warmup_step(model, optimizer_model, batch, device)
            phase = 'warmup'
        else:
            # Phase B: Co-evolution
            metrics = coevolution_step(
                model, agent,
                optimizer_model, optimizer_agent,
                batch, device
            )
            phase = 'coevolution'

        # Update metrics
        metrics_tracker.update(**metrics)

        iter_time = time.time() - iter_start_time

        # Logging
        if (iter_num + 1) % config.log_interval == 0:
            avg_metrics = metrics_tracker.get_all_means()
            avg_metrics['iter_time'] = iter_time
            log_metrics(avg_metrics, iter_num + 1, phase=phase, wandb_run=wandb_run)
            metrics_tracker.reset()

        # Evaluation
        if (iter_num + 1) % config.eval_interval == 0:
            print("\nRunning evaluation...")
            eval_metrics = evaluate(model, agent, val_loader, device, num_batches=config.eval_iters)
            log_metrics(eval_metrics, iter_num + 1, phase='eval', wandb_run=wandb_run)

            # Track best model
            if 'topology/correct' in eval_metrics:
                if eval_metrics['topology/correct'] > best_topo_correct:
                    best_topo_correct = eval_metrics['topology/correct']
                    print(f"  New best topology correctness: {best_topo_correct:.4f}")

        # Checkpointing
        if (iter_num + 1) % config.checkpoint_interval == 0:
            save_checkpoint(
                model, agent,
                optimizer_model, optimizer_agent,
                iter_num + 1,
                vars(config),
                exp_dir
            )

    # Final evaluation
    print("\n" + "=" * 80)
    print("Final Evaluation")
    print("=" * 80)
    final_metrics = evaluate(model, agent, val_loader, device, num_batches=config.eval_iters)
    log_metrics(final_metrics, config.max_iters, phase='final', wandb_run=wandb_run)

    # Save final checkpoint
    save_checkpoint(
        model, agent,
        optimizer_model, optimizer_agent,
        config.max_iters,
        vars(config),
        exp_dir
    )

    # Print final results
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best topology correctness: {best_topo_correct:.4f}")
    print(f"Final P(select_x0_first): {final_metrics.get('first_step/p_x0', 0):.4f}")
    print(f"Final P(select_x5_last): {final_metrics.get('last_step/p_x5', 0):.4f}")
    print(f"Final P(select_any_branch_second): {final_metrics.get('second_step/p_any_branch', 0):.4f}")
    print(f"Final topology correctness: {final_metrics.get('topology/correct', 0):.4f}")
    print(f"Final accuracy: {final_metrics.get('accuracy', 0):.4f}")
    print(f"Experiment directory: {exp_dir}")
    print("=" * 80)

    # Check success criteria
    success = (
        final_metrics.get('first_step/p_x0', 0) > 0.90 and
        final_metrics.get('last_step/p_x5', 0) > 0.90 and
        final_metrics.get('topology/correct', 0) > 0.90
    )

    if success:
        print("\n🎉 SUCCESS! Agent learned the 6-node DAG topology")
        print("   ✓ x0 (root) selected first in >90% of cases")
        print("   ✓ x5 (sink) selected last in >90% of cases")
        print("   ✓ Overall topological order correct in >90% of cases")
    else:
        print("\n⚠️  Training completed but convergence criteria not met")
        print("   Consider longer training or hyperparameter tuning")

    print("=" * 80)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
