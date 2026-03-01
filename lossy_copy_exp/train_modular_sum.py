"""
Training Script for Modular Sum Experiment

Extended 3-variable experiment with configurable causal structure.
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
from lossy_copy_exp.modular_sum_dataset import ModularSumDataset
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
import lossy_copy_exp.config_modular_sum as config


def compute_first_step_selection_probs(first_actions, logical_ids_batch):
    """
    Compute probability of selecting each logical position (x1, x2, y) first.

    Args:
        first_actions: [B] physical positions selected first
        logical_ids_batch: [B, L] logical IDs at each physical position

    Returns:
        dict with probabilities for x1, x2, y, and any_x
    """
    B = first_actions.shape[0]

    # Map physical position to logical ID
    # first_actions is [B] containing physical position indices
    # logical_ids_batch is [B, L] where L=3
    # We need to gather the logical ID at the selected physical position

    # Expand first_actions to [B, 1] for gathering
    first_actions_expanded = first_actions.unsqueeze(-1)  # [B, 1]

    # Gather logical IDs at selected positions
    chosen_logical_ids = torch.gather(logical_ids_batch, 1, first_actions_expanded).squeeze(-1)  # [B]

    # Count selections
    p_x1 = (chosen_logical_ids == 0).float().mean().item()
    p_x2 = (chosen_logical_ids == 1).float().mean().item()
    p_y = (chosen_logical_ids == 2).float().mean().item()
    p_any_x = p_x1 + p_x2  # Combined probability of selecting any x

    return {
        'p_select_x1_first': p_x1,
        'p_select_x2_first': p_x2,
        'p_select_y_first': p_y,
        'p_select_any_x_first': p_any_x,
    }


def get_batch(dataloader_iter, dataloader, device):
    """Get next batch from dataloader, handling epoch boundaries."""
    try:
        batch = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)

    # Move to device
    batch = {k: v.to(device) for k, v in batch.items()}

    return batch, dataloader_iter


def warmup_step(model, optimizer_model, batch, device):
    """
    Warmup training step: train model with random orders.
    """
    model.train()

    tokens = batch['tokens']  # [B, T]
    logical_ids = batch['logical_ids']  # [B, T]
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
    acc_x1 = compute_accuracy(logits, tokens, position_idx=0)
    acc_x2 = compute_accuracy(logits, tokens, position_idx=1)
    acc_y = compute_accuracy(logits, tokens, position_idx=2)

    return {
        'loss': loss.item(),
        'accuracy': acc,
        'accuracy_x1': acc_x1,
        'accuracy_x2': acc_x2,
        'accuracy_y': acc_y,
    }


def coevolution_step(model, agent, optimizer_model, optimizer_agent, batch, device):
    """
    Co-evolution training step: train both model and agent with REINFORCE.
    """
    model.train()
    agent.train()

    tokens = batch['tokens']  # [B, T]
    logical_ids = batch['logical_ids']  # [B, T]
    unshuffled_tokens = batch['unshuffled_tokens']  # [B, T]
    B, T = tokens.shape

    # ===== Phase 1: Agent Rollout =====
    filled_mask = torch.zeros(B, T, device=device)
    partial_tokens = torch.zeros(B, T, dtype=torch.long, device=device)
    generated_order = torch.zeros(B, T, dtype=torch.long, device=device)

    actions_list = []
    log_probs_list = []
    rewards_list = []

    # Track first step selection for monitoring
    first_actions = None

    for step in range(T):
        current_order = generated_order.clone()

        # Get hidden states
        with torch.no_grad():
            _, _, hidden_states = model.forward_with_hidden(
                partial_tokens,
                current_order,
                logical_ids=logical_ids,
                return_hidden_states=True
            )

        # Agent selects next position
        action_probs = agent(hidden_states, filled_mask)
        actions, log_probs = agent.sample_action(hidden_states, filled_mask)

        # Store first step actions for monitoring
        if step == 0:
            first_actions = actions.clone()

        # Store for REINFORCE
        actions_list.append(actions)
        log_probs_list.append(log_probs)

        # Update generated order
        generated_order[torch.arange(B), actions] = step

        # Fill selected positions with true tokens
        for b in range(B):
            action_b = actions[b].item()
            partial_tokens[b, action_b] = unshuffled_tokens[b, action_b]

        # Mark position as filled
        filled_mask[torch.arange(B), actions] = 1

        # Compute reward
        logits, _ = model.forward_with_hidden(
            partial_tokens,
            generated_order,
            logical_ids=logical_ids,
            return_hidden_states=False
        )

        log_probs_model = F.log_softmax(logits, dim=-1)
        batch_indices = torch.arange(B, device=device)

        correct_tokens = unshuffled_tokens[batch_indices, actions]
        pred_logits = log_probs_model[batch_indices, actions, correct_tokens]

        if config.reward_type == 'log_prob':
            reward = pred_logits
        elif config.reward_type == 'binary':
            preds = logits[batch_indices, actions, :].argmax(dim=-1)
            reward = (preds == correct_tokens).float()
        else:
            raise ValueError(f"Unknown reward type: {config.reward_type}")

        rewards_list.append(reward)

    # ===== Phase 2: REINFORCE Update for Agent =====
    returns = compute_returns(rewards_list, mode='immediate')

    if config.use_baseline:
        returns = normalize_returns(returns)

    policy_loss = 0
    for log_prob, ret in zip(log_probs_list, returns):
        policy_loss += -(log_prob * ret.detach()).mean()
    policy_loss = policy_loss / T

    optimizer_agent.zero_grad()
    policy_loss.backward()

    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(agent.parameters(), config.grad_clip)

    optimizer_agent.step()

    # ===== Phase 3: Supervised Update for Model =====
    logits_model, loss_model = model.forward_with_hidden(
        unshuffled_tokens,
        generated_order.detach(),
        logical_ids=logical_ids,
        return_hidden_states=False
    )

    optimizer_model.zero_grad()
    loss_model.backward()

    if config.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

    optimizer_model.step()

    # ===== Compute Metrics =====
    # First-step selection probabilities (CRITICAL METRIC)
    first_step_probs = compute_first_step_selection_probs(first_actions, logical_ids)

    # Model accuracy
    acc = compute_accuracy(logits_model, unshuffled_tokens)
    acc_x1 = compute_accuracy(logits_model, unshuffled_tokens, position_idx=0)
    acc_x2 = compute_accuracy(logits_model, unshuffled_tokens, position_idx=1)
    acc_y = compute_accuracy(logits_model, unshuffled_tokens, position_idx=2)

    # Average reward
    avg_reward = torch.stack(rewards_list).mean().item()

    metrics = {
        'loss': loss_model.item(),
        'policy_loss': policy_loss.item(),
        'accuracy': acc,
        'accuracy_x1': acc_x1,
        'accuracy_x2': acc_x2,
        'accuracy_y': acc_y,
        'avg_reward': avg_reward,
    }

    # Add first-step selection probabilities
    metrics.update(first_step_probs)

    return metrics


@torch.no_grad()
def evaluate(model, agent, dataloader, device, num_batches=10):
    """Evaluate model and agent."""
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

        # Rollout with greedy agent
        filled_mask = torch.zeros(B, T, device=device)
        partial_tokens = torch.zeros(B, T, dtype=torch.long, device=device)
        generated_order = torch.zeros(B, T, dtype=torch.long, device=device)
        first_actions = None

        for step in range(T):
            current_order = generated_order.clone()

            _, _, hidden_states = model.forward_with_hidden(
                partial_tokens,
                current_order,
                logical_ids=logical_ids,
                return_hidden_states=True
            )

            action_probs = agent(hidden_states, filled_mask)
            actions = action_probs.argmax(dim=-1)  # Greedy

            if step == 0:
                first_actions = actions

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
        first_step_probs = compute_first_step_selection_probs(first_actions, logical_ids)
        acc = compute_accuracy(logits, unshuffled_tokens)
        acc_x1 = compute_accuracy(logits, unshuffled_tokens, position_idx=0)
        acc_x2 = compute_accuracy(logits, unshuffled_tokens, position_idx=1)
        acc_y = compute_accuracy(logits, unshuffled_tokens, position_idx=2)

        batch_metrics = {
            'loss': loss.item(),
            'accuracy': acc,
            'accuracy_x1': acc_x1,
            'accuracy_x2': acc_x2,
            'accuracy_y': acc_y,
        }
        batch_metrics.update(first_step_probs)

        metrics_tracker.update(**batch_metrics)

    return metrics_tracker.get_all_means()


def main():
    """Main training loop."""
    print("=" * 80)
    print("LO-ARMs: Modular Sum Experiment")
    print("=" * 80)
    print(f"Mode: {'Lossy (y = (x1+x2)//2)' if config.use_lossy else 'Modular (y = (x1+x2) % P)'}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Sequence length: {config.seq_length}")
    print("=" * 80)

    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Device setup
    device = config.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    print(f"\nUsing device: {device}")

    # Create experiment directory
    exp_dir = create_experiment_dir(config.out_dir, config.wandb_run_name)
    save_config(config, exp_dir)

    # Initialize wandb
    wandb_run = None
    if config.wandb_log:
        try:
            import wandb
            # Filter config for JSON serialization
            config_dict = {}
            for key, value in vars(config).items():
                if not key.startswith('_'):
                    try:
                        import json
                        json.dumps(value)
                        config_dict[key] = value
                    except (TypeError, ValueError):
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

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ModularSumDataset(
        vocab_size=config.vocab_size,
        num_samples=config.num_train_samples,
        use_lossy=config.use_lossy,
        seed=config.dataset_seed
    )
    val_dataset = ModularSumDataset(
        vocab_size=config.vocab_size,
        num_samples=config.num_val_samples,
        use_lossy=config.use_lossy,
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

    # Create model
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

    # Create agent
    print("\nInitializing agent...")
    agent = OrderPolicyNet(
        d_model=config.n_embd,
        policy_dim=config.policy_dim,
        num_positions=config.seq_length  # 3 positions
    ).to(device)

    # Create optimizers
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
        weight_decay=0.0
    )

    # Training loop
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    print(f"Warmup steps: {config.warmup_steps}")
    print(f"Co-evolution steps: {config.max_iters - config.warmup_steps}")
    print(f"Total iterations: {config.max_iters}")
    print("=" * 80 + "\n")

    metrics_tracker = MetricsTracker(window_size=config.log_interval)
    train_loader_iter = iter(train_loader)
    best_p_any_x = 0.0 if config.use_lossy else 0.33  # Different targets for different modes

    for iter_num in range(config.max_iters):
        iter_start_time = time.time()

        # Get batch
        batch, train_loader_iter = get_batch(train_loader_iter, train_loader, device)

        # Training step
        if iter_num < config.warmup_steps:
            metrics = warmup_step(model, optimizer_model, batch, device)
            phase = 'warmup'
        else:
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

            # Track best based on mode
            if config.use_lossy:
                # Lossy mode: want P(select_any_x_first) -> 1.0
                if 'p_select_any_x_first' in eval_metrics:
                    if eval_metrics['p_select_any_x_first'] > best_p_any_x:
                        best_p_any_x = eval_metrics['p_select_any_x_first']
                        print(f"  New best P(select_any_x_first): {best_p_any_x:.4f}")
            else:
                # Modular mode: just track for observation
                if 'p_select_y_first' in eval_metrics:
                    print(f"  P(select_y_first): {eval_metrics['p_select_y_first']:.4f} (should stay ~0.33)")

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
    if config.use_lossy:
        print(f"Best P(select_any_x_first): {best_p_any_x:.4f} (target: >0.90)")
        print(f"Final P(select_y_first): {final_metrics.get('p_select_y_first', 0):.4f} (target: ~0.0)")
    else:
        print(f"Final P(select_y_first): {final_metrics.get('p_select_y_first', 0):.4f} (expected: ~0.33)")
    print(f"Final accuracy: {final_metrics.get('accuracy', 0):.4f}")
    print(f"Experiment directory: {exp_dir}")
    print("=" * 80)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    main()
