"""
Test script to verify all LO-ARMs components work together.

This runs a minimal training loop to ensure everything is properly integrated.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lossy_copy_exp.lossy_copy_dataset import LossyCopyDataset
from lossy_copy_exp.model_wrapper import AOGPTWithHiddenStates
from lossy_copy_exp.order_policy_net import OrderPolicyNet
from lossy_copy_exp.utils import compute_selection_probability, compute_accuracy
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPTConfig
from torch.utils.data import DataLoader


def test_integration():
    """Test integration of all components."""
    print("=" * 80)
    print("Testing LO-ARMs Integration")
    print("=" * 80)

    device = 'cpu'  # Use CPU for testing

    # Step 1: Create dataset
    print("\n[1/6] Testing dataset...")
    dataset = LossyCopyDataset(vocab_size=16, k=2, num_samples=100, seed=42)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print(f"  ✓ Dataset created: {len(dataset)} samples")
    print(f"  ✓ Batch shape: tokens={batch['tokens'].shape}, logical_ids={batch['logical_ids'].shape}")

    # Step 2: Create model
    print("\n[2/6] Testing model wrapper...")
    config = AOGPTConfig(
        vocab_size=16,
        block_size=2,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=True
    )
    model = AOGPTWithHiddenStates(config).to(device)
    print(f"  ✓ Model created: {model.get_num_params()/1e6:.2f}M parameters")

    # Step 3: Test forward pass
    print("\n[3/6] Testing model forward pass...")
    tokens = batch['tokens'].to(device)
    logical_ids = batch['logical_ids'].to(device)
    orders = model.sample_random_orders(tokens)

    logits, loss = model.forward_with_hidden(
        tokens, orders, logical_ids=logical_ids, return_hidden_states=False
    )
    print(f"  ✓ Forward pass (no hidden): loss={loss.item():.4f}")

    logits, loss, hidden = model.forward_with_hidden(
        tokens, orders, logical_ids=logical_ids, return_hidden_states=True
    )
    print(f"  ✓ Forward pass (with hidden): hidden shape={hidden.shape}")
    assert hidden.shape == (4, 3, 64), f"Expected hidden shape (4, 3, 64), got {hidden.shape}"

    # Step 4: Create agent
    print("\n[4/6] Testing agent...")
    agent = OrderPolicyNet(d_model=64, policy_dim=32, num_positions=2).to(device)
    print(f"  ✓ Agent created: {sum(p.numel() for p in agent.parameters())} parameters")

    # Step 5: Test agent forward pass
    print("\n[5/6] Testing agent forward pass...")
    filled_mask = torch.zeros(4, 2, device=device)
    action_probs = agent(hidden, filled_mask)
    print(f"  ✓ Action probs: {action_probs[0].tolist()}")
    assert torch.allclose(action_probs.sum(dim=-1), torch.ones(4)), "Probabilities don't sum to 1"

    actions, log_probs = agent.sample_action(hidden, filled_mask)
    print(f"  ✓ Sampled actions: {actions.tolist()}")
    print(f"  ✓ Log probs: {log_probs.tolist()}")

    # Step 6: Test full training step
    print("\n[6/6] Testing full training step...")
    optimizer_model = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_agent = torch.optim.Adam(agent.parameters(), lr=1e-4)

    # Warmup step
    model.train()
    orders = model.sample_random_orders(tokens)
    logits, loss = model.forward_with_hidden(
        tokens, orders, logical_ids=logical_ids, return_hidden_states=False
    )
    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()
    print(f"  ✓ Warmup step: loss={loss.item():.4f}")

    # Co-evolution step (simplified)
    model.train()
    agent.train()
    unshuffled_tokens = batch['unshuffled_tokens'].to(device)

    # Agent rollout
    filled_mask = torch.zeros(4, 2, device=device)
    partial_tokens = torch.zeros(4, 2, dtype=torch.long, device=device)
    generated_order = torch.zeros(4, 2, dtype=torch.long, device=device)

    first_actions = None
    for step in range(2):
        with torch.no_grad():
            _, _, hidden_states = model.forward_with_hidden(
                partial_tokens,
                generated_order,
                logical_ids=logical_ids,
                return_hidden_states=True
            )

        actions, log_probs = agent.sample_action(hidden_states, filled_mask)

        if step == 0:
            first_actions = actions

        generated_order[torch.arange(4), actions] = step
        for b in range(4):
            partial_tokens[b, actions[b]] = unshuffled_tokens[b, actions[b]]
        filled_mask[torch.arange(4), actions] = 1

    # Model update
    logits, loss = model.forward_with_hidden(
        unshuffled_tokens,
        generated_order,
        logical_ids=logical_ids,
        return_hidden_states=False
    )
    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()
    print(f"  ✓ Model update: loss={loss.item():.4f}")

    # Compute metrics
    prob_x_first = compute_selection_probability(first_actions, x_position=0)
    acc = compute_accuracy(logits, unshuffled_tokens)
    print(f"  ✓ P(select_x_first)={prob_x_first:.2f}, accuracy={acc:.2f}")

    print("\n" + "=" * 80)
    print("✅ All integration tests passed!")
    print("=" * 80)
    print("\nThe implementation is ready to run. To start training:")
    print("  python lossy_copy_exp/train_loarms.py")
    print("=" * 80)


if __name__ == '__main__':
    test_integration()
