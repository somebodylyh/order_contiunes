"""
Test script for Modular Sum experiment components.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lossy_copy_exp.modular_sum_dataset import ModularSumDataset
from lossy_copy_exp.model_wrapper import AOGPTWithHiddenStates
from lossy_copy_exp.order_policy_net import OrderPolicyNet
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPTConfig
from torch.utils.data import DataLoader


def test_3variable_integration():
    """Test integration of all components with 3 variables."""
    print("=" * 80)
    print("Testing Modular Sum (3-Variable) Integration")
    print("=" * 80)

    device = 'cpu'

    # Test 1: Dataset (Lossy mode)
    print("\n[1/8] Testing dataset (Lossy mode)...")
    dataset_lossy = ModularSumDataset(vocab_size=16, num_samples=100, use_lossy=True, seed=42)
    dataloader = DataLoader(dataset_lossy, batch_size=4, shuffle=True)
    batch = next(iter(dataloader))
    print(f"  ✓ Lossy dataset created: {len(dataset_lossy)} samples")
    print(f"  ✓ Batch shapes: tokens={batch['tokens'].shape}, logical_ids={batch['logical_ids'].shape}")
    assert batch['tokens'].shape == (4, 3), f"Expected shape (4, 3), got {batch['tokens'].shape}"

    # Verify lossy computation
    for i in range(4):
        unshuffled = batch['unshuffled_tokens'][i]
        x1, x2, y = unshuffled[0].item(), unshuffled[1].item(), unshuffled[2].item()
        expected_y = (x1 + x2) // 2
        assert y == expected_y, f"Lossy check failed: y={y}, expected={(x1+x2)//2}"
    print(f"  ✓ Lossy computation verified: y = (x1+x2)//2")

    # Test 2: Dataset (Modular mode)
    print("\n[2/8] Testing dataset (Modular mode)...")
    dataset_modular = ModularSumDataset(vocab_size=16, num_samples=100, use_lossy=False, seed=42)
    batch_mod = next(iter(DataLoader(dataset_modular, batch_size=4)))
    print(f"  ✓ Modular dataset created")

    # Verify modular computation
    for i in range(4):
        unshuffled = batch_mod['unshuffled_tokens'][i]
        x1, x2, y = unshuffled[0].item(), unshuffled[1].item(), unshuffled[2].item()
        expected_y = (x1 + x2) % 16
        assert y == expected_y, f"Modular check failed: y={y}, expected={(x1+x2)%16}"
    print(f"  ✓ Modular computation verified: y = (x1+x2) % P")

    # Test 3: Model with 3 positions
    print("\n[3/8] Testing model with seq_length=3...")
    config = AOGPTConfig(
        vocab_size=16,
        block_size=3,  # 3 positions
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        bias=True
    )
    model = AOGPTWithHiddenStates(config).to(device)
    print(f"  ✓ Model created: {model.get_num_params()/1e6:.2f}M parameters")
    print(f"  ✓ Logical pos emb shape: {model.logical_pos_emb.weight.shape}")
    assert model.logical_pos_emb.weight.shape[0] >= 3, "Logical pos emb must support at least 3 positions"

    # Test 4: Forward pass
    print("\n[4/8] Testing model forward pass...")
    tokens = batch['tokens'].to(device)
    logical_ids = batch['logical_ids'].to(device)
    orders = model.sample_random_orders(tokens)

    logits, loss = model.forward_with_hidden(
        tokens, orders, logical_ids=logical_ids, return_hidden_states=False
    )
    print(f"  ✓ Forward pass: logits shape={logits.shape}, loss={loss.item():.4f}")
    assert logits.shape == (4, 4, 16), f"Expected logits shape (4, 4, 16), got {logits.shape}"  # [B, T+1, vocab]

    logits, loss, hidden = model.forward_with_hidden(
        tokens, orders, logical_ids=logical_ids, return_hidden_states=True
    )
    print(f"  ✓ Hidden states: shape={hidden.shape}")
    assert hidden.shape == (4, 4, 64), f"Expected hidden shape (4, 4, 64), got {hidden.shape}"  # [B, T+1, n_embd]

    # Test 5: Agent with 3 positions
    print("\n[5/8] Testing agent with num_positions=3...")
    agent = OrderPolicyNet(d_model=64, policy_dim=32, num_positions=3).to(device)
    print(f"  ✓ Agent created: {sum(p.numel() for p in agent.parameters())} parameters")

    # Test 6: Agent forward pass
    print("\n[6/8] Testing agent forward pass...")
    filled_mask = torch.zeros(4, 3, device=device)
    action_probs = agent(hidden, filled_mask)
    print(f"  ✓ Action probs shape: {action_probs.shape}")
    print(f"  ✓ Sample probs: {action_probs[0].tolist()}")
    assert action_probs.shape == (4, 3), f"Expected shape (4, 3), got {action_probs.shape}"
    assert torch.allclose(action_probs.sum(dim=-1), torch.ones(4), atol=1e-5), "Probs don't sum to 1"

    # Test masking
    filled_mask[:, 0] = 1  # Mask first position
    action_probs_masked = agent(hidden, filled_mask)
    print(f"  ✓ Masked probs: {action_probs_masked[0].tolist()}")
    assert (action_probs_masked[:, 0] < 1e-6).all(), "Position 0 should be masked"

    # Test 7: Full 3-step rollout
    print("\n[7/8] Testing 3-step rollout...")
    model.train()
    agent.train()
    unshuffled_tokens = batch['unshuffled_tokens'].to(device)
    B, T = 4, 3

    filled_mask = torch.zeros(B, T, device=device)
    partial_tokens = torch.zeros(B, T, dtype=torch.long, device=device)
    generated_order = torch.zeros(B, T, dtype=torch.long, device=device)

    first_actions = None
    for step in range(T):
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

        generated_order[torch.arange(B), actions] = step
        for b in range(B):
            partial_tokens[b, actions[b]] = unshuffled_tokens[b, actions[b]]
        filled_mask[torch.arange(B), actions] = 1

        print(f"  Step {step}: actions={actions.tolist()}, filled={filled_mask[0].tolist()}")

    print(f"  ✓ 3-step rollout completed")
    print(f"  ✓ Final order: {generated_order[0].tolist()}")

    # Test 8: First-step selection probability computation
    print("\n[8/8] Testing first-step selection metrics...")
    # Map physical actions to logical IDs
    first_actions_expanded = first_actions.unsqueeze(-1)
    chosen_logical_ids = torch.gather(logical_ids, 1, first_actions_expanded).squeeze(-1)

    p_x1 = (chosen_logical_ids == 0).float().mean().item()
    p_x2 = (chosen_logical_ids == 1).float().mean().item()
    p_y = (chosen_logical_ids == 2).float().mean().item()

    print(f"  ✓ P(select x1 first): {p_x1:.2f}")
    print(f"  ✓ P(select x2 first): {p_x2:.2f}")
    print(f"  ✓ P(select y first): {p_y:.2f}")
    print(f"  ✓ Sum: {p_x1 + p_x2 + p_y:.2f} (should be 1.0)")

    assert abs((p_x1 + p_x2 + p_y) - 1.0) < 0.01, "Probabilities don't sum to 1"

    print("\n" + "=" * 80)
    print("✅ All 3-variable integration tests passed!")
    print("=" * 80)
    print("\nReady to run full training:")
    print("  python lossy_copy_exp/train_modular_sum.py")
    print("=" * 80)


if __name__ == '__main__':
    test_3variable_integration()
