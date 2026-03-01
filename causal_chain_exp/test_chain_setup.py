"""
Quick test script to verify causal chain experiment setup.

Tests:
1. Dataset creation and data loading
2. Model initialization
3. Agent initialization
4. Single training step (warmup)
5. Single training step (coevolution)

Run this before launching full training to catch any issues early.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_chain_exp.causal_chain_dataset import CausalChainDataset
from lossy_copy_exp.model_wrapper import AOGPTWithHiddenStates
from lossy_copy_exp.order_policy_net import OrderPolicyNet
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPTConfig
import causal_chain_exp.config_chain as config


def test_setup():
    """Test experiment setup."""
    print("=" * 80)
    print("Testing Causal Chain Experiment Setup")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Test 1: Dataset
    print("\n[Test 1] Dataset Creation")
    print("-" * 80)
    dataset = CausalChainDataset(
        vocab_size=config.vocab_size,
        num_samples=100,
        seed=config.dataset_seed
    )
    print(f"✓ Dataset created: {len(dataset)} samples")

    # Get a sample
    sample = dataset[0]
    print(f"✓ Sample keys: {list(sample.keys())}")
    print(f"  tokens shape: {sample['tokens'].shape}")
    print(f"  logical_ids shape: {sample['logical_ids'].shape}")
    print(f"  Example: tokens={sample['tokens'].tolist()}")
    print(f"           unshuffled={sample['unshuffled_tokens'].tolist()}")

    # Verify causal chain
    unshuffled = sample['unshuffled_tokens']
    a, b, c = unshuffled[0].item(), unshuffled[1].item(), unshuffled[2].item()
    assert b == a // 2, f"B={b} should be {a // 2}"
    assert c == b // 2, f"C={c} should be {b // 2}"
    print(f"✓ Causal chain verified: A={a} → B={b} (A//2) → C={c} (B//2)")

    # Test 2: Model
    print("\n[Test 2] Model Initialization")
    print("-" * 80)
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
    print(f"✓ Model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Config: {config.n_layer} layers, {config.n_head} heads, {config.n_embd} embd")

    # Test 3: Agent
    print("\n[Test 3] Agent Initialization")
    print("-" * 80)
    agent = OrderPolicyNet(
        d_model=config.n_embd,
        policy_dim=config.policy_dim,
        num_positions=config.seq_length
    ).to(device)
    print(f"✓ Agent created")
    print(f"  Parameters: {sum(p.numel() for p in agent.parameters()):,}")
    print(f"  Num positions: {config.seq_length}")

    # Test 4: Forward pass with random order (warmup phase)
    print("\n[Test 4] Forward Pass (Random Order)")
    print("-" * 80)
    batch_size = 4
    tokens = torch.stack([dataset[i]['tokens'] for i in range(batch_size)]).to(device)
    logical_ids = torch.stack([dataset[i]['logical_ids'] for i in range(batch_size)]).to(device)
    unshuffled = torch.stack([dataset[i]['unshuffled_tokens'] for i in range(batch_size)]).to(device)

    print(f"  Batch shape: {tokens.shape}")

    # Random order
    orders = model.sample_random_orders(tokens)
    print(f"  Random orders: {orders[0].tolist()}")

    # Forward pass
    logits, loss = model.forward_with_hidden(
        tokens, orders, logical_ids=logical_ids, return_hidden_states=False
    )
    print(f"✓ Forward pass successful")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # Test 5: Agent rollout (coevolution phase)
    print("\n[Test 5] Agent Rollout")
    print("-" * 80)

    model.eval()
    agent.eval()

    B, T = tokens.shape
    filled_mask = torch.zeros(B, T, device=device)
    partial_tokens = torch.zeros(B, T, dtype=torch.long, device=device)
    generated_order = torch.zeros(B, T, dtype=torch.long, device=device)

    first_actions = None

    for step in range(T):
        current_order = generated_order.clone()

        with torch.no_grad():
            _, _, hidden_states = model.forward_with_hidden(
                partial_tokens,
                current_order,
                logical_ids=logical_ids,
                return_hidden_states=True
            )

        action_probs = agent(hidden_states, filled_mask)
        actions = action_probs.argmax(dim=-1)

        if step == 0:
            first_actions = actions

        generated_order[torch.arange(B), actions] = step

        for b in range(B):
            action_b = actions[b].item()
            partial_tokens[b, action_b] = unshuffled[b, action_b]

        filled_mask[torch.arange(B), actions] = 1

    print(f"✓ Agent rollout successful")
    print(f"  Generated order (sample 0): {generated_order[0].tolist()}")
    print(f"  First actions: {first_actions.tolist()}")

    # Count first action distribution
    prob_root = (first_actions == 0).float().mean().item()
    prob_mid = (first_actions == 1).float().mean().item()
    prob_leaf = (first_actions == 2).float().mean().item()

    print(f"\n  First action distribution (untrained):")
    print(f"    P(select_root_first) = {prob_root:.2f}")
    print(f"    P(select_mid_first)  = {prob_mid:.2f}")
    print(f"    P(select_leaf_first) = {prob_leaf:.2f}")

    # Final forward pass
    with torch.no_grad():
        logits_final, loss_final = model.forward_with_hidden(
            unshuffled,
            generated_order,
            logical_ids=logical_ids,
            return_hidden_states=False
        )

    print(f"✓ Final prediction")
    print(f"  Loss with agent order: {loss_final.item():.4f}")

    # Test 6: Metric computation
    print("\n[Test 6] Metrics")
    print("-" * 80)
    from lossy_copy_exp.utils import compute_accuracy

    acc = compute_accuracy(logits_final, unshuffled)
    acc_a = compute_accuracy(logits_final, unshuffled, position_idx=0)
    acc_b = compute_accuracy(logits_final, unshuffled, position_idx=1)
    acc_c = compute_accuracy(logits_final, unshuffled, position_idx=2)

    print(f"✓ Accuracy (untrained):")
    print(f"    Overall: {acc:.2f}")
    print(f"    A: {acc_a:.2f}")
    print(f"    B: {acc_b:.2f}")
    print(f"    C: {acc_c:.2f}")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\nReady to run full training:")
    print("  bash causal_chain_exp/run_chain.sh")
    print("  OR")
    print("  python causal_chain_exp/train_chain.py")
    print("=" * 80)


if __name__ == '__main__':
    test_setup()
