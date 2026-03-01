"""
Quick test script to verify Diamond DAG experiment setup.

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

from dag_exp.dag_dataset import DiamondDAGDataset
from lossy_copy_exp.model_wrapper import AOGPTWithHiddenStates
from lossy_copy_exp.order_policy_net import OrderPolicyNet
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPTConfig
import dag_exp.config_dag as config


def test_setup():
    """Test experiment setup."""
    print("=" * 80)
    print("Testing Diamond DAG Experiment Setup")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Test 1: Dataset
    print("\n[Test 1] Dataset Creation")
    print("-" * 80)
    dataset = DiamondDAGDataset(
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

    # Verify DAG structure
    unshuffled = sample['unshuffled_tokens']
    x0, x1, x2, x3 = [unshuffled[i].item() for i in range(4)]
    assert x1 == x0 // 2, f"x1={x1} should be {x0 // 2}"
    assert x2 == (x0 + 1) // 2, f"x2={x2} should be {(x0 + 1) // 2}"
    assert x3 == (x1 + x2) % 16, f"x3={x3} should be {(x1 + x2) % 16}"
    print(f"✓ DAG structure verified:")
    print(f"    x0={x0} → x1={x1} (x0//2)")
    print(f"    x0={x0} → x2={x2} ((x0+1)//2)")
    print(f"    x1={x1}, x2={x2} → x3={x3} ((x1+x2)%16)")

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

    actions_list = []

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

        actions_list.append(actions)

        generated_order[torch.arange(B), actions] = step

        for b in range(B):
            action_b = actions[b].item()
            partial_tokens[b, action_b] = unshuffled[b, action_b]

        filled_mask[torch.arange(B), actions] = 1

    print(f"✓ Agent rollout successful")
    print(f"  Generated order (sample 0): {generated_order[0].tolist()}")
    print(f"\n  Action sequence (sample 0):")
    for step, actions in enumerate(actions_list):
        action = actions[0].item()
        node_name = ['x0(root)', 'x1(brA)', 'x2(brB)', 'x3(sink)'][action]
        print(f"    Step {step}: selected {node_name}")

    # Count first action distribution
    first_actions = actions_list[0]
    prob_x0 = (first_actions == 0).float().mean().item()
    prob_x1 = (first_actions == 1).float().mean().item()
    prob_x2 = (first_actions == 2).float().mean().item()
    prob_x3 = (first_actions == 3).float().mean().item()

    print(f"\n  First action distribution (untrained):")
    print(f"    P(select_x0_first) = {prob_x0:.2f} (should become ~1.0)")
    print(f"    P(select_x1_first) = {prob_x1:.2f} (should become ~0.0)")
    print(f"    P(select_x2_first) = {prob_x2:.2f} (should become ~0.0)")
    print(f"    P(select_x3_first) = {prob_x3:.2f} (should become ~0.0)")

    # Count last action distribution
    last_actions = actions_list[-1]
    prob_x3_last = (last_actions == 3).float().mean().item()
    print(f"\n  Last action distribution (untrained):")
    print(f"    P(select_x3_last) = {prob_x3_last:.2f} (should become ~1.0)")

    # Final forward pass
    with torch.no_grad():
        logits_final, loss_final = model.forward_with_hidden(
            unshuffled,
            generated_order,
            logical_ids=logical_ids,
            return_hidden_states=False
        )

    print(f"\n✓ Final prediction")
    print(f"  Loss with agent order: {loss_final.item():.4f}")

    # Test 6: Metric computation
    print("\n[Test 6] Metrics")
    print("-" * 80)
    from lossy_copy_exp.utils import compute_accuracy

    acc = compute_accuracy(logits_final, unshuffled)
    acc_x0 = compute_accuracy(logits_final, unshuffled, position_idx=0)
    acc_x1 = compute_accuracy(logits_final, unshuffled, position_idx=1)
    acc_x2 = compute_accuracy(logits_final, unshuffled, position_idx=2)
    acc_x3 = compute_accuracy(logits_final, unshuffled, position_idx=3)

    print(f"✓ Accuracy (untrained):")
    print(f"    Overall: {acc:.2f}")
    print(f"    x0: {acc_x0:.2f}")
    print(f"    x1: {acc_x1:.2f}")
    print(f"    x2: {acc_x2:.2f}")
    print(f"    x3: {acc_x3:.2f}")

    # Test topological correctness
    topo_correct = (first_actions == 0) & (last_actions == 3)
    topo_correct_rate = topo_correct.float().mean().item()
    print(f"\n✓ Topological correctness (untrained):")
    print(f"    P(x0 first AND x3 last): {topo_correct_rate:.2f} (should become ~1.0)")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\nReady to run full training:")
    print("  bash dag_exp/run_dag.sh")
    print("  OR")
    print("  python dag_exp/train_dag.py")
    print("\nExpected convergence:")
    print("  • P(select_x0_first) → 1.0 (x0 is the root)")
    print("  • P(select_x3_last) → 1.0 (x3 is the sink)")
    print("  • P(select_any_branch_second) → 1.0 (x1 or x2 after x0)")
    print("  • Topological correctness → 1.0")
    print("=" * 80)


if __name__ == '__main__':
    test_setup()
