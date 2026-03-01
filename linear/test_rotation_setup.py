"""
Test Setup for Linear Rotation Experiment

Validates that all components are working correctly before training.
"""

import os
import sys
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from linear_rotation_exp.rotation_dataset import LinearRotationDataset
from lossy_copy_exp.model_wrapper import AOGPTWithHiddenStates
from lossy_copy_exp.order_policy_net import OrderPolicyNet
from model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm import AOGPTConfig
import linear_rotation_exp.config_rotation as config


def test_dataset():
    """Test dataset creation and loading."""
    print("=" * 60)
    print("Test 1: Dataset")
    print("=" * 60)

    dataset = LinearRotationDataset(
        vocab_size=config.vocab_size,
        seq_length=config.seq_length,
        hidden_dim=config.hidden_dim,
        ortho_mode=config.ortho_mode,
        num_samples=100,
        seed=42
    )

    print(f"✓ Dataset created: {len(dataset)} samples")

    # Get a sample
    sample = dataset[0]
    print(f"✓ Sample keys: {list(sample.keys())}")
    print(f"✓ Tokens shape: {sample['tokens'].shape}")
    print(f"✓ Logical IDs shape: {sample['logical_ids'].shape}")
    print(f"✓ Order shape: {sample['order'].shape}")
    print(f"✓ Bag type: {type(sample['bag'])}")

    # Check shapes
    assert sample['tokens'].shape == (config.seq_length,), "Wrong tokens shape"
    assert sample['logical_ids'].shape == (config.seq_length,), "Wrong logical_ids shape"
    assert sample['order'].shape == (config.seq_length,), "Wrong order shape"

    # Check l2r order
    expected_order = list(range(config.seq_length))
    actual_order = sample['order'].tolist()
    assert actual_order == expected_order, f"Expected l2r order {expected_order}, got {actual_order}"

    # Get statistics
    stats = dataset.get_statistics()
    print(f"✓ Dataset statistics:")
    print(f"    Validity: {stats['validity_rate']*100:.1f}%")
    print(f"    Uniqueness: {stats['uniqueness_rate']*100:.1f}%")
    print(f"    Avg Margin: {stats['avg_margin']:.3f}")

    print("✅ PASSED\n")


def test_model():
    """Test model initialization and forward pass."""
    print("=" * 60)
    print("Test 2: Model")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize model
    model_config = AOGPTConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias
    )
    model = AOGPTWithHiddenStates(model_config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created: {num_params / 1e6:.2f}M parameters")

    # Test forward pass
    B, L = 4, config.seq_length
    tokens = torch.randint(0, config.vocab_size, (B, L), device=device)
    orders = torch.zeros(B, L, dtype=torch.long, device=device)
    for i in range(L):
        orders[:, i] = i
    logical_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)

    logits, loss = model.forward_with_hidden(
        tokens, orders, logical_ids=logical_ids, return_hidden_states=False
    )

    print(f"✓ Forward pass successful")
    print(f"    Logits shape: {logits.shape}")
    print(f"    Loss: {loss.item():.4f}")

    # Model returns [B, L+1, vocab_size] because of [None] token
    assert logits.shape == (B, L+1, config.vocab_size), f"Wrong logits shape: {logits.shape}"
    assert loss.item() > 0, "Loss should be positive"

    # Test with hidden states
    logits, loss, hidden_states = model.forward_with_hidden(
        tokens, orders, logical_ids=logical_ids, return_hidden_states=True
    )

    print(f"✓ Forward with hidden states successful")
    print(f"    Hidden states shape: {hidden_states.shape}")

    # Hidden states also include [None] token, so shape is [B, L+1, n_embd]
    assert hidden_states.shape == (B, L+1, config.n_embd), f"Wrong hidden states shape: {hidden_states.shape}"

    print("✅ PASSED\n")


def test_agent():
    """Test agent initialization and action sampling."""
    print("=" * 60)
    print("Test 3: Agent")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize agent
    agent = OrderPolicyNet(
        d_model=config.n_embd,
        policy_dim=config.policy_dim,
        num_positions=config.seq_length
    ).to(device)

    num_params = sum(p.numel() for p in agent.parameters())
    print(f"✓ Agent created: {num_params / 1e6:.2f}M parameters")

    # Test action sampling
    B, L = 4, config.seq_length
    hidden_states = torch.randn(B, L, config.n_embd, device=device)
    filled_mask = torch.zeros(B, L, device=device)
    filled_mask[:, 0] = 1  # Mark first position as filled

    action_probs = agent(hidden_states, filled_mask)
    print(f"✓ Agent forward pass successful")
    print(f"    Action probs shape: {action_probs.shape}")

    assert action_probs.shape == (B, L), f"Wrong action probs shape: {action_probs.shape}"

    # Check that filled positions have zero probability
    assert torch.all(action_probs[:, 0] == 0), "Filled positions should have zero probability"

    # Check that probabilities sum to 1
    probs_sum = action_probs.sum(dim=1)
    assert torch.allclose(probs_sum, torch.ones(B, device=device), atol=1e-5), "Probs should sum to 1"

    # Test action sampling
    actions, log_probs = agent.sample_action(hidden_states, filled_mask)
    print(f"✓ Action sampling successful")
    print(f"    Actions shape: {actions.shape}")
    print(f"    Log probs shape: {log_probs.shape}")

    assert actions.shape == (B,), f"Wrong actions shape: {actions.shape}"
    assert log_probs.shape == (B,), f"Wrong log probs shape: {log_probs.shape}"

    # Check that actions are not from filled positions
    assert torch.all(actions != 0), "Actions should not select filled positions"

    print("✅ PASSED\n")


def test_training_step():
    """Test a single training step."""
    print("=" * 60)
    print("Test 4: Training Step")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create small dataset
    dataset = LinearRotationDataset(
        vocab_size=config.vocab_size,
        seq_length=config.seq_length,
        hidden_dim=config.hidden_dim,
        ortho_mode=config.ortho_mode,
        num_samples=16,
        seed=42
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # Initialize model and agent
    model_config = AOGPTConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias
    )
    model = AOGPTWithHiddenStates(model_config).to(device)
    agent = OrderPolicyNet(
        d_model=config.n_embd,
        policy_dim=config.policy_dim,
        num_positions=config.seq_length
    ).to(device)

    # Initialize optimizers
    optimizer_model = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer_agent = torch.optim.AdamW(agent.parameters(), lr=1e-4)

    # Get a batch
    batch = next(iter(dataloader))
    batch = {k: v.to(device) if k != 'bag' else v for k, v in batch.items()}

    print(f"✓ Batch loaded:")
    print(f"    Tokens shape: {batch['tokens'].shape}")

    # Warmup step (train model with random orders)
    model.train()
    tokens = batch['tokens']
    logical_ids = batch['logical_ids']
    orders = model.sample_random_orders(tokens)

    logits, loss = model.forward_with_hidden(
        tokens, orders, logical_ids=logical_ids, return_hidden_states=False
    )

    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()

    print(f"✓ Warmup step successful")
    print(f"    Loss: {loss.item():.4f}")

    # Co-evolution step (train both model and agent)
    model.train()
    agent.train()

    unshuffled_tokens = batch['unshuffled_tokens']
    B, L = tokens.shape

    filled_mask = torch.zeros(B, L, device=device)
    partial_tokens = torch.zeros(B, L, dtype=torch.long, device=device)
    generated_order = torch.zeros(B, L, dtype=torch.long, device=device)
    actions_list = []

    # Simple rollout for 2 steps
    for step in range(2):
        current_order = generated_order.clone()

        with torch.no_grad():
            _, _, hidden_states = model.forward_with_hidden(
                partial_tokens,
                current_order,
                logical_ids=logical_ids,
                return_hidden_states=True
            )

        actions, log_probs = agent.sample_action(hidden_states, filled_mask)
        actions_list.append(actions)

        generated_order[torch.arange(B), actions] = step

        for b in range(B):
            action_b = actions[b].item()
            partial_tokens[b, action_b] = unshuffled_tokens[b, action_b]

        filled_mask[torch.arange(B), actions] = 1

    print(f"✓ Co-evolution rollout successful")
    print(f"    Actions collected: {len(actions_list)}")

    print("✅ PASSED\n")


def test_data_quality():
    """Test data quality metrics from Phase 0."""
    print("=" * 60)
    print("Test 5: Data Quality (Phase 0 Check)")
    print("=" * 60)

    from linear_rotation_exp.data_generator import LinearDynamicalGenerator

    generator = LinearDynamicalGenerator(
        vocab_size=config.vocab_size,
        hidden_dim=config.hidden_dim,
        ortho_mode=config.ortho_mode,
        seed=42
    )

    # Generate a few sequences and check quality
    num_samples = 100
    validity_count = 0
    uniqueness_count = 0

    for i in range(num_samples):
        start = i % config.vocab_size
        result = generator.generate_sequence(config.seq_length, start, mode='argmax')
        tokens = result['tokens']

        is_valid, is_unique = generator.verify_uniqueness(tokens)

        if is_valid:
            validity_count += 1
        if is_unique:
            uniqueness_count += 1

    validity_rate = validity_count / num_samples
    uniqueness_rate = uniqueness_count / num_samples

    print(f"✓ Data quality check:")
    print(f"    Validity: {validity_rate*100:.1f}% (target: 100%)")
    print(f"    Uniqueness: {uniqueness_rate*100:.1f}% (target: >90%)")

    assert validity_rate == 1.0, "Validity should be 100%"
    assert uniqueness_rate >= 0.9, f"Uniqueness {uniqueness_rate*100:.1f}% < 90%"

    print("✅ PASSED\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("🧪 Testing Linear Rotation Experiment Setup")
    print("=" * 60)
    print()

    tests = [
        ('Dataset', test_dataset),
        ('Model', test_model),
        ('Agent', test_agent),
        ('Training Step', test_training_step),
        ('Data Quality', test_data_quality),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {name}")
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()

    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("✅ All tests passed! Ready for training.")
        return True
    else:
        print("❌ Some tests failed. Please fix before training.")
        return False


if __name__ == '__main__':
    success = run_all_tests()
    import sys
    sys.exit(0 if success else 1)
