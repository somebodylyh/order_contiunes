"""
Causal Chain Dataset: A → B → C

Tests whether Agent can discover multi-level causal hierarchy:
- A (Root): Source node, uniformly random
- B (Middle): B = A // 2 (lossy, A→B deterministic but B↛A ambiguous)
- C (Leaf): C = B // 2 (lossy, B→C deterministic but C↛B ambiguous)

Key insight: Due to integer division irreversibility:
- Knowing A → Can determine B and C
- Knowing B → Can only determine C, cannot recover A
- Knowing C → Cannot determine anything

Agent should learn: Generate A first (root), then B, finally C (leaf)
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class CausalChainDataset(Dataset):
    """
    Three-variable causal chain dataset with hierarchical dependencies.

    Args:
        vocab_size: Range for A (root) [0, vocab_size)
        num_samples: Dataset size
        seed: Random seed for reproducibility

    Causal Structure:
        A (Root) → B = A // 2 → C = B // 2

        Information content: I(A) > I(B) > I(C)
        Causal depth: A (depth 0) > B (depth 1) > C (depth 2)

    Returns:
        Dictionary containing:
            'tokens': [3] shuffled sequence [could be any permutation of A, B, C]
            'logical_ids': [3] logical positions [0, 1, 2] where 0=A(root), 1=B(mid), 2=C(leaf)
            'unshuffled_tokens': [3] original [A, B, C] order
            'order': [3] shuffle permutation applied
    """

    def __init__(self, vocab_size=64, num_samples=10000, seed=42):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seed = seed

        # Validate parameters
        assert vocab_size >= 4, "vocab_size must be >= 4 for meaningful causal chain"

        print(f"CausalChainDataset initialized:")
        print(f"  Causal structure: A → B = A//2 → C = B//2")
        print(f"  Vocab size: {vocab_size}")
        print(f"  A range: [0, {vocab_size})")
        print(f"  B range: [0, {vocab_size//2})")
        print(f"  C range: [0, {vocab_size//4})")
        print(f"  Samples: {num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a single (A, B, C) causal chain.

        Returns:
            dict with keys:
                'tokens': shuffled [A, B, C] (physical order)
                'logical_ids': [0, 1, 2] (logical positions: 0=root, 1=mid, 2=leaf)
                'unshuffled_tokens': original [A, B, C]
                'order': permutation applied
        """
        # Use idx as part of seed for deterministic but varied sampling
        local_rng = np.random.RandomState(self.seed + idx)

        # Step 1: Sample A (Root) uniformly from [0, vocab_size)
        a = local_rng.randint(0, self.vocab_size)

        # Step 2: Compute B = A // 2 (Middle node, depends on A)
        b = a // 2

        # Step 3: Compute C = B // 2 (Leaf node, depends on B)
        c = b // 2

        # Create unshuffled sequence [A, B, C]
        unshuffled_tokens = torch.tensor([a, b, c], dtype=torch.long)

        # Logical IDs: 0=A(root), 1=B(mid), 2=C(leaf) (unchanging, identifies causal depth)
        logical_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        # Randomly shuffle physical order (creates all 3! = 6 permutations)
        indices = list(range(3))
        local_rng.shuffle(indices)
        order = torch.tensor(indices, dtype=torch.long)

        # Apply shuffle
        tokens = unshuffled_tokens[order]

        return {
            'tokens': tokens,
            'logical_ids': logical_ids,  # Always [0, 1, 2], unchanged
            'unshuffled_tokens': unshuffled_tokens,
            'order': order
        }


def test_causal_chain_dataset():
    """Test the CausalChainDataset implementation."""
    print("=" * 80)
    print("Testing CausalChainDataset")
    print("=" * 80)

    # Test 1: Basic dataset creation
    print("\n[Test 1] Basic Dataset Creation")
    print("-" * 80)
    dataset = CausalChainDataset(vocab_size=64, num_samples=100, seed=42)

    # Check dataset length
    assert len(dataset) == 100, "Dataset length incorrect"
    print(f"✓ Dataset length: {len(dataset)}")

    # Test 2: Verify causal chain logic
    print("\n[Test 2] Causal Chain Logic")
    print("-" * 80)
    print("First 10 samples:")
    for i in range(10):
        sample = dataset[i]
        unshuffled = sample['unshuffled_tokens']
        a, b, c = unshuffled[0].item(), unshuffled[1].item(), unshuffled[2].item()

        # Verify causal relationships
        assert b == a // 2, f"B={b} should be {a // 2} (A//2)"
        assert c == b // 2, f"C={c} should be {b // 2} (B//2)"

        # Verify ranges
        assert 0 <= a < 64, f"A={a} out of range [0, 64)"
        assert 0 <= b < 32, f"B={b} out of range [0, 32)"
        assert 0 <= c < 16, f"C={c} out of range [0, 16)"

        print(f"  Sample {i:2d}: A={a:2d} → B={b:2d} (A//2) → C={c:2d} (B//2)")

    # Test 3: Information content analysis
    print("\n[Test 3] Information Content Analysis")
    print("-" * 80)
    print("Given each variable, how many possibilities for the others?")

    # Case 1: Given A (root)
    a_test = 20
    b_from_a = a_test // 2
    c_from_a = b_from_a // 2
    print(f"\nGiven A={a_test}:")
    print(f"  B is uniquely determined: B = {b_from_a}")
    print(f"  C is uniquely determined: C = {c_from_a}")
    print(f"  Ambiguity: 1 possibility (fully determined)")

    # Case 2: Given B (middle)
    b_test = 10
    c_from_b = b_test // 2
    # Find all possible A values
    possible_a = [a for a in range(64) if a // 2 == b_test]
    print(f"\nGiven B={b_test}:")
    print(f"  C is uniquely determined: C = {c_from_b}")
    print(f"  A is ambiguous: {len(possible_a)} possibilities {possible_a}")

    # Case 3: Given C (leaf)
    c_test = 5
    # Find all possible B values
    possible_b = [b for b in range(32) if b // 2 == c_test]
    # For each B, find possible A values
    possible_a_from_c = []
    for b_val in possible_b:
        possible_a_from_c.extend([a for a in range(64) if a // 2 == b_val])
    print(f"\nGiven C={c_test}:")
    print(f"  B is ambiguous: {len(possible_b)} possibilities {possible_b}")
    print(f"  A is highly ambiguous: {len(possible_a_from_c)} possibilities")

    print("\n✓ Information hierarchy: I(A) > I(B) > I(C)")

    # Test 4: Shuffle distribution
    print("\n[Test 4] Shuffle Distribution")
    print("-" * 80)
    from collections import Counter
    order_counts = Counter()
    for i in range(600):
        sample = dataset[i % 100]
        order_tuple = tuple(sample['order'].tolist())
        order_counts[order_tuple] += 1

    print(f"Order distribution (should be ~uniform over 6 permutations):")
    for order, count in sorted(order_counts.items()):
        print(f"  {list(order)}: {count:3d} ({count/600*100:.1f}%)")

    # Test 5: Logical IDs binding
    print("\n[Test 5] Logical IDs Binding")
    print("-" * 80)
    sample = dataset[0]
    tokens = sample['tokens']
    logical_ids = sample['logical_ids']
    unshuffled = sample['unshuffled_tokens']
    order = sample['order']

    print(f"Unshuffled: A={unshuffled[0]}, B={unshuffled[1]}, C={unshuffled[2]}")
    print(f"Order applied: {order.tolist()}")
    print(f"Shuffled tokens: {tokens.tolist()}")
    print(f"Logical IDs (always [0,1,2]): {logical_ids.tolist()}")

    # Verify tokens match shuffled unshuffled
    expected_tokens = unshuffled[order]
    assert torch.equal(tokens, expected_tokens), "Tokens don't match shuffled unshuffled"
    print("✓ Token-logical_id binding verified")

    # Test 6: Causal depth analysis
    print("\n[Test 6] Causal Depth Analysis")
    print("-" * 80)
    print("Causal depth = number of steps from root")
    print("  A (Root): depth 0 - source, no dependencies")
    print("  B (Mid):  depth 1 - depends on A")
    print("  C (Leaf): depth 2 - depends on B (and transitively on A)")
    print("\nOptimal generation order: A → B → C (follow causal chain)")

    # Test 7: Edge cases
    print("\n[Test 7] Edge Cases")
    print("-" * 80)

    # Small vocab
    dataset_small = CausalChainDataset(vocab_size=8, num_samples=50, seed=123)
    sample_small = dataset_small[0]
    a, b, c = sample_small['unshuffled_tokens'].tolist()
    print(f"Small vocab (8): A={a}, B={b}, C={c}")
    assert 0 <= a < 8 and 0 <= b < 4 and 0 <= c < 2, "Range check failed"
    print("✓ Small vocab works")

    # Large vocab
    dataset_large = CausalChainDataset(vocab_size=128, num_samples=50, seed=456)
    sample_large = dataset_large[0]
    a, b, c = sample_large['unshuffled_tokens'].tolist()
    print(f"Large vocab (128): A={a}, B={b}, C={c}")
    assert b == a // 2 and c == b // 2, "Causal chain check failed"
    print("✓ Large vocab works")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\nKey Insights:")
    print("  • A (root) has FULL information about the chain")
    print("  • B (mid) has PARTIAL information (can determine C, not A)")
    print("  • C (leaf) has MINIMAL information (cannot determine B or A)")
    print("  • Agent should learn: Generate A first, then B, finally C")
    print("=" * 80)


if __name__ == '__main__':
    test_causal_chain_dataset()
