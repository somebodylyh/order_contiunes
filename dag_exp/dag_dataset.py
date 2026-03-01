"""
Diamond DAG Dataset: x0 → (x1, x2) → x3

Tests whether Agent can discover complex DAG topology with both
fork and join structures.

Graph Structure:
       x0 (Root)
      /    \
     /      \
   x1        x2
 (Branch A) (Branch B)
     \      /
      \    /
       x3 (Sink)

Mathematical Definition:
- x0 (Root): Uniformly sampled from [0, vocab_size)
- x1 (Branch A): x1 = x0 // 2 (lossy, forces x0 → x1 causality)
- x2 (Branch B): x2 = (x0 + 1) // 2 (lossy, forces x0 → x2 causality)
- x3 (Sink): x3 = (x1 + x2) % 16 (joins both branches)

Key Insights:
1. x0 is the ONLY root - must be generated first
2. x1 and x2 are symmetric branches - can be generated in any order after x0
3. x3 is the sink - must be generated last (depends on both x1 and x2)

Expected Topological Sort: x0 → {x1, x2} → x3
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class DiamondDAGDataset(Dataset):
    """
    Four-variable Diamond DAG dataset with fork and join structures.

    Args:
        vocab_size: Range for x0 (root) [0, vocab_size)
        num_samples: Dataset size
        seed: Random seed for reproducibility

    DAG Structure:
        x0 (Root, depth 0) → x1 (Branch A, depth 1)
                          → x2 (Branch B, depth 1)
        x1, x2 → x3 (Sink, depth 2)

        Information content: I(x0) > I(x1) ≈ I(x2) > I(x3)
        Causal depth: x0 (0) < x1, x2 (1) < x3 (2)

    Returns:
        Dictionary containing:
            'tokens': [4] shuffled sequence [any permutation of x0, x1, x2, x3]
            'logical_ids': [4] logical positions [0, 1, 2, 3] where:
                0 = x0 (root)
                1 = x1 (branch A)
                2 = x2 (branch B)
                3 = x3 (sink)
            'unshuffled_tokens': [4] original [x0, x1, x2, x3] order
            'order': [4] shuffle permutation applied
    """

    def __init__(self, vocab_size=64, num_samples=10000, seed=42):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_samples = num_samples
        self.seed = seed

        # Validate parameters
        assert vocab_size >= 4, "vocab_size must be >= 4 for meaningful DAG"

        print(f"DiamondDAGDataset initialized:")
        print(f"  DAG structure: x0 → (x1, x2) → (x3, x4) → x5")
        print(f"  Vocab size: {vocab_size}")
        print(f"  x0 range: [0, {vocab_size})")
        print(f"  x1 range: [0, {vocab_size//2})")
        print(f"  x2 range: [0, {(vocab_size+1)//2})")
        print(f"  x3 range: [0, {vocab_size//4})")
        print(f"  x4 range: [0, {vocab_size//4})")
        print(f"  x5 range: [0, {vocab_size//2})")
        print(f"  Samples: {num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a single (x0, x1, x2, x3, x4, x5) Diamond DAG sample.

        Returns:
            dict with keys:
                'tokens': shuffled [x0, x1, x2, x3, x4, x5] (physical order)
                'logical_ids': [0, 1, 2, 3, 4, 5] (logical positions)
                'unshuffled_tokens': original [x0, x1, x2, x3, x4, x5]
                'order': permutation applied
        """
        # Use idx as part of seed for deterministic but varied sampling
        local_rng = np.random.RandomState(self.seed + idx)

        # Step 1: Sample x0 (Root) uniformly from [0, vocab_size)
        x0 = local_rng.randint(0, self.vocab_size)

        # Step 2: Compute x1 (Branch A) = x0 // 2
        x1 = x0 // 2

        # Step 3: Compute x2 (Branch B) = (x0 + 1) // 2
        # Slightly different mapping to create diversity
        x2 = (x0 + 1) // 2

        # Step 4: Compute x3 (Sink) = (x1 + x2) % 16
        # Joins both branches, modulo to keep vocab small
        x3 = (x1 + x2) % (self.vocab_size//4)

        # Step 5: Compute x4 (Branch C) = (x3 + 1) // 2
        x4 = (x3 + 1) // 2

        # Step 6: Compute x5 (Sink) = (x3 + x4) % 16
        # Joins x3 and x4, modulo to keep vocab small
        x5 = (x3 + x4) % 16

        # Create unshuffled sequence [x0, x1, x2, x3, x4, x5]
        unshuffled_tokens = torch.tensor([x0, x1, x2, x3, x4, x5], dtype=torch.long)

        # Logical IDs: 0=x0(root), 1=x1(branchA), 2=x2(branchB), 3=x3(sink), 4=x4(branchC), 5=x5(sink)
        logical_ids = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)

        # Randomly shuffle physical order (creates all 4! = 24 permutations)
        indices = list(range(6))
        local_rng.shuffle(indices)
        order = torch.tensor(indices, dtype=torch.long)

        # Apply shuffle
        tokens = unshuffled_tokens[order]

        return {
            'tokens': tokens,
            'logical_ids': logical_ids,  # Always [0, 1, 2, 3, 4, 5], unchanged
            'unshuffled_tokens': unshuffled_tokens,
            'order': order
        }


def test_diamond_dag_dataset():
    """Test the DiamondDAGDataset implementation."""
    print("=" * 80)
    print("Testing DiamondDAGDataset")
    print("=" * 80)

    # Test 1: Basic dataset creation
    print("\n[Test 1] Basic Dataset Creation")
    print("-" * 80)
    dataset = DiamondDAGDataset(vocab_size=64, num_samples=100, seed=42)

    # Check dataset length
    assert len(dataset) == 100, "Dataset length incorrect"
    print(f"✓ Dataset length: {len(dataset)}")

    # Test 2: Verify DAG logic
    print("\n[Test 2] DAG Structure Logic")
    print("-" * 80)
    print("First 10 samples:")
    for i in range(10):
        sample = dataset[i]
        unshuffled = sample['unshuffled_tokens']
        x0, x1, x2, x3, x4, x5 = [unshuffled[j].item() for j in range(6)]

        # Verify DAG relationships
        assert x1 == x0 // 2, f"x1={x1} should be {x0 // 2} (x0//2)"
        assert x2 == (x0 + 1) // 2, f"x2={x2} should be {(x0 + 1) // 2} ((x0+1)//2)"
        assert x3 == (x1 + x2) % 16, f"x3={x3} should be {(x1 + x2) % 16} ((x1+x2)%16)"
        assert x4 == (x3 + 1) // 2, f"x4={x4} should be {(x3 + 1) // 2} ((x3+1)//2)"
        assert x5 == (x3 + x4) % 16, f"x5={x5} should be {(x3 + x4) % 16} ((x3+x4)%16)"

        # Verify ranges
        assert 0 <= x0 < 64, f"x0={x0} out of range [0, 64)"
        assert 0 <= x1 < 32, f"x1={x1} out of range [0, 32)"
        assert 0 <= x2 < 33, f"x2={x2} out of range [0, 33)"    
        assert 0 <= x3 < 16, f"x3={x3} out of range [0, 16)"
        assert 0 <= x4 < 16, f"x4={x4} out of range [0, 16)"
        assert 0 <= x5 < 32, f"x5={x5} out of range [0, 32)"

        print(f"  Sample {i:2d}: x0={x0:2d} → x1={x1:2d} (x0//2)")
        print(f"             x0={x0:2d} → x2={x2:2d} ((x0+1)//2)")
        print(f"             x1={x1:2d}, x2={x2:2d} → x3={x3:2d} ((x1+x2)%16)")
        print(f"             x3={x3:2d} → x4={x4:2d} ((x3+1)//2)")
        print(f"             x3={x3:2d}, x4={x4:2d} → x5={x5:2d} ((x3+x4)%16)")

    # Test 3: Information content analysis
    print("\n[Test 3] Information Content Analysis")
    print("-" * 80)
    print("Given each variable, how many possibilities for the others?")

    # Case 1: Given x0 (root)
    x0_test = 20
    x1_from_x0 = x0_test // 2
    x2_from_x0 = (x0_test + 1) // 2
    x3_from_x0 = (x1_from_x0 + x2_from_x0) % 16
    x4_from_x0 = (x3_from_x0 + 1) // 2
    x5_from_x0 = (x3_from_x0 + x4_from_x0) % 16
    print(f"\nGiven x0={x0_test} (Root):")
    print(f"  x1 is uniquely determined: x1 = {x1_from_x0}")
    print(f"  x2 is uniquely determined: x2 = {x2_from_x0}")
    print(f"  x3 is uniquely determined: x3 = {x3_from_x0}")
    print(f"  x4 is uniquely determined: x4 = {x4_from_x0}")
    print(f"  x5 is uniquely determined: x5 = {x5_from_x0}")
    print(f"  Ambiguity: 1 possibility (fully determined)")

    # Test 4: Topological properties
    print("\n[Test 4] Topological Properties")
    print("-" * 80)
    print("Topological depths:")
    print("  x0 (Root):     depth 0 - no dependencies")
    print("  x1 (Branch A): depth 1 - depends on x0")
    print("  x2 (Branch B): depth 1 - depends on x0")
    print("  x3 (Sink):     depth 2 - depends on x1 and x2")
    print("  x4 (Branch C): depth 2 - depends on x3")
    print("  x5 (Sink):     depth 3 - depends on x4 and x5")
    print("\nValid topological sorts:")
    print("  x0 → x1 → x2 → x3 → x4 → x5 ✓")
    print("  x0 → x2 → x1 → x3 → x4 → x5 ✓")
    print("\nInvalid orderings:")
    print("  x1 → x0 → x2 → x3 → x4 → x5 ✗ (x1 before x0)")
    print("  x0 → x1 → x3 → x2 → x4 → x5 ✗ (x3 before x2)")
    print("  x0 → x1 → x3 → x4 → x2 → x5 ✗ (x4 before x2)")
    print("  x0 → x1 → x3 → x4 → x5 → x2 ✗ (x5 before x2)")

    # Test 5: Shuffle distribution
    print("\n[Test 5] Shuffle Distribution")
    print("-" * 80)
    from collections import Counter
    order_counts = Counter()
    for i in range(120):  # Multiple of 24 for even distribution
        sample = dataset[i % 100]
        order_tuple = tuple(sample['order'].tolist())
        order_counts[order_tuple] += 1

    print(f"Order distribution (should be ~uniform over 24 permutations):")
    print(f"Total unique orders seen: {len(order_counts)}")
    print(f"First 5 orders and their counts:")
    for order, count in list(sorted(order_counts.items()))[:5]:
        print(f"  {list(order)}: {count:3d}")

    # Test 6: Branch symmetry
    print("\n[Test 6] Branch Symmetry")
    print("-" * 80)
    print("x1 and x2 are symmetric branches - both have the same causal depth")
    print("However, they use different mappings:")

    test_x0 = 10
    test_x1 = test_x0 // 2
    test_x2 = (test_x0 + 1) // 2
    print(f"\nExample with x0={test_x0}:")
    print(f"  x1 = x0 // 2 = {test_x1}")
    print(f"  x2 = (x0+1) // 2 = {test_x2}")
    print(f"  Note: x1 and x2 can be different!")

    # Test 7: Edge cases
    print("\n[Test 7] Edge Cases")
    print("-" * 80)

    # Small vocab
    dataset_small = DiamondDAGDataset(vocab_size=8, num_samples=50, seed=123)
    sample_small = dataset_small[0]
    x0, x1, x2, x3, x4, x5 = sample_small['unshuffled_tokens'].tolist()
    print(f"Small vocab (8): x0={x0}, x1={x1}, x2={x2}, x3={x3}, x4={x4}, x5={x5}")
    # For vocab_size=8: x3 should be (x1+x2) % (8//4) = (x1+x2) % 2
    assert x1 == x0 // 2 and x2 == (x0 + 1) // 2 and x3 == (x1 + x2) % (dataset_small.vocab_size // 4) and x4 == (x3 + 1) // 2 and x5 == (x3 + x4) % 16
    print("✓ Small vocab works")

    # Large vocab
    dataset_large = DiamondDAGDataset(vocab_size=128, num_samples=50, seed=456)
    sample_large = dataset_large[0]
    x0, x1, x2, x3, x4, x5 = sample_large['unshuffled_tokens'].tolist()
    print(f"Large vocab (128): x0={x0}, x1={x1}, x2={x2}, x3={x3}, x4={x4}, x5={x5}")
    # For vocab_size=128: x3 should be (x1+x2) % (128//4) = (x1+x2) % 32
    assert x1 == x0 // 2 and x2 == (x0 + 1) // 2 and x3 == (x1 + x2) % (dataset_large.vocab_size // 4) and x4 == (x3 + 1) // 2 and x5 == (x3 + x4) % 16
    print("✓ Large vocab works")

    print("\n" + "=" * 80)
    print("✅ All tests passed!")
    print("=" * 80)
    print("\nKey Insights:")
    print("  • x0 (root) has FULL information about entire DAG")
    print("  • x1 and x2 (branches) have PARTIAL information (can determine x3)")
    print("  • x3 (sink) has MINIMAL information (cannot determine parents)")
    print("  • x4 (branch C) has PARTIAL information (can determine x5)")
    print("  • x5 (sink) has MINIMAL information (cannot determine parents)")
    print("  • Agent should learn: Generate x0 first, then x1/x2, finally x3, x4, x5")
    print("  • Branch order (x1 vs x2) is symmetric and can be arbitrary")
    print("=" * 80)


if __name__ == '__main__':
    test_diamond_dag_dataset()
