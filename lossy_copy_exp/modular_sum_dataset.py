"""
Modular Sum Dataset with Lossy Switch

Three-variable task [x1, x2, y] with configurable causal structure:
- Lossy mode (use_lossy=True): y = (x1 + x2) // 2  -> Strong causality x1,x2 -> y
- Modular mode (use_lossy=False): y = (x1 + x2) % P -> Symmetric, any two determine third
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class ModularSumDataset(Dataset):
    """
    Three-variable synthetic dataset with configurable causal structure.

    Args:
        vocab_size: Range for variables [0, vocab_size), also serves as modulus P
        num_samples: Dataset size
        use_lossy:
            - True:  y = (x1 + x2) // 2  (lossy, strong x->y causality)
            - False: y = (x1 + x2) % P   (lossless, complete symmetry)
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            'tokens': [3] shuffled sequence [could be any permutation of x1, x2, y]
            'logical_ids': [3] logical positions [0, 1, 2] where 0=x1, 1=x2, 2=y
            'unshuffled_tokens': [3] original [x1, x2, y] order
            'order': [3] shuffle permutation applied
    """

    def __init__(self, vocab_size=64, num_samples=10000, use_lossy=True, seed=42):
        super().__init__()
        self.vocab_size = vocab_size
        self.p = vocab_size  # Modulus for modular mode
        self.num_samples = num_samples
        self.use_lossy = use_lossy
        self.seed = seed

        # Validate parameters
        assert vocab_size > 0, "vocab_size must be positive"

        print(f"ModularSumDataset initialized:")
        print(f"  Mode: {'Lossy (y = (x1+x2)//2)' if use_lossy else f'Modular (y = (x1+x2) % {self.p})'}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Samples: {num_samples}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a single (x1, x2, y) triple.

        Returns:
            dict with keys:
                'tokens': shuffled [x1, x2, y] (physical order)
                'logical_ids': [0, 1, 2] (logical positions, always same)
                'unshuffled_tokens': original [x1, x2, y]
                'order': permutation applied
        """
        # Use idx as part of seed for deterministic but varied sampling
        local_rng = np.random.RandomState(self.seed + idx)

        # Sample x1 and x2 uniformly from [0, vocab_size)
        x1 = local_rng.randint(0, self.vocab_size)
        x2 = local_rng.randint(0, self.vocab_size)

        # Compute y based on mode
        if self.use_lossy:
            # Lossy: y = (x1 + x2) // 2
            # Strong causality: x1, x2 -> y (deterministic)
            # But y -> x1, x2 is ambiguous (multiple solutions)
            y = (x1 + x2) // 2
        else:
            # Modular: y = (x1 + x2) % P
            # Complete symmetry: any two variables determine the third
            # x1 + x2 = y (mod P) -> all three are on equal footing
            y = (x1 + x2) % self.p

        # Create unshuffled sequence [x1, x2, y]
        unshuffled_tokens = torch.tensor([x1, x2, y], dtype=torch.long)

        # Logical IDs: 0=x1, 1=x2, 2=y (unchanging, identifies token role)
        logical_ids = torch.tensor([0, 1, 2], dtype=torch.long)

        # Randomly shuffle physical order
        # Create combined list to maintain token-logical_id binding
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


def test_modular_sum_dataset():
    """Test the ModularSumDataset implementation."""
    print("=" * 80)
    print("Testing ModularSumDataset")
    print("=" * 80)

    # Test 1: Lossy mode
    print("\n[Test 1] Lossy Mode (y = (x1 + x2) // 2)")
    print("-" * 80)
    dataset_lossy = ModularSumDataset(vocab_size=16, num_samples=100, use_lossy=True, seed=42)

    # Check dataset length
    assert len(dataset_lossy) == 100, "Dataset length incorrect"
    print(f"✓ Dataset length: {len(dataset_lossy)}")

    # Check first few samples
    print("\nFirst 5 samples (lossy mode):")
    for i in range(5):
        sample = dataset_lossy[i]
        tokens = sample['tokens']
        logical_ids = sample['logical_ids']
        unshuffled = sample['unshuffled_tokens']
        order = sample['order']

        x1, x2, y = unshuffled[0].item(), unshuffled[1].item(), unshuffled[2].item()

        # Verify x1, x2 in range
        assert 0 <= x1 < 16, f"x1={x1} out of range"
        assert 0 <= x2 < 16, f"x2={x2} out of range"

        # Verify y = (x1 + x2) // 2
        expected_y = (x1 + x2) // 2
        assert y == expected_y, f"y={y} should be {expected_y}"

        # Verify logical_ids are always [0, 1, 2]
        assert torch.equal(logical_ids, torch.tensor([0, 1, 2])), "logical_ids should be [0, 1, 2]"

        # Verify tokens match shuffled unshuffled
        expected_tokens = unshuffled[order]
        assert torch.equal(tokens, expected_tokens), "Tokens don't match shuffled unshuffled"

        print(f"  Sample {i}: x1={x1:2d}, x2={x2:2d}, y={y:2d} | order={order.tolist()} | tokens={tokens.tolist()}")

    # Test 2: Modular mode
    print("\n[Test 2] Modular Mode (y = (x1 + x2) % P)")
    print("-" * 80)
    dataset_modular = ModularSumDataset(vocab_size=16, num_samples=100, use_lossy=False, seed=42)

    print("\nFirst 5 samples (modular mode):")
    for i in range(5):
        sample = dataset_modular[i]
        unshuffled = sample['unshuffled_tokens']
        x1, x2, y = unshuffled[0].item(), unshuffled[1].item(), unshuffled[2].item()

        # Verify y = (x1 + x2) % 16
        expected_y = (x1 + x2) % 16
        assert y == expected_y, f"y={y} should be {expected_y}"

        print(f"  Sample {i}: x1={x1:2d}, x2={x2:2d}, y={y:2d} | (x1+x2)={x1+x2:2d} | y mod 16={y}")

    # Test 3: Shuffle distribution (should be uniform across 6 permutations)
    print("\n[Test 3] Shuffle Distribution")
    print("-" * 80)
    from collections import Counter
    order_counts = Counter()
    for i in range(600):
        sample = dataset_lossy[i % 100]  # Cycle through dataset
        order_tuple = tuple(sample['order'].tolist())
        order_counts[order_tuple] += 1

    print(f"Order distribution (should be ~uniform):")
    for order, count in sorted(order_counts.items()):
        print(f"  {list(order)}: {count:3d} ({count/600*100:.1f}%)")

    # Test 4: Large vocab
    print("\n[Test 4] Large Vocab Test")
    print("-" * 80)
    dataset_large = ModularSumDataset(vocab_size=64, num_samples=1000, use_lossy=True, seed=123)
    sample = dataset_large[0]
    x1, x2, y = sample['unshuffled_tokens'].tolist()
    print(f"✓ Large vocab: x1={x1}, x2={x2}, y={y}, y_expected={(x1+x2)//2}")
    assert y == (x1 + x2) // 2, "Lossy computation failed for large vocab"

    # Test 5: Causality difference
    print("\n[Test 5] Causality Analysis")
    print("-" * 80)
    print("Lossy mode: Given y, how many (x1, x2) pairs are possible?")
    y_test = 10
    valid_pairs_lossy = []
    for x1 in range(16):
        for x2 in range(16):
            if (x1 + x2) // 2 == y_test:
                valid_pairs_lossy.append((x1, x2))
    print(f"  y={y_test} -> {len(valid_pairs_lossy)} valid (x1,x2) pairs (e.g., {valid_pairs_lossy[:3]}...)")

    print("\nModular mode: Given y, how many (x1, x2) pairs are possible?")
    valid_pairs_modular = []
    for x1 in range(16):
        for x2 in range(16):
            if (x1 + x2) % 16 == y_test:
                valid_pairs_modular.append((x1, x2))
    print(f"  y={y_test} -> {len(valid_pairs_modular)} valid (x1,x2) pairs (e.g., {valid_pairs_modular[:3]}...)")

    print("\n✅ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_modular_sum_dataset()
