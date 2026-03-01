"""
Lossy Copy Dataset for LO-ARMs Experiment

Generates (x, y) pairs where y = x // k, with random shuffling to test
whether the Agent can learn the optimal generation order.
"""

import torch
from torch.utils.data import Dataset
import numpy as np


class LossyCopyDataset(Dataset):
    """
    Synthetic dataset for the Lossy Copy task.

    Each sample consists of:
    - x: randomly sampled from [0, vocab_size)
    - y: computed as x // k (integer division)

    The physical order of (x, y) is randomly shuffled with 50% probability.

    Args:
        vocab_size: Range for x values [0, vocab_size)
        k: Divisor for lossy copy (y = x // k)
        num_samples: Dataset size
        seed: Random seed for reproducibility

    Returns:
        Dictionary containing:
            'tokens': [2] shuffled sequence
            'logical_ids': [2] original positions [0, 1] where 0=x, 1=y
            'unshuffled_tokens': [2] original [x, y] order
            'order': [2] shuffle permutation applied
    """

    def __init__(self, vocab_size=64, k=2, num_samples=10000, seed=42):
        super().__init__()
        self.vocab_size = vocab_size
        self.k = k
        self.num_samples = num_samples
        self.seed = seed

        # Set random seed for reproducibility
        self.rng = np.random.RandomState(seed)

        # Validate parameters
        assert vocab_size > 0, "vocab_size must be positive"
        assert k > 0, "k must be positive"
        assert k < vocab_size, "k must be less than vocab_size"

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Generate a single (x, y) pair with random shuffling.

        Returns:
            dict with keys:
                'tokens': shuffled [x, y] or [y, x]
                'logical_ids': [0, 1] indicating logical positions
                'unshuffled_tokens': original [x, y]
                'order': permutation applied (e.g., [0, 1] or [1, 0])
        """
        # Use idx as part of seed for deterministic but varied sampling
        local_rng = np.random.RandomState(self.seed + idx)

        # Sample x uniformly from [0, vocab_size)
        x = local_rng.randint(0, self.vocab_size)

        # Compute y = x // k
        y = x // self.k

        # Create unshuffled sequence [x, y]
        unshuffled_tokens = torch.tensor([x, y], dtype=torch.long)

        # Logical IDs: 0 for x, 1 for y (unchanging)
        logical_ids = torch.tensor([0, 1], dtype=torch.long)

        # Shuffle with 50% probability
        if local_rng.rand() < 0.5:
            # Swap order: [y, x]
            order = torch.tensor([1, 0], dtype=torch.long)
            tokens = torch.tensor([y, x], dtype=torch.long)
        else:
            # Keep order: [x, y]
            order = torch.tensor([0, 1], dtype=torch.long)
            tokens = torch.tensor([x, y], dtype=torch.long)

        return {
            'tokens': tokens,
            'logical_ids': logical_ids,
            'unshuffled_tokens': unshuffled_tokens,
            'order': order
        }


def test_dataset():
    """Test the LossyCopyDataset implementation."""
    print("Testing LossyCopyDataset...")

    # Test with small vocab
    dataset = LossyCopyDataset(vocab_size=8, k=2, num_samples=100, seed=42)

    # Check dataset length
    assert len(dataset) == 100, "Dataset length incorrect"
    print(f"✓ Dataset length: {len(dataset)}")

    # Check first few samples
    for i in range(5):
        sample = dataset[i]
        tokens = sample['tokens']
        logical_ids = sample['logical_ids']
        unshuffled_tokens = sample['unshuffled_tokens']
        order = sample['order']

        x = unshuffled_tokens[0].item()
        y = unshuffled_tokens[1].item()

        # Verify x is in range
        assert 0 <= x < 8, f"x={x} out of range [0, 8)"

        # Verify y = x // 2
        assert y == x // 2, f"y={y} should be {x // 2}"

        # Verify logical_ids are always [0, 1]
        assert torch.equal(logical_ids, torch.tensor([0, 1])), "logical_ids should be [0, 1]"

        # Verify tokens match shuffled unshuffled_tokens
        if torch.equal(order, torch.tensor([0, 1])):
            assert torch.equal(tokens, unshuffled_tokens), "Tokens should match [x, y]"
        else:
            assert torch.equal(tokens, torch.tensor([y, x])), "Tokens should be [y, x]"

        print(f"  Sample {i}: x={x}, y={y}, order={order.tolist()}, tokens={tokens.tolist()}")

    # Check shuffle distribution (should be ~50% each)
    num_shuffled = 0
    for i in range(100):
        sample = dataset[i]
        if not torch.equal(sample['order'], torch.tensor([0, 1])):
            num_shuffled += 1

    shuffle_ratio = num_shuffled / 100
    print(f"✓ Shuffle ratio: {shuffle_ratio:.2f} (expected ~0.50)")
    assert 0.3 < shuffle_ratio < 0.7, "Shuffle ratio should be around 0.5"

    # Test with larger vocab
    dataset_large = LossyCopyDataset(vocab_size=64, k=2, num_samples=1000, seed=123)
    sample = dataset_large[0]
    x = sample['unshuffled_tokens'][0].item()
    y = sample['unshuffled_tokens'][1].item()
    assert 0 <= x < 64, "x out of range for large vocab"
    assert y == x // 2, "y = x // 2 check failed for large vocab"
    print(f"✓ Large vocab test: x={x}, y={y}")

    print("\n✅ All dataset tests passed!")


if __name__ == '__main__':
    test_dataset()
