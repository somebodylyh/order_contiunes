"""
PyTorch Dataset wrapper for Continuous Vector Linear Rotation Experiment

Provides:
- ContinuousRotationDataset: Dataset class for continuous vector sequences
- create_continuous_dataloaders: Factory function for train/val/test dataloaders
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple
import numpy as np

from .continuous_data_generator import ContinuousDenseARGenerator


def block_wise_shuffle(vectors: torch.Tensor, num_chunks: int = 4):
    """
    Block-wise shuffle: split sequence into chunks, preserve internal order,
    shuffle chunk order.

    Args:
        vectors: [L, D] tensor
        num_chunks: number of chunks to split into

    Returns:
        shuffled_vectors: [L, D]
        shuffle_indices: [L] permutation indices such that shuffled_vectors = vectors[shuffle_indices]
    """
    L = vectors.shape[0]
    chunk_size = L // num_chunks

    # Build index ranges for each chunk: [0..chunk_size-1], [chunk_size..2*chunk_size-1], ...
    chunk_indices = [torch.arange(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]

    # Handle remainder if L is not evenly divisible
    if L % num_chunks != 0:
        chunk_indices[-1] = torch.arange((num_chunks - 1) * chunk_size, L)

    # Shuffle chunk order
    chunk_order = torch.randperm(num_chunks)

    # Concatenate indices in shuffled chunk order
    shuffle_indices = torch.cat([chunk_indices[i] for i in chunk_order])
    shuffled_vectors = vectors[shuffle_indices]

    return shuffled_vectors, shuffle_indices


class ContinuousRotationDataset(Dataset):
    """
    PyTorch Dataset for continuous vector sequences from Dense AR process.

    Supports:
    - 'train' mode: Online generation (new samples each epoch)
    - 'eval' mode: Static pre-generated samples (reproducible)
    """

    def __init__(
        self,
        generator: ContinuousDenseARGenerator,
        seq_length: int,
        init_mode: str = 'positive_first',
        mode: str = 'train',
        num_samples: int = 10000,
        virtual_size: Optional[int] = None,
        seed: Optional[int] = None,
        num_chunks: int = 4
    ):
        """
        Initialize the dataset.

        Args:
            generator: ContinuousDenseARGenerator instance
            seq_length: Sequence length (L)
            init_mode: 'positive_first', 'negative_first', or 'random'
            mode: 'train' (online) or 'eval' (static pre-generated)
            num_samples: Number of samples (actual for eval, virtual for train)
            virtual_size: Virtual epoch size for training (default: num_samples)
            seed: Random seed for reproducibility
            num_chunks: Number of chunks for block-wise shuffle
        """
        self.generator = generator
        self.seq_length = seq_length
        self.init_mode = init_mode
        self.mode = mode
        self.num_samples = num_samples
        self.virtual_size = virtual_size if virtual_size is not None else num_samples
        self.seed = seed
        self.num_chunks = num_chunks

        # Pre-generate samples for eval mode
        if mode == 'eval':
            self._pregenerate_samples()
        else:
            self.samples = None

    def _pregenerate_samples(self) -> None:
        """Pre-generate all samples for eval mode."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        print(f"[INFO] Pre-generating {self.num_samples} {self.init_mode} samples...")
        result = self.generator.generate_sequence(
            length=self.seq_length,
            init_mode=self.init_mode,
            batch_size=self.num_samples
        )
        self.samples = {
            'vectors': result['vectors'],  # [num_samples, L, D]
            'init_vectors': result['init_vectors']  # [num_samples, k, D]
        }
        print(f"[INFO] Generated {self.num_samples} {self.init_mode} sequences")

    def __len__(self) -> int:
        if self.mode == 'eval':
            return self.num_samples
        else:
            return self.virtual_size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dictionary containing:
                'vectors': [L, D] - original time-ordered sequence
                'shuffled_vectors': [L, D] - randomly shuffled sequence
                'shuffle_indices': [L] - indices used for shuffling
                'order': [L] - ground truth order (inverse of shuffle_indices)
                'init_vectors': [k, D] - initial k vectors
        """
        if self.mode == 'eval':
            vectors = self.samples['vectors'][idx]
            init_vectors = self.samples['init_vectors'][idx]
        else:
            # Online generation for train mode
            result = self.generator.generate_single_sequence(
                length=self.seq_length,
                init_mode=self.init_mode
            )
            vectors = result['vectors']
            init_vectors = result['init_vectors']

        # Create shuffled version
        shuffled_vectors, shuffle_indices = block_wise_shuffle(vectors, num_chunks=self.num_chunks)

        # Ground truth order: mapping from shuffled position to original position
        # If shuffle_indices[i] = j, then shuffled_vectors[i] = vectors[j]
        # To recover original order, we need order[i] = position of vectors[i] in shuffled_vectors
        order = torch.argsort(shuffle_indices)

        return {
            'vectors': vectors,  # [L, D] - ground truth time-ordered
            'shuffled_vectors': shuffled_vectors,  # [L, D] - input (shuffled set)
            'shuffle_indices': shuffle_indices,  # [L] - shuffling permutation
            'order': order,  # [L] - ground truth generation order (L2R = 0,1,2,...)
            'init_vectors': init_vectors  # [k, D]
        }


def create_continuous_dataloaders(
    vector_dim: int = 32,
    seq_length: int = 16,
    dependency_window: int = 5,
    num_matrices: Optional[int] = None,
    train_samples: int = 10000,
    val_samples: int = 1000,
    test_samples: int = 1000,
    batch_size: int = 64,
    num_workers: int = 4,
    seed: int = 42,
    fixed_matrices_path: Optional[str] = None,
    train_init_mode: str = 'positive_first',
    val_init_mode: str = 'negative_first',
    num_chunks: int = 4,
    noise_scale: float = 0.0
) -> Tuple[DataLoader, DataLoader, DataLoader, ContinuousDenseARGenerator]:
    """
    Create train/val/test dataloaders with shared generator.

    Args:
        vector_dim: Dimension of vectors (D)
        seq_length: Sequence length (L)
        dependency_window: AR dependency window (k)
        train_samples: Number of training samples
        val_samples: Number of validation samples
        test_samples: Number of test samples
        batch_size: Batch size for all loaders
        num_workers: Number of dataloader workers
        seed: Random seed for reproducibility
        fixed_matrices_path: Path to save/load orthogonal matrices
        train_init_mode: Initialization mode for training ('positive_first')
        val_init_mode: Initialization mode for val/test ('negative_first' for OOD)

    Returns:
        Tuple of (train_loader, val_loader, test_loader, generator)
    """
    # Create shared generator with fixed matrices
    generator = ContinuousDenseARGenerator(
        vector_dim=vector_dim,
        dependency_window=dependency_window,
        num_matrices=num_matrices,
        seed=seed,
        fixed_matrices_path=fixed_matrices_path,
        noise_scale=noise_scale
    )

    # Create datasets
    train_dataset = ContinuousRotationDataset(
        generator=generator,
        seq_length=seq_length,
        init_mode=train_init_mode,
        mode='train',
        num_samples=train_samples,
        seed=seed,
        num_chunks=num_chunks
    )

    val_dataset = ContinuousRotationDataset(
        generator=generator,
        seq_length=seq_length,
        init_mode=val_init_mode,
        mode='eval',
        num_samples=val_samples,
        seed=seed + 1000,  # Different seed for val
        num_chunks=num_chunks
    )

    test_dataset = ContinuousRotationDataset(
        generator=generator,
        seq_length=seq_length,
        init_mode=val_init_mode,  # Same as val (OOD)
        mode='eval',
        num_samples=test_samples,
        seed=seed + 2000,  # Different seed for test
        num_chunks=num_chunks
    )

    print(f"[INFO] Generated {train_samples} train sequences (init: {train_init_mode})")
    print(f"[INFO] Generated {val_samples} val sequences (init: {val_init_mode}) [OOD]")
    print(f"[INFO] Generated {test_samples} test sequences (init: {val_init_mode}) [OOD]")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader, generator


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching.

    Args:
        batch: List of sample dictionaries

    Returns:
        Batched dictionary with tensors stacked along dim 0
    """
    return {
        'vectors': torch.stack([s['vectors'] for s in batch]),
        'shuffled_vectors': torch.stack([s['shuffled_vectors'] for s in batch]),
        'shuffle_indices': torch.stack([s['shuffle_indices'] for s in batch]),
        'order': torch.stack([s['order'] for s in batch]),
        'init_vectors': torch.stack([s['init_vectors'] for s in batch])
    }


def test_dataset():
    """Test the dataset and dataloader."""
    print("=" * 60)
    print("Testing ContinuousRotationDataset")
    print("=" * 60)

    # Create dataloaders
    train_loader, val_loader, test_loader, generator = create_continuous_dataloaders(
        vector_dim=32,
        seq_length=16,
        dependency_window=5,
        train_samples=1000,
        val_samples=100,
        test_samples=100,
        batch_size=32,
        num_workers=0,  # For testing
        seed=42,
        fixed_matrices_path='linear_rotation_exp/test_matrices.pt',
        train_init_mode='positive_first',
        val_init_mode='negative_first'
    )

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    # Test a batch
    print("\n1. Testing train batch...")
    batch = next(iter(train_loader))
    print(f"   vectors shape: {batch['vectors'].shape}")
    print(f"   shuffled_vectors shape: {batch['shuffled_vectors'].shape}")
    print(f"   shuffle_indices shape: {batch['shuffle_indices'].shape}")
    print(f"   order shape: {batch['order'].shape}")
    print(f"   init_vectors shape: {batch['init_vectors'].shape}")

    # Verify shuffling is consistent
    print("\n2. Verifying shuffling consistency...")
    vectors = batch['vectors'][0]
    shuffled = batch['shuffled_vectors'][0]
    shuffle_idx = batch['shuffle_indices'][0]
    order = batch['order'][0]

    # Check that shuffled_vectors[i] = vectors[shuffle_idx[i]]
    reconstructed = vectors[shuffle_idx]
    assert torch.allclose(shuffled, reconstructed), "Shuffling inconsistent"
    print("   ✓ shuffled_vectors = vectors[shuffle_indices]")

    # Check that order is the inverse permutation
    # vectors[order[i]] should give the i-th element in original order
    assert torch.allclose(order, torch.argsort(shuffle_idx)), "Order is not argsort(shuffle_indices)"
    print("   ✓ order = argsort(shuffle_indices)")

    # Test OOD split
    print("\n3. Testing OOD split...")
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    train_first_components = train_batch['init_vectors'][:, :, 0]  # [B, k]
    val_first_components = val_batch['init_vectors'][:, :, 0]  # [B, k]

    print(f"   Train init first components (sample): {train_first_components[0, :3].tolist()}")
    print(f"   Val init first components (sample): {val_first_components[0, :3].tolist()}")

    # Verify train has positive first components, val has negative
    assert (train_first_components > 0).all(), "Train should have positive first components"
    assert (val_first_components < 0).all(), "Val should have negative first components"
    print("   ✓ OOD split verified (train: positive, val: negative)")

    # Clean up
    import os
    if os.path.exists('linear_rotation_exp/test_matrices.pt'):
        os.remove('linear_rotation_exp/test_matrices.pt')

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_dataset()
