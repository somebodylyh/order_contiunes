"""
Disk-backed dataset using numpy memmap for lazy loading.

Memory usage is near-zero regardless of dataset size — only accessed
pages are loaded into RAM by the OS.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional

from linear.continuous_data_generator import ContinuousDenseARGenerator


def block_wise_shuffle(vectors: torch.Tensor, num_chunks: int = 4):
    """Block-wise shuffle: shuffle chunk order, preserve internal order."""
    L = vectors.shape[0]
    chunk_size = L // num_chunks
    chunk_indices = [torch.arange(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    if L % num_chunks != 0:
        chunk_indices[-1] = torch.arange((num_chunks - 1) * chunk_size, L)
    chunk_order = torch.randperm(num_chunks)
    shuffle_indices = torch.cat([chunk_indices[i] for i in chunk_order])
    shuffled_vectors = vectors[shuffle_indices]
    return shuffled_vectors, shuffle_indices


class MemmapDataset(Dataset):
    """Dataset backed by numpy memmap files on disk."""

    def __init__(self, data_dir: str, split: str, num_chunks: int = 4):
        vectors_path = os.path.join(data_dir, f'{split}_vectors.npy')
        init_vectors_path = os.path.join(data_dir, f'{split}_init_vectors.npy')

        if not os.path.exists(vectors_path):
            raise FileNotFoundError(f"Data not found: {vectors_path}\n"
                                    f"Run: python baseline_continuous/pregenerate_data.py")

        # Open as read-only memmap — no memory allocation
        self.vectors = np.memmap(vectors_path, dtype='float32', mode='r')
        self.init_vectors = np.memmap(init_vectors_path, dtype='float32', mode='r')

        # Load config to get shapes
        config = torch.load(os.path.join(data_dir, 'data_config.pt'), weights_only=False)
        n = config[f'{split}_samples']
        seq_length = config['seq_length']
        vector_dim = config['vector_dim']
        num_init = config['num_init']

        self.vectors = self.vectors.reshape(n, seq_length, vector_dim)
        self.init_vectors = self.init_vectors.reshape(n, num_init, vector_dim)
        self.num_samples = n
        self.num_chunks = num_chunks

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Read from disk (memmap handles caching)
        vectors = torch.from_numpy(self.vectors[idx].copy())
        init_vectors = torch.from_numpy(self.init_vectors[idx].copy())

        # Create shuffled version
        shuffled_vectors, shuffle_indices = block_wise_shuffle(vectors, num_chunks=self.num_chunks)
        order = torch.argsort(shuffle_indices)

        return {
            'vectors': vectors,
            'shuffled_vectors': shuffled_vectors,
            'shuffle_indices': shuffle_indices,
            'order': order,
            'init_vectors': init_vectors,
        }


def create_disk_dataloaders(
    data_dir: str,
    batch_size: int = 128,
    num_workers: int = 4,
    num_chunks: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders from pre-generated disk data."""

    train_ds = MemmapDataset(data_dir, 'train', num_chunks=num_chunks)
    val_ds = MemmapDataset(data_dir, 'val', num_chunks=num_chunks)
    test_ds = MemmapDataset(data_dir, 'test', num_chunks=num_chunks)

    print(f"[disk] train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
