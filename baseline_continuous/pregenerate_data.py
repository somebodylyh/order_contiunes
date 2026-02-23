"""
Pre-generate large dataset to disk using numpy memmap.

Usage:
    python baseline_continuous/pregenerate_data.py [--train_samples 500000] [--val_samples 10000] [--test_samples 10000]

Data is saved as numpy memmap files for lazy loading during training.
Memory usage is near-zero regardless of dataset size.
"""

import sys
import os
import argparse
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baseline_continuous import config as cfg
from linear.continuous_data_generator import ContinuousDenseARGenerator


def pregenerate_split(generator, num_samples, seq_length, init_mode, seed, out_dir, split_name, chunk_size=5000):
    """Pre-generate one split and save as memmap."""
    os.makedirs(out_dir, exist_ok=True)

    D = generator.D
    num_init = generator.num_init

    vectors_path = os.path.join(out_dir, f'{split_name}_vectors.npy')
    init_vectors_path = os.path.join(out_dir, f'{split_name}_init_vectors.npy')

    # Create memmap files
    vectors_mmap = np.memmap(vectors_path, dtype='float32', mode='w+',
                             shape=(num_samples, seq_length, D))
    init_mmap = np.memmap(init_vectors_path, dtype='float32', mode='w+',
                          shape=(num_samples, num_init, D))

    torch.manual_seed(seed)
    np.random.seed(seed)

    generated = 0
    while generated < num_samples:
        bs = min(chunk_size, num_samples - generated)
        result = generator.generate_sequence(
            length=seq_length,
            init_mode=init_mode,
            batch_size=bs
        )
        vectors_mmap[generated:generated + bs] = result['vectors'].numpy()
        init_mmap[generated:generated + bs] = result['init_vectors'].numpy()
        generated += bs
        print(f"  [{split_name}] {generated}/{num_samples}")

    # Flush to disk
    vectors_mmap.flush()
    init_mmap.flush()
    del vectors_mmap, init_mmap

    print(f"  [{split_name}] Saved to {out_dir}")


def get_current_data_config(args):
    """Build a dict of data-generation parameters for comparison."""
    return {
        'vector_dim': cfg.vector_dim,
        'seq_length': cfg.seq_length,
        'dependency_window': cfg.dependency_window,
        'num_matrices': cfg.num_matrices,
        'noise_scale': cfg.noise_scale,
        'alpha': cfg.alpha,
        'seed': cfg.seed,
        'train_init_mode': cfg.train_init_mode,
        'val_init_mode': cfg.val_init_mode,
        'train_samples': args.train_samples,
        'val_samples': args.val_samples,
        'test_samples': args.test_samples,
    }


def data_config_matches(config_path, current_config):
    """Check if existing data_config.pt matches current config."""
    if not os.path.exists(config_path):
        return False
    try:
        saved = torch.load(config_path, map_location='cpu')
        for key, val in current_config.items():
            if key not in saved or saved[key] != val:
                print(f"  Config mismatch: {key} (saved={saved.get(key)}, current={val})")
                return False
        return True
    except Exception as e:
        print(f"  Failed to load existing config: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_samples', type=int, default=cfg.train_samples)
    parser.add_argument('--val_samples', type=int, default=cfg.val_samples)
    parser.add_argument('--test_samples', type=int, default=cfg.test_samples)
    parser.add_argument('--out_dir', type=str, default=os.path.join(os.path.dirname(__file__), 'data'))
    parser.add_argument('--force', action='store_true', help='Force regeneration even if config matches')
    args = parser.parse_args()

    print("=" * 60)
    print("Pre-generating dataset to disk (memmap)")
    print("=" * 60)
    print(f"  vector_dim={cfg.vector_dim}, seq_length={cfg.seq_length}")
    print(f"  dependency_window={cfg.dependency_window}, num_matrices={cfg.num_matrices}")
    print(f"  noise_scale={cfg.noise_scale}")
    print(f"  train={args.train_samples}, val={args.val_samples}, test={args.test_samples}")

    # Check if existing data matches current config
    config_path = os.path.join(args.out_dir, 'data_config.pt')
    current_config = get_current_data_config(args)

    if not args.force and data_config_matches(config_path, current_config):
        print("\n  Data config unchanged — reusing existing data. Use --force to regenerate.")
        print("=" * 60)
        return

    # Estimate disk usage
    per_sample_bytes = cfg.seq_length * cfg.vector_dim * 4
    total_bytes = (args.train_samples + args.val_samples + args.test_samples) * per_sample_bytes
    print(f"  Estimated disk usage: {total_bytes / 1e9:.1f} GB")
    print()

    # Create generator
    generator = ContinuousDenseARGenerator(
        vector_dim=cfg.vector_dim,
        dependency_window=cfg.dependency_window,
        num_matrices=cfg.num_matrices,
        seed=cfg.seed,
        noise_scale=cfg.noise_scale,
        alpha=cfg.alpha
    )

    # Save generator config for verification at load time
    os.makedirs(args.out_dir, exist_ok=True)
    save_dict = dict(current_config)
    save_dict['num_init'] = generator.num_init
    save_dict['A_matrices'] = generator.A_matrices
    torch.save(save_dict, config_path)

    # Generate each split (overwrites existing files)
    print("Generating train split...")
    pregenerate_split(generator, args.train_samples, cfg.seq_length,
                      cfg.train_init_mode, cfg.seed, args.out_dir, 'train')

    print("Generating val split...")
    pregenerate_split(generator, args.val_samples, cfg.seq_length,
                      cfg.val_init_mode, cfg.seed + 1000, args.out_dir, 'val')

    print("Generating test split...")
    pregenerate_split(generator, args.test_samples, cfg.seq_length,
                      cfg.val_init_mode, cfg.seed + 2000, args.out_dir, 'test')

    print()
    print("=" * 60)
    print("Done! Data saved to:", args.out_dir)
    print("=" * 60)


if __name__ == '__main__':
    main()
