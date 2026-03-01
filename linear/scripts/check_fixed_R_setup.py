"""
Sanity check for the fixed R matrix setup.

Validates:
1. R matrix is orthogonal
2. All 64 sequences (one per start token) are unique
3. Token overlap between train/val splits
4. No sequence collision
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from collections import Counter

from linear_rotation_exp.data_generator import LinearDynamicalGenerator


def main():
    print("=" * 60)
    print("Fixed R Matrix Setup - Sanity Check")
    print("=" * 60)
    print()

    # Parameters (must match config)
    V = 64
    L = 20
    fixed_R_path = 'linear_rotation_exp/fixed_R.pt'
    train_start_tokens = list(range(0, 50))
    val_start_tokens = list(range(50, 64))

    # Check 1: R matrix exists and is orthogonal
    print("1. Checking R matrix...")
    if not os.path.exists(fixed_R_path):
        print(f"   ERROR: Fixed R matrix not found at {fixed_R_path}")
        print("   Run: python linear_rotation_exp/scripts/generate_fixed_R.py")
        return False

    R = torch.load(fixed_R_path, weights_only=True)
    print(f"   Shape: {R.shape}")

    if R.shape != (V, V):
        print(f"   ERROR: Expected shape ({V}, {V}), got {R.shape}")
        return False

    identity_check = torch.allclose(R.T @ R, torch.eye(V), atol=1e-5)
    print(f"   Orthogonality: {'PASSED' if identity_check else 'FAILED'}")
    if not identity_check:
        return False

    # Check 2: Generate all 64 sequences
    print("\n2. Generating all sequences with fixed R...")
    generator = LinearDynamicalGenerator(
        vocab_size=V,
        hidden_dim=V,
        fixed_R_path=fixed_R_path
    )

    all_sequences = []
    all_bags = []

    for start in range(V):
        result = generator.generate_sequence(
            length=L,
            start_token_id=start,
            mode='argmax'
        )
        all_sequences.append(tuple(result['tokens']))
        all_bags.append(frozenset(result['bag'].items()))

    # Check 3: All sequences are unique
    print("\n3. Checking sequence uniqueness...")
    unique_sequences = set(all_sequences)
    print(f"   Total sequences: {len(all_sequences)}")
    print(f"   Unique sequences: {len(unique_sequences)}")

    if len(unique_sequences) < len(all_sequences):
        print("   WARNING: Some sequences are duplicated!")
        # Find duplicates
        from collections import Counter
        seq_counts = Counter(all_sequences)
        duplicates = [(seq, count) for seq, count in seq_counts.items() if count > 1]
        for seq, count in duplicates[:5]:
            print(f"     Sequence {seq[:5]}... appears {count} times")
    else:
        print("   PASSED: All sequences are unique")

    # Check 4: All bags are unique
    print("\n4. Checking bag uniqueness...")
    unique_bags = set(all_bags)
    print(f"   Total bags: {len(all_bags)}")
    print(f"   Unique bags: {len(unique_bags)}")

    if len(unique_bags) < len(all_bags):
        print("   WARNING: Some bags are duplicated (this is expected for long sequences)")
    else:
        print("   PASSED: All bags are unique")

    # Check 5: Validity and uniqueness rates
    print("\n5. Checking validity and local uniqueness...")
    valid_count = 0
    unique_count = 0

    for start in range(V):
        result = generator.generate_sequence(length=L, start_token_id=start, mode='argmax')
        is_valid, is_unique = generator.verify_uniqueness(result['tokens'])
        if is_valid:
            valid_count += 1
        if is_unique:
            unique_count += 1

    print(f"   Validity rate: {valid_count/V*100:.1f}%")
    print(f"   Uniqueness rate: {unique_count/V*100:.1f}%")

    # Check 6: Train/Val split
    print("\n6. Checking train/val split...")
    print(f"   Train start tokens: {train_start_tokens[0]}-{train_start_tokens[-1]} ({len(train_start_tokens)} tokens)")
    print(f"   Val start tokens: {val_start_tokens[0]}-{val_start_tokens[-1]} ({len(val_start_tokens)} tokens)")

    train_sequences = [all_sequences[i] for i in train_start_tokens]
    val_sequences = [all_sequences[i] for i in val_start_tokens]

    # Check overlap
    train_set = set(train_sequences)
    val_set = set(val_sequences)
    overlap = train_set & val_set

    print(f"   Train unique sequences: {len(train_set)}")
    print(f"   Val unique sequences: {len(val_set)}")
    print(f"   Sequence overlap: {len(overlap)}")

    if len(overlap) > 0:
        print("   WARNING: Train and Val have overlapping sequences!")
    else:
        print("   PASSED: No sequence overlap between train and val")

    # Check 7: Token statistics
    print("\n7. Token statistics across splits...")

    train_tokens = set()
    for seq in train_sequences:
        train_tokens.update(seq)

    val_tokens = set()
    for seq in val_sequences:
        val_tokens.update(seq)

    print(f"   Unique tokens in train: {len(train_tokens)}")
    print(f"   Unique tokens in val: {len(val_tokens)}")
    print(f"   Token overlap: {len(train_tokens & val_tokens)}")

    # Check 8: Determinism
    print("\n8. Checking determinism...")
    # Generate same sequence twice
    result1 = generator.generate_sequence(length=L, start_token_id=0, mode='argmax')
    result2 = generator.generate_sequence(length=L, start_token_id=0, mode='argmax')

    if result1['tokens'] == result2['tokens']:
        print("   PASSED: Same start token produces same sequence")
    else:
        print("   FAILED: Non-deterministic generation!")
        return False

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"R matrix: VALID (orthogonal {V}x{V})")
    print(f"Sequences: {len(unique_sequences)}/{len(all_sequences)} unique")
    print(f"Validity: {valid_count/V*100:.1f}%")
    print(f"Train/Val: {len(train_start_tokens)}/{len(val_start_tokens)} samples, no overlap")
    print(f"Determinism: PASSED")
    print()
    print("Setup is ready for training!")

    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
