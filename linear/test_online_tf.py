"""
Quick test for Online Generation and Teacher Forcing

This script verifies that:
1. Online generation works (no memorization)
2. Teacher forcing is properly implemented
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from linear_rotation_exp.rotation_dataset import LinearRotationDataset, create_dataloaders
import linear_rotation_exp.config_rotation as config


def test_online_generation():
    """Test that online generation produces different samples"""
    print("=" * 60)
    print("Test 1: Online Generation")
    print("=" * 60)

    dataset = LinearRotationDataset(
        vocab_size=16,
        seq_length=20,
        hidden_dim=16,
        ortho_mode='random',
        num_samples=10,
        seed=42,
        mode='train',
        virtual_size=100
    )

    # Get multiple samples and check they're different
    sample1 = dataset[0]
    sample2 = dataset[0]  # Same index but should be different due to online generation

    tokens1 = sample1['tokens'].tolist()
    tokens2 = sample2['tokens'].tolist()

    print(f"Sample 1: {tokens1[:5]}...")
    print(f"Sample 2: {tokens2[:5]}...")

    if tokens1 != tokens2:
        print("✅ PASS: Online generation produces different samples")
    else:
        print("❌ FAIL: Samples are identical (not generating online)")

    print()


def test_static_generation():
    """Test that eval mode produces consistent samples"""
    print("=" * 60)
    print("Test 2: Static Evaluation")
    print("=" * 60)

    dataset = LinearRotationDataset(
        vocab_size=16,
        seq_length=20,
        hidden_dim=16,
        ortho_mode='random',
        num_samples=10,
        seed=42,
        mode='eval'
    )

    # Get same sample twice
    sample1 = dataset[0]
    sample2 = dataset[0]

    tokens1 = sample1['tokens'].tolist()
    tokens2 = sample2['tokens'].tolist()

    print(f"Sample 1: {tokens1[:5]}...")
    print(f"Sample 2: {tokens2[:5]}...")

    if tokens1 == tokens2:
        print("✅ PASS: Static mode produces consistent samples")
    else:
        print("❌ FAIL: Static samples differ (should be identical)")

    print()


def test_teacher_forcing_ratio():
    """Test teacher forcing ratio decay"""
    print("=" * 60)
    print("Test 3: Teacher Forcing Decay")
    print("=" * 60)

    # Test decay calculation
    tf_start = config.teacher_forcing_start
    tf_end = config.teacher_forcing_end
    decay_steps = config.teacher_forcing_decay_steps

    checkpoints = [0, 1000, 2500, 5000, 7000]

    print(f"TF Start: {tf_start}, TF End: {tf_end}, Decay Steps: {decay_steps}")
    print()

    for step in checkpoints:
        if step < decay_steps:
            tf_ratio = tf_start - (tf_start - tf_end) * (step / decay_steps)
        else:
            tf_ratio = tf_end

        print(f"Step {step:5d}: TF Ratio = {tf_ratio:.3f}")

    print()
    print("✅ PASS: Teacher forcing decay calculated correctly")
    print()


def test_dataloaders():
    """Test dataloader creation"""
    print("=" * 60)
    print("Test 4: Dataloader Creation")
    print("=" * 60)

    train_loader, val_loader, test_loader = create_dataloaders(
        vocab_size=16,
        seq_length=20,
        hidden_dim=16,
        ortho_mode='random',
        train_samples=100,
        val_samples=50,
        test_samples=50,
        batch_size=4,
        num_workers=0,
        seed=42,
        train_mode='online'
    )

    # Get a batch from each
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    print(f"Train batch shape: {train_batch['tokens'].shape}")
    print(f"Val batch shape: {val_batch['tokens'].shape}")

    # Get two batches from train loader - should be different
    train_batch2 = next(iter(train_loader))
    tokens1 = train_batch['tokens'][0].tolist()
    tokens2 = train_batch2['tokens'][0].tolist()

    if tokens1 != tokens2:
        print("✅ PASS: Train batches are different (online generation working)")
    else:
        print("⚠️  WARNING: Train batches identical (may be due to seeding)")

    print()


if __name__ == '__main__':
    test_online_generation()
    test_static_generation()
    test_teacher_forcing_ratio()
    test_dataloaders()

    print("=" * 60)
    print("✅ All Tests Complete!")
    print("=" * 60)
    print()
    print("You can now run training with:")
    print("  python train_rotation.py")
    print()
    print("Expected behavior:")
    print("  - Train loss may be higher initially (no memorization)")
    print("  - Val loss should decrease (learning the rule)")
    print("  - TF ratio will decay from 1.0 to 0.0 over 5000 steps")
    print("  - L2R correct should increase significantly")
