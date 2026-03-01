"""
Sanity Check: Verify the trained model learned real causal dynamics,
not a statistical shortcut.

Primary metric: Cosine Similarity (not MSE).
In D=64 with unit-norm vectors, MSE baseline is ~2/D ~ 0.03, making
raw MSE differences misleading. Cosine similarity is the correct measure:
  - Random vectors in R^64: E[CosSim] ~ 0
  - A model that learned physics: CosSim >> 0

Test A: Real OOD physics data  -> expect HIGH CosSim (> 0.3)
Test B: Random spherical noise -> expect ZERO CosSim (~ 0.0)

Usage:
    python -m linear_rotation_exp.sanity_check
"""

import torch
import torch.nn.functional as F
import os

from .continuous_model import ContinuousTransformer, ContinuousTransformerConfig
from .set_to_seq_agent import SetToSeqAgent
from .continuous_data_generator import ContinuousDenseARGenerator
from . import config_continuous_rotation as config


def run_sanity_check():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 1. Load checkpoint and recover config
    # ------------------------------------------------------------------
    checkpoint_path = os.path.join(config.exp_dir, 'best_model.pt')
    print(f"\n[1] Loading checkpoint from {checkpoint_path} ...")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_cfg = checkpoint.get('config', {})

    seq_length = ckpt_cfg.get('seq_length', config.seq_length)
    block_size = ckpt_cfg.get('block_size', config.block_size)
    vector_dim = ckpt_cfg.get('vector_dim', config.vector_dim)
    dep_window = ckpt_cfg.get('dependency_window', config.dependency_window)
    num_matrices = ckpt_cfg.get('num_matrices', getattr(config, 'num_matrices', 6))

    print(f"    seq_length={seq_length}, vector_dim={vector_dim}, "
          f"dependency_window={dep_window}, num_matrices={num_matrices}")

    # ------------------------------------------------------------------
    # 2. Rebuild Model & Agent with checkpoint dimensions
    # ------------------------------------------------------------------
    model_config = ContinuousTransformerConfig(
        vector_dim=vector_dim,
        n_layer=ckpt_cfg.get('n_layer', config.n_layer),
        n_head=ckpt_cfg.get('n_head', config.n_head),
        n_embd=ckpt_cfg.get('n_embd', config.n_embd),
        block_size=block_size,
        dropout=0.0,
        bias=ckpt_cfg.get('bias', config.bias),
    )
    model = ContinuousTransformer(model_config).to(device)

    agent = SetToSeqAgent(
        vector_dim=vector_dim,
        d_model=ckpt_cfg.get('agent_d_model', config.agent_d_model),
        encoder_layers=ckpt_cfg.get('agent_encoder_layers', config.agent_encoder_layers),
        encoder_heads=ckpt_cfg.get('agent_encoder_heads', config.agent_encoder_heads),
        decoder_layers=ckpt_cfg.get('agent_decoder_layers', config.agent_decoder_layers),
        decoder_heads=ckpt_cfg.get('agent_decoder_heads', config.agent_decoder_heads),
        max_len=seq_length,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    agent.load_state_dict(checkpoint['agent_state_dict'])
    model.eval()
    agent.eval()
    print("    Checkpoint loaded successfully.")

    # ------------------------------------------------------------------
    # 3. Prepare data generator (same matrices as training)
    # ------------------------------------------------------------------
    generator = ContinuousDenseARGenerator(
        vector_dim=vector_dim,
        dependency_window=dep_window,
        num_matrices=num_matrices,
        seed=config.seed,
        fixed_matrices_path=config.fixed_matrices_path,
    )

    B = 256

    # ------------------------------------------------------------------
    # Test A: Real OOD Physics Data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Test A] Real Physics Data (OOD, negative_first)")
    print("=" * 60)

    result = generator.generate_sequence(
        length=seq_length, init_mode='negative_first', batch_size=B
    )
    real_vectors = result['vectors'].to(device)

    # Shuffle each sequence independently
    shuffle_indices = torch.stack([torch.randperm(seq_length) for _ in range(B)]).to(device)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, seq_length)
    real_shuffled = real_vectors[batch_idx, shuffle_indices]
    gt_order = torch.argsort(shuffle_indices, dim=1)

    with torch.no_grad():
        # Agent proposes ordering
        permutation, _, _ = agent(real_shuffled, teacher_forcing_ratio=0.0)

        # Model predicts under agent's ordering
        preds_agent, mse_agent, _ = model.forward_with_hidden(
            real_shuffled, permutation, targets=real_vectors
        )
        # Reorder targets to match prediction order for cosine sim
        ordered_targets_agent = real_vectors[batch_idx, permutation]
        cos_sim_agent = F.cosine_similarity(
            preds_agent[:, :-1], ordered_targets_agent[:, 1:], dim=-1
        ).mean().item()

        # Model predicts under GT ordering (upper bound)
        preds_gt, mse_gt, _ = model.forward_with_hidden(
            real_shuffled, gt_order, targets=real_vectors
        )
        ordered_targets_gt = real_vectors[batch_idx, gt_order]
        cos_sim_gt = F.cosine_similarity(
            preds_gt[:, :-1], ordered_targets_gt[:, 1:], dim=-1
        ).mean().item()

        # Agent accuracy
        l2r_correct = (permutation == gt_order).all(dim=-1).float().mean().item()

    print(f"    MSE  (Agent order):   {mse_agent.item():.6f}")
    print(f"    MSE  (GT order):      {mse_gt.item():.6f}")
    print(f"    CosSim (Agent order): {cos_sim_agent:.4f}")
    print(f"    CosSim (GT order):    {cos_sim_gt:.4f}")
    print(f"    Agent L2R accuracy:   {l2r_correct:.2%}")

    # ------------------------------------------------------------------
    # Test B: Random Spherical Noise (no causality)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Test B] Spherical Random Noise (Zero Causality)")
    print("=" * 60)

    noise = torch.randn(B, seq_length, vector_dim, device=device)
    fake_vectors = F.normalize(noise, p=2, dim=-1)

    with torch.no_grad():
        perm_fake, _, _ = agent(fake_vectors, teacher_forcing_ratio=0.0)
        ordered_fake = fake_vectors[batch_idx, perm_fake]

        preds_fake, _ = model(ordered_fake)
        mse_fake = F.mse_loss(preds_fake[:, :-1], ordered_fake[:, 1:]).item()
        cos_sim_fake = F.cosine_similarity(
            preds_fake[:, :-1], ordered_fake[:, 1:], dim=-1
        ).mean().item()

    print(f"    MSE  (Noise):         {mse_fake:.6f}")
    print(f"    CosSim (Noise):       {cos_sim_fake:.4f}")

    # ------------------------------------------------------------------
    # Test C: Random Ordering on Real Data (ablation)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[Test C] Real Data + Random Order (Agent Ablation)")
    print("=" * 60)

    with torch.no_grad():
        random_order = torch.stack([torch.randperm(seq_length) for _ in range(B)]).to(device)
        preds_rand, mse_rand, _ = model.forward_with_hidden(
            real_shuffled, random_order, targets=real_vectors
        )
        ordered_targets_rand = real_vectors[batch_idx, random_order]
        cos_sim_rand = F.cosine_similarity(
            preds_rand[:, :-1], ordered_targets_rand[:, 1:], dim=-1
        ).mean().item()

    print(f"    MSE  (Random order):  {mse_rand.item():.6f}")
    print(f"    CosSim (Rand order):  {cos_sim_rand:.4f}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SANITY CHECK SUMMARY")
    print("=" * 60)
    print()
    print(f"  {'Condition':<32} {'MSE':>8}  {'CosSim':>8}")
    print(f"  {'-'*32} {'-'*8}  {'-'*8}")
    print(f"  {'Real OOD + GT order':<32} {mse_gt.item():>8.4f}  {cos_sim_gt:>8.4f}")
    print(f"  {'Real OOD + Agent order':<32} {mse_agent.item():>8.4f}  {cos_sim_agent:>8.4f}")
    print(f"  {'Real OOD + Random order':<32} {mse_rand.item():>8.4f}  {cos_sim_rand:>8.4f}")
    print(f"  {'Spherical Noise + Agent':<32} {mse_fake:>8.4f}  {cos_sim_fake:>8.4f}")
    print(f"  {'Theoretical random baseline':<32} {'~0.031':>8}  {'~0.000':>8}")
    print()

    # Judgment based on cosine similarity
    if cos_sim_agent > 0.3 and cos_sim_fake < 0.05:
        print("  PASS - The model learned REAL CAUSAL DYNAMICS.")
        print(f"  CosSim gap: {cos_sim_agent:.3f} (real) vs {cos_sim_fake:.3f} (noise)")
        print(f"  Agent perfectly recovers order: {l2r_correct:.0%}")
    elif cos_sim_agent > 0.15 and cos_sim_agent > cos_sim_fake + 0.1:
        print("  LIKELY PASS - Model shows meaningful causal signal.")
        print(f"  CosSim: {cos_sim_agent:.3f} (real) vs {cos_sim_fake:.3f} (noise)")
    elif cos_sim_fake > cos_sim_agent - 0.05:
        print("  FAIL - Model cannot distinguish causal data from noise.")
    else:
        print("  INCONCLUSIVE - Check values manually.")


if __name__ == '__main__':
    run_sanity_check()
