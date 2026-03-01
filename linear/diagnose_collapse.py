"""
诊断脚本：检测模型是否发生"均值塌陷"（Mean Collapse）

运行方式:
    python -m linear_rotation_exp.diagnose_collapse [--checkpoint PATH]

诊断内容:
    1. 预测向量的模长 vs 目标向量的模长
    2. 预测向量的方差（是否趋于常数）
    3. 预测是否趋同于某个固定向量
    4. Agent 输出的排列多样性
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter

from . import config_continuous_rotation as config
from .continuous_dataset import create_continuous_dataloaders
from .continuous_model import ContinuousTransformer, ContinuousTransformerConfig
from .set_to_seq_agent import SetToSeqAgent


def find_latest_checkpoint(exp_dir: str) -> str:
    """查找最新的 checkpoint 文件"""
    candidates = [
        os.path.join(exp_dir, 'best_model.pt'),
        os.path.join(exp_dir, 'final_model.pt'),
    ]

    # 也检查 numbered checkpoints
    if os.path.exists(exp_dir):
        for f in sorted(os.listdir(exp_dir), reverse=True):
            if f.startswith('checkpoint_') and f.endswith('.pt'):
                candidates.append(os.path.join(exp_dir, f))

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def diagnose_model_collapse(
    model: ContinuousTransformer,
    agent: SetToSeqAgent,
    dataloader,
    device: torch.device,
    num_batches: int = 10
):
    """
    诊断模型是否发生均值塌陷

    Returns:
        dict: 诊断结果
    """
    model.eval()
    if agent is not None:
        agent.eval()

    all_pred_norms = []
    all_target_norms = []
    all_pred_vars = []
    all_similarities_to_mean = []
    all_cos_sims = []
    all_mses = []
    all_permutations = []
    all_gt_orders = []

    print("=" * 70)
    print("模型塌陷诊断 (Model Collapse Diagnosis)")
    print("=" * 70)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            vectors = batch['vectors'].to(device)  # [B, L, D] GT ordered
            shuffled_vectors = batch['shuffled_vectors'].to(device)  # [B, L, D]
            gt_order = batch['order'].to(device)  # [B, L]
            B, L, D = vectors.shape

            # Agent 预测排列
            if agent is not None:
                permutation, _, _ = agent(shuffled_vectors, teacher_forcing_ratio=0.0)
            else:
                # 使用随机排列（baseline 模式）
                permutation = torch.stack([torch.randperm(L, device=device) for _ in range(B)])

            all_permutations.append(permutation.cpu())
            all_gt_orders.append(gt_order.cpu())

            # 按 Agent 排列重排序
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
            X_ordered = shuffled_vectors[batch_idx, permutation]  # [B, L, D]

            # Model 预测
            predictions, loss, _ = model.forward_with_hidden(
                shuffled_vectors, permutation, targets=vectors
            )

            # 预测 vs 目标（按排列顺序）
            pred = predictions[:, :-1]  # [B, L-1, D]
            target = X_ordered[:, 1:]   # [B, L-1, D]

            # ===== 诊断 1: 模长 =====
            pred_norm = pred.norm(dim=-1)  # [B, L-1]
            target_norm = target.norm(dim=-1)  # [B, L-1]
            all_pred_norms.append(pred_norm.cpu())
            all_target_norms.append(target_norm.cpu())

            # ===== 诊断 2: 方差 =====
            # 每个位置的预测在 batch 内的方差
            pred_var = pred.var(dim=0).mean(dim=-1)  # [L-1]
            all_pred_vars.append(pred_var.cpu())

            # ===== 诊断 3: 趋同性 =====
            # 所有预测的平均向量
            pred_flat = pred.reshape(-1, D)  # [B*(L-1), D]
            pred_mean = pred_flat.mean(dim=0)  # [D]
            # 每个预测与平均向量的相似度
            similarity_to_mean = F.cosine_similarity(
                pred_flat, pred_mean.unsqueeze(0).expand(pred_flat.size(0), -1), dim=-1
            )
            all_similarities_to_mean.append(similarity_to_mean.cpu())

            # ===== 诊断 4: 预测质量 =====
            cos_sim = F.cosine_similarity(pred, target, dim=-1)  # [B, L-1]
            mse = (pred - target).pow(2).mean(dim=-1)  # [B, L-1]
            all_cos_sims.append(cos_sim.cpu())
            all_mses.append(mse.cpu())

    # 汇总结果
    pred_norms = torch.cat(all_pred_norms).flatten()
    target_norms = torch.cat(all_target_norms).flatten()
    pred_vars = torch.cat(all_pred_vars)
    similarities = torch.cat(all_similarities_to_mean)
    cos_sims = torch.cat(all_cos_sims).flatten()
    mses = torch.cat(all_mses).flatten()
    permutations = torch.cat(all_permutations)
    gt_orders = torch.cat(all_gt_orders)

    results = {}

    # ===== 报告 1: 模长分析 =====
    print("\n[1] 模长分析 (Norm Analysis)")
    print("-" * 50)
    results['pred_norm_mean'] = pred_norms.mean().item()
    results['pred_norm_std'] = pred_norms.std().item()
    results['target_norm_mean'] = target_norms.mean().item()
    results['target_norm_std'] = target_norms.std().item()
    results['norm_ratio'] = results['pred_norm_mean'] / (results['target_norm_mean'] + 1e-8)

    print(f"    预测模长: {results['pred_norm_mean']:.4f} ± {results['pred_norm_std']:.4f}")
    print(f"    目标模长: {results['target_norm_mean']:.4f} ± {results['target_norm_std']:.4f}")
    print(f"    模长比值: {results['norm_ratio']:.4f}")

    if results['norm_ratio'] < 0.5:
        print("    [!] 警告: 预测模长显著小于目标，可能存在模长塌陷")
    elif results['norm_ratio'] > 2.0:
        print("    [!] 警告: 预测模长显著大于目标，可能存在模长爆炸")
    else:
        print("    [✓] 模长比值正常")

    # ===== 报告 2: 方差分析 =====
    print("\n[2] 方差分析 (Variance Analysis)")
    print("-" * 50)
    results['pred_var_mean'] = pred_vars.mean().item()
    results['pred_var_std'] = pred_vars.std().item()

    print(f"    预测方差 (跨 batch): {results['pred_var_mean']:.6f} ± {results['pred_var_std']:.6f}")

    if results['pred_var_mean'] < 0.01:
        print("    [!] 警告: 方差极小，模型可能输出近似常数")
    else:
        print("    [✓] 方差正常")

    # ===== 报告 3: 趋同性分析 =====
    print("\n[3] 趋同性分析 (Convergence to Mean)")
    print("-" * 50)
    results['similarity_to_mean'] = similarities.mean().item()
    results['similarity_to_mean_std'] = similarities.std().item()

    print(f"    与平均预测的相似度: {results['similarity_to_mean']:.4f} ± {results['similarity_to_mean_std']:.4f}")

    if results['similarity_to_mean'] > 0.95:
        print("    [!] 严重警告: 所有预测高度趋同，模型发生均值塌陷!")
    elif results['similarity_to_mean'] > 0.8:
        print("    [!] 警告: 预测趋同性较高，可能存在部分塌陷")
    else:
        print("    [✓] 预测多样性正常")

    # ===== 报告 4: 预测质量 =====
    print("\n[4] 预测质量 (Prediction Quality)")
    print("-" * 50)
    results['cos_sim_mean'] = cos_sims.mean().item()
    results['cos_sim_std'] = cos_sims.std().item()
    results['mse_mean'] = mses.mean().item()
    results['mse_std'] = mses.std().item()

    print(f"    余弦相似度: {results['cos_sim_mean']:.4f} ± {results['cos_sim_std']:.4f}")
    print(f"    MSE: {results['mse_mean']:.4f} ± {results['mse_std']:.4f}")

    # ===== 报告 5: Agent 排列分析 =====
    if agent is not None:
        print("\n[5] Agent 排列分析 (Permutation Analysis)")
        print("-" * 50)

        # Kendall Tau
        from scipy.stats import kendalltau
        taus = []
        for i in range(min(100, permutations.size(0))):
            tau, _ = kendalltau(permutations[i].numpy(), gt_orders[i].numpy())
            if not np.isnan(tau):
                taus.append(tau)

        results['kendall_tau_mean'] = np.mean(taus) if taus else 0.0
        results['kendall_tau_std'] = np.std(taus) if taus else 0.0
        print(f"    Kendall Tau: {results['kendall_tau_mean']:.4f} ± {results['kendall_tau_std']:.4f}")

        # 完全正确率
        exact_match = (permutations == gt_orders).all(dim=-1).float().mean().item()
        results['exact_match'] = exact_match
        print(f"    完全正确率: {exact_match:.2%}")

        # 首位正确率
        first_correct = (permutations[:, 0] == gt_orders[:, 0]).float().mean().item()
        results['first_position_correct'] = first_correct
        print(f"    首位正确率: {first_correct:.2%}")

        # 排列多样性：检查 Agent 是否总是输出相同的排列
        unique_perms = set()
        for p in permutations[:100]:
            unique_perms.add(tuple(p.tolist()))
        results['unique_permutations'] = len(unique_perms)
        print(f"    前100个样本中的唯一排列数: {results['unique_permutations']}")

        if results['unique_permutations'] < 10:
            print("    [!] 警告: Agent 输出排列多样性极低，可能发生策略塌陷")

    # ===== 综合诊断 =====
    print("\n" + "=" * 70)
    print("综合诊断 (Summary)")
    print("=" * 70)

    issues = []
    if results['norm_ratio'] < 0.5:
        issues.append("模长塌陷 (Norm Collapse)")
    if results['pred_var_mean'] < 0.01:
        issues.append("方差塌陷 (Variance Collapse)")
    if results['similarity_to_mean'] > 0.9:
        issues.append("均值塌陷 (Mean Collapse)")
    if agent is not None and results.get('unique_permutations', 100) < 10:
        issues.append("策略塌陷 (Policy Collapse)")

    if issues:
        print(f"\n发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n建议:")
        if "模长塌陷" in str(issues):
            print("    - 考虑使用 Cosine Loss 替代 MSE Loss")
            print("    - 或在 loss 中加入模长正则项")
        if "均值塌陷" in str(issues) or "方差塌陷" in str(issues):
            print("    - 检查学习率是否过大")
            print("    - 考虑增加 dropout 或其他正则化")
            print("    - 检查数据预处理是否正确")
        if "策略塌陷" in str(issues):
            print("    - 增加 exploration (如 entropy bonus)")
            print("    - 检查 reward 信号是否足够")
    else:
        print("\n[✓] 未检测到明显的塌陷问题")

    return results


def main():
    parser = argparse.ArgumentParser(description='Diagnose model collapse')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file')
    parser.add_argument('--num_batches', type=int, default=10,
                        help='Number of batches to analyze')
    args = parser.parse_args()

    # 查找 checkpoint
    if args.checkpoint is None:
        checkpoint_path = find_latest_checkpoint(config.exp_dir)
        if checkpoint_path is None:
            print(f"[ERROR] 未找到 checkpoint，请指定 --checkpoint 参数")
            print(f"        搜索目录: {config.exp_dir}")
            return
    else:
        checkpoint_path = args.checkpoint

    print(f"[INFO] 加载 checkpoint: {checkpoint_path}")

    # 设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint.get('config', {})

    print(f"[INFO] Checkpoint step: {checkpoint.get('step', 'unknown')}")
    print(f"[INFO] Checkpoint metrics: {checkpoint.get('metrics', {})}")

    # 创建数据加载器
    print("\n[INFO] 创建数据加载器...")
    _, val_loader, _, _ = create_continuous_dataloaders(
        vector_dim=saved_config.get('vector_dim', config.vector_dim),
        seq_length=saved_config.get('seq_length', config.seq_length),
        dependency_window=saved_config.get('dependency_window', config.dependency_window),
        num_matrices=saved_config.get('num_matrices', getattr(config, 'num_matrices', None)),
        train_samples=1000,  # 少量样本即可
        val_samples=saved_config.get('val_samples', config.val_samples),
        test_samples=1000,
        batch_size=saved_config.get('batch_size', config.batch_size),
        num_workers=0,  # 诊断时使用单线程
        seed=saved_config.get('seed', config.seed),
        fixed_matrices_path=saved_config.get('fixed_matrices_path', config.fixed_matrices_path),
        train_init_mode=saved_config.get('train_init_mode', config.train_init_mode),
        val_init_mode=saved_config.get('val_init_mode', config.val_init_mode),
        num_chunks=saved_config.get('num_chunks', getattr(config, 'num_chunks', 4))
    )

    # 创建模型
    print("[INFO] 创建模型...")
    from .continuous_model import ContinuousTransformerConfig
    model_config = ContinuousTransformerConfig(
        vector_dim=saved_config.get('vector_dim', config.vector_dim),
        n_layer=saved_config.get('n_layer', config.n_layer),
        n_head=saved_config.get('n_head', config.n_head),
        n_embd=saved_config.get('n_embd', config.n_embd),
        block_size=saved_config.get('block_size', config.block_size),
        dropout=0.0,  # 评估时关闭 dropout
        bias=saved_config.get('bias', config.bias)
    )
    model = ContinuousTransformer(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 创建 Agent (如果存在)
    agent = None
    if 'agent_state_dict' in checkpoint:
        print("[INFO] 创建 Agent...")
        agent = SetToSeqAgent(
            vector_dim=saved_config.get('vector_dim', config.vector_dim),
            d_model=saved_config.get('agent_d_model', config.agent_d_model),
            encoder_layers=saved_config.get('agent_encoder_layers', config.agent_encoder_layers),
            encoder_heads=saved_config.get('agent_encoder_heads', config.agent_encoder_heads),
            decoder_layers=saved_config.get('agent_decoder_layers', config.agent_decoder_layers),
            decoder_heads=saved_config.get('agent_decoder_heads', config.agent_decoder_heads),
            max_len=saved_config.get('seq_length', config.seq_length),
            dropout=0.0
        ).to(device)
        agent.load_state_dict(checkpoint['agent_state_dict'])
    else:
        print("[INFO] Checkpoint 中无 Agent (Baseline 模式)")

    # 运行诊断
    print()
    results = diagnose_model_collapse(model, agent, val_loader, device, args.num_batches)

    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
