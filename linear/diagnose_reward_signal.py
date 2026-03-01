"""
诊断脚本：验证 Reward 信号的可区分性 (Reward Discriminability)

核心假设验证："闭环自洽陷阱" (Closed-Loop Self-Consistency Trap)
- Model 是否对 GT 序列和随机序列给出相同的 "满意度"？
- 如果 Score_GT ≈ Score_Random，则 Reward 无法指导 Agent 学习正确排序

实验设计：
- Case A (GT): 正确的时间顺序 → 计算 Cosine Reward
- Case B (Random): 100 个随机排列 → 计算 Cosine Reward 的分布
- 统计检验: Z-Score = (Score_GT - Score_Random_Mean) / Score_Random_Std

运行方式:
    python -m linear_rotation_exp.diagnose_reward_signal [--checkpoint PATH]
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from . import config_continuous_rotation as config


def simple_table(data: List[List], headers: List[str]) -> str:
    """简单的表格格式化函数（替代 tabulate）"""
    # 计算每列的最大宽度
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # 构建分隔线
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    # 构建表格
    lines = [sep]

    # 表头
    header_row = "|" + "|".join(f" {h:^{col_widths[i]}} " for i, h in enumerate(headers)) + "|"
    lines.append(header_row)
    lines.append(sep)

    # 数据行
    for row in data:
        data_row = "|" + "|".join(f" {str(cell):^{col_widths[i]}} " for i, cell in enumerate(row)) + "|"
        lines.append(data_row)

    lines.append(sep)
    return "\n".join(lines)
from .continuous_dataset import ContinuousRotationDataset, create_continuous_dataloaders
from .continuous_data_generator import ContinuousDenseARGenerator
from .continuous_model import ContinuousTransformer, ContinuousTransformerConfig


def find_latest_checkpoint(exp_dir: str) -> Optional[str]:
    """查找最新的 checkpoint 文件"""
    candidates = [
        os.path.join(exp_dir, 'best_model.pt'),
        os.path.join(exp_dir, 'final_model.pt'),
    ]

    # 添加所有 checkpoint 文件，按数字排序（最新的优先）
    if os.path.exists(exp_dir):
        checkpoint_files = []
        for f in os.listdir(exp_dir):
            if f.startswith('checkpoint_') and f.endswith('.pt'):
                try:
                    step = int(f.replace('checkpoint_', '').replace('.pt', ''))
                    checkpoint_files.append((step, os.path.join(exp_dir, f)))
                except ValueError:
                    pass
        # 按 step 降序排序
        checkpoint_files.sort(key=lambda x: -x[0])
        for _, path in checkpoint_files:
            candidates.append(path)

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def compute_cosine_reward(
    model: ContinuousTransformer,
    shuffled_vectors: torch.Tensor,
    order: torch.Tensor,
    device: torch.device
) -> float:
    """
    计算给定排列下的 Cosine Reward (Model 的 "满意度")

    Reward 定义:
    - 对于位置 t (t=0,...,L-2)，Model 用前 t 个 token 预测第 t+1 个 token
    - Reward = mean(CosSim(pred[t], target[t+1]))

    Args:
        model: ContinuousTransformer 模型
        shuffled_vectors: [L, D] 打乱的输入向量
        order: [L] 排列顺序
        device: 计算设备

    Returns:
        mean_cosine_similarity: 平均余弦相似度
    """
    L, D = shuffled_vectors.shape

    # 按 order 重排，得到当前排列下的序列
    ordered_vectors = shuffled_vectors[order]  # [L, D]

    # 准备输入：添加 batch 维度
    X = ordered_vectors.unsqueeze(0)  # [1, L, D]

    # 输入投影 + 位置编码 + Transformer
    x = model.input_proj(X)  # [1, L, n_embd]
    positions = torch.arange(0, L, dtype=torch.long, device=device)
    pos_emb = model.wpe(positions)  # [L, n_embd]
    x = x + pos_emb

    for block in model.blocks:
        x = block(x, use_causal_mask=True)

    x = model.ln_f(x)
    predictions = model.output_proj(x)  # [1, L, D]

    # 预测：pred[t] 预测 target[t+1]
    # predictions[:, :-1] -> [1, L-1, D] (预测值)
    # X[:, 1:] -> [1, L-1, D] (目标值)
    pred = predictions[0, :-1]  # [L-1, D]
    target = X[0, 1:]  # [L-1, D]

    # 计算逐位置的余弦相似度
    cos_sim = F.cosine_similarity(pred, target, dim=-1)  # [L-1]

    return cos_sim.mean().item()


def generate_random_permutation(L: int, device: torch.device) -> torch.Tensor:
    """生成完全随机的排列"""
    return torch.randperm(L, device=device)


def generate_block_shuffle_permutation(
    L: int,
    num_chunks: int,
    device: torch.device
) -> torch.Tensor:
    """
    生成 block-wise shuffle 排列（保持块内顺序，打乱块间顺序）

    这更贴近实际的 Agent 任务：Agent 需要恢复 chunk 级别的顺序
    """
    chunk_size = L // num_chunks
    chunk_indices = [torch.arange(i * chunk_size, (i + 1) * chunk_size, device=device)
                     for i in range(num_chunks)]

    # 处理不能整除的情况
    if L % num_chunks != 0:
        chunk_indices[-1] = torch.arange((num_chunks - 1) * chunk_size, L, device=device)

    # 随机打乱 chunk 顺序
    chunk_order = torch.randperm(num_chunks, device=device)
    perm = torch.cat([chunk_indices[i] for i in chunk_order])

    return perm


def diagnose_reward_discriminability(
    model: ContinuousTransformer,
    generator: ContinuousDenseARGenerator,
    device: torch.device,
    num_test_sequences: int = 100,
    num_random_perms: int = 100,
    num_chunks: int = 16,
    seq_length: int = 64,
    init_mode: str = 'positive_first',
    seed: int = 12345,
    use_block_shuffle: bool = True
) -> Dict:
    """
    诊断 Reward 信号的可区分性

    核心问题: Score_GT 是否显著高于 Score_Random？

    Args:
        model: 训练好的 ContinuousTransformer
        generator: 数据生成器
        device: 计算设备
        num_test_sequences: 测试的序列数量
        num_random_perms: 每个序列测试的随机排列数量
        num_chunks: block shuffle 的 chunk 数量
        seq_length: 序列长度
        init_mode: 初始化模式
        seed: 随机种子
        use_block_shuffle: True = block-wise shuffle, False = 完全随机

    Returns:
        诊断结果字典
    """
    model.eval()
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=" * 80)
    print("    REWARD DISCRIMINABILITY DIAGNOSIS")
    print("    验证 Reward 信号的可区分性")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  - 测试序列数: {num_test_sequences}")
    print(f"  - 每序列随机排列数: {num_random_perms}")
    print(f"  - 序列长度: {seq_length}")
    print(f"  - Chunk 数量: {num_chunks}")
    print(f"  - 随机排列类型: {'Block Shuffle' if use_block_shuffle else 'Full Random'}")
    print(f"  - 初始化模式: {init_mode}")
    print()

    # 存储所有结果
    results_per_sequence = []

    with torch.no_grad():
        print("正在生成测试数据并计算 Rewards...")
        print("-" * 80)

        for seq_idx in range(num_test_sequences):
            # 生成新的测试序列
            result = generator.generate_single_sequence(
                length=seq_length,
                init_mode=init_mode
            )
            vectors = result['vectors'].to(device)  # [L, D] GT ordered
            L = vectors.shape[0]

            # ===== Case A: GT 顺序 (Identity Permutation) =====
            gt_order = torch.arange(L, device=device)  # [0, 1, 2, ..., L-1]
            score_gt = compute_cosine_reward(model, vectors, gt_order, device)

            # ===== Case B: 随机排列 =====
            random_scores = []
            for _ in range(num_random_perms):
                if use_block_shuffle:
                    random_order = generate_block_shuffle_permutation(L, num_chunks, device)
                else:
                    random_order = generate_random_permutation(L, device)

                score_random = compute_cosine_reward(model, vectors, random_order, device)
                random_scores.append(score_random)

            random_scores = np.array(random_scores)
            score_random_mean = random_scores.mean()
            score_random_std = random_scores.std()
            score_random_max = random_scores.max()
            score_random_min = random_scores.min()

            # 计算 Z-Score
            if score_random_std > 1e-8:
                z_score = (score_gt - score_random_mean) / score_random_std
            else:
                z_score = 0.0 if abs(score_gt - score_random_mean) < 1e-8 else float('inf')

            # GT 在随机分布中的百分位
            percentile = (random_scores < score_gt).sum() / len(random_scores) * 100

            # 是否显著 (GT > Mean + 2*Std)
            is_significant = score_gt > score_random_mean + 2 * score_random_std

            results_per_sequence.append({
                'seq_idx': seq_idx,
                'score_gt': score_gt,
                'score_random_mean': score_random_mean,
                'score_random_std': score_random_std,
                'score_random_max': score_random_max,
                'score_random_min': score_random_min,
                'z_score': z_score,
                'percentile': percentile,
                'is_significant': is_significant
            })

            # 每 10 个序列打印一次进度
            if (seq_idx + 1) % 10 == 0:
                print(f"  进度: {seq_idx + 1}/{num_test_sequences} 序列已处理")

    print()

    # ==================== 汇总统计 ====================
    print("=" * 80)
    print("    SUMMARY TABLE (汇总表)")
    print("=" * 80)

    # 打印前 20 个样本的详细表格
    table_data = []
    for r in results_per_sequence[:20]:
        table_data.append([
            r['seq_idx'] + 1,
            f"{r['score_gt']:.4f}",
            f"{r['score_random_mean']:.4f}",
            f"{r['score_random_std']:.4f}",
            f"{r['z_score']:.2f}",
            f"{r['percentile']:.1f}%",
            "Yes" if r['is_significant'] else "No"
        ])

    headers = ["Seq#", "Score_GT", "Score_Rand_Mean", "Score_Rand_Std", "Z-Score", "Percentile", "GT>u+2s"]
    print(simple_table(table_data, headers))

    if num_test_sequences > 20:
        print(f"\n(仅显示前 20 个样本，共 {num_test_sequences} 个)")

    # ==================== 关键指标 ====================
    print("\n" + "=" * 80)
    print("    KEY METRICS (关键指标)")
    print("=" * 80)

    all_gt_scores = np.array([r['score_gt'] for r in results_per_sequence])
    all_random_means = np.array([r['score_random_mean'] for r in results_per_sequence])
    all_z_scores = np.array([r['z_score'] for r in results_per_sequence])
    all_percentiles = np.array([r['percentile'] for r in results_per_sequence])
    num_significant = sum(1 for r in results_per_sequence if r['is_significant'])

    metrics = {
        'gt_score_mean': all_gt_scores.mean(),
        'gt_score_std': all_gt_scores.std(),
        'random_score_mean': all_random_means.mean(),
        'random_score_std': all_random_means.std(),
        'reward_gap': all_gt_scores.mean() - all_random_means.mean(),
        'z_score_mean': all_z_scores.mean(),
        'z_score_std': all_z_scores.std(),
        'percentile_mean': all_percentiles.mean(),
        'percentile_std': all_percentiles.std(),
        'discriminability_ratio': num_significant / num_test_sequences,
        'num_significant': num_significant,
        'num_total': num_test_sequences
    }

    print(f"\n1. Reward Scores:")
    print(f"   Score_GT (平均):      {metrics['gt_score_mean']:.4f} ± {metrics['gt_score_std']:.4f}")
    print(f"   Score_Random (平均):  {metrics['random_score_mean']:.4f} ± {metrics['random_score_std']:.4f}")
    print(f"   Reward Gap (差距):    {metrics['reward_gap']:.4f}")

    print(f"\n2. Statistical Significance:")
    print(f"   Z-Score (平均):       {metrics['z_score_mean']:.2f} ± {metrics['z_score_std']:.2f}")
    print(f"   GT Percentile (平均): {metrics['percentile_mean']:.1f}% ± {metrics['percentile_std']:.1f}%")

    print(f"\n3. Discriminability Ratio (区分度):")
    print(f"   GT > Mean + 2*Std:    {metrics['discriminability_ratio']:.1%} ({num_significant}/{num_test_sequences})")

    # ==================== 诊断结论 ====================
    print("\n" + "=" * 80)
    print("    DIAGNOSIS (诊断结论)")
    print("=" * 80)

    # 判断标准
    disc_ratio = metrics['discriminability_ratio']
    z_mean = metrics['z_score_mean']
    pct_mean = metrics['percentile_mean']

    if disc_ratio >= 0.8 and z_mean >= 2.0:
        diagnosis = 'STRONG_SIGNAL'
        print("\n[PASS] Reward 信号具有 **强区分度**")
        print("       - GT 排列的 reward 显著高于随机排列")
        print("       - RL 方法 (REINFORCE/GRPO) 应该能有效工作")
        print("       - 建议继续当前的训练策略")

    elif disc_ratio >= 0.5 or (z_mean >= 1.0 and pct_mean >= 70):
        diagnosis = 'MEDIUM_SIGNAL'
        print("\n[WARNING] Reward 信号具有 **中等区分度**")
        print("          - GT 排列有一定优势，但不够显著")
        print("          - RL 方法可能需要更多辅助信号")
        print("\n   建议:")
        print("   1. 增加 smoothness/continuity reward")
        print("   2. 使用 curriculum learning (少 chunks → 多 chunks)")
        print("   3. 考虑更稳定的 RL 方法 (GRPO/PPO)")

    elif disc_ratio > 0.1:
        diagnosis = 'WEAK_SIGNAL'
        print("\n[FAIL] Reward 信号具有 **弱区分度**")
        print("       - GT 排列的 reward 仅略高于随机排列")
        print("       - 当前 reward 设计难以指导 Agent 学习")
        print("\n   建议:")
        print("   1. 先用 Behavior Cloning (BC) 预训练 Model")
        print("   2. 考虑更换 reward 函数设计")
        print("   3. 简化问题 (减少 chunks 数量)")

    else:
        diagnosis = 'NO_SIGNAL'
        print("\n[CRITICAL] Reward 信号 **没有区分度**!")
        print("           - GT 排列的 reward 与随机排列基本相同")
        print("           - 这证实了 '闭环自洽陷阱' 假设")
        print("           - Model 对任何排列都给出相似的预测")
        print("\n   原因分析:")
        print("   1. Model 可能只学习了向量的统计特性，而非时序依赖")
        print("   2. Cosine Loss 可能不足以捕捉顺序信息")
        print("   3. 数据生成过程的 AR 结构可能太弱")
        print("\n   建议:")
        print("   1. 检查数据生成器的 AR 结构是否正确")
        print("   2. 考虑使用 L2R 监督信号而非 Cosine Loss")
        print("   3. 先用 GT 顺序训练 Model，验证 AR 可学习性")

    metrics['diagnosis'] = diagnosis

    # ==================== 附加诊断 ====================
    print("\n" + "-" * 80)
    print("    ADDITIONAL ANALYSIS (附加分析)")
    print("-" * 80)

    # Model 在 GT 顺序下的预测质量
    print(f"\n1. Model 在 GT 顺序下的预测质量:")
    print(f"   平均 Cosine Similarity: {metrics['gt_score_mean']:.4f}")

    if metrics['gt_score_mean'] < 0.3:
        print("   [!] 警告: Model 即使用 GT 顺序，预测质量也很低")
        print("       这说明 Model 可能未正确学习 AR 动态")
    elif metrics['gt_score_mean'] < 0.5:
        print("   [~] Model 预测质量偏低")
    elif metrics['gt_score_mean'] < 0.7:
        print("   [~] Model 预测质量中等")
    else:
        print("   [OK] Model 预测质量良好")

    # Z-Score 分布
    print(f"\n2. Z-Score 分布:")
    z_positive = (all_z_scores > 0).sum() / len(all_z_scores) * 100
    z_gt_1 = (all_z_scores > 1).sum() / len(all_z_scores) * 100
    z_gt_2 = (all_z_scores > 2).sum() / len(all_z_scores) * 100
    print(f"   Z > 0:  {z_positive:.1f}% 的样本")
    print(f"   Z > 1:  {z_gt_1:.1f}% 的样本")
    print(f"   Z > 2:  {z_gt_2:.1f}% 的样本")

    # 最差和最好的样本
    print(f"\n3. 极端情况:")
    worst_idx = np.argmin(all_z_scores)
    best_idx = np.argmax(all_z_scores)
    print(f"   最差样本 (Seq #{worst_idx+1}): Z = {all_z_scores[worst_idx]:.2f}")
    print(f"   最好样本 (Seq #{best_idx+1}): Z = {all_z_scores[best_idx]:.2f}")

    print("\n" + "=" * 80)
    print("    诊断完成")
    print("=" * 80)

    return metrics, results_per_sequence


def main():
    parser = argparse.ArgumentParser(
        description='诊断 Reward 信号的可区分性 (Reward Discriminability Diagnosis)'
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint 文件路径 (默认: 自动查找最新的)')
    parser.add_argument('--num_sequences', type=int, default=100,
                        help='测试的序列数量 (默认: 100)')
    parser.add_argument('--num_random', type=int, default=100,
                        help='每个序列的随机排列数量 (默认: 100)')
    parser.add_argument('--num_chunks', type=int, default=16,
                        help='Block shuffle 的 chunk 数量 (默认: 16)')
    parser.add_argument('--full_random', action='store_true',
                        help='使用完全随机排列而非 block shuffle')
    parser.add_argument('--seed', type=int, default=12345,
                        help='随机种子 (默认: 12345)')
    args = parser.parse_args()

    # 查找 checkpoint
    if args.checkpoint is None:
        checkpoint_path = find_latest_checkpoint(config.exp_dir)
        if checkpoint_path is None:
            print(f"[ERROR] 未找到 checkpoint 文件")
            print(f"        搜索目录: {config.exp_dir}")
            print(f"        请使用 --checkpoint 参数指定路径")
            sys.exit(1)
    else:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint 文件不存在: {checkpoint_path}")
            sys.exit(1)

    print(f"[INFO] 加载 checkpoint: {checkpoint_path}")

    # 设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint.get('config', {})

    step = checkpoint.get('step', 'unknown')
    print(f"[INFO] Checkpoint step: {step}")

    # 获取配置参数
    vector_dim = saved_config.get('vector_dim', config.vector_dim)
    seq_length = saved_config.get('seq_length', config.seq_length)
    dependency_window = saved_config.get('dependency_window', config.dependency_window)
    num_matrices = saved_config.get('num_matrices', getattr(config, 'num_matrices', None))
    fixed_matrices_path = saved_config.get('fixed_matrices_path', config.fixed_matrices_path)
    train_init_mode = saved_config.get('train_init_mode', config.train_init_mode)

    print(f"[INFO] 数据配置: vector_dim={vector_dim}, seq_length={seq_length}")
    print(f"[INFO] AR 配置: dependency_window={dependency_window}, num_matrices={num_matrices}")

    # 创建数据生成器
    print("[INFO] 创建数据生成器...")
    generator = ContinuousDenseARGenerator(
        vector_dim=vector_dim,
        dependency_window=dependency_window,
        num_matrices=num_matrices,
        seed=args.seed,
        fixed_matrices_path=fixed_matrices_path
    )

    # 创建模型
    print("[INFO] 创建模型...")
    model_config = ContinuousTransformerConfig(
        vector_dim=vector_dim,
        n_layer=saved_config.get('n_layer', config.n_layer),
        n_head=saved_config.get('n_head', config.n_head),
        n_embd=saved_config.get('n_embd', config.n_embd),
        block_size=saved_config.get('block_size', config.block_size),
        dropout=0.0,
        bias=saved_config.get('bias', config.bias)
    )
    model = ContinuousTransformer(model_config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print()

    # 运行诊断
    metrics, results = diagnose_reward_discriminability(
        model=model,
        generator=generator,
        device=device,
        num_test_sequences=args.num_sequences,
        num_random_perms=args.num_random,
        num_chunks=args.num_chunks,
        seq_length=seq_length,
        init_mode=train_init_mode,
        seed=args.seed,
        use_block_shuffle=not args.full_random
    )

    # 打印最终结论
    print(f"\n最终诊断结果: {metrics['diagnosis']}")
    print(f"Discriminability Ratio: {metrics['discriminability_ratio']:.1%}")

    if metrics['discriminability_ratio'] < 0.1:
        print("\n>>> 证实了 '闭环自洽陷阱' 假设 <<<")
        print(">>> 需要更改 Reward 函数或训练方法 <<<")


if __name__ == '__main__':
    main()
