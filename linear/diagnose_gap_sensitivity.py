"""
诊断脚本：测试 Contrastive Gap Reward 的敏感性

核心假设验证："Cold Start" 问题
- Gap = Loss(Random_Baseline) - Loss(Agent_Perm)
- 如果 Agent 输出纯随机排列，Gap ≈ 0，学习无法启动

实验设计：
- Model_Random: 随机初始化的模型（模拟训练开始时的状态）
- Model_Oracle: 用 GT 顺序训练过的模型（模拟已经"顿悟"的状态）

测试三种排列质量：
- Perm_Random: 完全随机打乱
- Perm_Half: 一半正确，一半打乱
- Perm_GT: 完全正确（Ground Truth）

关键问题：
1. Cold Start: Model_Random 能否区分 Perm_Half vs Perm_Random？
2. Ideal State: Model_Oracle 能否给出 Gap_GT >> Gap_Half >> Gap_Random？

运行方式:
    python -m linear_rotation_exp.diagnose_gap_sensitivity
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from . import config_continuous_rotation as config
from .continuous_data_generator import ContinuousDenseARGenerator
from .continuous_model import ContinuousTransformer, ContinuousTransformerConfig


def simple_table(data: List[List], headers: List[str]) -> str:
    """简单的表格格式化函数"""
    col_widths = [len(h) for h in headers]
    for row in data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    lines = [sep]
    header_row = "|" + "|".join(f" {h:^{col_widths[i]}} " for i, h in enumerate(headers)) + "|"
    lines.append(header_row)
    lines.append(sep)

    for row in data:
        data_row = "|" + "|".join(f" {str(cell):^{col_widths[i]}} " for i, cell in enumerate(row)) + "|"
        lines.append(data_row)

    lines.append(sep)
    return "\n".join(lines)


def create_model(vector_dim: int, seq_length: int, device: torch.device) -> ContinuousTransformer:
    """创建一个新的 ContinuousTransformer 模型"""
    model_config = ContinuousTransformerConfig(
        vector_dim=vector_dim,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=seq_length,
        dropout=0.0,
        bias=config.bias
    )
    return ContinuousTransformer(model_config).to(device)


def compute_cosine_loss(
    model: ContinuousTransformer,
    vectors: torch.Tensor,
    order: torch.Tensor,
    device: torch.device
) -> float:
    """
    计算给定排列下的 Cosine Loss (1 - CosSim)

    Args:
        model: ContinuousTransformer 模型
        vectors: [B, L, D] 原始向量序列（GT 顺序）
        order: [B, L] 排列顺序
        device: 计算设备

    Returns:
        mean_cosine_loss: 平均余弦损失
    """
    B, L, D = vectors.shape

    # 按 order 重排
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, L)
    ordered_vectors = vectors[batch_idx, order]  # [B, L, D]

    # 前向传播
    x = model.input_proj(ordered_vectors)
    positions = torch.arange(0, L, dtype=torch.long, device=device)
    pos_emb = model.wpe(positions)
    x = x + pos_emb

    for block in model.blocks:
        x = block(x, use_causal_mask=True)

    x = model.ln_f(x)
    predictions = model.output_proj(x)  # [B, L, D]

    # 计算 loss: pred[t] 预测 target[t+1]
    pred = predictions[:, :-1]  # [B, L-1, D]
    target = ordered_vectors[:, 1:]  # [B, L-1, D]

    cos_sim = F.cosine_similarity(pred, target, dim=-1)  # [B, L-1]
    cos_loss = (1.0 - cos_sim).mean()

    return cos_loss.item()


def generate_permutation_types(
    L: int,
    num_chunks: int,
    batch_size: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """
    生成三种类型的排列

    Args:
        L: 序列长度
        num_chunks: chunk 数量
        batch_size: batch 大小
        device: 计算设备

    Returns:
        字典包含三种排列：'random', 'half', 'gt'
    """
    chunk_size = L // num_chunks
    perms = {}

    # 1. Perm_GT: Identity permutation
    perm_gt = torch.arange(L, device=device).unsqueeze(0).expand(batch_size, L).clone()
    perms['gt'] = perm_gt

    # 2. Perm_Random: 完全随机的 chunk 顺序
    perm_random = []
    for _ in range(batch_size):
        chunk_indices = [torch.arange(i * chunk_size, (i + 1) * chunk_size, device=device)
                         for i in range(num_chunks)]
        chunk_order = torch.randperm(num_chunks, device=device)
        perm = torch.cat([chunk_indices[i] for i in chunk_order])
        perm_random.append(perm)
    perms['random'] = torch.stack(perm_random)

    # 3. Perm_Half: 前一半 chunks 正确，后一半 chunks 随机打乱
    half_chunks = num_chunks // 2
    perm_half = []
    for _ in range(batch_size):
        # 前一半 chunks 保持顺序
        correct_part = [torch.arange(i * chunk_size, (i + 1) * chunk_size, device=device)
                        for i in range(half_chunks)]

        # 后一半 chunks 随机打乱
        remaining_chunks = [torch.arange(i * chunk_size, (i + 1) * chunk_size, device=device)
                           for i in range(half_chunks, num_chunks)]
        remaining_order = torch.randperm(len(remaining_chunks), device=device)
        shuffled_part = [remaining_chunks[i] for i in remaining_order]

        perm = torch.cat(correct_part + shuffled_part)
        perm_half.append(perm)
    perms['half'] = torch.stack(perm_half)

    return perms


def generate_random_baseline_perm(
    L: int,
    num_chunks: int,
    batch_size: int,
    device: torch.device
) -> torch.Tensor:
    """生成随机基线排列（用于计算 Gap）"""
    chunk_size = L // num_chunks
    perms = []
    for _ in range(batch_size):
        chunk_indices = [torch.arange(i * chunk_size, (i + 1) * chunk_size, device=device)
                         for i in range(num_chunks)]
        chunk_order = torch.randperm(num_chunks, device=device)
        perm = torch.cat([chunk_indices[i] for i in chunk_order])
        perms.append(perm)
    return torch.stack(perms)


def train_oracle_model(
    model: ContinuousTransformer,
    generator: ContinuousDenseARGenerator,
    device: torch.device,
    seq_length: int,
    num_steps: int = 1000,
    batch_size: int = 32,
    lr: float = 1e-3
) -> ContinuousTransformer:
    """
    用 GT 顺序快速训练一个 Oracle 模型

    Args:
        model: 待训练的模型
        generator: 数据生成器
        device: 计算设备
        seq_length: 序列长度
        num_steps: 训练步数
        batch_size: batch 大小
        lr: 学习率

    Returns:
        训练后的模型
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\n训练 Oracle Model ({num_steps} steps, GT 顺序)...")

    for step in range(num_steps):
        # 生成数据
        result = generator.generate_sequence(
            length=seq_length,
            init_mode='positive_first',
            batch_size=batch_size
        )
        vectors = result['vectors'].to(device)  # [B, L, D]
        B, L, D = vectors.shape

        # GT 顺序
        gt_order = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        # 前向传播
        x = model.input_proj(vectors)
        positions = torch.arange(0, L, dtype=torch.long, device=device)
        pos_emb = model.wpe(positions)
        x = x + pos_emb

        for block in model.blocks:
            x = block(x, use_causal_mask=True)

        x = model.ln_f(x)
        predictions = model.output_proj(x)

        # Cosine Loss
        pred = predictions[:, :-1]
        target = vectors[:, 1:]
        cos_sim = F.cosine_similarity(pred, target, dim=-1)
        loss = (1.0 - cos_sim).mean()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(f"  Step {step + 1}/{num_steps}, Loss: {loss.item():.4f}, CosSim: {cos_sim.mean().item():.4f}")

    model.eval()
    print(f"  训练完成! 最终 Loss: {loss.item():.4f}")
    return model


def diagnose_gap_sensitivity(
    generator: ContinuousDenseARGenerator,
    device: torch.device,
    seq_length: int = 64,
    num_chunks: int = 8,
    num_test_batches: int = 50,
    batch_size: int = 32,
    oracle_train_steps: int = 1000
) -> Dict:
    """
    诊断 Contrastive Gap Reward 的敏感性

    Args:
        generator: 数据生成器
        device: 计算设备
        seq_length: 序列长度
        num_chunks: chunk 数量
        num_test_batches: 测试 batch 数量
        batch_size: batch 大小
        oracle_train_steps: Oracle 模型训练步数

    Returns:
        诊断结果
    """
    print("=" * 80)
    print("    CONTRASTIVE GAP SENSITIVITY DIAGNOSIS")
    print("    测试 Gap Reward 在 Cold Start 问题下的敏感性")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  - 序列长度: {seq_length}")
    print(f"  - Chunk 数量: {num_chunks}")
    print(f"  - 测试 Batch 数: {num_test_batches}")
    print(f"  - Batch 大小: {batch_size}")
    print(f"  - Oracle 训练步数: {oracle_train_steps}")

    vector_dim = generator.D

    # ==================== 创建两个模型 ====================
    print("\n" + "-" * 80)
    print("1. 创建模型")
    print("-" * 80)

    print("\n创建 Model_Random (随机初始化)...")
    model_random = create_model(vector_dim, seq_length, device)
    model_random.eval()

    print("\n创建 Model_Oracle (GT 训练)...")
    model_oracle = create_model(vector_dim, seq_length, device)
    model_oracle = train_oracle_model(
        model_oracle, generator, device,
        seq_length=seq_length,
        num_steps=oracle_train_steps,
        batch_size=batch_size
    )

    # ==================== 测试 Gap 敏感性 ====================
    print("\n" + "-" * 80)
    print("2. 测试 Gap 敏感性")
    print("-" * 80)

    # 存储结果
    results = {
        'random_model': {'gt': [], 'half': [], 'random': []},
        'oracle_model': {'gt': [], 'half': [], 'random': []}
    }

    print(f"\n正在测试 {num_test_batches} 个 batches...")

    with torch.no_grad():
        for batch_idx in range(num_test_batches):
            # 生成测试数据
            result = generator.generate_sequence(
                length=seq_length,
                init_mode='positive_first',
                batch_size=batch_size
            )
            vectors = result['vectors'].to(device)  # [B, L, D]
            L = vectors.shape[1]

            # 生成三种排列
            perms = generate_permutation_types(L, num_chunks, batch_size, device)

            # 生成随机基线排列（每次都重新生成）
            baseline_perm = generate_random_baseline_perm(L, num_chunks, batch_size, device)

            # 对两个模型分别计算
            for model_name, model in [('random_model', model_random), ('oracle_model', model_oracle)]:
                # 计算基线 loss
                loss_baseline = compute_cosine_loss(model, vectors, baseline_perm, device)

                # 计算三种排列的 Gap
                for perm_type in ['gt', 'half', 'random']:
                    loss_target = compute_cosine_loss(model, vectors, perms[perm_type], device)
                    gap = loss_baseline - loss_target  # Gap > 0 表示 target 比 baseline 好
                    results[model_name][perm_type].append(gap)

            if (batch_idx + 1) % 10 == 0:
                print(f"  进度: {batch_idx + 1}/{num_test_batches} batches")

    # ==================== 统计分析 ====================
    print("\n" + "=" * 80)
    print("    RESULTS (结果)")
    print("=" * 80)

    # 计算统计量
    stats = {}
    for model_name in ['random_model', 'oracle_model']:
        stats[model_name] = {}
        for perm_type in ['gt', 'half', 'random']:
            gaps = np.array(results[model_name][perm_type])
            stats[model_name][perm_type] = {
                'mean': gaps.mean(),
                'std': gaps.std(),
                'min': gaps.min(),
                'max': gaps.max()
            }

    # 打印表格
    print("\n### Gap Statistics (Gap = Loss_Baseline - Loss_Target)")
    print("### Gap > 0 表示 Target 排列比随机基线更好\n")

    table_data = []
    for model_name, display_name in [('random_model', 'Model_Random'), ('oracle_model', 'Model_Oracle')]:
        for perm_type, perm_display in [('gt', 'Perm_GT'), ('half', 'Perm_Half'), ('random', 'Perm_Random')]:
            s = stats[model_name][perm_type]
            table_data.append([
                display_name,
                perm_display,
                f"{s['mean']:.4f}",
                f"{s['std']:.4f}",
                f"{s['min']:.4f}",
                f"{s['max']:.4f}"
            ])

    headers = ["Model", "Permutation", "Gap_Mean", "Gap_Std", "Gap_Min", "Gap_Max"]
    print(simple_table(table_data, headers))

    # ==================== Z-Score 分析 ====================
    print("\n" + "-" * 80)
    print("3. Z-Score 分析 (差异显著性)")
    print("-" * 80)

    z_scores = {}

    # Scenario A: Cold Start (Model_Random)
    print("\n### Scenario A: Cold Start (Model_Random)")
    print("### 问题: 随机初始化的模型能否区分不同质量的排列？\n")

    # Gap(GT) vs Gap(Random)
    gt_gaps = np.array(results['random_model']['gt'])
    half_gaps = np.array(results['random_model']['half'])
    random_gaps = np.array(results['random_model']['random'])

    # Z-score: (Mean_A - Mean_B) / sqrt(Std_A^2/n + Std_B^2/n)
    n = len(gt_gaps)

    z_gt_vs_random = (gt_gaps.mean() - random_gaps.mean()) / np.sqrt(
        gt_gaps.std()**2/n + random_gaps.std()**2/n + 1e-8
    )
    z_half_vs_random = (half_gaps.mean() - random_gaps.mean()) / np.sqrt(
        half_gaps.std()**2/n + random_gaps.std()**2/n + 1e-8
    )
    z_gt_vs_half = (gt_gaps.mean() - half_gaps.mean()) / np.sqrt(
        gt_gaps.std()**2/n + half_gaps.std()**2/n + 1e-8
    )

    z_scores['cold_start'] = {
        'gt_vs_random': z_gt_vs_random,
        'half_vs_random': z_half_vs_random,
        'gt_vs_half': z_gt_vs_half
    }

    print(f"  Gap(GT) vs Gap(Random):    Z = {z_gt_vs_random:.2f}")
    print(f"  Gap(Half) vs Gap(Random):  Z = {z_half_vs_random:.2f}")
    print(f"  Gap(GT) vs Gap(Half):      Z = {z_gt_vs_half:.2f}")

    if z_half_vs_random > 2:
        print("\n  [OK] Cold Start 时，Model 能区分 Half vs Random (Z > 2)")
        print("       → Pure RL 可能有效")
    elif z_half_vs_random > 1:
        print("\n  [~] Cold Start 时，Model 对 Half vs Random 有微弱区分 (1 < Z < 2)")
        print("       → 需要 GRPO 放大信号")
    else:
        print("\n  [!] Cold Start 时，Model 无法区分 Half vs Random (Z < 1)")
        print("       → 需要 Curriculum Learning 或 BC Warmup")

    # Scenario B: Ideal State (Model_Oracle)
    print("\n### Scenario B: Ideal State (Model_Oracle)")
    print("### 问题: 训练过的模型能否正确排序 Gap_GT >> Gap_Half >> Gap_Random？\n")

    gt_gaps_oracle = np.array(results['oracle_model']['gt'])
    half_gaps_oracle = np.array(results['oracle_model']['half'])
    random_gaps_oracle = np.array(results['oracle_model']['random'])

    z_gt_vs_random_oracle = (gt_gaps_oracle.mean() - random_gaps_oracle.mean()) / np.sqrt(
        gt_gaps_oracle.std()**2/n + random_gaps_oracle.std()**2/n + 1e-8
    )
    z_half_vs_random_oracle = (half_gaps_oracle.mean() - random_gaps_oracle.mean()) / np.sqrt(
        half_gaps_oracle.std()**2/n + random_gaps_oracle.std()**2/n + 1e-8
    )
    z_gt_vs_half_oracle = (gt_gaps_oracle.mean() - half_gaps_oracle.mean()) / np.sqrt(
        gt_gaps_oracle.std()**2/n + half_gaps_oracle.std()**2/n + 1e-8
    )

    z_scores['ideal_state'] = {
        'gt_vs_random': z_gt_vs_random_oracle,
        'half_vs_random': z_half_vs_random_oracle,
        'gt_vs_half': z_gt_vs_half_oracle
    }

    print(f"  Gap(GT) vs Gap(Random):    Z = {z_gt_vs_random_oracle:.2f}")
    print(f"  Gap(Half) vs Gap(Random):  Z = {z_half_vs_random_oracle:.2f}")
    print(f"  Gap(GT) vs Gap(Half):      Z = {z_gt_vs_half_oracle:.2f}")

    # 检查排序是否正确
    oracle_gt_mean = gt_gaps_oracle.mean()
    oracle_half_mean = half_gaps_oracle.mean()
    oracle_random_mean = random_gaps_oracle.mean()

    ordering_correct = oracle_gt_mean > oracle_half_mean > oracle_random_mean

    if ordering_correct and z_gt_vs_random_oracle > 2:
        print("\n  [OK] Ideal State: Gap_GT > Gap_Half > Gap_Random (排序正确)")
        print("       → Reward 函数在模型学习后是有效的")
    elif ordering_correct:
        print("\n  [~] Ideal State: 排序正确但差异不够显著")
        print("       → 可能需要更长的训练或更强的 AR 结构")
    else:
        print("\n  [!] Ideal State: 排序不正确!")
        print(f"       GT: {oracle_gt_mean:.4f}, Half: {oracle_half_mean:.4f}, Random: {oracle_random_mean:.4f}")
        print("       → Reward 函数可能存在根本问题")

    # ==================== 诊断结论 ====================
    print("\n" + "=" * 80)
    print("    DIAGNOSIS (诊断结论)")
    print("=" * 80)

    diagnosis = {}

    # Cold Start 诊断
    if z_half_vs_random > 2:
        diagnosis['cold_start'] = 'CAN_START'
        cold_start_msg = "Pure RL 可以启动学习"
    elif z_half_vs_random > 0.5:
        diagnosis['cold_start'] = 'WEAK_SIGNAL'
        cold_start_msg = "需要 GRPO 放大弱信号"
    else:
        diagnosis['cold_start'] = 'CANNOT_START'
        cold_start_msg = "需要 Curriculum Learning 或 BC Warmup"

    # Ideal State 诊断
    if ordering_correct and z_gt_vs_random_oracle > 3:
        diagnosis['ideal_state'] = 'VALID'
        ideal_state_msg = "Reward 函数有效"
    elif ordering_correct:
        diagnosis['ideal_state'] = 'WEAK_VALID'
        ideal_state_msg = "Reward 函数基本有效但信号弱"
    else:
        diagnosis['ideal_state'] = 'INVALID'
        ideal_state_msg = "Reward 函数可能有问题"

    print(f"\n1. Cold Start 诊断: {diagnosis['cold_start']}")
    print(f"   → {cold_start_msg}")

    print(f"\n2. Ideal State 诊断: {diagnosis['ideal_state']}")
    print(f"   → {ideal_state_msg}")

    # 综合建议
    print("\n" + "-" * 80)
    print("综合建议:")
    print("-" * 80)

    if diagnosis['cold_start'] == 'CAN_START' and diagnosis['ideal_state'] == 'VALID':
        print("\n  [最佳情况] 可以直接使用 REINFORCE 进行训练")
    elif diagnosis['cold_start'] == 'CANNOT_START':
        print("\n  [需要预热] 建议方案:")
        print("    1. 使用 Curriculum Learning: 从 num_chunks=2 开始，逐步增加")
        print("    2. 或: 先用 BC (Behavior Cloning) 预训练 Agent")
        print("    3. 或: 使用 GRPO 替代 REINFORCE（更好的信号放大）")
    elif diagnosis['cold_start'] == 'WEAK_SIGNAL':
        print("\n  [信号较弱] 建议方案:")
        print("    1. 使用 GRPO 替代 REINFORCE")
        print("    2. 增加 exploration (温度参数)")
        print("    3. 考虑 Curriculum Learning")

    if diagnosis['ideal_state'] != 'VALID':
        print("\n  [Reward 问题] 即使模型训练后，Reward 信号也不够强")
        print("    建议检查:")
        print("    1. AR 数据生成过程是否有足够强的时序依赖")
        print("    2. 是否需要更复杂的 Reward 函数")

    print("\n" + "=" * 80)
    print("    诊断完成")
    print("=" * 80)

    return {
        'stats': stats,
        'z_scores': z_scores,
        'diagnosis': diagnosis
    }


def main():
    parser = argparse.ArgumentParser(
        description='诊断 Contrastive Gap Reward 的敏感性 (Cold Start 问题)'
    )
    parser.add_argument('--seq_length', type=int, default=64,
                        help='序列长度 (默认: 64)')
    parser.add_argument('--num_chunks', type=int, default=8,
                        help='Chunk 数量 (默认: 8)')
    parser.add_argument('--num_batches', type=int, default=50,
                        help='测试 batch 数量 (默认: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch 大小 (默认: 32)')
    parser.add_argument('--oracle_steps', type=int, default=1000,
                        help='Oracle 模型训练步数 (默认: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 设备
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] 使用设备: {device}")

    # 创建数据生成器
    print("[INFO] 创建数据生成器...")
    generator = ContinuousDenseARGenerator(
        vector_dim=config.vector_dim,
        dependency_window=config.dependency_window,
        num_matrices=getattr(config, 'num_matrices', None),
        seed=args.seed,
        fixed_matrices_path=config.fixed_matrices_path
    )

    # 运行诊断
    results = diagnose_gap_sensitivity(
        generator=generator,
        device=device,
        seq_length=args.seq_length,
        num_chunks=args.num_chunks,
        num_test_batches=args.num_batches,
        batch_size=args.batch_size,
        oracle_train_steps=args.oracle_steps
    )

    # 打印最终结论
    print(f"\n最终诊断:")
    print(f"  Cold Start: {results['diagnosis']['cold_start']}")
    print(f"  Ideal State: {results['diagnosis']['ideal_state']}")


if __name__ == '__main__':
    main()
