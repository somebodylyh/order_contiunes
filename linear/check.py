import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add project root to path for imports
# Get the project root (parent of linear_rotation_exp directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from linear_rotation_exp.continuous_model import ContinuousTransformer, ContinuousTransformerConfig
from linear_rotation_exp.set_to_seq_agent import SetToSeqAgent
from linear_rotation_exp.continuous_data_generator import ContinuousDenseARGenerator
import linear_rotation_exp.config_continuous_rotation as config

def run_sanity_check():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. 加载配置和模型
    print("\n[1] Loading Model and Agent...")
    # 强制覆盖为测试时的配置 (L=16)
    config.seq_length = 16
    config.dependency_window = -1 # Full History
    
    # Create model config
    model_config = ContinuousTransformerConfig(
        vector_dim=config.vector_dim,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        dropout=config.dropout,
        bias=config.bias
    )
    model = ContinuousTransformer(model_config).to(device)
    
    agent = SetToSeqAgent(
        vector_dim=config.vector_dim,
        d_model=config.agent_d_model,
        encoder_layers=config.agent_encoder_layers,
        encoder_heads=config.agent_encoder_heads,
        decoder_layers=config.agent_decoder_layers,
        decoder_heads=config.agent_decoder_heads,
        max_len=config.seq_length,
        dropout=config.dropout
    ).to(device)

    # 加载你刚刚训练好的权重
    checkpoint_path = '/home/admin/lyuyuhuan/AO-GPT-MDM/linear_rotation_exp/checkpoints_continuous/best_model.pt'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        agent.load_state_dict(checkpoint['agent_state_dict'])
        print("✅ Checkpoint loaded successfully.")
    except FileNotFoundError:
        print("❌ Checkpoint not found! Please run training first.")
        return

    model.eval()
    agent.eval()

    # 2. 准备数据生成器 (用于生成真实的物理数据)
    generator = ContinuousDenseARGenerator(
        vector_dim=config.vector_dim,
        dependency_window=config.dependency_window,
        num_matrices=config.num_matrices,
        seed=config.seed,
        fixed_matrices_path=config.fixed_matrices_path
    )

    batch_size = 100
    
    # ==========================================
    # 测试 A: 真实的物理数据 (Real Physics)
    # ==========================================
    print("\n[Test A] Real Physics Data (Valid Causality)")
    # 生成正常的、有因果关系的数据
    result = generator.generate_sequence(
        length=config.seq_length,
        init_mode='negative_first',  # 用 OOD 数据测
        batch_size=batch_size
    )
    real_seq = result['vectors'].to(device)  # [B, L, D]
    # 打乱
    indices = torch.randperm(config.seq_length).repeat(batch_size, 1).to(device)
    real_shuffled = torch.gather(real_seq, 1, indices.unsqueeze(-1).expand(-1, -1, config.vector_dim))
    
    # Agent 排序 -> Model 预测
    with torch.no_grad():
        logits, _ = agent(real_shuffled)
        permutation = logits.argmax(dim=-1) # 取概率最大的顺序
        
        # 根据 Agent 的建议重排
        perm_expanded = permutation.unsqueeze(-1).expand(-1, -1, config.vector_dim)
        ordered_seq = torch.gather(real_shuffled, 1, perm_expanded)
        
        # Model 预测
        preds = model(ordered_seq)
        # 计算 MSE (只看 t=0 到 L-2 预测 t=1 到 L-1)
        targets = ordered_seq[:, 1:, :]
        preds_truncated = preds[:, :-1, :]
        mse_real = F.mse_loss(preds_truncated, targets).item()
        
    print(f"   -> Model MSE on Real Data: {mse_real:.6f} (Should be very low, e.g., < 0.02)")

    # ==========================================
    # 测试 B: 球面随机噪声 (Spherical Noise)
    # ==========================================
    # 这是一个"陷阱"。数据分布和真实数据一模一样（都在单位球面上），
    # 唯一的区别是：它们之间没有 W 的因果联系。
    print("\n[Test B] Spherical Random Noise (Broken Causality)")
    
    # 生成高斯噪声并归一化 -> 得到单位球面上的均匀分布
    noise = torch.randn(batch_size, config.seq_length, config.vector_dim).to(device)
    fake_seq = F.normalize(noise, p=2, dim=-1) # [B, L, D]
    
    with torch.no_grad():
        # 让 Agent 强行去排序这个噪声
        logits, _ = agent(fake_seq)
        permutation = logits.argmax(dim=-1)
        
        perm_expanded = permutation.unsqueeze(-1).expand(-1, -1, config.vector_dim)
        ordered_fake = torch.gather(fake_seq, 1, perm_expanded)
        
        # Model 强行预测
        preds = model(ordered_fake)
        targets = ordered_fake[:, 1:, :]
        preds_truncated = preds[:, :-1, :]
        mse_fake = F.mse_loss(preds_truncated, targets).item()

    print(f"   -> Model MSE on Fake Data: {mse_fake:.6f} (Should be HIGH, e.g., > 0.5)")

    # ==========================================
    # 3. 结果判定
    # ==========================================
    print("\n" + "="*30)
    print("📢 SANITY CHECK RESULT")
    print("="*30)
    
    ratio = mse_fake / mse_real
    print(f"Ratio (Fake MSE / Real MSE): {ratio:.2f}")
    
    if mse_real < 0.05 and mse_fake > 0.5:
        print("\n✅ PASS! The model has learned REAL PHYSICS.")
        print("   It performs well on causal data but fails on random noise.")
        print("   This confirms no statistical shortcuts were used.")
    elif mse_fake < 0.1:
        print("\n❌ FAIL! The model is cheating.")
        print("   It can predict random noise too well, meaning it's ignoring the causal link.")
    else:
        print("\n⚠️  INCONCLUSIVE. Check the values manually.")

if __name__ == "__main__":
    run_sanity_check()