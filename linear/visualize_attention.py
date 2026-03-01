"""
Visualize Attention Maps for Continuous Transformer

This script extracts and visualizes attention weights from the trained
ContinuousTransformer model to inspect what "neural physics" it has learned.

Expected: For a Dense AR process with dependency window K=6, we should see
attention concentrated on the K previous positions (a diagonal band pattern).
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional

# Import from local modules
from linear_rotation_exp import config_continuous_rotation as config
from linear_rotation_exp.continuous_model import ContinuousTransformer, ContinuousTransformerConfig
from linear_rotation_exp.continuous_data_generator import ContinuousDenseARGenerator


class AttentionExtractor:
    """
    Non-invasive attention weight extractor using PyTorch forward hooks.

    Registers hooks on CausalSelfAttention layers to capture attention weights
    during the forward pass without modifying the original model code.
    """

    def __init__(self, model: ContinuousTransformer):
        self.model = model
        self.attention_weights = {}
        self.hooks = []

    def _create_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # We need to capture attention weights during the forward pass
            # Since the standard forward only returns output, we'll re-compute
            # attention weights from the input
            x = input[0]  # [B, T, C]
            B, T, C = x.size()

            # Get attention parameters
            n_head = module.n_head
            head_dim = module.head_dim

            # Compute Q, K, V
            qkv = module.c_attn(x)
            q, k, v = qkv.split(module.n_embd, dim=2)

            # Reshape for multi-head attention
            k = k.view(B, T, n_head, head_dim).transpose(1, 2)
            q = q.view(B, T, n_head, head_dim).transpose(1, 2)

            # Compute attention scores
            att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(head_dim))

            # Apply causal mask
            causal_mask = module.causal_mask[:, :, :T, :T]
            att = att.masked_fill(causal_mask == 0, float('-inf'))

            # Softmax to get attention weights
            att = F.softmax(att, dim=-1)

            # Store attention weights: [B, n_head, T, T]
            self.attention_weights[f'layer_{layer_idx}'] = att.detach().cpu()

        return hook_fn

    def register_hooks(self, layer_indices: Optional[list] = None):
        """
        Register forward hooks on specified attention layers.

        Args:
            layer_indices: List of layer indices to hook. If None, hooks all layers.
        """
        self.clear_hooks()

        if layer_indices is None:
            layer_indices = list(range(len(self.model.blocks)))

        for idx in layer_indices:
            block = self.model.blocks[idx]
            attn_module = block.attn
            hook = attn_module.register_forward_hook(self._create_hook(idx))
            self.hooks.append(hook)

        print(f"[INFO] Registered hooks on layers: {layer_indices}")

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_weights = {}

    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Return captured attention weights."""
        return self.attention_weights


def load_model(checkpoint_path: str, device: str = 'cuda') -> ContinuousTransformer:
    """
    Load the trained ContinuousTransformer model.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    print(f"[INFO] Loading model from {checkpoint_path}")

    # Create model config from config file
    model_config = ContinuousTransformerConfig(
        vector_dim=config.vector_dim,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        dropout=config.dropout,
        bias=config.bias
    )

    # Initialize model
    model = ContinuousTransformer(model_config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded from training checkpoint (iter: {checkpoint.get('iter', 'N/A')})")
    else:
        model.load_state_dict(checkpoint)
        print("[INFO] Loaded from state dict")

    model.to(device)
    model.eval()

    return model


def generate_ground_truth_sequence(device: str = 'cuda') -> torch.Tensor:
    """
    Generate a single ground truth sequence in correct order.

    Returns:
        vectors: [1, L, D] tensor of the sequence
    """
    print(f"[INFO] Generating ground truth sequence...")
    print(f"       Vector dim (D): {config.vector_dim}")
    print(f"       Sequence length (L): {config.seq_length}")
    print(f"       Dependency window (K): {config.dependency_window}")

    generator = ContinuousDenseARGenerator(
        vector_dim=config.vector_dim,
        dependency_window=config.dependency_window,
        seed=42,
        fixed_matrices_path=config.fixed_matrices_path
    )

    result = generator.generate_sequence(
        length=config.seq_length,
        init_mode='positive_first',
        batch_size=1
    )

    vectors = result['vectors'].to(device)  # [1, L, D]

    # Verify the sequence follows AR dynamics
    is_valid, max_error = generator.verify_ar_dynamics(vectors.cpu())
    print(f"[INFO] AR dynamics verified: {is_valid}, max error: {max_error:.2e}")

    return vectors


def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    layer_name: str,
    save_path: str,
    dependency_window: int = 6
):
    """
    Plot attention heatmap with visual guides for expected dependency structure.

    Args:
        attention_weights: [B, n_head, L, L] attention tensor
        layer_name: Name of the layer (for title)
        save_path: Path to save the figure
        dependency_window: K value for visual guide (-1 means full history)
    """
    # Average across batch and heads: [L, L]
    avg_attention = attention_weights.mean(dim=(0, 1)).numpy()

    L = avg_attention.shape[0]
    full_history = (dependency_window == -1)
    K_label = "Full" if full_history else str(dependency_window)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap with high-contrast colormap
    sns.heatmap(
        avg_attention,
        ax=ax,
        cmap='Blues',
        vmin=0,
        vmax=avg_attention.max(),
        square=True,
        cbar_kws={'label': 'Attention Weight', 'shrink': 0.8}
    )

    # Draw diagonal line (self-attention / causal boundary)
    ax.plot([0, L], [0, L], color='red', linewidth=1.5,
            linestyle='--', alpha=0.8, label='Causal boundary (t=t)')

    # Draw K-offset diagonal line only for fixed window
    if not full_history and dependency_window < L:
        ax.plot([0, L-dependency_window], [dependency_window, L],
               color='cyan', linewidth=1.5, linestyle='--', alpha=0.8,
               label=f'K={dependency_window} offset diagonal')

    # Labels and title
    ax.set_xlabel('Key Position (Source)', fontsize=12)
    ax.set_ylabel('Query Position (Target)', fontsize=12)
    ax.set_title(
        f'Attention Map - {layer_name}\n'
        f'(L={L}, K={K_label}, averaged over all heads)',
        fontsize=14
    )

    # Set tick labels
    if L <= 32:
        ax.set_xticks(np.arange(L) + 0.5)
        ax.set_yticks(np.arange(L) + 0.5)
        ax.set_xticklabels(range(L), fontsize=7)
        ax.set_yticklabels(range(L), fontsize=7)
    else:
        step = max(1, L // 16)
        ax.set_xticks(np.arange(0, L, step) + 0.5)
        ax.set_yticks(np.arange(0, L, step) + 0.5)
        ax.set_xticklabels(range(0, L, step), fontsize=7)
        ax.set_yticklabels(range(0, L, step), fontsize=7)

    # Add legend
    ax.legend(loc='upper left', fontsize=10)

    # Add annotation
    if full_history:
        note = 'Full history mode: each position\nattends to ALL previous positions'
    else:
        note = (f'Expected: For row t, high attention\n'
                f'on columns max(0, t-{dependency_window}) to t-1')
    ax.text(
        0.02, 0.98, note,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved attention heatmap to {save_path}")
    plt.close()


def plot_per_head_attention(
    attention_weights: torch.Tensor,
    layer_name: str,
    save_path: str,
    dependency_window: int = 6
):
    """
    Plot attention heatmap for each head separately.

    Args:
        attention_weights: [B, n_head, L, L] attention tensor
        layer_name: Name of the layer
        save_path: Path to save the figure
        dependency_window: K value (-1 means full history)
    """
    # Get attention from first batch item: [n_head, L, L]
    attention = attention_weights[0].numpy()
    n_head = attention.shape[0]
    L = attention.shape[1]
    full_history = (dependency_window == -1)
    K_label = "Full" if full_history else str(dependency_window)

    # Create subplot grid
    n_cols = 4
    n_rows = (n_head + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_head > 1 else [axes]

    for head_idx in range(n_head):
        ax = axes[head_idx]
        head_attention = attention[head_idx]

        sns.heatmap(
            head_attention,
            ax=ax,
            cmap='Blues',
            vmin=0,
            vmax=head_attention.max(),
            square=True,
            cbar_kws={'shrink': 0.6}
        )

        # Draw causal boundary
        ax.plot([0, L], [0, L], color='red', linewidth=1,
                linestyle='--', alpha=0.7)

        # Draw K-offset diagonal only for fixed window
        if not full_history and dependency_window < L:
            ax.plot([0, L-dependency_window], [dependency_window, L],
                   color='cyan', linewidth=1.5, linestyle='--', alpha=0.7)

        ax.set_xlabel('Key Position', fontsize=9)
        ax.set_ylabel('Query Position', fontsize=9)
        ax.set_title(f'Head {head_idx}', fontsize=11, fontweight='bold')
        step = max(1, L // 8)
        ax.set_xticks(np.arange(0, L, step) + 0.5)
        ax.set_yticks(np.arange(0, L, step) + 0.5)
        ax.set_xticklabels(range(0, L, step), fontsize=7)
        ax.set_yticklabels(range(0, L, step), fontsize=7)

    # Hide unused subplots
    for idx in range(n_head, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f'{layer_name} - Per-Head Attention (L={L}, K={K_label})',
                 fontsize=14, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved per-head attention to {save_path}")
    plt.close()


def analyze_attention_pattern(
    attention_weights: torch.Tensor,
    dependency_window: int = 6
) -> Dict[str, float]:
    """
    Analyze attention pattern to quantify if it matches expected AR dynamics.

    Args:
        attention_weights: [B, n_head, L, L] attention tensor
        dependency_window: K value (-1 means full history)

    Returns:
        Dictionary of analysis metrics
    """
    # Average across batch and heads: [L, L]
    avg_attention = attention_weights.mean(dim=(0, 1)).numpy()
    L = avg_attention.shape[0]

    # Effective window: -1 means all previous positions
    full_history = (dependency_window == -1)
    K = L if full_history else dependency_window

    # Metrics
    metrics = {}
    metrics['full_history'] = full_history

    # 1. Attention mass in dependency window
    # For each row t, sum attention in columns [t-K, t-1]
    in_window_mass = 0
    out_window_mass = 0
    valid_rows = 0

    for t in range(1, L):  # Skip t=0 (no valid keys)
        row = avg_attention[t, :t]  # Only consider causal positions
        if full_history:
            window_start = 0
        else:
            window_start = max(0, t - K)

        in_window = row[window_start:t].sum()
        out_window = row[:window_start].sum() if window_start > 0 else 0

        in_window_mass += in_window
        out_window_mass += out_window
        valid_rows += 1

    total_mass = in_window_mass + out_window_mass
    metrics['in_window_ratio'] = in_window_mass / total_mass if total_mass > 0 else 0
    metrics['in_window_mass'] = in_window_mass / valid_rows
    metrics['out_window_mass'] = out_window_mass / valid_rows

    # 2. Peak attention distance from query position
    peak_distances = []
    start_t = 1 if full_history else max(1, K)
    for t in range(start_t, L):
        row = avg_attention[t, :t]
        if len(row) == 0:
            continue
        peak_pos = np.argmax(row)
        distance = t - peak_pos
        peak_distances.append(distance)

    metrics['avg_peak_distance'] = np.mean(peak_distances) if peak_distances else 0
    metrics['std_peak_distance'] = np.std(peak_distances) if peak_distances else 0

    # 3. Attention entropy (lower = more focused)
    entropies = []
    for t in range(1, L):
        row = avg_attention[t, :t]
        if len(row) == 0:
            continue
        row_sum = row.sum()
        if row_sum > 0:
            row = row / row_sum  # Normalize
        entropy = -np.sum(row * np.log(row + 1e-10))
        entropies.append(entropy)

    metrics['avg_entropy'] = np.mean(entropies) if entropies else 0

    # 4. Attention concentration per row (top-k coverage)
    top1_coverages = []
    top3_coverages = []
    for t in range(1, L):
        row = avg_attention[t, :t]
        if len(row) == 0:
            continue
        sorted_row = np.sort(row)[::-1]
        row_sum = row.sum()
        if row_sum > 0:
            top1_coverages.append(sorted_row[0] / row_sum)
            top3_coverages.append(sorted_row[:min(3, len(sorted_row))].sum() / row_sum)

    metrics['avg_top1_coverage'] = np.mean(top1_coverages) if top1_coverages else 0
    metrics['avg_top3_coverage'] = np.mean(top3_coverages) if top3_coverages else 0

    return metrics


def main():
    """Main function to visualize attention maps."""
    print("=" * 70)
    print("Attention Map Visualization for Continuous Transformer")
    print("=" * 70)

    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[CONFIG] Device: {device}")
    print(f"[CONFIG] Vector dim (D): {config.vector_dim}")
    print(f"[CONFIG] Sequence length (L): {config.seq_length}")
    print(f"[CONFIG] Dependency window (K): {config.dependency_window}")
    print(f"[CONFIG] Model layers: {config.n_layer}")
    print(f"[CONFIG] Attention heads: {config.n_head}")

    # Find checkpoint
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoints_continuous')
    checkpoint_path = os.path.join(checkpoint_dir, 'final_model.pt')

    if not os.path.exists(checkpoint_path):
        # Try to find the latest checkpoint
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_')])
        if checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
        else:
            raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")

    print(f"\n[INFO] Using checkpoint: {checkpoint_path}")

    # Load model
    model = load_model(checkpoint_path, device)

    # Create attention extractor
    extractor = AttentionExtractor(model)

    # Register hooks on the LAST layer only (as specified in requirements)
    last_layer_idx = config.n_layer - 1
    extractor.register_hooks([last_layer_idx])

    # Generate ground truth sequence
    vectors = generate_ground_truth_sequence(device)

    # Run inference to capture attention
    print("\n[INFO] Running model inference...")
    with torch.no_grad():
        predictions, _ = model(vectors)

    # Get attention weights
    attention_weights = extractor.get_attention_weights()

    # Output directory
    output_dir = os.path.dirname(__file__)

    # Plot attention heatmap for last layer
    layer_key = f'layer_{last_layer_idx}'
    if layer_key in attention_weights:
        attn = attention_weights[layer_key]
        print(f"\n[INFO] Attention shape: {attn.shape}")
        print(f"       (Batch, Heads, Query, Key)")

        # Save averaged attention map
        output_path = os.path.join(
            output_dir,
            f'attention_map_L{config.seq_length}_K{config.dependency_window}.png'
        )
        plot_attention_heatmap(
            attn,
            f'Layer {last_layer_idx} (Last Layer)',
            output_path,
            config.dependency_window
        )

        # Save per-head attention maps
        output_path_heads = os.path.join(
            output_dir,
            f'attention_map_L{config.seq_length}_K{config.dependency_window}_per_head.png'
        )
        plot_per_head_attention(
            attn,
            f'Layer {last_layer_idx}',
            output_path_heads,
            config.dependency_window
        )

        # Analyze attention pattern
        print("\n" + "=" * 50)
        print("Attention Pattern Analysis")
        print("=" * 50)
        metrics = analyze_attention_pattern(attn, config.dependency_window)

        K_label = "Full History" if metrics.get('full_history') else f"K={config.dependency_window}"
        print(f"\nDependency Window ({K_label}) Analysis:")
        print(f"  - Attention in window ratio: {metrics['in_window_ratio']:.4f}")
        if metrics.get('full_history'):
            print(f"    (Full history: all causal positions are in-window)")
        else:
            print(f"    (Higher is better, indicates model focuses on last K positions)")
        print(f"  - Avg attention mass in window: {metrics['in_window_mass']:.4f}")
        print(f"  - Avg attention mass outside window: {metrics['out_window_mass']:.4f}")
        print(f"\nPeak Attention Analysis:")
        print(f"  - Avg peak distance from query: {metrics['avg_peak_distance']:.2f}")
        if metrics.get('full_history'):
            print(f"    (Full history: peak can be at any previous position)")
        else:
            print(f"    (Should be around 1-{config.dependency_window} if learned correctly)")
        print(f"  - Std of peak distance: {metrics['std_peak_distance']:.2f}")
        print(f"\nAttention Focus:")
        print(f"  - Avg attention entropy: {metrics['avg_entropy']:.4f}")
        print(f"    (Lower entropy = more focused attention)")
        print(f"  - Avg top-1 coverage: {metrics['avg_top1_coverage']:.4f}")
        print(f"    (Fraction of attention on the single most-attended position)")
        print(f"  - Avg top-3 coverage: {metrics['avg_top3_coverage']:.4f}")
        print(f"    (Fraction of attention on the top 3 most-attended positions)")

    else:
        print(f"[ERROR] No attention weights captured for {layer_key}")

    # Also visualize all layers for comparison
    print("\n[INFO] Generating attention maps for all layers...")
    extractor.register_hooks(None)  # All layers

    with torch.no_grad():
        predictions, _ = model(vectors)

    all_attention = extractor.get_attention_weights()

    # Create a comparison figure for all layers
    n_layers = len(all_attention)
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_layers > 1 else [axes]

    K = config.dependency_window
    full_history = (K == -1)
    K_label = "Full" if full_history else str(K)

    for idx, (layer_key, attn) in enumerate(all_attention.items()):
        if idx >= len(axes):
            break
        avg_attn = attn.mean(dim=(0, 1)).numpy()
        L = avg_attn.shape[0]

        ax = axes[idx]
        sns.heatmap(avg_attn, ax=ax, cmap='mako', square=True,
                   cbar_kws={'shrink': 0.6})

        # Draw causal boundary
        ax.plot([0, L], [0, L], color='red', linewidth=1,
                linestyle='--', alpha=0.7)

        # Draw K-offset diagonal only for fixed window
        if not full_history and K < L:
            ax.plot([0, L-K], [K, L], color='cyan', linewidth=1.5,
                   linestyle='--', alpha=0.7)

        ax.set_xlabel('Key Position', fontsize=9)
        ax.set_ylabel('Query Position', fontsize=9)
        ax.set_title(f'{layer_key.replace("_", " ").title()} (Avg over heads)',
                     fontsize=10, fontweight='bold')
        step = max(1, L // 8)
        ax.set_xticks(np.arange(0, L, step) + 0.5)
        ax.set_yticks(np.arange(0, L, step) + 0.5)
        ax.set_xticklabels(range(0, L, step), fontsize=7)
        ax.set_yticklabels(range(0, L, step), fontsize=7)

    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        f'Attention Maps Across All Layers\n'
        f'(L={config.seq_length}, K={K_label})',
        fontsize=14, y=1.02
    )
    plt.tight_layout()

    all_layers_path = os.path.join(
        output_dir,
        f'attention_map_L{config.seq_length}_K{config.dependency_window}_all_layers.png'
    )
    plt.savefig(all_layers_path, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved all-layers attention map to {all_layers_path}")
    plt.close()

    # Cleanup
    extractor.clear_hooks()

    print("\n" + "=" * 70)
    print("Visualization Complete!")
    print("=" * 70)
    print(f"\nGenerated files:")
    print(f"  1. attention_map_L{config.seq_length}_K{config.dependency_window}.png")
    print(f"     - Averaged attention map for the last layer")
    print(f"  2. attention_map_L{config.seq_length}_K{config.dependency_window}_per_head.png")
    print(f"     - Per-head attention maps for the last layer")
    print(f"  3. attention_map_L{config.seq_length}_K{config.dependency_window}_all_layers.png")
    print(f"     - Comparison across all {config.n_layer} layers")
    print(f"\nExpected pattern:")
    if config.dependency_window == -1:
        print(f"  - Full history mode: each position attends to ALL previous positions")
        print(f"  - Expected: a lower-triangular pattern with learned structure")
        print(f"  - Green dashed line marks the causal boundary")
    else:
        print(f"  - For each row t (query position), attention should be concentrated")
        print(f"    on columns t-{config.dependency_window} to t-1 (last K positions)")
        print(f"  - This appears as a diagonal 'band' or 'glow' pattern")
        print(f"  - The cyan dashed line marks the K-offset diagonal (lower bound)")


if __name__ == '__main__':
    main()
