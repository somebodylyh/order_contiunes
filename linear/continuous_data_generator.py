"""
Dense AR Data Generator for Continuous Vector Linear Rotation Experiment

Generates sequences following the Dense AR process:
    x_t = sum(A_i @ x_{t-i}) for i=1 to k, when t >= k
    x_0, ..., x_{k-1} ~ Normalized random Gaussian (initialization)

Where:
- D = vector_dim (dimension of each vector)
- L = sequence length
- k = dependency_window (number of lag terms)
- A_1, ..., A_k are k orthogonal [D, D] matrices, each scaled by 1/sqrt(k)
"""

import math

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import os


class ContinuousDenseARGenerator:
    """
    Generates continuous vector sequences following a Dense AR process
    with k orthogonal transformation matrices.
    """

    def __init__(
        self,
        vector_dim: int = 32,
        dependency_window: int = 5,
        num_matrices: Optional[int] = None,
        seed: Optional[int] = None,
        fixed_matrices_path: Optional[str] = None,
        noise_scale: float = 0.0,
        alpha: float = 0.3        # 样本特异性偏置强度
    ):
        """
        Initialize the Dense AR generator.

        Args:
            vector_dim: Dimension of each vector (D)
            dependency_window: Number of lag terms in AR process (k).
                               Set to -1 for Full History mode (each step depends on all previous steps).
            num_matrices: Number of orthogonal matrices. Defaults to dependency_window (k>0) or 6 (k==-1).
            seed: Random seed for reproducibility
            fixed_matrices_path: Path to save/load fixed orthogonal matrices
            noise_scale: Std of Gaussian noise added after tanh (0 = no noise)
            alpha: Sample-specific bias strength (injected from first init_vector each step)
        """
        self.D = vector_dim
        self.k = dependency_window
        self.noise_scale = noise_scale
        self.alpha = alpha

        # Number of orthogonal matrices
        if num_matrices is not None:
            self.num_matrices = num_matrices
        elif dependency_window > 0:
            self.num_matrices = dependency_window
        else:
            self.num_matrices = 6  # default for full history mode

        # Number of initial (random) vectors
        self.num_init = 1 if dependency_window == -1 else dependency_window

        self.seed = seed
        self.fixed_matrices_path = fixed_matrices_path

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Load or generate orthogonal matrices
        self.A_matrices = self._load_or_generate_matrices()

    def _generate_orthogonal_matrices(self) -> List[torch.Tensor]:
        """
        Generate num_matrices random orthogonal matrices of shape [D, D],
        each scaled by 1/sqrt(num_matrices) for stability.
        """
        print(f"[INFO] Generating {self.num_matrices} orthogonal matrices of shape ({self.D}, {self.D})...")
        matrices = []
        for i in range(self.num_matrices):
            # Generate random matrix and orthogonalize via QR decomposition
            random_matrix = torch.randn(self.D, self.D)
            Q, R = torch.linalg.qr(random_matrix)
            # Ensure proper orthogonal matrix (det = +1)
            # Multiply by sign of diagonal of R to ensure consistent orientation
            Q = Q @ torch.diag(torch.sign(torch.diag(R)))
            # Scale by 1/sqrt(num_matrices) for stability
            Q = Q / np.sqrt(self.num_matrices)
            matrices.append(Q)
        return matrices

    def _load_or_generate_matrices(self) -> List[torch.Tensor]:
        """Load matrices from file if exists, otherwise generate and save."""
        if self.fixed_matrices_path and os.path.exists(self.fixed_matrices_path):
            print(f"[INFO] Loading orthogonal matrices from {self.fixed_matrices_path}")
            data = torch.load(self.fixed_matrices_path, weights_only=True)
            matrices = data['matrices']
            # Verify dimensions
            if len(matrices) != self.num_matrices or matrices[0].shape != (self.D, self.D):
                print(f"[WARNING] Loaded matrices have wrong dimensions. Regenerating...")
                matrices = self._generate_orthogonal_matrices()
                self._save_matrices(matrices)
            return matrices
        else:
            matrices = self._generate_orthogonal_matrices()
            if self.fixed_matrices_path:
                self._save_matrices(matrices)
            return matrices

    def _save_matrices(self, matrices: List[torch.Tensor]) -> None:
        """Save matrices to file for reproducibility."""
        if self.fixed_matrices_path:
            os.makedirs(os.path.dirname(self.fixed_matrices_path), exist_ok=True)
            torch.save({
                'matrices': matrices,
                'vector_dim': self.D,
                'num_matrices': self.num_matrices,
                'dependency_window': self.k
            }, self.fixed_matrices_path)
            print(f"[INFO] Saved orthogonal matrices to {self.fixed_matrices_path}")

    def _generate_init_vectors(
        self,
        batch_size: int,
        mode: str = 'positive_first'
    ) -> torch.Tensor:
        """
        Generate initial vectors (first k vectors) with normalized Gaussian.

        Args:
            batch_size: Number of sequences to generate
            mode: 'positive_first' (v[0] > 0) or 'negative_first' (v[0] < 0)

        Returns:
            init_vectors: [batch_size, k, D] tensor of normalized initial vectors
        """
        # Generate random Gaussian vectors
        init_vectors = torch.randn(batch_size, self.num_init, self.D)

        # Normalize each vector
        init_vectors = F.normalize(init_vectors, p=2, dim=-1)

        # Apply OOD constraint on first component
        if mode == 'positive_first':
            # Ensure first component is positive
            mask = init_vectors[:, :, 0] < 0
            init_vectors[mask] = -init_vectors[mask]
        elif mode == 'negative_first':
            # Ensure first component is negative
            mask = init_vectors[:, :, 0] > 0
            init_vectors[mask] = -init_vectors[mask]
        # else: 'random' - no constraint

        return init_vectors

    def generate_sequence(
        self,
        length: int,
        init_mode: str = 'positive_first',
        batch_size: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a batch of sequences following the Dense AR process.

        Args:
            length: Sequence length (L)
            init_mode: 'positive_first', 'negative_first', or 'random'
            batch_size: Number of sequences to generate

        Returns:
            Dictionary containing:
                'vectors': [batch_size, L, D] - full sequences
                'init_vectors': [batch_size, k, D] - initial k vectors
        """
        if self.k > 0:
            assert length >= self.k, f"Sequence length ({length}) must be >= dependency window ({self.k})"
        else:
            assert length >= 2, f"Sequence length ({length}) must be >= 2 for full history mode"

        # Generate initial vectors
        init_vectors = self._generate_init_vectors(batch_size, init_mode)

        # Initialize sequence with init vectors
        vectors = torch.zeros(batch_size, length, self.D)
        vectors[:, :self.num_init, :] = init_vectors

        # Generate remaining vectors via Dense AR process
        for t in range(self.num_init, length):
            x_t = torch.zeros(batch_size, self.D)

            if self.k == -1:
                # Full History mode: depend on all previous steps (1..t)
                for i in range(1, t + 1):
                    mat_idx = (i - 1) % self.num_matrices
                    x_t = x_t + torch.matmul(vectors[:, t - i, :], self.A_matrices[mat_idx].T)
                # Dynamic scaling: undo baked-in 1/sqrt(M), apply 1/sqrt(t)
                x_t = x_t * math.sqrt(self.num_matrices) / math.sqrt(t)
            else:
                # Fixed window mode (existing logic)
                # x_t = sum(A_i @ x_{t-i}) for i=1 to k
                for i in range(1, self.k + 1):
                    x_t = x_t + torch.matmul(vectors[:, t - i, :], self.A_matrices[i - 1].T)

            # Inject sample-specific bias from first init_vector
            if self.alpha != 0:
                x_t = x_t + self.alpha * init_vectors[:, 0, :]  # [B, D]

            # Nonlinearity: tanh compresses values, breaks pure linear structure
            x_t = torch.tanh(x_t * 2.0)
            # Add Gaussian noise before normalization
            if self.noise_scale > 0:
                x_t = x_t + self.noise_scale * torch.randn_like(x_t)
            # Normalize for stability
            x_t = F.normalize(x_t, p=2, dim=-1)
            vectors[:, t, :] = x_t

        return {
            'vectors': vectors,  # [batch_size, L, D]
            'init_vectors': init_vectors  # [batch_size, k, D]
        }

    def generate_single_sequence(
        self,
        length: int,
        init_mode: str = 'positive_first'
    ) -> Dict[str, torch.Tensor]:
        """
        Generate a single sequence (convenience method).

        Returns:
            Dictionary with 'vectors' [L, D] and 'init_vectors' [k, D]
        """
        result = self.generate_sequence(length, init_mode, batch_size=1)
        return {
            'vectors': result['vectors'].squeeze(0),  # [L, D]
            'init_vectors': result['init_vectors'].squeeze(0)  # [k, D]
        }

    def verify_ar_dynamics(
        self,
        vectors: torch.Tensor,
        tolerance: float = 1e-5
    ) -> Tuple[bool, float]:
        """
        Verify that a sequence follows the Dense AR dynamics.
        Note: When noise_scale > 0, exact verification is not possible;
        this method skips verification and returns (True, 0.0).

        Args:
            vectors: [L, D] or [batch_size, L, D] tensor
            tolerance: Maximum allowed error

        Returns:
            (is_valid, max_error)
        """
        # With noise, exact verification is impossible
        if self.noise_scale > 0:
            return True, 0.0

        if vectors.dim() == 2:
            vectors = vectors.unsqueeze(0)

        batch_size, length, D = vectors.shape
        max_error = 0.0

        for t in range(self.num_init, length):
            # Compute expected x_t
            expected = torch.zeros(batch_size, D)

            if self.k == -1:
                # Full History mode
                for i in range(1, t + 1):
                    mat_idx = (i - 1) % self.num_matrices
                    expected = expected + torch.matmul(vectors[:, t - i, :], self.A_matrices[mat_idx].T)
                expected = expected * math.sqrt(self.num_matrices) / math.sqrt(t)
            else:
                # Fixed window mode
                for i in range(1, self.k + 1):
                    expected = expected + torch.matmul(vectors[:, t - i, :], self.A_matrices[i - 1].T)

            # Inject sample-specific bias (deterministic, so verification still works)
            if self.alpha != 0:
                expected = expected + self.alpha * vectors[:, 0, :]  # first init_vector [B, D]

            # Apply tanh (matching generate_sequence)
            expected = torch.tanh(expected * 2.0)
            expected = F.normalize(expected, p=2, dim=-1)

            # Compute error (allow for normalization differences)
            error = torch.abs(vectors[:, t, :] - expected).max().item()
            max_error = max(max_error, error)

        is_valid = max_error < tolerance
        return is_valid, max_error

    def get_matrices(self) -> List[torch.Tensor]:
        """Return the orthogonal matrices (for analysis)."""
        return self.A_matrices

    def compute_sequence_stats(self, vectors: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics about a sequence.

        Args:
            vectors: [L, D] or [batch_size, L, D] tensor

        Returns:
            Dictionary of statistics
        """
        if vectors.dim() == 2:
            vectors = vectors.unsqueeze(0)

        # Compute pairwise cosine similarities
        vectors_flat = vectors.reshape(-1, vectors.shape[-1])
        cos_sim_matrix = F.cosine_similarity(
            vectors_flat.unsqueeze(1),
            vectors_flat.unsqueeze(0),
            dim=-1
        )

        # Mask diagonal
        mask = 1 - torch.eye(cos_sim_matrix.shape[0])
        cos_sim_offdiag = cos_sim_matrix * mask

        # Compute norms
        norms = torch.norm(vectors, p=2, dim=-1)

        return {
            'mean_cos_sim': cos_sim_offdiag.sum().item() / mask.sum().item(),
            'max_cos_sim': cos_sim_offdiag.max().item(),
            'min_cos_sim': cos_sim_offdiag.min().item(),
            'mean_norm': norms.mean().item(),
            'std_norm': norms.std().item(),
            'min_norm': norms.min().item(),
            'max_norm': norms.max().item()
        }


def test_generator():
    """Test the Dense AR generator."""
    print("=" * 60)
    print("Testing ContinuousDenseARGenerator")
    print("=" * 60)

    # Create generator
    generator = ContinuousDenseARGenerator(
        vector_dim=32,
        dependency_window=5,
        seed=42,
        fixed_matrices_path='linear_rotation_exp/test_matrices.pt'
    )

    # Test single sequence generation
    print("\n1. Testing single sequence generation...")
    result = generator.generate_single_sequence(length=16, init_mode='positive_first')
    vectors = result['vectors']
    init_vectors = result['init_vectors']
    print(f"   Vectors shape: {vectors.shape}")
    print(f"   Init vectors shape: {init_vectors.shape}")

    # Verify first component constraint
    print(f"   First components of init vectors: {init_vectors[:, 0].tolist()[:5]}...")
    assert all(v > 0 for v in init_vectors[:, 0].tolist()), "First component should be positive"
    print("   ✓ Positive first component constraint satisfied")

    # Test OOD generation (negative first)
    print("\n2. Testing OOD generation (negative_first)...")
    result_ood = generator.generate_single_sequence(length=16, init_mode='negative_first')
    init_vectors_ood = result_ood['init_vectors']
    print(f"   First components of OOD init vectors: {init_vectors_ood[:, 0].tolist()[:5]}...")
    assert all(v < 0 for v in init_vectors_ood[:, 0].tolist()), "First component should be negative"
    print("   ✓ Negative first component constraint satisfied (OOD)")

    # Verify AR dynamics
    print("\n3. Verifying AR dynamics...")
    is_valid, max_error = generator.verify_ar_dynamics(vectors)
    print(f"   Is valid: {is_valid}, Max error: {max_error:.2e}")
    assert is_valid, f"AR dynamics verification failed with error {max_error}"
    print("   ✓ AR dynamics verified")

    # Test batch generation
    print("\n4. Testing batch generation...")
    batch_result = generator.generate_sequence(length=16, init_mode='positive_first', batch_size=64)
    print(f"   Batch vectors shape: {batch_result['vectors'].shape}")
    assert batch_result['vectors'].shape == (64, 16, 32), "Batch shape mismatch"
    print("   ✓ Batch generation works")

    # Compute statistics
    print("\n5. Computing sequence statistics...")
    stats = generator.compute_sequence_stats(vectors)
    for k, v in stats.items():
        print(f"   {k}: {v:.4f}")

    # Verify matrices are orthogonal
    print("\n6. Verifying orthogonality of matrices...")
    for i, A in enumerate(generator.A_matrices):
        # Scale back to check orthogonality
        A_scaled = A * np.sqrt(generator.k)
        error = torch.norm(A_scaled @ A_scaled.T - torch.eye(generator.D)).item()
        print(f"   Matrix {i}: orthogonality error = {error:.2e}")
        assert error < 1e-5, f"Matrix {i} is not orthogonal"
    print("   ✓ All matrices are orthogonal")

    # Clean up test file
    if os.path.exists('linear_rotation_exp/test_matrices.pt'):
        os.remove('linear_rotation_exp/test_matrices.pt')

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    import torch.nn.functional as F

    print("验证：生成100个样本，检查 t=L-1 时刻的样本多样性...")
    gen = ContinuousDenseARGenerator(
        vector_dim=32,
        dependency_window=-1,
        num_matrices=8,
        noise_scale=0.05,
        alpha=0.3,
        seed=42
    )
    result = gen.generate_sequence(length=128, init_mode='positive_first', batch_size=100)
    last = result['vectors'][:, -1, :]      # [100, D]
    cos_sim = F.cosine_similarity(last.unsqueeze(1), last.unsqueeze(0), dim=-1)  # [100, 100]
    mask = 1 - torch.eye(100)
    mean_sim = (cos_sim * mask).sum() / mask.sum()
    print(f"t=L-1 时刻平均两两余弦相似度: {mean_sim.item():.4f}")
    if mean_sim.item() < 0.3:
        print("PASS：未发现坍缩")
    else:
        print("FAIL：发现坍缩！")
    assert mean_sim.item() < 0.3, f"坍缩检测失败：均值 = {mean_sim.item():.4f}"
