"""
Generate and save a fixed orthogonal R matrix for consistent dynamics across all samples.

This script creates a single R matrix that will be used by all train/val/test samples,
ensuring the same physics (h_{t+1} = R @ h_t + x_t) applies everywhere.
"""

import torch
import os
from scipy.stats import ortho_group

# Vocabulary size (must match config)
V = 64

# Generate random orthogonal matrix using Haar measure
R = ortho_group.rvs(dim=V).astype('float32')
R = torch.tensor(R, dtype=torch.float32)

# Verify orthogonality
identity_check = torch.allclose(R.T @ R, torch.eye(V), atol=1e-5)
if not identity_check:
    raise ValueError("Generated matrix is not orthogonal!")

# Save to file
output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
output_path = os.path.join(output_dir, 'fixed_R.pt')

torch.save(R, output_path)
print(f"Saved fixed R matrix: {R.shape}")
print(f"Output path: {output_path}")
print(f"Orthogonality check: {'PASSED' if identity_check else 'FAILED'}")

# Print some statistics
eigenvalues = torch.linalg.eigvals(R)
print(f"Eigenvalue magnitudes (should all be ~1): min={eigenvalues.abs().min():.4f}, max={eigenvalues.abs().max():.4f}")
