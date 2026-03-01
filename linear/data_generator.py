"""
Linear Dynamical System Data Generator

Core component for generating sequences based on orthogonal matrix dynamics:
    h_{t+1} = R * h_t + x_t
    next_token = argmax(h_{t+1})

Where:
- R is a fixed orthogonal matrix (vocab_size × vocab_size)
- x_t is the one-hot encoding of token t
- History propagates through powers of R, forming long-range dependencies
"""

import os
import numpy as np
import torch
from collections import Counter
from scipy.stats import ortho_group
from typing import Dict, List, Tuple


class LinearDynamicalGenerator:
    """
    Generates sequences using linear orthogonal matrix dynamics.

    The system evolves as:
        h_0 = one_hot(start_token)
        h_{t+1} = R @ h_t + one_hot(x_t)
        x_{t+1} = argmax(h_{t+1})
    """

    def __init__(self, vocab_size: int = 64, hidden_dim: int = 64, ortho_mode: str = 'random',
                 seed: int = None, fixed_R_path: str = None):
        """
        Initialize the generator with an orthogonal transition matrix.

        Args:
            vocab_size: Vocabulary size V
            hidden_dim: Hidden dimension D (usually D = V)
            ortho_mode: 'random' (scipy.stats.ortho_group) or 'permutation'
            seed: Random seed for reproducibility
            fixed_R_path: Path to pre-generated fixed R matrix (overrides ortho_mode)
        """
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.ortho_mode = ortho_mode

        if seed is not None:
            np.random.seed(seed)

        # Load or generate orthogonal matrix R
        need_regenerate = False
        if fixed_R_path and os.path.exists(fixed_R_path):
            # Load FIXED R matrix (same for all samples)
            R_tensor = torch.load(fixed_R_path, weights_only=True)
            if R_tensor.shape != (vocab_size, vocab_size):
                print(f"[WARNING] Fixed R matrix shape {R_tensor.shape} doesn't match vocab_size {vocab_size}")
                print(f"[WARNING] Regenerating R matrix and saving to {fixed_R_path}")
                need_regenerate = True
            else:
                self.R = R_tensor.numpy()
        else:
            need_regenerate = True

        if need_regenerate:
            if ortho_mode == 'random':
                # Generate random orthogonal matrix using Haar measure
                print(f"[INFO] Generating new {vocab_size}x{vocab_size} orthogonal matrix...")
                self.R = ortho_group.rvs(dim=vocab_size).astype(np.float32)
            elif ortho_mode == 'permutation':
                # Generate permutation matrix (sparse orthogonal)
                perm = np.random.permutation(vocab_size)
                self.R = np.eye(vocab_size)[perm].astype(np.float32)
            else:
                raise ValueError(f"Unknown ortho_mode: {ortho_mode}")

            # Save to fixed_R_path if specified
            if fixed_R_path:
                os.makedirs(os.path.dirname(fixed_R_path), exist_ok=True)
                R_tensor = torch.tensor(self.R, dtype=torch.float32)
                torch.save(R_tensor, fixed_R_path)
                print(f"[INFO] Saved new R matrix to {fixed_R_path}")

        # Verify orthogonality: R^T @ R ≈ I
        identity_check = np.allclose(self.R.T @ self.R, np.eye(vocab_size), atol=1e-5)
        if not identity_check:
            raise ValueError("Generated matrix is not orthogonal!")

    def _one_hot(self, token_id: int) -> np.ndarray:
        """Convert token ID to one-hot vector."""
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        vec[token_id] = 1.0
        return vec

    def generate_sequence(
        self,
        length: int,
        start_token_id: int,
        temperature: float = 1.0,
        mode: str = 'argmax'
    ) -> Dict:
        """
        Generate a sequence using the linear dynamical system.

        Args:
            length: Target sequence length L
            start_token_id: Starting token (tokens[0])
            temperature: Temperature for sampling (only used if mode='sample')
            mode: 'argmax' or 'sample'

        Returns:
            {
                'tokens': List[int],              # Ground truth sequence [t0, t1, ..., t_{L-1}]
                'bag': Counter,                   # Multiset of tokens (for shuffled input)
                'states': List[np.ndarray],       # Hidden state trajectory (for debugging)
                'logits_history': List[np.ndarray]  # Logits at each step (for margin computation)
            }
        """
        if start_token_id < 0 or start_token_id >= self.vocab_size:
            raise ValueError(f"start_token_id {start_token_id} out of range [0, {self.vocab_size})")

        tokens = [start_token_id]
        states = []
        logits_history = []

        # Initialize: h_0 = one_hot(start_token)
        h = self._one_hot(start_token_id)
        states.append(h.copy())

        # Generate sequence
        for t in range(length - 1):
            # Update state: h_{t+1} = R @ h_t + x_t
            x_t = self._one_hot(tokens[t])
            h = self.R @ h + x_t

            states.append(h.copy())
            logits_history.append(h.copy())

            # Select next token
            if mode == 'argmax':
                next_token = int(np.argmax(h))
            elif mode == 'sample':
                # Apply temperature and softmax
                logits = h / temperature
                logits = logits - np.max(logits)  # Numerical stability
                probs = np.exp(logits) / np.sum(np.exp(logits))
                next_token = int(np.random.choice(self.vocab_size, p=probs))
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Verify token is in valid range
            if next_token < 0 or next_token >= self.vocab_size:
                raise ValueError(f"Generated token {next_token} out of range!")

            tokens.append(next_token)

        # Create bag (multiset)
        bag = Counter(tokens)

        return {
            'tokens': tokens,
            'bag': bag,
            'states': states,
            'logits_history': logits_history
        }

    def verify_uniqueness(self, tokens: List[int]) -> Tuple[bool, bool]:
        """
        Verify sequence uniqueness and validity.

        Checks two conditions:
        1. Validity: Does the ground truth path follow argmax rule?
        2. Uniqueness: At each step, can other tokens in the bag also satisfy argmax?

        This is a LOCAL GREEDY CHECK (conservative):
        - If local uniqueness passes, global uniqueness is guaranteed
        - We check if any remaining token in bag has the same argmax value

        Returns:
            (is_valid: bool, is_unique: bool)
        """
        if len(tokens) == 0:
            return False, False

        bag = Counter(tokens)
        h = self._one_hot(tokens[0])

        for t in range(len(tokens) - 1):
            # Update state: h_{t+1} = R @ h_t + x_t
            x_t = self._one_hot(tokens[t])
            h = self.R @ h + x_t

            # Get predicted token
            predicted = int(np.argmax(h))
            ground_truth = tokens[t + 1]

            # Check validity
            if predicted != ground_truth:
                return False, False

            # Check uniqueness
            # Remove tokens already used up to this point
            used_tokens = Counter(tokens[:t + 1])
            remaining = bag - used_tokens

            # Check if any other remaining token has the same argmax value
            max_value = h[predicted]
            for other_token in remaining.keys():
                if remaining[other_token] > 0 and other_token != ground_truth:
                    if np.abs(h[other_token] - max_value) < 1e-6:
                        # Another token also satisfies argmax
                        return True, False

        return True, True

    def compute_margin(self, tokens: List[int]) -> List[float]:
        """
        Compute margin (top1 - top2 logit) at each step.

        Args:
            tokens: Ground truth token sequence

        Returns:
            List of margins at each prediction step
        """
        margins = []
        h = self._one_hot(tokens[0])

        for t in range(len(tokens) - 1):
            # Update state
            x_t = self._one_hot(tokens[t])
            h = self.R @ h + x_t

            # Get top-2 values
            sorted_vals = np.sort(h)[::-1]
            margin = sorted_vals[0] - sorted_vals[1]
            margins.append(margin)

        return margins


def main():
    """
    Phase 0: Data Quality Health Check

    Validates that the data generation mechanism produces:
    - 100% valid sequences (follow argmax rule)
    - >90% unique sequences (bag uniquely determines order)
    - Reasonable margins (0.5-2.0 range)
    """
    print("=" * 60)
    print("🚀 Starting Phase 0: Data Quality Validation")
    print("=" * 60)
    print()

    # Parameters
    V = 16
    D = 16
    L = 20
    num_samples = 1000

    print(f"Parameters:")
    print(f"  Vocab Size (V):      {V}")
    print(f"  Hidden Dim (D):      {D}")
    print(f"  Sequence Length (L): {L}")
    print(f"  Num Samples:         {num_samples}")
    print()

    # Initialize generator
    generator = LinearDynamicalGenerator(V, D, ortho_mode='random')

    # Statistics
    validity_count = 0
    uniqueness_count = 0
    all_margins = []

    print("Generating and validating sequences...")
    for i in range(num_samples):
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i + 1}/{num_samples}")

        # Generate sequence with random start
        start = np.random.randint(0, V)
        result = generator.generate_sequence(L, start, mode='argmax')
        tokens = result['tokens']

        # Verify validity and uniqueness
        is_valid, is_unique = generator.verify_uniqueness(tokens)

        if is_valid:
            validity_count += 1
        if is_unique:
            uniqueness_count += 1

        # Compute margins
        margins = generator.compute_margin(tokens)
        all_margins.extend(margins)

    print()

    # Compute statistics
    validity_rate = validity_count / num_samples
    uniqueness_rate = uniqueness_count / num_samples
    avg_margin = np.mean(all_margins)
    std_margin = np.std(all_margins)
    min_margin = np.min(all_margins)
    max_margin = np.max(all_margins)

    # Print report
    print("=" * 60)
    print("📊 Data Quality Report (Phase 0 Validation)")
    print("=" * 60)
    print(f"Validity Rate:    {validity_rate * 100:6.2f}%  (Target: 100%)")
    print(f"Uniqueness Rate:  {uniqueness_rate * 100:6.2f}%  (Target: >90%)")
    print(f"Avg Margin:       {avg_margin:6.3f}     (Target: 0.5-2.0)")
    print(f"Std Margin:       {std_margin:6.3f}")
    print(f"Min Margin:       {min_margin:6.3f}")
    print(f"Max Margin:       {max_margin:6.3f}")
    print("=" * 60)
    print()

    # Decision logic
    if validity_count < num_samples:
        print("❌ FAIL: Validity < 100%, check code logic")
        print("   Action: Debug the generation/verification code")
        return False
    elif uniqueness_rate < 0.80:
        print("⚠️  WARNING: Uniqueness < 80%, suggest increasing vocab_size")
        print(f"   Action: Try V=32 or V=64 instead of V={V}")
        return False
    elif uniqueness_rate < 0.90:
        print("⚡ MARGINAL: Uniqueness in [80%, 90%), consider tuning")
        print(f"   Action: Monitor closely or increase V from {V} to {V*2}")
        return True
    else:
        print("✅ SUCCESS: Data quality excellent, proceed to Phase 1")
        print("   Action: Implement full training pipeline")
        return True


if __name__ == '__main__':
    success = main()
    import sys
    sys.exit(0 if success else 1)
