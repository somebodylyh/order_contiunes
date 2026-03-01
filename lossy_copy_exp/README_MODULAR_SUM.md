# Modular Sum Experiment: Extended 3-Variable Causal Discovery

## Overview

This experiment extends the Lossy Copy task from 2 variables `[x, y]` to 3 variables `[x1, x2, y]`, testing whether the Agent can discover different causal structures:

### Two Modes:

**Mode A: Lossy** (`use_lossy=True`)
```python
y = (x1 + x2) // 2
```
- **Strong causal asymmetry**: x1, x2 → y (deterministic)
- **Many-to-one**: Given y, multiple (x1, x2) pairs are possible
- **Expected behavior**: Agent should learn to generate x1, x2 first, then y

**Mode B: Modular** (`use_lossy=False`)
```python
y = (x1 + x2) % P  # where P = vocab_size
```
- **Complete symmetry**: Any two variables determine the third
- **No causal preference**: x1, x2, y are on equal footing
- **Expected behavior**: Agent should show no preference (or random collapse)

## Quick Start

### 1. Test Components

```bash
# Test dataset
python lossy_copy_exp/modular_sum_dataset.py

# Test integration
python lossy_copy_exp/test_modular_sum.py
```

### 2. Run Experiment

**Case A: Lossy Mode (Strong Causality)**
```bash
# Edit config: use_lossy = True
python lossy_copy_exp/train_modular_sum.py
```

**Case B: Modular Mode (Symmetric)**
```bash
# Edit config: use_lossy = False
python lossy_copy_exp/train_modular_sum.py
```

## Expected Results

### Case A: Lossy Mode (`use_lossy=True`)

| Metric | Initial | After Training | Target |
|--------|---------|----------------|--------|
| `P(select_y_first)` | ~33% | → **~0%** | <5% |
| `P(select_x1_first)` | ~33% | → **~50%** | 40-60% |
| `P(select_x2_first)` | ~33% | → **~50%** | 40-60% |
| `P(select_any_x_first)` | ~66% | → **~100%** | >95% |

**Interpretation**: Agent discovers that y depends on x1 and x2, so it must generate x before y.

### Case B: Modular Mode (`use_lossy=False`)

| Metric | Initial | After Training | Interpretation |
|--------|---------|----------------|----------------|
| `P(select_y_first)` | ~33% | **~33%** | No preference |
| `P(select_x1_first)` | ~33% | **~33%** | Symmetric |
| `P(select_x2_first)` | ~33% | **~33%** | Symmetric |

**OR**: Mode collapse to any single variable (e.g., always pick x1 first) due to initialization bias.

**Interpretation**: Agent correctly recognizes complete symmetry, or collapses to arbitrary order.

## Key Metrics to Monitor

### Critical: First-Step Selection Probabilities

```python
# In wandb or logs, watch:
probs/select_x1_first   # Should → ~50% in lossy mode
probs/select_x2_first   # Should → ~50% in lossy mode
probs/select_y_first    # Should → ~0% in lossy mode, ~33% in modular mode
probs/select_any_x_first # Should → ~100% in lossy mode
```

### Model Performance

- `accuracy_x1`, `accuracy_x2`, `accuracy_y`: Should all reach >95%
- `loss`: Should decrease smoothly
- `policy_loss`: Should stabilize

## Configuration

Edit `config_modular_sum.py`:

```python
# Core switch
use_lossy = True  # True for Case A, False for Case B

# Task parameters
vocab_size = 64
seq_length = 3

# Training
max_iters = 10000  # Longer training for 3-variable task
warmup_steps = 1000
batch_size = 64
```

## File Structure

```
lossy_copy_exp/
├── modular_sum_dataset.py      # 3-variable dataset with use_lossy switch
├── config_modular_sum.py       # Configuration for modular sum
├── train_modular_sum.py        # Training script with 3-step rollout
├── test_modular_sum.py         # Integration tests
└── README_MODULAR_SUM.md       # This file
```

## Differences from 2-Variable Task

### Dataset
- **Sequence length**: 2 → 3
- **Logical IDs**: [0, 1] → [0, 1, 2]
- **Shuffle**: 2! = 2 permutations → 3! = 6 permutations

### Model
- **Block size**: 2 → 3
- **Logical pos emb**: vocab 2 → 3

### Agent
- **Num positions**: 2 → 3
- **Rollout steps**: 2 steps → 3 steps

### Monitoring
- **New metrics**: Added `p_select_x1_first`, `p_select_x2_first`, `p_select_any_x_first`
- **Accuracy**: Split into `accuracy_x1`, `accuracy_x2`, `accuracy_y`

## Experiment Plan

### Phase 1: Verify Lossy Mode

1. Set `use_lossy = True` in config
2. Run training for 10k iterations
3. Verify `P(select_any_x_first) → 1.0`
4. Verify `P(select_y_first) → 0.0`

### Phase 2: Test Modular Mode

1. Set `use_lossy = False` in config
2. Run training for 10k iterations
3. Verify `P(select_y_first) ≈ 0.33` (stays flat) OR mode collapse

### Phase 3: Compare

1. Plot both curves on same chart
2. Analyze difference in convergence behavior
3. Confirm hypothesis: Agent can distinguish causal structures

## Troubleshooting

### Agent Not Learning in Lossy Mode

- Increase warmup: `warmup_steps = 2000`
- Check reward signal: Should improve from negative to less negative
- Verify dataset: Run `python lossy_copy_exp/modular_sum_dataset.py`

### Unexpected Behavior in Modular Mode

- Mode collapse (e.g., always picks x1) is actually OK - shows no causal preference
- Check if probabilities are truly uniform or collapsed
- Compare with random baseline

### Model Accuracy Low

- Increase training iterations: `max_iters = 20000`
- Check model size: Try `n_embd = 256, n_layer = 4`
- Verify logical position embeddings are working

## Visualization

Use wandb to plot:

```python
# Compare two runs
wandb.init(project="LO-ARMs-ModularSum")

# Key plot: First-step selection over time
plt.plot(steps, p_select_y_first_lossy, label='Lossy: P(y first)')
plt.plot(steps, p_select_y_first_modular, label='Modular: P(y first)')
plt.axhline(0.33, color='gray', linestyle='--', label='Random')
plt.legend()
plt.xlabel('Training Steps')
plt.ylabel('P(select y first)')
plt.title('Agent Learning: Lossy vs Modular')
```

## Citation

If you use this code, please cite:

```bibtex
@article{loarms2024,
  title={Learning Optimal Order via Reinforcement Learning for Autoregressive Models},
  author={...},
  year={2024}
}
```

## Notes

- This experiment is self-contained in `lossy_copy_exp/`
- No original codebase files are modified
- All components are reusable for other multi-variable tasks
