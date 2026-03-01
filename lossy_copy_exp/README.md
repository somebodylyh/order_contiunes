# LO-ARMs: Learning Optimal Order via RL

## Overview

This experiment verifies whether an RL-trained Agent can autonomously discover the optimal generation order for the **Lossy Copy** task:
- Given sequence `[x, y]` where `y = x // k`
- Agent should learn to generate `x` first (since `x` determines `y`)
- Core hypothesis: Due to causal asymmetry (x → y is deterministic, but y ↛ x), the ordering policy will converge to "generate x first, then y"

## Quick Start

### 1. Run Integration Tests

```bash
python lossy_copy_exp/test_all.py
```

This verifies all components work together.

### 2. Start Training

```bash
python lossy_copy_exp/train_loarms.py
```

Training will run for 5000 iterations with:
- **Phase A (0-1000)**: Warmup with random orders
- **Phase B (1000-5000)**: Co-evolution of model and agent

### 3. Monitor Results

If wandb is installed, results will be logged to Weights & Biases. Otherwise, check console output.

Key metrics to watch:
- `prob_select_x_first`: Should converge to ~1.0
- `accuracy_x`, `accuracy_y`: Should reach >95%
- `policy_loss`: Should stabilize

## File Structure

```
lossy_copy_exp/
├── __init__.py                  # Package init
├── lossy_copy_dataset.py        # Synthetic [x, y] dataset
├── model_wrapper.py             # AOGPT with hidden states & logical embeddings
├── order_policy_net.py          # Agent (OrderPolicyNet)
├── train_loarms.py              # Main training script
├── config_lossy_copy.py         # Experiment configuration
├── utils.py                     # Metrics & logging helpers
├── test_all.py                  # Integration test script
├── README.md                    # This file
└── checkpoints/                 # Output directory (created during training)
```

## Key Implementation Details

### 1. Logical Position Embeddings (CRITICAL)

**Problem**: With `vocab_size=64`, x ∈ [0,63], y ∈ [0,31]. Model sees value `15` but doesn't know if it's x or y.

**Solution**: Added `logical_pos_emb` in `model_wrapper.py`:
```python
self.logical_pos_emb = nn.Embedding(block_size, n_embd)
```

This tells the model "which token is x, which is y" based on logical position, not physical position in the shuffled sequence.

**Without this**: Agent sees meaningless hidden states → experiment will fail.

### 2. Continuous Reward Function

**Problem**: Binary rewards (0/1) give no signal when model predicts wrong in early training.

**Solution**: Use `reward = log P(correct_token)` (negative log-likelihood):
- Even when argmax is wrong, probability improvements still reward Agent
- Smoother gradients → faster convergence

**Range**: [-inf, 0], where 0 = perfect confidence

### 3. Two-Phase Training

**Phase A: Warmup (steps 0-1000)**
- Freeze Agent (no gradient updates)
- Random order generation
- Train model with standard cross-entropy loss
- Purpose: Initialize model with basic prediction capability

**Phase B: Co-evolution (steps 1000-5000)**
- Train both model and agent
- Agent uses REINFORCE to learn ordering policy
- Model learns to predict under agent-selected orders

## Configuration

Edit `config_lossy_copy.py` to change experiment settings:

```python
# Task
vocab_size = 64      # x ∈ [0, vocab_size), y ∈ [0, vocab_size//k)
k_divisor = 2        # y = x // k

# Model (tiny for fast iteration)
n_layer = 2
n_head = 2
n_embd = 128

# Training
batch_size = 64
max_iters = 5000
warmup_steps = 1000

# Agent
agent_learning_rate = 1e-4
reward_type = 'log_prob'  # or 'binary'
```

### Experiment Variants

Uncomment sections at the bottom of `config_lossy_copy.py` to test:
- Larger model (n_layer=4, n_embd=256)
- Harder task (vocab_size=128, k=4)
- Binary reward (reward_type='binary')
- No warmup (warmup_steps=0)
- Longer training (max_iters=10000)

## Expected Results

### Success Criteria

1. `prob_select_x_first > 0.9` after training
2. Model accuracy >95% on both x and y
3. Smooth loss decrease (no divergence)

### Training Dynamics

- **Early (0-500)**: `prob_select_x_first ≈ 0.5` (random)
- **Mid (500-2000)**: `prob_select_x_first` increases rapidly
- **Late (2000+)**: `prob_select_x_first → 1.0` (converged)

### Failure Modes

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `prob_select_x_first ≈ 0.5` | Agent not learning | Check reward signal, increase agent LR |
| `prob_select_x_first < 0.1` | Learned wrong order | Verify y = x // k logic |
| Loss increases | Training instability | Reduce LR, add grad clipping |
| No improvement | Cold start problem | Increase warmup_steps |

## Verification Plan

### Step 1: Component Tests

```bash
# Test dataset
python lossy_copy_exp/lossy_copy_dataset.py

# Test model wrapper
python lossy_copy_exp/model_wrapper.py

# Test agent
python lossy_copy_exp/order_policy_net.py

# Test utilities
python lossy_copy_exp/utils.py
```

### Step 2: Integration Test

```bash
python lossy_copy_exp/test_all.py
```

Should print:
```
✅ All integration tests passed!
```

### Step 3: Mini Training Run

Edit `config_lossy_copy.py`:
```python
max_iters = 100
vocab_size = 8
```

Run:
```bash
python lossy_copy_exp/train_loarms.py
```

Verify:
- No errors
- Loss decreases
- Checkpoints saved

### Step 4: Full Experiment

Reset config and run full training:
```bash
python lossy_copy_exp/train_loarms.py
```

Should complete in ~10-30 minutes on GPU.

## Monitoring Training

### Console Output

```
[WARMUP iter 10] loss: 3.5234 | accuracy: 0.1523 | ...
[COEVOLUTION iter 1100] loss: 1.2341 | prob_select_x_first: 0.5234 | ...
[EVAL iter 1200] prob_select_x_first: 0.5847 | accuracy: 0.8234 | ...
```

### Weights & Biases

If `wandb_log=True` in config:
1. Login: `wandb login`
2. Check dashboard at https://wandb.ai/your-username/LO-ARMs

Plots to monitor:
- `train/prob_select_x_first`: Should converge to 1.0
- `train/loss`: Should decrease smoothly
- `eval/accuracy`: Should reach >0.95

## Checkpoints

Saved to `lossy_copy_exp/checkpoints/`:
- `checkpoint_latest.pt`: Latest checkpoint (updated every checkpoint_interval)
- `checkpoint_500.pt`, `checkpoint_1000.pt`, ...: Periodic checkpoints
- `config.json`: Experiment configuration

### Loading Checkpoints

```python
from lossy_copy_exp.utils import load_checkpoint

checkpoint_path = 'lossy_copy_exp/checkpoints/checkpoint_latest.pt'
iter_num, config = load_checkpoint(
    checkpoint_path, model, agent,
    optimizer_model, optimizer_agent
)
print(f"Resumed from iteration {iter_num}")
```

## Troubleshooting

### Import Errors

Ensure parent directory is in Python path:
```python
import sys
sys.path.append('/home/admin/lyuyuhuan/AO-GPT-MDM')
```

### CUDA Out of Memory

Reduce batch size in `config_lossy_copy.py`:
```python
batch_size = 32  # or 16
```

### Training Diverges

1. Reduce learning rates:
   ```python
   learning_rate = 5e-4
   agent_learning_rate = 5e-5
   ```

2. Increase gradient clipping:
   ```python
   grad_clip = 0.5
   ```

3. Use binary rewards:
   ```python
   reward_type = 'binary'
   ```

### Agent Not Learning

1. Increase warmup to initialize model better:
   ```python
   warmup_steps = 2000
   ```

2. Check reward signal is correct:
   - Add print statements in `coevolution_step`
   - Verify rewards are in expected range

3. Try larger agent:
   ```python
   policy_dim = 256
   ```

## Design Decisions

### Why Logical Position Embeddings?

Standard positional embeddings only encode "where in sequence", not "what role". With overlapping value ranges (x ∈ [0,63], y ∈ [0,31]), the model cannot distinguish x=15 from y=15 without role information.

Logical embeddings solve this by encoding "I am x" vs "I am y", independent of physical shuffling.

### Why Continuous Rewards?

Binary rewards (correct=1, wrong=0) provide sparse feedback. In early training, model might be 100% wrong → all 0 rewards → no gradient signal.

Continuous rewards (log probabilities) give dense feedback: even when wrong, improvements in probability still reward the Agent.

### Why Two-Phase Training?

Cold-start problem: Random model + untrained agent → both perform poorly → no learning signal.

Solution: Warmup phase trains model to basic competence, then co-evolution can leverage meaningful rewards.

## Related Files in Main Codebase

This experiment **imports but does not modify**:
- `model_AOGPT_AdaLN6_NoRep_cond_128_trunc_qknorm.py`: Base AOGPT model
- `ema.py`: Exponential moving average (optional, for model stability)
- `configurator.py`: Configuration system (optional)

All experimental code is self-contained in `lossy_copy_exp/`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{loarms2024,
  title={Learning Optimal Order via Reinforcement Learning for Autoregressive Models},
  author={...},
  year={2024}
}
```

## License

[Same as parent repository]
