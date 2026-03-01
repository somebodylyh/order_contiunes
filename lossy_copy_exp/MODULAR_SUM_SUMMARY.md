# Modular Sum Experiment Implementation Summary

## ✅ Implementation Complete

Extended the 2-variable Lossy Copy experiment to 3-variable Modular Sum with configurable causal structure.

---

## 📁 New Files Created

### Core Implementation (4 files)

1. **`modular_sum_dataset.py`** (7.8 KB)
   - Three-variable dataset: `[x1, x2, y]`
   - **Lossy mode**: `y = (x1 + x2) // 2` (strong causality)
   - **Modular mode**: `y = (x1 + x2) % P` (complete symmetry)
   - Shuffles all 3! = 6 permutations randomly
   - Maintains logical_ids binding: 0=x1, 1=x2, 2=y
   - Includes comprehensive tests

2. **`config_modular_sum.py`** (3.1 KB)
   - Core switch: `use_lossy = True/False`
   - Extended parameters: `seq_length = 3`, `block_size = 3`
   - Training: `max_iters = 10000` (increased for complexity)
   - Dynamic wandb run name based on mode
   - Separate output directories for each mode

3. **`train_modular_sum.py`** (15.2 KB)
   - Extended training loop for 3-step rollout
   - **New metric**: `compute_first_step_selection_probs()`
     - Tracks P(select x1 first), P(select x2 first), P(select y first)
     - Computes P(select any_x first) = P(x1) + P(x2)
   - Updated warmup_step for 3 variables
   - Updated coevolution_step with detailed first-step monitoring
   - Evaluation with greedy agent policy
   - Mode-specific best metric tracking

4. **`test_modular_sum.py`** (4.9 KB)
   - Integration tests for 3-variable setup
   - Tests both lossy and modular modes
   - Verifies 3-step rollout
   - Tests first-step selection probability computation
   - Confirms all shapes and dimensions

### Documentation & Scripts (3 files)

5. **`README_MODULAR_SUM.md`** (5.2 KB)
   - Complete experiment guide
   - Expected results for both modes
   - Key metrics to monitor
   - Troubleshooting tips
   - Visualization recommendations

6. **`run_modular_sum.sh`** (3.4 KB)
   - Easy-to-use launch script
   - Flags: `--lossy`, `--modular`, `--test`
   - Automatic config switching
   - Clear output with expected results

7. **`MODULAR_SUM_SUMMARY.md`** (This file)

---

## 🔑 Key Extensions from 2-Variable Task

### 1. Dataset: 2 → 3 Variables

| Aspect | 2-Variable (Lossy Copy) | 3-Variable (Modular Sum) |
|--------|-------------------------|--------------------------|
| Variables | `[x, y]` | `[x1, x2, y]` |
| Computation | `y = x // k` | Lossy: `y = (x1+x2)//2`<br>Modular: `y = (x1+x2)%P` |
| Permutations | 2! = 2 | 3! = 6 |
| Logical IDs | `[0, 1]` | `[0, 1, 2]` |

### 2. Model: Block Size Extension

```python
# Before: block_size = 2
# After:  block_size = 3

model_config = AOGPTConfig(
    block_size=3,  # Extended
    ...
)

# Logical pos emb automatically adjusts:
# self.logical_pos_emb = nn.Embedding(config.block_size, config.n_embd)
# Now supports 3 positions: 0=x1, 1=x2, 2=y
```

### 3. Agent: 3-Position Support

```python
# Before: num_positions = 2
# After:  num_positions = 3

agent = OrderPolicyNet(
    d_model=128,
    policy_dim=128,
    num_positions=3  # Extended
)

# Output now: [B, 3] probability distribution
# Mask now: [B, 3] filled positions
```

### 4. Training: 3-Step Rollout

```python
# Before: 2 steps (select x, then y)
# After:  3 steps (select x1, x2, y in some order)

for step in range(3):  # Extended from 2
    # Agent selects next position
    actions, log_probs = agent.sample_action(hidden_states, filled_mask)

    # Track first step for analysis
    if step == 0:
        first_actions = actions  # Critical metric

    # ... fill, mask, reward
```

### 5. Monitoring: New Metrics

**Added in `train_modular_sum.py`:**

```python
def compute_first_step_selection_probs(first_actions, logical_ids_batch):
    """
    Map physical actions to logical IDs and compute probabilities.

    Returns:
        {
            'p_select_x1_first': float,  # NEW
            'p_select_x2_first': float,  # NEW
            'p_select_y_first': float,
            'p_select_any_x_first': float,  # NEW (x1 + x2)
        }
    """
```

**Logged to wandb:**
- `probs/select_x1_first`
- `probs/select_x2_first`
- `probs/select_y_first`
- `probs/select_any_x_first`

---

## 🎯 Experimental Design

### Hypothesis

**Lossy Mode** (`use_lossy=True`): Strong causal asymmetry
- Agent should discover: x1, x2 → y (deterministic)
- Expected behavior: P(select_any_x_first) → 1.0, P(select_y_first) → 0.0

**Modular Mode** (`use_lossy=False`): Complete symmetry
- Agent should discover: No causal preference
- Expected behavior: P(select_y_first) stays ~0.33 OR mode collapse

### Test Protocol

1. **Phase 1**: Run lossy mode → verify strong causality learning
2. **Phase 2**: Run modular mode → verify symmetry recognition
3. **Phase 3**: Compare curves → confirm hypothesis

---

## 📊 Expected Results

### Case A: Lossy Mode

```
Iteration 0:
  P(select_x1_first): ~33%
  P(select_x2_first): ~33%
  P(select_y_first): ~33%
  P(select_any_x_first): ~66%

Iteration 5000:
  P(select_x1_first): ~50%  ✓
  P(select_x2_first): ~50%  ✓
  P(select_y_first): ~0%    ✓✓✓ (KEY RESULT)
  P(select_any_x_first): ~100%  ✓✓✓
```

### Case B: Modular Mode

```
Iteration 0:
  P(select_x1_first): ~33%
  P(select_x2_first): ~33%
  P(select_y_first): ~33%

Iteration 5000:
  P(select_x1_first): ~33%  (unchanged)
  P(select_x2_first): ~33%  (unchanged)
  P(select_y_first): ~33%   (unchanged) ✓✓✓ (KEY RESULT)

  OR: Mode collapse to arbitrary order
  (e.g., always x1 → x2 → y, but no causal reason)
```

---

## 🚀 Quick Start

### Test Everything

```bash
./lossy_copy_exp/run_modular_sum.sh --test
```

Expected output:
```
✓ Lossy dataset verified
✓ Modular dataset verified
✓ Model supports 3 positions
✓ Agent supports 3 positions
✓ 3-step rollout works
✓ First-step metrics computed correctly
```

### Run Lossy Mode

```bash
./lossy_copy_exp/run_modular_sum.sh --lossy
```

Monitor: `P(select_any_x_first)` should → 1.0

### Run Modular Mode

```bash
./lossy_copy_exp/run_modular_sum.sh --modular
```

Monitor: `P(select_y_first)` should stay ~0.33

---

## 🔍 Code Changes Verification

### No Modifications to Existing Files ✓

All code is new and self-contained in `lossy_copy_exp/`:
- ✅ `model_wrapper.py` - NOT modified (already supports `block_size` parameter)
- ✅ `order_policy_net.py` - NOT modified (already supports `num_positions` parameter)
- ✅ `utils.py` - NOT modified (already supports variable-length sequences)

### Reused Components ✓

From 2-variable experiment:
- ✅ `AOGPTWithHiddenStates` (flexible block_size)
- ✅ `OrderPolicyNet` (flexible num_positions)
- ✅ `compute_accuracy` (supports position_idx)
- ✅ `compute_returns`, `normalize_returns` (sequence-agnostic)
- ✅ Checkpointing, logging, metrics tracking

### New Components ✓

Only 3 new Python files needed:
1. `modular_sum_dataset.py` - New dataset with use_lossy switch
2. `config_modular_sum.py` - New config for 3-variable task
3. `train_modular_sum.py` - Extended training with new metrics

---

## 📈 Monitoring Checklist

### During Training (Console/Wandb)

**Warmup Phase (0-1000 iters):**
- [ ] Loss decreasing
- [ ] Accuracy improving
- [ ] No first-step metrics yet (agent frozen)

**Co-evolution Phase (1000-10000 iters):**

**For Lossy Mode:**
- [ ] `P(select_y_first)` dropping from ~0.33 to <0.05
- [ ] `P(select_any_x_first)` rising from ~0.66 to >0.95
- [ ] `P(select_x1_first)` and `P(select_x2_first)` balanced (~0.5 each)
- [ ] Accuracy >0.95 for all positions

**For Modular Mode:**
- [ ] `P(select_y_first)` staying flat around 0.33
- [ ] All three probabilities similar (OR mode collapse)
- [ ] Accuracy >0.95 for all positions

### After Training

- [ ] Final metrics logged
- [ ] Checkpoint saved
- [ ] Wandb run completed
- [ ] Compare with expected results

---

## 🐛 Troubleshooting

### Issue: "Shape mismatch" errors

**Cause**: Hardcoded dimensions from 2-variable task

**Solution**:
- Check all uses of `2` and replace with `config.seq_length` or `T`
- Verify `block_size = 3` in config
- Check logical_pos_emb has at least 3 embeddings

### Issue: Agent not learning in lossy mode

**Possible causes**:
1. Insufficient warmup → increase to 2000 steps
2. Reward signal too weak → check `reward_type = 'log_prob'`
3. Learning rate too high → reduce `agent_learning_rate` to 5e-5

**Debug**:
```python
# Add to training loop:
print(f"First actions: {first_actions}")
print(f"Chosen logical IDs: {chosen_logical_ids}")
print(f"Avg reward: {avg_reward}")
```

### Issue: Modular mode shows preference (not symmetric)

**This might be OK!** Two possibilities:
1. **True mode collapse**: Agent converges to arbitrary order due to initialization bias (not causal)
2. **Bug**: Check that dataset correctly implements `y = (x1 + x2) % P`

**Verification**:
```bash
python lossy_copy_exp/modular_sum_dataset.py
# Check modular computation is correct
```

---

## 📝 Files Summary

```
lossy_copy_exp/
├── modular_sum_dataset.py        # NEW: 3-variable dataset
├── config_modular_sum.py         # NEW: 3-variable config
├── train_modular_sum.py          # NEW: 3-step training
├── test_modular_sum.py           # NEW: 3-variable tests
├── run_modular_sum.sh            # NEW: Easy launcher
├── README_MODULAR_SUM.md         # NEW: Documentation
├── MODULAR_SUM_SUMMARY.md        # NEW: This file
│
├── model_wrapper.py              # UNCHANGED (flexible)
├── order_policy_net.py           # UNCHANGED (flexible)
├── utils.py                      # UNCHANGED (reused)
└── ...                           # Original 2-variable files
```

**Total new code**: ~31 KB across 7 files
**Total lines**: ~1200 lines

---

## ✅ Implementation Checklist

### Task 1: Dataset ✅
- [x] ModularSumDataset with use_lossy switch
- [x] Lossy mode: `y = (x1 + x2) // 2`
- [x] Modular mode: `y = (x1 + x2) % P`
- [x] 3-variable shuffling with logical_ids binding
- [x] Standalone tests

### Task 2: Config ✅
- [x] `use_lossy` parameter
- [x] `seq_length = 3`
- [x] `block_size = 3`
- [x] Dynamic wandb run names
- [x] Separate output directories

### Task 3: Training Loop ✅
- [x] 3-step rollout
- [x] First-step selection tracking
- [x] `compute_first_step_selection_probs()`
- [x] Logging: p_select_x1/x2/y/any_x_first
- [x] Extended evaluation loop

### Task 4: Agent Adaptation ✅
- [x] Verified `num_positions` flexibility
- [x] Tested with 3 positions
- [x] Masking works for 3 positions
- [x] Probability normalization correct

### Task 5: Testing ✅
- [x] Dataset tests (both modes)
- [x] Integration tests
- [x] 3-step rollout verification
- [x] Metric computation tests

### Task 6: Documentation ✅
- [x] README_MODULAR_SUM.md
- [x] Run script with --lossy/--modular flags
- [x] This implementation summary
- [x] Expected results documented

---

## 🎯 Ready to Run!

```bash
# Quick test
./lossy_copy_exp/run_modular_sum.sh --test

# Run lossy mode (strong causality)
./lossy_copy_exp/run_modular_sum.sh --lossy

# Run modular mode (symmetric)
./lossy_copy_exp/run_modular_sum.sh --modular
```

**Estimated runtime**: 20-40 minutes per experiment on GPU

---

## 📚 Next Steps

1. **Run Case A** (lossy): Verify agent learns causal structure
2. **Run Case B** (modular): Verify agent recognizes symmetry
3. **Compare results**: Plot both curves side-by-side
4. **Analyze**: Write up findings on causal discovery capability

---

**Implementation complete! All components tested and ready for experimentation.**
