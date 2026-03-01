# Critical Fixes: Online Generation + Teacher Forcing

**Date**: 2026-01-27
**Status**: ✅ IMPLEMENTED & TESTED

---

## Problem Diagnosed

Previous training showed a **co-evolution trap**:

1. **Overfitting**: Train Loss = 0.13, Val Loss = 15.0 (100x gap!)
2. **Memorization**: Model memorized fixed training bags instead of learning R matrix
3. **Agent Failure**: L2R Correct = 0.00% (Agent learned nothing)

**Root Cause**: The model found a shortcut - memorize the 8000 fixed samples instead of learning the physical rule `h_{t+1} = R @ h_t + x_t`.

---

## Solution 1: Online Data Generation

### What Changed

Modified `rotation_dataset.py` to support two modes:

#### Train Mode (Online)
```python
LinearRotationDataset(mode='train', virtual_size=10000)
```
- **No pre-generation**: Data is generated on-the-fly in `__getitem__`
- **Infinite variety**: Every call to `dataset[i]` produces a NEW sequence
- **Virtual epoch size**: Uses `virtual_size` to define epoch length
- **Kills memorization**: Model cannot memorize - must learn R

#### Eval Mode (Static)
```python
LinearRotationDataset(mode='eval', num_samples=1000)
```
- **Pre-generated**: Fixed data for reproducible evaluation
- **Consistent**: Same index always returns same sample
- **Fair comparison**: Val/Test use static data for proper benchmarking

### Key Implementation Details

```python
def __getitem__(self, idx):
    if self.mode == 'train':
        # Generate fresh sample every time
        start_token = np.random.randint(0, self.vocab_size)
        result = self.generator.generate_sequence(...)
        return format_as_tensors(result)
    else:
        # Return pre-generated static sample
        return self.samples[idx]
```

---

## Solution 2: Teacher Forcing

### What Changed

Modified `train_rotation.py` to guide the Agent during early training:

#### Configuration
```python
# config_rotation.py
teacher_forcing_start = 1.0       # Start with 100% forcing
teacher_forcing_end = 0.0         # Decay to 0%
teacher_forcing_decay_steps = 5000  # Linear decay
```

#### Training Logic
```python
def coevolution_step(..., global_step):
    # Compute current TF ratio (linear decay)
    if global_step < decay_steps:
        tf_ratio = start - (start - end) * (global_step / decay_steps)
    else:
        tf_ratio = end

    for step in range(L):
        # Agent samples action
        actions = agent.sample_action(...)

        # Teacher Forcing: Override with correct action
        if tf_ratio > 0:
            force_mask = torch.rand(B) < tf_ratio
            correct_actions = torch.full((B,), step)  # L2R: step 0 -> pos 0, step 1 -> pos 1
            actions = torch.where(force_mask, correct_actions, actions)

        # Use action to update sequence
        partial_tokens[..., actions] = unshuffled_tokens[..., actions]
```

### Why This Works

1. **Early Training (TF=1.0)**:
   - Agent's actions are 100% overridden with correct L2R order
   - Model sees valid trajectories: `h_0 -> h_1 -> h_2 -> ...`
   - Model learns the R matrix dynamics (physical rule)
   - Model gives accurate rewards (high for valid, low for invalid)

2. **Mid Training (TF=0.5)**:
   - 50% teacher, 50% agent
   - Model is mostly trained, gives good rewards
   - Agent starts learning from accurate reward signal

3. **Late Training (TF=0.0)**:
   - No forcing, pure REINFORCE
   - Agent learned the order, Model provides perfect rewards
   - System converges to optimal policy

---

## Configuration Updates

### `config_rotation.py`

**Before**:
```python
num_train_samples = 8000  # Fixed pre-generated samples
```

**After**:
```python
train_mode = 'online'              # Enable online generation
num_train_samples = 10000          # Virtual epoch size
teacher_forcing_start = 1.0        # Start with 100% forcing
teacher_forcing_end = 0.0          # Decay to 0%
teacher_forcing_decay_steps = 5000 # Decay duration
```

---

## Testing Results

All tests passed ✅:

1. **Online Generation**: Different samples on repeated calls
2. **Static Evaluation**: Consistent samples for validation
3. **TF Decay**: Correct linear decay from 1.0 → 0.0
4. **Dataloaders**: Train (online) and Val (static) work correctly

```
Test 1: Online Generation - ✅ PASS
Test 2: Static Evaluation - ✅ PASS
Test 3: Teacher Forcing Decay - ✅ PASS
Test 4: Dataloader Creation - ✅ PASS
```

---

## Expected Training Behavior

### Phase 1: Warmup (0-2000 iter)
- TF not active yet
- Model learns with random orders
- Loss should converge to ~1.5

### Phase 2: Early Co-evolution (2000-4000 iter, TF=1.0→0.6)
- **Train Loss**: May be higher than before (no memorization!)
- **Val Loss**: Should start DECREASING (learning the rule)
- **L2R Correct**: Should rise from 0% toward 50%+
- **TF Ratio**: 1.0 → 0.6

### Phase 3: Mid Co-evolution (4000-7000 iter, TF=0.6→0.0)
- **Val Loss**: Continues decreasing
- **L2R Correct**: Should reach 70-80%
- **TF Ratio**: 0.6 → 0.0

### Phase 4: Late Co-evolution (7000-12000 iter, TF=0.0)
- **Pure REINFORCE**: No teacher forcing
- **L2R Correct**: Should exceed 80%
- **Topology Correct**: Should approach 100%

---

## Key Metrics to Monitor

### ⚠️ Warning Signs (Old behavior)
- Train Loss << Val Loss (e.g., 0.13 vs 15.0) → Memorization
- L2R Correct stuck at 0% → Agent not learning
- Val Loss increasing → Model not generalizing

### ✅ Good Signs (New behavior)
- Train Loss ≈ Val Loss (e.g., 0.8 vs 1.0) → Learning the rule
- L2R Correct rising → Agent learning order
- Val Loss decreasing → Model generalizing
- TF Ratio decaying → Gradually removing training wheels

---

## Running the Training

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM/linear_rotation_exp

# Test the fixes first
python test_online_tf.py

# Run full training
python train_rotation.py
```

### What to Watch

Open WandB and monitor:
1. `train/loss` vs `val/loss` (should be similar)
2. `val/l2r_order_correct` (should rise)
3. `train/tf_ratio` (should decay)
4. `val/first_step/p_t0` (should approach 1.0)

---

## Files Modified

1. **`rotation_dataset.py`**
   - Added `mode` parameter ('train' or 'eval')
   - Added `virtual_size` for online mode
   - Modified `__getitem__` to generate on-the-fly for train mode
   - Modified `__len__` to return virtual_size for train mode

2. **`train_rotation.py`**
   - Added `global_step` parameter to `coevolution_step`
   - Implemented TF ratio calculation (linear decay)
   - Added action override logic with `force_mask`
   - Added `tf_ratio` to logged metrics

3. **`config_rotation.py`**
   - Added `train_mode = 'online'`
   - Increased `num_train_samples` to 10000 (virtual epoch)
   - Added `teacher_forcing_start`, `teacher_forcing_end`, `teacher_forcing_decay_steps`

4. **`test_online_tf.py`** (New)
   - Validation script for both fixes

---

## Theory: Why This Fixes the Problem

### The Memorization Trap

**Old System**:
```
Model sees 8000 fixed bags → Memorizes "Bag A leads to Sequence X"
↓
Gives high reward for memorized (but wrong) sequences
↓
Agent learns wrong policy (memorized shortcuts)
↓
System collapses: Neither learns the true rule
```

**New System with Online + TF**:
```
Model sees infinite variety of bags → Cannot memorize
↓
Teacher Forcing ensures valid trajectories early on
↓
Model learns h_{t+1} = R @ h_t + x_t (true rule)
↓
Model gives accurate rewards (valid = high, invalid = low)
↓
Agent learns correct L2R order from accurate rewards
↓
System converges: Both learn correctly
```

### The Teacher Forcing Strategy

Think of it like teaching a child to ride a bike:

1. **TF = 1.0 (Training wheels on)**: You hold the bike 100% of the time
   - Model sees correct trajectories, learns physics
   - Agent doesn't fall, builds confidence

2. **TF = 0.5 (Partial support)**: You hold the bike 50% of the time
   - Model is trained, gives good feedback
   - Agent starts balancing on its own

3. **TF = 0.0 (No support)**: You let go completely
   - Agent has learned, rides independently
   - Model provides accurate corrections

---

## Success Criteria

### Must Achieve
- ✅ Val Loss < 2.0 (was 15.0 before)
- ✅ L2R Correct > 50% (was 0% before)
- ✅ Train Loss ≈ Val Loss (was 100x gap before)

### Stretch Goals
- 🎯 L2R Correct > 80%
- 🎯 First Step p(t0) > 90%
- 🎯 Topology Correct > 90%

---

## Theoretical Justification

This approach is grounded in:

1. **Curriculum Learning**: Start easy (with guidance), gradually increase difficulty
2. **Scheduled Sampling**: Gradually transition from supervised to reinforcement learning
3. **Anti-Memorization**: Online generation prevents overfitting to fixed dataset
4. **Co-training**: Model and Agent help each other converge

**Reference Pattern**: Similar to how language models use teacher forcing during pre-training, then switch to autoregressive generation.

---

## If It Still Doesn't Work

### Debug Checklist

1. **Check TF is active**: Monitor `train/tf_ratio` in WandB (should decay)
2. **Check online generation**: Train loss should be higher than 0.13 initially
3. **Check val loss**: Should be close to train loss (within 2x)
4. **Check L2R correct**: Should be > 0% by iteration 3000

### Potential Adjustments

If L2R Correct < 20% by iteration 5000:
```python
# Extend teacher forcing
teacher_forcing_decay_steps = 8000  # Slower decay

# Or stronger forcing
teacher_forcing_start = 1.0
teacher_forcing_end = 0.2  # Keep 20% forcing even late
```

If Val Loss still high:
```python
# Increase model capacity
n_layer = 4
n_embd = 256
```

---

## Summary

**Two critical fixes implemented**:
1. ✅ **Online Generation**: Kills memorization
2. ✅ **Teacher Forcing**: Guides agent to learn

**Expected outcome**: Model learns the true R matrix rule, Agent discovers L2R order, system converges correctly.

**Next step**: Run `python train_rotation.py` and monitor WandB!
