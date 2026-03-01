# Curriculum & Dense Rewards Implementation Summary

**Date**: 2026-01-27
**Status**: ✅ IMPLEMENTED

---

## Changes Made

### 1. Extended Teacher Forcing Curriculum ✅

**File**: `config_rotation.py`

```python
# BEFORE
teacher_forcing_decay_steps = 5000

# AFTER
teacher_forcing_decay_steps = 20000  # 4x longer guidance period
```

**Rationale**:
- Agent was losing guidance too quickly (5000 steps)
- Model needs time to reach >80% accuracy before agent goes independent
- Extended to 20000 steps ensures gradual, gentle transition

---

### 2. Dense Step-wise Rewards ✅

**File**: `config_rotation.py`

```python
# NEW PARAMETERS
use_stepwise_rewards = True       # Enable dense feedback
stepwise_reward_weight = 2.0      # Weight for immediate correctness (alpha)
```

**File**: `train_rotation.py` - Reward Calculation

```python
# BEFORE: Only prediction reward
if config.reward_type == 'log_prob':
    reward = pred_logits

# AFTER: Combined rewards
# Prediction reward (from model accuracy)
prediction_reward = pred_logits

# Step-wise reward (immediate feedback)
correct_position = step  # Ground truth L2R: step 0 -> pos 0, step 1 -> pos 1
stepwise_reward = (actions == correct_position).float()  # 1.0 or 0.0

# Combined with weighting
total_reward = prediction_reward + config.stepwise_reward_weight * stepwise_reward
```

**Rationale**:
- Old reward was sparse: only feedback from final model predictions
- New reward is dense: immediate feedback at every step
- Agent knows RIGHT NOW if it picked the correct position
- `alpha=2.0` makes step-wise signal dominant early on

---

### 3. New Monitoring Metric ✅

**File**: `train_rotation.py` - Metrics

```python
# Calculate step-wise correctness
stepwise_correct = []
for step, actions in enumerate(actions_list):
    correct_position = step
    correct_rate = (actions == correct_position).float().mean().item()
    stepwise_correct.append(correct_rate)
avg_stepwise_correct = sum(stepwise_correct) / len(stepwise_correct)

# Added to metrics dict
metrics['avg_stepwise_correct'] = avg_stepwise_correct
```

**Purpose**: Track how often agent picks the correct position at each step

---

## Reward Structure Explained

### Two-Component Reward System

At each generation step `t`, the agent receives:

#### Component 1: Prediction Reward (Sparse)
```
prediction_reward = log P(correct_token | current_sequence)
Range: [-∞, 0]
```
- Based on model's confidence in predicting the token
- Sparse: Only meaningful when model is well-trained
- Noisy early on when model is still learning

#### Component 2: Step-wise Reward (Dense)
```
stepwise_reward = 1.0 if (action == correct_position) else 0.0
Range: [0, 1]
```
- Binary feedback: Did you pick the right position?
- Dense: Every step provides clear signal
- Immediate: No need to wait for model to evaluate

#### Combined Formula
```
total_reward = prediction_reward + alpha × stepwise_reward
             = log_prob + 2.0 × binary_correctness
```

### Why Alpha = 2.0?

| Component | Typical Range | With Alpha |
|-----------|---------------|------------|
| Prediction | [-2, 0] | [-2, 0] |
| Step-wise | [0, 1] | [0, 2.0] |

With `alpha=2.0`, a **correct position pick** (+2.0) can override a **poor prediction** (-2.0), ensuring the agent learns the order even before the model is perfect.

---

## Training Timeline

### Current Configuration
```python
max_iters = 7000
warmup_steps = 2000
teacher_forcing_decay_steps = 20000
```

⚠️ **Important Note**: With only 5000 co-evolution steps (7000 - 2000), the TF ratio will only decay from **1.0 → 0.75**. This is intentional for this short training run - the agent stays in "high guidance" mode throughout.

### Recommended for Full Training
```python
max_iters = 25000           # Allow full curriculum
warmup_steps = 2000         # Model learns basics
# Co-evolution: 23000 steps
# TF decay: 20000 steps → reaches TF=0.0 at iter 22000
# Final polish: 3000 steps with TF=0.0
```

### Phase Breakdown (Full Training)

| Phase | Iterations | TF Ratio | Agent Role | Goal |
|-------|-----------|----------|------------|------|
| Warmup | 0-2000 | N/A | Frozen | Model learns token prediction |
| High Guidance | 2000-7000 | 1.0→0.75 | 25% decisions | Agent learns basics with heavy guidance |
| Medium Guidance | 7000-12000 | 0.75→0.5 | 50% decisions | Agent builds confidence |
| Low Guidance | 12000-22000 | 0.5→0.0 | 50-100% decisions | Agent achieves independence |
| Independent | 22000-25000 | 0.0 | 100% decisions | Pure REINFORCE, polish policy |

---

## Expected Behavior

### ✅ Good Signs

1. **`avg_stepwise_correct` rising quickly**
   - Starts: ~5% (random guessing, 1/20 positions)
   - By iter 5000: >30%
   - By iter 10000: >50%
   - By iter 15000: >70%
   - **This is the leading indicator!**

2. **`l2r_order_correct` rising slowly**
   - Lags behind `avg_stepwise_correct`
   - Much harder: needs ALL 20 positions correct
   - By iter 10000: >10%
   - By iter 20000: >30%
   - Target: >50% by end

3. **`tf_ratio` decaying linearly**
   - Should form a smooth line from 1.0 → 0.0
   - No jumps or plateaus

4. **Model `accuracy` stable**
   - Should reach 75-80% during warmup
   - Should stay >70% throughout
   - If drops below 60%: agent is forcing bad trajectories

5. **Train loss ≈ Val loss**
   - Should be within 2x (e.g., 0.8 vs 1.2)
   - If gap >5x: memorization returning (shouldn't happen with online generation)

### ⚠️ Warning Signs

| Problem | Diagnosis | Fix |
|---------|-----------|-----|
| `avg_stepwise_correct` stuck at ~5-10% | Agent not learning from dense rewards | Increase `stepwise_reward_weight` to 3.0 or 4.0 |
| `l2r_order_correct` = 0% after 10k iters | TF too fast OR agent LR too low | Increase `agent_learning_rate` to 1e-4 or 5e-5 |
| Model accuracy drops to <60% | Agent forcing invalid trajectories | Slow down TF: increase `teacher_forcing_decay_steps` to 25000 |
| Train loss << Val loss | Memorization (shouldn't happen) | Verify `train_mode = 'online'` |

---

## Key Monitoring Metrics

### Primary Indicators (Watch These!)

1. **`train/avg_stepwise_correct`** 📈
   - Leading indicator of agent learning
   - Should rise steadily from 5% → 70%+

2. **`val/l2r_order_correct`** 🎯
   - Ultimate success metric
   - Target: >50% by end of training

3. **`train/tf_ratio`** 📉
   - Should decay smoothly
   - Verify curriculum is working

4. **`val/accuracy`** 🎓
   - Model quality indicator
   - Should stay >70%

### Secondary Indicators

5. **`val/first_step/p_t0`** - Should approach 1.0
6. **`val/last_step/p_t19`** - Should approach 1.0
7. **`val/kendall_tau`** - Rank correlation, should approach 0.9
8. **`train/avg_reward`** - Should increase (less negative)

---

## Theory: Why This Works

### The Problem Before

```
Sparse Reward + Fast Curriculum = Failure
      ↓                   ↓
Agent gets feedback     Agent loses guidance
only from final loss    before learning basics
      ↓                   ↓
No idea which step      Makes random guesses
was right or wrong      without correction
      ↓                   ↓
        Policy collapses
```

### The Solution Now

```
Dense Reward + Slow Curriculum = Success
      ↓                   ↓
Agent knows IMMEDIATELY Agent keeps guidance
which step is correct   while building skills
      ↓                   ↓
Learns step-by-step     Confidence grows gradually
which positions are     before going independent
correct
      ↓                   ↓
        Policy converges
```

### Analogy: Teaching a Child to Ride a Bike

**Before** (Fast Curriculum + Sparse Reward):
- Hold the bike for 30 seconds
- Let go completely
- Only tell them "you fell" after they crash
- Child never learns

**Now** (Slow Curriculum + Dense Reward):
- Hold the bike for 10 minutes (extended guidance)
- Gradually reduce support over 20 minutes
- Say "good!" immediately when they balance (dense feedback)
- Child learns to ride

---

## Code Snippets

### TF Ratio Calculation
```python
# In coevolution_step()
if global_step < config.teacher_forcing_decay_steps:
    tf_ratio = config.teacher_forcing_start - \
               (config.teacher_forcing_start - config.teacher_forcing_end) * \
               (global_step / config.teacher_forcing_decay_steps)
else:
    tf_ratio = config.teacher_forcing_end
```

### Action Override with TF
```python
# Sample from agent
actions, log_probs = agent.sample_action(hidden_states, filled_mask)

# Teacher forcing: override with correct action
if tf_ratio > 0:
    force_mask = torch.rand(B, device=device) < tf_ratio
    correct_actions = torch.full((B,), step, dtype=torch.long, device=device)
    actions = torch.where(force_mask, correct_actions, actions)
```

### Dense Reward Calculation
```python
# Prediction reward (sparse, from model)
prediction_reward = log_probs_model[batch_indices, actions, correct_tokens]

# Step-wise reward (dense, immediate)
if config.use_stepwise_rewards:
    correct_position = step  # L2R: step 0 -> position 0
    stepwise_reward = (actions == correct_position).float()

    # Combine
    total_reward = prediction_reward + \
                   config.stepwise_reward_weight * stepwise_reward
```

---

## Validation

Run the verification script to confirm all changes:

```bash
python linear_rotation_exp/verify_curriculum_rewards.py
```

Expected output:
```
✅ TF Decay Steps: 20000
✅ Dense Rewards: Enabled
✅ Reward Weight: 2.0
✅ All configurations verified
```

---

## Next Steps

### 1. Run Short Test (Current Config)
```bash
python train_rotation.py  # 7000 iterations
```

**What to expect**:
- TF stays in "high guidance" (1.0 → 0.75 only)
- `avg_stepwise_correct` should reach >30%
- `l2r_order_correct` may still be low (<10%)
- This is a **proof of concept** - shows dense rewards work

### 2. Full Training (Recommended)

Edit `config_rotation.py`:
```python
max_iters = 25000  # Allow full curriculum
```

Then run:
```bash
python train_rotation.py  # 25000 iterations
```

**What to expect**:
- Full TF decay: 1.0 → 0.0
- `avg_stepwise_correct` → 70%+
- `l2r_order_correct` → 50%+
- Agent achieves independence

---

## If Results Are Still Poor

### Tuning Knobs

1. **Slower Curriculum**:
   ```python
   teacher_forcing_decay_steps = 30000
   max_iters = 35000
   ```

2. **Stronger Dense Rewards**:
   ```python
   stepwise_reward_weight = 3.0  # or 4.0
   ```

3. **Higher Agent LR**:
   ```python
   agent_learning_rate = 5e-5  # or 1e-4
   ```

4. **Larger Model** (if Model accuracy < 70%):
   ```python
   n_layer = 4
   n_embd = 256
   ```

---

## Files Modified

1. ✅ `config_rotation.py`
   - Extended `teacher_forcing_decay_steps`: 5000 → 20000
   - Added `use_stepwise_rewards = True`
   - Added `stepwise_reward_weight = 2.0`

2. ✅ `train_rotation.py`
   - Modified reward calculation in `coevolution_step()`
   - Added step-wise reward component
   - Added `avg_stepwise_correct` metric

3. ✅ `verify_curriculum_rewards.py` (New)
   - Validation and explanation script

4. ✅ `CURRICULUM_REWARDS_SUMMARY.md` (This file)
   - Complete documentation

---

## Success Criteria

### Minimum Success (Short Training - 7000 iters)
- ✅ `avg_stepwise_correct` > 30%
- ✅ Model accuracy > 70%
- ✅ No overfitting (train loss ≈ val loss)

### Full Success (Extended Training - 25000 iters)
- 🎯 `avg_stepwise_correct` > 70%
- 🎯 `l2r_order_correct` > 50%
- 🎯 `first_step/p_t0` > 90%
- 🎯 Model accuracy > 75%

---

## Summary

**Three critical improvements**:
1. ✅ Online Generation (from previous fix) - Prevents memorization
2. ✅ Extended Curriculum (4x longer) - Gentler learning curve
3. ✅ Dense Step-wise Rewards - Immediate feedback

**Expected outcome**: Agent learns L2R order through combination of:
- Gentle guidance (slow TF decay)
- Immediate feedback (dense rewards)
- Diverse data (online generation)

**Next**: Run training and monitor `avg_stepwise_correct` - this is the key indicator! 🚀
