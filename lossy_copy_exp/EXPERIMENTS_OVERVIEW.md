# LO-ARMs Experiments Overview

This directory contains two complementary experiments for learning optimal generation order via reinforcement learning.

---

## 📚 Experiment 1: Lossy Copy (2 Variables)

**File**: `train_loarms.py`
**Config**: `config_lossy_copy.py`
**Dataset**: `lossy_copy_dataset.py`

### Task
- Variables: `[x, y]` where `y = x // k`
- Causal structure: x → y (deterministic)
- Sequence length: 2

### Hypothesis
Agent should learn to generate x before y.

### Expected Results
- `P(select_x_first)` → 1.0
- Model accuracy > 95%

### Run
```bash
python lossy_copy_exp/train_loarms.py
```

---

## 📚 Experiment 2: Modular Sum (3 Variables)

**File**: `train_modular_sum.py`
**Config**: `config_modular_sum.py`
**Dataset**: `modular_sum_dataset.py`

### Task
- Variables: `[x1, x2, y]`
- Two modes controlled by `use_lossy` switch:

#### Mode A: Lossy (`use_lossy=True`)
```python
y = (x1 + x2) // 2
```
- Strong causal asymmetry: x1, x2 → y
- Many-to-one mapping

**Expected**: Agent learns to select x1 or x2 first, never y first
- `P(select_any_x_first)` → 100%
- `P(select_y_first)` → 0%

#### Mode B: Modular (`use_lossy=False`)
```python
y = (x1 + x2) % P
```
- Complete symmetry: any two determine the third
- No causal preference

**Expected**: Agent shows no preference
- `P(select_y_first)` stays ~33%
- OR: Mode collapse to arbitrary order

### Run
```bash
# Lossy mode
./lossy_copy_exp/run_modular_sum.sh --lossy

# Modular mode
./lossy_copy_exp/run_modular_sum.sh --modular
```

---

## 🔍 Comparison

| Aspect | Lossy Copy | Modular Sum (Lossy) | Modular Sum (Modular) |
|--------|------------|---------------------|----------------------|
| Variables | 2 | 3 | 3 |
| Computation | y = x // k | y = (x1+x2) // 2 | y = (x1+x2) % P |
| Causality | Strong (x→y) | Strong (x1,x2→y) | None (symmetric) |
| Agent Goal | Learn x first | Learn x before y | Discover symmetry |
| Key Metric | P(x first) → 1.0 | P(y first) → 0.0 | P(y first) ~ 0.33 |

---

## 📁 File Organization

```
lossy_copy_exp/
│
├── Experiment 1: Lossy Copy (2-variable)
│   ├── lossy_copy_dataset.py
│   ├── config_lossy_copy.py
│   ├── train_loarms.py
│   └── run_experiment.sh
│
├── Experiment 2: Modular Sum (3-variable)
│   ├── modular_sum_dataset.py
│   ├── config_modular_sum.py
│   ├── train_modular_sum.py
│   ├── test_modular_sum.py
│   ├── run_modular_sum.sh
│   ├── README_MODULAR_SUM.md
│   └── MODULAR_SUM_SUMMARY.md
│
├── Shared Components
│   ├── model_wrapper.py          # AOGPT with hidden states & logical embeddings
│   ├── order_policy_net.py       # Agent (OrderPolicyNet)
│   └── utils.py                  # Metrics, logging, checkpointing
│
└── Documentation
    ├── README.md                  # Original experiment guide
    ├── IMPLEMENTATION_SUMMARY.md  # Lossy Copy implementation
    └── EXPERIMENTS_OVERVIEW.md    # This file
```

---

## 🎯 Research Questions

### Q1: Can agents discover simple causality? (Experiment 1)
- Task: `y = x // k`
- Test: Does agent learn x → y order?
- **Status**: ✅ Implemented

### Q2: Can agents distinguish causal structures? (Experiment 2)
- Task A: Lossy `y = (x1+x2)//2` (strong causality)
- Task B: Modular `y = (x1+x2)%P` (symmetric)
- Test: Does agent behave differently in A vs B?
- **Status**: ✅ Implemented

### Q3: Does causal discovery scale to longer sequences?
- Future work: Extend to 4+ variables
- Future work: More complex causal graphs

---

## 🚀 Getting Started

### First Time Setup
```bash
conda activate order_lando
cd /home/admin/lyuyuhuan/AO-GPT-MDM
```

### Run Tests
```bash
# Test 2-variable experiment
./lossy_copy_exp/run_experiment.sh --test

# Test 3-variable experiment
./lossy_copy_exp/run_modular_sum.sh --test
```

### Run Experiments

**Experiment 1: Lossy Copy**
```bash
python lossy_copy_exp/train_loarms.py
```

**Experiment 2A: Modular Sum (Lossy)**
```bash
./lossy_copy_exp/run_modular_sum.sh --lossy
```

**Experiment 2B: Modular Sum (Modular)**
```bash
./lossy_copy_exp/run_modular_sum.sh --modular
```

---

## 📊 Key Metrics Summary

### Experiment 1 (2-variable)
Monitor: `prob_select_x_first`
- Target: > 0.9
- Interpretation: Agent learns causal order

### Experiment 2A (3-variable Lossy)
Monitor: `probs/select_y_first`
- Target: < 0.05
- Interpretation: Agent never selects y first

Monitor: `probs/select_any_x_first`
- Target: > 0.95
- Interpretation: Agent always selects x before y

### Experiment 2B (3-variable Modular)
Monitor: `probs/select_y_first`
- Target: ~ 0.33 (flat)
- Interpretation: Agent recognizes symmetry

---

## 🔬 Analysis Workflow

1. **Run Exp 1**: Verify basic causal learning works
2. **Run Exp 2A**: Verify agent learns complex causality (x1,x2→y)
3. **Run Exp 2B**: Verify agent recognizes symmetry
4. **Compare**: Plot Exp 2A vs 2B curves side-by-side
5. **Conclude**: Agent can distinguish causal structures

---

## 📖 Documentation

- **Quick Start**: `README.md`
- **Lossy Copy Details**: `IMPLEMENTATION_SUMMARY.md`
- **Modular Sum Details**: `MODULAR_SUM_SUMMARY.md`
- **Full Guide**: `README_MODULAR_SUM.md`
- **This Overview**: `EXPERIMENTS_OVERVIEW.md`

---

## ✅ Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| 2-variable dataset | ✅ | Lossy Copy |
| 2-variable training | ✅ | train_loarms.py |
| 3-variable dataset | ✅ | With use_lossy switch |
| 3-variable training | ✅ | Extended rollout & metrics |
| Model flexibility | ✅ | Supports any block_size |
| Agent flexibility | ✅ | Supports any num_positions |
| Tests | ✅ | Both experiments tested |
| Documentation | ✅ | Comprehensive guides |

---

**All experiments ready for execution! 🚀**
