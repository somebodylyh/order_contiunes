# Align AR & MDM Baseline Experiments Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `train_ar.py` and `train_mdm.py` use统一的 AR-order 评估指标，两者出现在同一 wandb 图上，直接可比。

**Architecture:** Extract a shared `evaluate_ar()` function into `baseline_continuous/eval_utils.py`. Both scripts import并使用它。统一 wandb metric naming，加 `group` 参数让 runs 关联。**评估标准：都用 AR 顺序生成 loss + cosine similarity**，直接回答"MDM 的随机序训练是否影响了 AR 顺序下的预测能力"。

**Tech Stack:** PyTorch, wandb, existing ContinuousAOGPT model

---

## Problem Analysis

| Issue | train_ar.py | train_mdm.py |
|-------|------------|--------------|
| val metric names | `val/loss`, `val/cos_sim` | `val/ar_loss`, `val/ar_cos_sim`, `val/random_loss`, `val/random_cos_sim` |
| Evaluation method | AR-order only (correct) | AR + Random MC (unnecessarily complex) |
| wandb group | none | none |
| wandb step | `iter` as data field | `iter` as data field |

## Design Decision

**统一用 AR 顺序评估。** 理由：
- 两个模型都能做 AR 顺序的 `model(vectors, mode='AR')` — 同一个前向过程，完全可比
- 衡量的是同一件事："给定前面的向量，预测下一个的能力"
- 无需 MC 采样，确定性指标，评估更快
- 直接回答核心问题：random-order training 对 AR 预测能力有帮助还是有伤害？

## Unified Metric Schema

After changes, **both** scripts will log:

```
train/loss        — training loss (各自的训练 mode：AR用升序，MDM用随机序)
train/lr          — learning rate
val/loss          — validation loss under AR order (1 - cosine_similarity)
val/cos_sim       — validation cosine similarity under AR order
```

Final evaluation adds `final/` prefix versions for val and test.

---

### Task 1: Create shared evaluation module `eval_utils.py`

**Files:**
- Create: `baseline_continuous/eval_utils.py`
- Create: `tests/test_eval_utils.py`

**Step 1: Write the failing test**

Create `tests/test_eval_utils.py`:

```python
"""Tests for shared evaluation utility."""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import pytest
from baseline_continuous.eval_utils import evaluate_ar
from baseline_continuous.continuous_aogpt import ContinuousAOGPT, ContinuousAOGPTConfig


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, n=16, seq_len=8, dim=16):
        self.vectors = torch.randn(n, seq_len, dim)
        self.shuffled = self.vectors[:, torch.randperm(seq_len), :]

    def __len__(self):
        return len(self.vectors)

    def __getitem__(self, idx):
        return {'vectors': self.vectors[idx], 'shuffled_vectors': self.shuffled[idx]}


@pytest.fixture
def model_and_loader():
    config = ContinuousAOGPTConfig(block_size=8, vector_dim=16, n_layer=1, n_head=2, n_embd=32)
    model = ContinuousAOGPT(config)
    ds = FakeDataset(n=16, seq_len=8, dim=16)
    loader = torch.utils.data.DataLoader(ds, batch_size=4)
    return model, loader


def test_evaluate_ar_returns_expected_keys(model_and_loader):
    model, loader = model_and_loader
    results = evaluate_ar(model, loader, device='cpu', max_batches=2)
    expected_keys = {'val_loss', 'val_cos_sim'}
    assert set(results.keys()) == expected_keys


def test_evaluate_ar_values_are_finite(model_and_loader):
    model, loader = model_and_loader
    results = evaluate_ar(model, loader, device='cpu', max_batches=2)
    for k, v in results.items():
        assert isinstance(v, float), f"{k} is not float"
        assert not (v != v), f"{k} is NaN"


def test_evaluate_ar_loss_in_range(model_and_loader):
    """Cosine loss = 1 - cos_sim, should be in [0, 2]."""
    model, loader = model_and_loader
    results = evaluate_ar(model, loader, device='cpu', max_batches=2)
    assert 0.0 <= results['val_loss'] <= 2.0


def test_evaluate_ar_cos_sim_in_range(model_and_loader):
    """Cosine similarity should be in [-1, 1]."""
    model, loader = model_and_loader
    results = evaluate_ar(model, loader, device='cpu', max_batches=2)
    assert -1.0 <= results['val_cos_sim'] <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/AO-GPT-MDM && python -m pytest tests/test_eval_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'baseline_continuous.eval_utils'`

**Step 3: Write minimal implementation**

Create `baseline_continuous/eval_utils.py`:

```python
"""Shared evaluation utilities for baseline experiments."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def evaluate_ar(model, val_loader, device, max_batches=None):
    """
    Evaluate model using AR (ascending) order.

    Both AR and MDM models use this same function so metrics are directly comparable.
    Input is always batch['shuffled_vectors'] — the model sees shuffled data
    and predicts in ascending order via mode='AR'.

    Returns:
        dict with keys: val_loss, val_cos_sim
    """
    model.eval()
    total_loss = 0.0
    total_cos_sim = 0.0
    total_batches = 0

    for i, batch in enumerate(val_loader):
        if max_batches is not None and i >= max_batches:
            break

        vectors = batch['shuffled_vectors'].to(device)
        predictions, loss = model(vectors, mode='AR')

        # predictions[:, :-1, :] predicts vectors (targets in AR order = vectors themselves)
        shift_preds = predictions[:, :-1, :]
        cos_sim = F.cosine_similarity(shift_preds, vectors, dim=-1).mean()

        total_loss += loss.item()
        total_cos_sim += cos_sim.item()
        total_batches += 1

    n = max(total_batches, 1)
    model.train()
    return {
        'val_loss': total_loss / n,
        'val_cos_sim': total_cos_sim / n,
    }
```

**Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/AO-GPT-MDM && python -m pytest tests/test_eval_utils.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add baseline_continuous/eval_utils.py tests/test_eval_utils.py
git commit -m "feat: add shared evaluate_ar() for consistent baseline evaluation"
```

---

### Task 2: Update `train_ar.py` to use shared evaluation and unified wandb metrics

**Files:**
- Modify: `baseline_continuous/train_ar.py`

**Step 1: Write the failing test**

Add to `tests/test_eval_utils.py`:

```python
def test_train_ar_imports_eval_utils():
    """Verify train_ar uses the shared evaluate_ar function."""
    import baseline_continuous.train_ar as mod
    source = open(mod.__file__).read()
    assert 'from baseline_continuous.eval_utils import evaluate_ar' in source
    assert "group='baseline-comparison'" in source
```

**Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/AO-GPT-MDM && python -m pytest tests/test_eval_utils.py::test_train_ar_imports_eval_utils -v`
Expected: FAIL

**Step 3: Modify `train_ar.py`**

Changes:

1. **Add import, remove old evaluate function:**
   - Add: `from baseline_continuous.eval_utils import evaluate_ar`
   - Delete the entire local `evaluate()` function (lines 52-78)

2. **Update wandb.init** (line 150) — add `group`:
   ```python
   wandb.init(project=cfg.wandb_project, name=run_name, group='baseline-comparison', config={...})
   ```

3. **Update train wandb.log** (line 204) — add `step=it`, remove `'iter': it`:
   ```python
   wandb.log({'train/loss': loss.item(), 'train/lr': lr}, step=it)
   ```

4. **Update evaluation block** (line 212) — replace `evaluate(...)` call:
   ```python
   eval_results = evaluate_ar(model, val_loader, device)
   print(f"  [eval] val_loss: {eval_results['val_loss']:.4f} | val_cos_sim: {eval_results['val_cos_sim']:.4f}")
   ```

5. **Update eval wandb.log** (line 217) — unified names + step:
   ```python
   wandb.log({
       'val/loss': eval_results['val_loss'],
       'val/cos_sim': eval_results['val_cos_sim'],
   }, step=it)
   ```

6. **Update best model check** (line 224):
   ```python
   if cfg.save_best_model and eval_results['val_loss'] < best_val_loss:
       best_val_loss = eval_results['val_loss']
   ```

7. **Update final evaluation** (lines 238-257) — use `evaluate_ar` for val and test:
   ```python
   final_results = evaluate_ar(model, val_loader, device)
   print(f"  val_loss: {final_results['val_loss']:.4f}")
   print(f"  val_cos_sim: {final_results['val_cos_sim']:.4f}")

   test_results = evaluate_ar(model, test_loader, device)
   print(f"  test_loss: {test_results['val_loss']:.4f}")
   print(f"  test_cos_sim: {test_results['val_cos_sim']:.4f}")

   if wandb_log:
       import wandb
       wandb.log({
           'final/val_loss': final_results['val_loss'],
           'final/val_cos_sim': final_results['val_cos_sim'],
           'final/test_loss': test_results['val_loss'],
           'final/test_cos_sim': test_results['val_cos_sim'],
       })
   ```

8. **Remove `no_shuffle` from evaluate calls** — `evaluate_ar` always uses `shuffled_vectors`.

**Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/AO-GPT-MDM && python -m pytest tests/test_eval_utils.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add baseline_continuous/train_ar.py tests/test_eval_utils.py
git commit -m "refactor: train_ar uses shared evaluate_ar with unified wandb metrics"
```

---

### Task 3: Update `train_mdm.py` to use shared evaluation and unified wandb metrics

**Files:**
- Modify: `baseline_continuous/train_mdm.py`

**Step 1: Write the failing test**

Add to `tests/test_eval_utils.py`:

```python
def test_train_mdm_imports_eval_utils():
    """Verify train_mdm uses the shared evaluate_ar function."""
    import baseline_continuous.train_mdm as mod
    source = open(mod.__file__).read()
    assert 'from baseline_continuous.eval_utils import evaluate_ar' in source
    assert "group='baseline-comparison'" in source
```

**Step 2: Run test to verify it fails**

Run: `cd /home/admin/lyuyuhuan/AO-GPT-MDM && python -m pytest tests/test_eval_utils.py::test_train_mdm_imports_eval_utils -v`
Expected: FAIL

**Step 3: Modify `train_mdm.py`**

Changes:

1. **Add import, remove old evaluate function:**
   - Add: `from baseline_continuous.eval_utils import evaluate_ar`
   - Delete the entire local `evaluate()` function (lines 54-105)

2. **Update wandb.init** (line 177) — add `group`:
   ```python
   wandb.init(project=cfg.wandb_project, name=f'mdm-{train_mode}', group='baseline-comparison', config={...})
   ```

3. **Update train wandb.log** (line 233) — add `step=it`, remove `'iter': it`:
   ```python
   wandb.log({'train/loss': loss.item(), 'train/lr': lr}, step=it)
   ```

4. **Update evaluation block** (line 241) — replace old evaluate call:
   ```python
   eval_results = evaluate_ar(model, val_loader, device)
   print(f"  [eval] val_loss: {eval_results['val_loss']:.4f} | val_cos_sim: {eval_results['val_cos_sim']:.4f}")
   ```

5. **Update eval wandb.log** (line 248) — unified names + step:
   ```python
   wandb.log({
       'val/loss': eval_results['val_loss'],
       'val/cos_sim': eval_results['val_cos_sim'],
   }, step=it)
   ```

6. **Update best model check** (line 258) — use `val_loss`:
   ```python
   if cfg.save_best_model and eval_results['val_loss'] < best_val_loss:
       best_val_loss = eval_results['val_loss']
   ```

7. **Update final evaluation** (lines 273-300) — use `evaluate_ar`, same format as train_ar.py:
   ```python
   final_results = evaluate_ar(model, val_loader, device)
   print(f"  val_loss: {final_results['val_loss']:.4f}")
   print(f"  val_cos_sim: {final_results['val_cos_sim']:.4f}")

   test_results = evaluate_ar(model, test_loader, device)
   print(f"  test_loss: {test_results['val_loss']:.4f}")
   print(f"  test_cos_sim: {test_results['val_cos_sim']:.4f}")

   if wandb_log:
       import wandb
       wandb.log({
           'final/val_loss': final_results['val_loss'],
           'final/val_cos_sim': final_results['val_cos_sim'],
           'final/test_loss': test_results['val_loss'],
           'final/test_cos_sim': test_results['val_cos_sim'],
       })
   ```

**Step 4: Run test to verify it passes**

Run: `cd /home/admin/lyuyuhuan/AO-GPT-MDM && python -m pytest tests/test_eval_utils.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add baseline_continuous/train_mdm.py tests/test_eval_utils.py
git commit -m "refactor: train_mdm uses shared evaluate_ar with unified wandb metrics"
```

---

### Task 4: Verify end-to-end consistency

**Step 1: Dry-run both scripts** (short iteration, no wandb)

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM
python baseline_continuous/train_ar.py --max_iters 20 --wandb_log false --batch_size 8
python baseline_continuous/train_mdm.py --max_iters 20 --wandb_log false --batch_size 8
```

Expected: Both print eval results with identical metric names: `val_loss`, `val_cos_sim`

**Step 2: Run full test suite**

```bash
cd /home/admin/lyuyuhuan/AO-GPT-MDM && python -m pytest tests/test_eval_utils.py -v
```

Expected: All tests PASS

**Step 3: Commit**

```bash
git add -A
git commit -m "test: verify aligned baseline experiments end-to-end"
```

---

## Summary of Changes

| File | Action | Purpose |
|------|--------|---------|
| `baseline_continuous/eval_utils.py` | CREATE | Shared `evaluate_ar()` — AR-order eval only |
| `baseline_continuous/train_ar.py` | MODIFY | Use `evaluate_ar`, unified metric names, wandb group |
| `baseline_continuous/train_mdm.py` | MODIFY | Use `evaluate_ar`, unified metric names, wandb group |
| `tests/test_eval_utils.py` | CREATE | Tests for shared evaluation |

After implementation, both runs will:
- Appear under the same wandb group `baseline-comparison`
- Log identical metric names (`val/loss`, `val/cos_sim`)
- Use `step=it` for aligned x-axes
- Share the same `evaluate_ar()` function for guaranteed consistency
- 直接回答：MDM 随机序训练 vs AR 升序训练，谁在 AR 预测任务上更强？
