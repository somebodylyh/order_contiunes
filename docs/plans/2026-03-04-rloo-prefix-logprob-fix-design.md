# Design: Fix RLOO Prefix Log-Prob Bug

Date: 2026-03-04

## Problem

In the current LO-ARM RLOO implementation, the REINFORCE term uses the wrong
`log_q`. The paper (Eq. 11 / Eq. 16) requires two distinct quantities:

| Usage | Symbol | Meaning |
|-------|--------|---------|
| Inside F_θ | `log q(z_i \| z_{<i}, x)` | log prob of **current step k** only |
| REINFORCE coefficient | `log q(z_{<i} \| x)` | **cumulative prefix** log prob = Σ_{j=0}^{k-1} log PL(z_j\|z_{<j}) |

Current code passes the same single-step `log_q` to both places. This causes:
- The q-net REINFORCE gradient to be wrong (step k instead of prefix 0..k-1)
- F term and REINFORCE term to compete on the same log_q, causing instability
- Policy entropy collapse (pol_ent → 0) and training loss spikes

## Fix Design

### New function: `plackett_luce_prefix_logprob` in `loarm_utils.py`

Vectorized computation of the prefix cumulative log prob:

```
log q(z_{<k}|x) = Σ_{j=0}^{k-1} [logits[z_j] - logsumexp(logits[z_j:])]
```

Implementation: reorder logits by permutation z, compute reversed cumulative
logsumexp, subtract to get per-step log probs, sum over j < k.
Edge case: k=0 → return zeros [B] (empty prefix).

### Modified signature: `compute_rloo_loss` in `rloo.py`

Split the single `log_q` parameter into two:
- `log_q1_step`, `log_q2_step`: used inside F_θ (already embedded, passed for symmetry)
- `log_q1_prefix`, `log_q2_prefix`: used in the REINFORCE coefficient

```python
direct_term    = -0.5 * L * (F1 + F2)
reinforce_term =  0.5 * L * (F1-F2).detach() * (log_q1_prefix - log_q2_prefix)
loss = (direct_term + reinforce_term).mean()
```

### Updated caller: `compute_batch_loss` in `train_loarm.py`

Compute both quantities and pass them separately:

```python
log_q1_step   = plackett_luce_logprob(q_logits, z1, step_k=k)
log_q2_step   = plackett_luce_logprob(q_logits, z2, step_k=k)
log_q1_prefix = plackett_luce_prefix_logprob(q_logits, z1, step_k=k)
log_q2_prefix = plackett_luce_prefix_logprob(q_logits, z2, step_k=k)

F1 = log_p_gen1 + log_p_pol1 - log_q1_step
F2 = log_p_gen2 + log_p_pol2 - log_q2_step
loss = compute_rloo_loss(F1, F2, log_q1_prefix, log_q2_prefix, L_len=L)
```

## Files Changed

| File | Change |
|------|--------|
| `baseline_continuous/loarm_utils.py` | Add `plackett_luce_prefix_logprob()` |
| `baseline_continuous/rloo.py` | Split log_q into step vs prefix params |
| `baseline_continuous/train_loarm.py` | Compute and pass both log_q variants |

Unit tests for `rloo.py` and `loarm_utils.py` must be updated/extended.

## Expected Effect

- q-net receives correct REINFORCE gradient across the full prefix
- Competing gradient directions (F vs REINFORCE on same step) eliminated
- Expected: more stable training loss, slower but healthier policy entropy decay
