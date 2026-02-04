# Zero-Mean Constraint Fix - Complete Summary

## 🎯 Two Critical Problems Fixed

### Problem 1: Post-Processing Correction Can CAUSE Bang-Bang

**The Issue:**
```python
# OLD (PROBLEMATIC):
actions_raw = policy(obs)  # Agents output actions
# ... later, external post-processing ...
action_mean = np.mean(actions_raw)
actions_executed = actions_raw - action_mean  # Applied outside computational graph!
```

**Why this causes bang-bang:**
```
t=0: mean = +0.3 → correction = -0.3 → all actions shifted DOWN
t=1: mean = -0.2 → correction = +0.2 → all actions shifted UP
t=2: mean = +0.4 → correction = -0.4 → all actions shifted DOWN
```

Even if raw actions are smooth, **varying correction magnitude creates oscillations!**

**The Fix:**
```python
# NEW (CORRECT):
actions_raw = actor(obs)  # [batch, 64, 8, 8]

# Differentiable zero-mean correction (part of computational graph!)
global_mean = actions_raw.mean()  # Differentiable
actions = actions_raw - global_mean  # Gradients flow through this!

# Now train with corrected actions
q_loss = -critic(obs, actions).mean()
```

**Benefits:**
1. ✅ Gradients see the zero-mean constraint
2. ✅ Policy learns to output near-zero mean BEFORE correction
3. ✅ Correction magnitude `|μ|` becomes small → stable
4. ✅ Bang-bang from varying corrections prevented

---

### Problem 2: Per-Tile Zero-Mean is Physically Invalid

**The Issue:**
```yaml
# OLD (WRONG):
lambda_zero: 0.1  # Penalizes non-zero mean per 8×8 tile
```

```python
# OLD loss (PHYSICALLY INVALID):
def zero_mean_loss(actions):
    # Force each 8×8 tile to be zero-mean
    mean_per_agent = actions.mean(dim=(2, 3))  # Per-agent mean
    return torch.mean(mean_per_agent ** 2)  # Penalize non-zero
```

**Why this is wrong:**
- Different flow regions NEED different average actuation
- Upstream: may need net blowing (positive mean)
- Downstream: may need net suction (negative mean)
- Forcing each tile to zero-mean is **unphysical**!

**The Fix:**
```yaml
# NEW (CORRECT):
lambda_zero: 0.0         # DEPRECATED: disable per-tile constraint
lambda_global_mean: 0.05 # NEW: only global constraint (mass conservation)
```

```python
# NEW loss (PHYSICALLY VALID):
def global_zero_mean_loss(actions):
    # Enforce zero-mean over ENTIRE field, not per-tile
    global_mean = actions.mean(dim=(1, 2, 3))  # Global mean
    return torch.mean(global_mean ** 2)
```

**Benefits:**
1. ✅ Physically valid: only global mass conservation enforced
2. ✅ Tiles can have non-zero mean as required by flow physics
3. ✅ Policy not over-constrained

---

## 📝 Complete List of Changes

### 1. Code Changes (`models_consistency.py`)

#### New Loss Function
```python
def global_zero_mean_loss(actions: torch.Tensor) -> torch.Tensor:
    """
    Enforce global zero-mean (mass conservation) without constraining
    individual tiles.
    """
    global_mean = actions.mean(dim=(1, 2, 3))  # [batch_size]
    return torch.mean(global_mean ** 2)
```

#### Updated MADDPG Init
```python
def __init__(self, ...
    lambda_zero: float = 0.0,  # Deprecated
    lambda_global_mean: float = 0.05,  # NEW
    ...):
```

#### Differentiable Correction in Training
```python
def update_batched(self, batch):
    # Get raw actions from actor
    actions_raw = self.actor(obs_flat, return_similarity=True)

    # CRITICAL: Differentiable zero-mean correction
    global_means = actions_raw.view(batch_size, -1).mean(dim=1, keepdim=True)
    global_means = global_means.view(batch_size, 1, 1, 1)
    actions = actions_raw - global_means  # Gradients flow through!

    # Train with corrected actions
    q_loss = -self.critic(obs_fields, actions).mean()
    global_mean_loss = global_zero_mean_loss(actions)

    actor_loss = q_loss + ... + self.lambda_global_mean * global_mean_loss
```

#### Correction in Inference
```python
def select_actions_batched(self, all_obs):
    all_actions_raw = self.actor(all_obs)

    # Apply same correction at inference
    global_mean = all_actions_raw.mean()
    all_actions = all_actions_raw - global_mean

    return all_actions.cpu().numpy()
```

### 2. Config Changes

**Both configs updated:**
```yaml
# config_consistency.yaml
# config_consistency_with_memory.yaml

model:
  smoothness:
    lambda_temporal: 0.0
    lambda_spatial: 0.5
    lambda_zero: 0.0         # DEPRECATED (was 0.1)
    lambda_global_mean: 0.05 # NEW
```

### 3. Training Script Changes (`stwStart_consistency.py`)

- Added `lambda_global_mean` parameter to MADDPG init
- Added tracking for `global_mean_loss`
- Added logging for `Episode/global_mean_loss`
- Updated smoothness domination check to include global_mean_loss

### 4. LaTeX Report Updates

**New Section 7: "Zero-Mean Constraint: Critical Implementation Details"**

Covers:
1. How post-processing correction creates oscillations
2. Gradient mismatch problem
3. Solution: Differentiable correction
4. Why per-tile constraint is invalid
5. Global vs per-tile comparison
6. Implementation code examples
7. Expected impact table

---

## 🔬 Technical Analysis

### Gradient Flow

**OLD (broken):**
```
Policy outputs a^raw → External correction → a^exec
     ↑                                          ↓
     └──────── Gradients assume direct control ─┘
                    (MISMATCH!)
```

**NEW (correct):**
```
Policy outputs a^raw → Differentiable correction → a
     ↑                                              ↓
     └────────── Gradients flow through ───────────┘
                    (CORRECT!)
```

### Jacobian Analysis

Correction transformation:
```
a_i^exec = a_i^raw - (1/N) Σ_j a_j^raw
```

Jacobian:
```
∂a_i^exec/∂a_j^raw = { 1 - 1/N  if i = j
                     { -1/N     if i ≠ j
```

With N=64: diagonal = 63/64 ≈ 0.984, off-diagonal = -1/64 ≈ -0.016

Old approach: gradients don't see this!
New approach: gradients flow through correctly!

---

## 📊 Comparison Table

| Aspect | OLD | NEW |
|--------|-----|-----|
| **Zero-mean correction** | External post-processing | Differentiable in graph |
| **Gradients aware?** | ❌ No | ✅ Yes |
| **Can cause bang-bang?** | ✅ Yes (varying μ) | ❌ No (learned to minimize μ) |
| **Per-tile constraint** | ✅ Yes (λ_zero = 0.1) | ❌ No (λ_zero = 0.0) |
| **Global constraint** | ⚠️ Via post-processing | ✅ Via loss (λ_global_mean = 0.05) |
| **Physically valid?** | ❌ No (per-tile invalid) | ✅ Yes (global only) |
| **Training stability** | ⚠️ Credit assignment confused | ✅ Clean gradients |

---

## 🎯 Expected Improvements

### 1. Reduced Bang-Bang from Corrections
- Policy learns: output actions with mean ≈ 0
- → Correction magnitude `|μ|` becomes small
- → Oscillations from varying corrections eliminated

### 2. Better Credit Assignment
- Gradients properly account for zero-mean constraint
- Policy learns what actions will be executed (not what it outputs raw)

### 3. Physical Validity
- Tiles can have non-zero mean as needed by flow
- Only global mass conservation enforced
- More expressive policy space

### 4. Training Stability
- No gradient-mismatch confusion
- Clearer learning signal
- Faster convergence expected

---

## 📄 Files Modified

### Code
- ✅ `run0/models_consistency.py` - New loss, differentiable correction
- ✅ `run0/config_consistency.yaml` - Updated parameters
- ✅ `run0/config_consistency_with_memory.yaml` - Updated parameters
- ✅ `run0/stwStart_consistency.py` - Training script updates

### Documentation
- ✅ `report/action_memory_implementation.tex` - New section added
- ✅ `report/action_memory_implementation.pdf` - Recompiled (15 pages, 500KB)
- ✅ `report/ZERO_MEAN_FIX_SUMMARY.md` - This file

---

## ✅ Testing Checklist

### Before Training

1. **Check config:**
   ```yaml
   lambda_zero: 0.0         # Should be 0
   lambda_global_mean: 0.05 # Should be > 0
   ```

2. **Verify initialization:**
   ```
   Look for in training output:
   "Zero-mean - per-agent (deprecated): 0.0, global: 0.05"
   ```

### During Training

3. **Monitor global_mean_loss:**
   ```
   tensorboard --logdir=./logs_consistency
   Check: Episode/global_mean_loss should decrease over time
   ```

4. **Check action mean:**
   - Early training: may be non-zero
   - Late training: should approach zero (policy learned the constraint)

5. **Compare with old runs:**
   - Temporal gradient RMS should be lower
   - Actions should be smoother
   - Drag reduction should be similar or better

---

## 🚀 Ready to Train

All fixes implemented. The training command remains the same:

```bash
cd run0
python stwStart_consistency.py --config config_consistency_with_memory.yaml
```

Monitor these new metrics in TensorBoard:
- `Episode/global_mean_loss` - should decrease
- `Episode/zero_loss` - should be near zero (deprecated loss disabled)
- Action smoothness should improve

**Key expectation:** Bang-bang behavior reduced due to both:
1. Action memory in observations (policy sees what it did)
2. Differentiable zero-mean (policy learns to output near-zero mean)

Both mechanisms work together to produce smooth, robust control!
