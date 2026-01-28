# Observation-Similarity Consistency Loss Implementation

## Problem

Multi-agent DRL for turbulent flow control produces spatially aggressive actions with discontinuities at patch boundaries. Each of 64 agents controls an 8×8 patch, but they can't coordinate during decentralized execution.

## Solution

Implemented **observation-similarity consistency loss**: if two observations are similar in learned feature space, their corresponding actions should be similar. This encourages the shared policy to be smooth in observation space, which translates to spatial smoothness when adjacent flow regions have similar features.

## Files Created

| File | Description |
|------|-------------|
| `models_consistency.py` | Actor with similarity branch, Critic with circular padding, consistency loss function, modified MADDPG class |
| `config_consistency.yaml` | Configuration with consistency parameters and tuning guide |
| `stwStart_consistency.py` | Training script with consistency loss logging |

## Key Changes

### 1. Actor Network (`CNNActorWithSimilarity`)
- **Standard padding** (not circular) — 8×8 patches are local windows, not periodic
- Added **similarity branch**: Conv 1×1 → GlobalAvgPool → 16-dim feature vector
- Features used for consistency loss during training only

### 2. Critic Network (`CNNCriticCircular`)
- **Circular padding** — full 64×64 domain IS periodic in x and y
- Better handles periodic boundary conditions in the flow field

### 3. Consistency Loss
```python
def observation_consistency_loss(sim_features, actions, tau=0.9, margin=0.1, boundary_only=True):
    # 1. Compute cosine similarity between agent observations
    # 2. Find pairs with similarity > tau
    # 3. Penalize action differences beyond margin
    # 4. Focus on boundary pixels for smoother patch transitions
```

### 4. Temporal Loss Disabled
- Set `lambda_temporal=0.0` (was 0.5)
- Comparing current actions to replay buffer actions from OLD policy creates confusing gradients
- Observation-consistency handles spatial coordination instead

## Default Parameters

```yaml
smoothness:
  lambda_temporal: 0.0      # Disabled
  lambda_spatial: 0.5       # Within-patch smoothness
  lambda_zero: 0.1          # Per-agent zero-mean
  consistency:
    enable: true
    lambda: 0.1             # Start low, increase if still aggressive
    tau_similarity: 0.9     # Threshold for "similar" observations
    margin: 0.1             # Allowed action difference
    boundary_only: true     # Focus on patch edges
    warmup_steps: 5000      # Delay before enabling
```

## Tuning Guide

**If policy becomes too stiff** (drag reduction drops):
- Decrease `consistency.lambda`: 0.1 → 0.01
- Increase `tau_similarity`: 0.9 → 0.95
- Increase `margin`: 0.1 → 0.2

**If policy is still too aggressive** (sharp boundaries):
- Increase `consistency.lambda`: 0.1 → 0.5
- Decrease `tau_similarity`: 0.9 → 0.85
- Decrease `margin`: 0.1 → 0.05

## Usage

```bash
cd run0
mpirun -np 1 python stwStart_consistency.py --config config_consistency.yaml
```

## TensorBoard Metrics

- `Loss/consistency` — consistency loss value (should decrease)
- `Training/consistency_active` — 1 when warmup complete
- `Episode/consistency_loss` — per-episode average
