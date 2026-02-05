# Quick Start: Point-Based DRL System

## TL;DR

**Point-based system** = 4096 agents (64×64 grid), each controlling 1 point with tiny MLP actor (385 params).

**Patch-based system** = 64 agents (8×8 grid), each controlling 8×8 patch with CNN actor (50K params).

Both control the same 64×64 action field (4096 DOF).

## Files Overview

```
run0/
├── Patch-Based System (Original)
│   ├── stwEnv_pettingzoo.py              # 64 agents, patch observations
│   ├── models_consistency.py             # CNN networks
│   ├── stwStart_consistency.py           # Training script
│   └── config_consistency_with_memory.yaml
│
└── Point-Based System (New)
    ├── stwEnv_pettingzoo_point.py        # 4096 agents, scalar observations
    ├── models_point.py                   # MLP actor + hybrid CNN critic
    ├── stwStart_point.py                 # Training script
    ├── config_point.yaml                 # Configuration
    ├── test_point_models.py              # Unit tests (no MPI)
    └── POINT_BASED_IMPLEMENTATION_SUMMARY.md
```

## Quick Test (No MPI)

```bash
cd run0
conda run -n torch_modern python test_point_models.py
```

**Expected:** All tests pass ✓ (~30 seconds)

## Start Training (With MPI)

```bash
cd run0
python stwStart_point.py --config config_point.yaml
```

## Monitor Training

```bash
tensorboard --logdir=./logs_point --port=6006
```

Open browser: http://localhost:6006

## Key Differences

| What | Patch-Based | Point-Based |
|------|-------------|-------------|
| **Agents** | 64 | 4096 |
| **Obs shape** | (3, 8, 8) | (3,) |
| **Action shape** | (8, 8) | (1,) |
| **Actor** | CNN | Tiny MLP |
| **Actor params** | ~50K | 385 |
| **Buffer (500K)** | ~10 GB | ~76 GB |
| **Speed** | Slower | Faster |

## Memory Warning

**Point-based uses ~76 GB RAM** with 500K buffer capacity.

**If you have <100 GB RAM:**

Edit `config_point.yaml`:
```yaml
model:
  buffer_size: 100_000  # Reduce from 500K → ~15 GB
```

## Critical Settings (Don't Change!)

These ensure proper behavior:

```yaml
observation:
  include_prev_action: true  # REQUIRED for smoothness

model:
  smoothness:
    lambda_temporal: 0.0        # Disabled (use action memory)
    lambda_spatial: 0.0         # Disabled (no patches)
    lambda_global_mean: 0.05    # REQUIRED for mass conservation
```

## What to Watch

### During Training

1. **Action mean → 0** (`Episode/action_mean`)
   - Should be < 0.01
   - If not, zero-mean constraint is broken!

2. **Drag reduction** (`Episode/dpdx_mean`)
   - Uncontrolled: -0.0042
   - Target: -0.002
   - Lower is better

3. **Gradient norms** (`Episode/actor_grad_norm`)
   - Should be stable (1-10)
   - If >50: exploding gradients

4. **Training step** (`Training/training_step`)
   - Starts at 0, increases after warmup (~5400 steps)

### First 3 Episodes (Warmup)

- Only basic metrics logged
- No training updates yet
- Buffer filling (need 5400 steps)

### After Episode 4+

- All loss metrics appear
- Training updates active
- Check for NaN/Inf

## Troubleshooting

### "Out of Memory"
→ Reduce `buffer_size` in config

### "Actions not zero-mean"
→ Check TensorBoard: `Episode/action_mean` should be ~0
→ If not, zero-mean constraint is broken (bug in code)

### "Training too slow"
→ Reduce `gradient_steps` (64 → 32)
→ Reduce `buffer_size` (500K → 200K)

### "NaN/Inf in losses"
→ Check `Episode/actor_grad_norm` (should be <50)
→ Reduce learning rates
→ Increase `gradient_clip`

### "Drag not reducing"
→ Wait for more episodes (may need 10-20)
→ Check if consistency loss is dominating (disable if so)
→ Increase `lambda_global_mean` for smoother actions

## Comparison with Patch-Based

Run both systems side-by-side to compare:

**Patch-based:**
```bash
python stwStart_consistency.py --config config_consistency_with_memory.yaml
tensorboard --logdir=./logs_consistency --port=6007
```

**Point-based:**
```bash
python stwStart_point.py --config config_point.yaml
tensorboard --logdir=./logs_point --port=6006
```

Compare:
- Training speed (steps/sec)
- Drag reduction (dpdx_mean)
- Action smoothness (visualizations)
- Memory usage

## Next Steps

1. **Test models** (no MPI): `python test_point_models.py`
2. **Adjust memory** if needed: edit `buffer_size` in config
3. **Start training**: `python stwStart_point.py --config config_point.yaml`
4. **Monitor**: `tensorboard --logdir=./logs_point`
5. **Wait ~5-10 episodes** for meaningful results

## Expected Results

- **First 3 episodes:** Warmup, no training
- **Episodes 4-10:** Training starts, losses appear
- **Episodes 10-20:** Drag starts reducing
- **Episodes 20+:** Convergence (if successful)

**Successful training:**
- `action_mean` < 0.01 (zero-mean verified)
- `dpdx_mean` decreasing (drag reducing)
- `actor_grad_norm` stable (1-10)
- No NaN/Inf

## Documentation

- **Summary:** `POINT_BASED_IMPLEMENTATION_SUMMARY.md` (this directory)
- **Project guide:** `../CLAUDE.md`
- **Zero-mean details:** `../report/ZERO_MEAN_FIX_SUMMARY.md`

---

**Questions?** Check the full implementation summary or project instructions.
