# Quick Start: Evaluating Training Results

## Overview

After downloading training results from HPC, use these scripts to evaluate performance and smoothness.

---

## 1. Download Results from HPC

```bash
# Download everything
scp -r hpc:~/project/logs* ./
scp -r hpc:~/project/checkpoints* ./
```

---

## 2. Analyze Training Metrics (Both Configs)

```bash
cd run0

# For fix1 (patch-based)
python analyze_tensorboard.py ../logs_consistency/maddpg_consistency_agents64_TIMESTAMP

# For point (point-based)
python analyze_tensorboard.py ../logs_point/maddpg_point_agents4096_TIMESTAMP
```

**Output:** Terminal summary with:
- Drag reduction progress
- Gradient clipping analysis
- Zero-mean constraint status
- Loss trends
- Recommendations

---

## 3. Test Policy Smoothness

### Option A: Auto-Detect (Recommended)

```bash
# Works for BOTH configs automatically!
python test_policy_smoothness_auto.py \
  --checkpoint ../checkpoints/model.pt \
  --config config_consistency_with_memory_fix1.yaml \
  --num_episodes 5

python test_policy_smoothness_auto.py \
  --checkpoint ../checkpoints_point/model.pt \
  --config config_point.yaml \
  --num_episodes 5
```

### Option B: Manual (Explicit Architecture)

**For fix1 (patch-based):**
```bash
python test_policy_smoothness.py \
  --checkpoint ../checkpoints_consistency/latest_model.pt \
  --config config_consistency_with_memory_fix1.yaml \
  --num_episodes 5
```

**For point (point-based):**
```bash
python test_policy_smoothness_point.py \
  --checkpoint ../checkpoints_point/latest_model.pt \
  --config config_point.yaml \
  --num_episodes 5
```

**Output:** Creates directory `smoothness_analysis_[point_]TIMESTAMP/` with:
- `summary.json` - Numerical metrics
- `smoothness_analysis.png` - Variance and FFT plots
- `input_output_comparison.png` - Time series at tracked point
- `episode_N_data.npz` - Raw data for further analysis

---

## 4. Offline Analysis (Detailed Visualizations)

After running step 3, analyze the saved episode data:

**For fix1 (patch-based):**
```bash
python analyze_smoothness_offline.py
# Reads from: smoothness_check/episode_4_data.npz
```

**For point (point-based):**
```bash
python analyze_smoothness_offline_point.py \
  --data_dir smoothness_analysis_point_TIMESTAMP \
  --episode 0 \
  --t_start 0 \
  --t_end 1800
```

**Output:** Additional plots in same directory:
- `action_evolution.png` - Action field snapshots over time
- `temporal_gradients.png` - Action changes between timesteps
- `spatial_gradients.png` - Gradients between adjacent points/agents
- `smoothness_metrics_timeseries.png` - RMS metrics over time
- `fft_analysis.png` - Frequency spectrum analysis
- `input_output_point32_32.png` - Correlation at specific point

---

## Key Metrics to Check

### From TensorBoard Analysis:

**Good signs:**
- ✅ Drag reduction progress >50% (dpdx improving toward -0.002)
- ✅ Action mean <1e-5 (perfect zero-mean constraint)
- ✅ Actor grad norm stable and varying (not stuck at clip threshold)
- ✅ Consistency loss stable or decreasing

**Warning signs:**
- ⚠️ Gradient clipping frequency >50% (increase `gradient_clip`)
- ⚠️ Consistency/spatial loss increasing (smoothness penalties too weak)
- ⚠️ Drag reduction plateaued <50% (may need more training or tuning)

### From Smoothness Analysis:

**Goal metrics (from report):**
- Temporal gradient RMS < 0.2 (smooth temporal evolution)
- FFT dominated by low frequencies (< 1 Hz preferred)
- Strong W vs Action correlation (opposition control learned)
- Spatial gradients smooth (no checkerboard patterns)

**Compare fix1 vs point:**
- Point-based may have sharper spatial gradients (finer control)
- Patch-based may have smoother fields (spatial smoothness penalty)
- Both should achieve similar drag reduction

---

## Example Workflow

```bash
# 1. Download from HPC
scp -r hpc:~/project/logs_consistency ./
scp -r hpc:~/project/checkpoints_consistency ./

# 2. Quick analysis
cd run0
python analyze_tensorboard.py ../logs_consistency/maddpg_consistency_agents64_20260204_144414

# 3. Test policy
python test_policy_smoothness_auto.py \
  --checkpoint ../checkpoints_consistency/latest_model.pt \
  --config config_consistency_with_memory_fix1.yaml \
  --num_episodes 5

# 4. Detailed offline analysis
python analyze_smoothness_custom.py \
  --data_dir smoothness_analysis_20260209_123456 \
  --episode 0 \
  --t_start 0 \
  --t_end 1800

# 5. Review plots
open smoothness_analysis_20260209_123456/*.png
```

---

## Troubleshooting

### "No module named 'stwEnv_pettingzoo_point'"
- You're using `test_policy_smoothness.py` (patch-only) on a point-based checkpoint
- Solution: Use `test_policy_smoothness_point.py` or `test_policy_smoothness_auto.py`

### "Checkpoint not found"
- Check path is correct relative to `run0/`
- Example: `../checkpoints_consistency/latest_model.pt`

### "Shape mismatch" during evaluation
- Checkpoint architecture doesn't match config
- Ensure using correct config file for the checkpoint

### Evaluation runs but crash during episode
- May need CaNS solver running (testing requires MPI environment)
- Alternative: Only analyze TensorBoard logs (step 2 only)

---

## Scripts Summary

| Script | Purpose | Configs |
|--------|---------|---------|
| `analyze_tensorboard.py` | Training metrics from logs | Both |
| `test_policy_smoothness_auto.py` | Auto-detect + test policy | Both |
| `test_policy_smoothness.py` | Test patch-based policy | fix1 only |
| `test_policy_smoothness_point.py` | Test point-based policy | point only |
| `analyze_smoothness_offline.py` | Detailed offline (patch) | fix1 only |
| `analyze_smoothness_offline_point.py` | Detailed offline (point) | point only |
| `analyze_smoothness_custom.py` | Flexible offline (patch) | fix1 only |

---

**Recommendation:** Always start with `test_policy_smoothness_auto.py` - it auto-detects the architecture and works for both configs!
