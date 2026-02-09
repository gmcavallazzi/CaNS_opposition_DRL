# Evaluation Scripts Guide

## Overview
This guide documents which evaluation scripts work with which training configurations after downloading results from HPC.

---

## Configuration Types

### 1. **config_consistency_with_memory_fix1** (Patch-Based)
- **Architecture:** 64 agents, each controlling 8×8 patch
- **Observations:** `[64, 3, 8, 8]` (u, w, prev_action)
- **Actions:** `[64, 8, 8]`
- **Total DOF:** 4096 points

### 2. **config_point** (Point-Based)
- **Architecture:** 4096 agents, each controlling 1 point
- **Observations:** `[4096, 3]` scalars (u, w, prev_action)
- **Actions:** `[4096, 1]` scalars
- **Total DOF:** 4096 points

---

## Evaluation Scripts Compatibility

### ✅ **Works for BOTH configs** (with minor adaptations)

#### 1. `analyze_tensorboard.py`
**Purpose:** Extract and analyze TensorBoard training metrics

**Usage:**
```bash
python analyze_tensorboard.py <log_directory>
```

**Compatibility:**
- ✅ **fix1:** Full support
- ✅ **point:** Full support
- **Data agnostic** - only reads scalar metrics from TensorBoard logs

**Key Metrics Analyzed:**
- Drag reduction (dpdx)
- Actor/critic losses
- Gradient norms
- Smoothness penalties
- Zero-mean constraint

---

### ⚠️ **Patch-based ONLY** (needs modification for point)

#### 2. `test_policy_smoothness.py`
**Purpose:** Run episodes with trained policy and compute smoothness metrics

**Usage:**
```bash
python test_policy_smoothness.py \
  --checkpoint ../checkpoints/model.pt \
  --config config_consistency_with_memory_fix1.yaml \
  --num_episodes 5
```

**Compatibility:**
- ✅ **fix1:** Full support (designed for patch-based)
- ❌ **point:** **REQUIRES MODIFICATION**

**Issues for point-based:**
```python
# Line 164: Assumes obs is [64, 2-3, 8, 8]
obs_array = np.stack([obs[agent] for agent in agents])

# Line 169: Assumes actions are [64, 8, 8]
actions = maddpg.select_actions_batched(obs_torch)  # [64, 8, 8]

# Line 225: Hardcoded reshape for 8×8 agent grid
actions_grid = actions.reshape(T, 8, 8, H, W)  # [T, 8_i, 8_j, 8, 8]
```

**Fix needed:**
- Import `stwEnv_pettingzoo_point.py` instead of `stwEnv_pettingzoo.py`
- Import `models_point.py` instead of `models_consistency.py`
- Handle scalar observations `[4096, 3]` instead of spatial `[64, 3, 8, 8]`
- Handle scalar actions `[4096, 1]` instead of spatial `[64, 8, 8]`
- Adjust spatial analysis (no 8×8 agent grid for point-based)

---

#### 3. `analyze_smoothness_offline.py`
**Purpose:** Analyze saved episode data (NPZ files) from test runs

**Usage:**
```bash
python analyze_smoothness_offline.py
# Expects data in: smoothness_check/episode_4_data.npz
```

**Compatibility:**
- ✅ **fix1:** Full support (designed for patch-based)
- ❌ **point:** **REQUIRES MODIFICATION**

**Issues for point-based:**
```python
# Line 16-17: Expects patch-based data shapes
observations = data['observations']  # [T, 64, 2, 8, 8]
actions = data['actions']            # [T, 64, 8, 8]

# Line 58-63: Hardcoded patch reconstruction
for i in range(8):
    for j in range(8):
        action_field[i*8:(i+1)*8, j*8:(j+1)*8] = actions[t, agent_idx]

# Line 130-134: Assumes 8×8 agent grid
actions_grid = actions_focused.reshape(-1, 8, 8, 8, 8)
```

**Fix needed:**
- Handle `actions = [T, 4096, 1]` instead of `[T, 64, 8, 8]`
- Simple reshape: `action_field = actions.reshape(T, 64, 64)` (no patch assembly)
- Skip agent-grid-based spatial gradient analysis (not applicable)

---

#### 4. `analyze_smoothness_custom.py`
**Purpose:** Flexible version of offline analysis with CLI arguments

**Usage:**
```bash
python analyze_smoothness_custom.py \
  --data_dir smoothness_check1 \
  --episode 2 \
  --t_start 0 \
  --t_end 1000
```

**Compatibility:**
- ✅ **fix1:** Full support
- ❌ **point:** **Same issues as `analyze_smoothness_offline.py`**

**Advantage:**
- More flexible (can specify data directory, episode, time range)
- Same fix needed as offline version

---

## Recommended Workflow

### For `config_consistency_with_memory_fix1` (Patch-Based)

#### Step 1: Download from HPC
```bash
# Download TensorBoard logs
scp -r hpc:~/project/logs_consistency ./

# Download checkpoints
scp -r hpc:~/project/checkpoints_consistency ./
```

#### Step 2: Analyze Training Metrics
```bash
cd run0
python analyze_tensorboard.py ../logs_consistency/maddpg_consistency_agents64_TIMESTAMP
```

**Expected output:**
- Drag reduction progress (should reach ~88-90%)
- Gradient clipping analysis (check if >50% clipping)
- Zero-mean constraint status (should be <1e-5)
- Smoothness loss trends

#### Step 3: Test Policy Smoothness
```bash
python test_policy_smoothness.py \
  --checkpoint ../checkpoints_consistency/latest_model.pt \
  --config config_consistency_with_memory_fix1.yaml \
  --num_episodes 5
```

**Generates:**
- `smoothness_analysis_TIMESTAMP/`
  - `summary.json` (temporal/spatial metrics)
  - `smoothness_analysis.png` (variance, FFT)
  - `input_output_comparison.png` (u, w, action time series)
  - `episode_N_data.npz` (raw data for further analysis)

#### Step 4: Detailed Offline Analysis
```bash
python analyze_smoothness_custom.py \
  --data_dir smoothness_analysis_TIMESTAMP \
  --episode 0 \
  --t_start 0 \
  --t_end 1800
```

**Generates:**
- Action evolution plots
- Temporal/spatial gradient fields
- FFT spectrum analysis
- Input-output correlation

---

### For `config_point` (Point-Based)

#### Step 1: Download from HPC
```bash
scp -r hpc:~/project/logs_point ./
scp -r hpc:~/project/checkpoints_point ./
```

#### Step 2: Analyze Training Metrics
```bash
cd run0
python analyze_tensorboard.py ../logs_point/maddpg_point_agents4096_TIMESTAMP
```

**✅ Works without modification** (TensorBoard logs are identical format)

#### Step 3: Test Policy Smoothness
**✅ NOW AVAILABLE** - Use point-specific script:

```bash
python test_policy_smoothness_point.py \
  --checkpoint ../checkpoints_point/latest_model.pt \
  --config config_point.yaml \
  --num_episodes 5
```

**Or use auto-detect wrapper:**
```bash
python test_policy_smoothness_auto.py \
  --checkpoint ../checkpoints_point/latest_model.pt \
  --config config_point.yaml \
  --num_episodes 5
```

**Generates:**
- `smoothness_analysis_point_TIMESTAMP/`
  - `summary.json` (includes `"architecture": "point-based"`)
  - `smoothness_analysis.png` (variance, FFT)
  - `input_output_comparison.png` (u, w, action time series)
  - `episode_N_data.npz` (raw data: actions=[T, 4096])

#### Step 4: Offline Analysis
**✅ NOW AVAILABLE** - Use point-specific script:

```bash
python analyze_smoothness_offline_point.py \
  --data_dir smoothness_analysis_point_TIMESTAMP \
  --episode 0 \
  --t_start 0 \
  --t_end 1800
```

**Generates:**
- Action evolution plots (64×64 field visualization)
- Temporal/spatial gradient fields
- FFT spectrum analysis
- Input-output correlation

---

## Quick Compatibility Table

| Script | fix1 (Patch) | point | Notes |
|--------|-------------|-------|-------|
| `analyze_tensorboard.py` | ✅ | ✅ | Works for both |
| `test_policy_smoothness.py` | ✅ | ❌ | Patch-only |
| `test_policy_smoothness_point.py` | ❌ | ✅ | Point-only (NEW) |
| `test_policy_smoothness_auto.py` | ✅ | ✅ | **Auto-detects** (NEW) |
| `analyze_smoothness_offline.py` | ✅ | ❌ | Patch-only |
| `analyze_smoothness_offline_point.py` | ❌ | ✅ | Point-only (NEW) |
| `analyze_smoothness_custom.py` | ✅ | ❌ | Patch-only |

---

## Key Differences to Handle

### Data Shapes

| Aspect | Patch-Based (fix1) | Point-Based |
|--------|-------------------|-------------|
| **Observations** | `[64, 3, 8, 8]` | `[4096, 3]` |
| **Actions** | `[64, 8, 8]` | `[4096, 1]` |
| **Reconstruction** | Assemble 64 patches | Simple reshape |
| **Spatial analysis** | Per-agent gradients | Per-point gradients |

### Analysis Considerations

**Patch-based:**
- Spatial gradients between agents (patch boundaries)
- Within-patch smoothness
- Agent-level consistency

**Point-based:**
- Direct field gradients (simpler)
- No patch boundaries
- Point-level consistency (expensive - 4096² comparisons)

---

## ✅ NEW: Point-Based Analysis Scripts (CREATED)

### ✅ `test_policy_smoothness_point.py`
**Status:** COMPLETE

**Features:**
- Handles point-based models (4096 agents, scalar obs/actions)
- Imports `stwEnv_pettingzoo_point` and `models_point`
- Direct reshape: `actions.reshape(T, 64, 64)` (no patch assembly)
- Computes spatial gradients on 64×64 field
- Same output format as patch-based version

**Usage:**
```bash
python test_policy_smoothness_point.py \
  --checkpoint ../checkpoints_point/model.pt \
  --config config_point.yaml \
  --num_episodes 5
```

### ✅ `analyze_smoothness_offline_point.py`
**Status:** COMPLETE

**Features:**
- Analyzes NPZ files from point-based test runs
- Expects `actions` shape `[T, 4096]` instead of `[T, 64, 8, 8]`
- Direct field reconstruction (no patch assembly)
- Same visualizations as patch-based version

**Usage:**
```bash
python analyze_smoothness_offline_point.py \
  --data_dir smoothness_analysis_point_TIMESTAMP \
  --episode 0 \
  --t_start 0 \
  --t_end 1800
```

### ✅ `test_policy_smoothness_auto.py`
**Status:** COMPLETE

**Features:**
- **Auto-detects architecture** from config file
- Dispatches to appropriate script (patch or point)
- Single unified entry point

**Detection logic:**
1. Check config filename for "point"
2. Check `net_arch.actor_hidden_dim` (point) vs `net_arch.actor_channels` (patch)
3. Default to patch-based

**Usage:**
```bash
# Works for BOTH configs!
python test_policy_smoothness_auto.py \
  --checkpoint ../checkpoints/model.pt \
  --config config_consistency_with_memory_fix1.yaml \
  --num_episodes 5

python test_policy_smoothness_auto.py \
  --checkpoint ../checkpoints_point/model.pt \
  --config config_point.yaml \
  --num_episodes 5

# Or force architecture:
python test_policy_smoothness_auto.py \
  --checkpoint model.pt \
  --config config.yaml \
  --force_arch point
```

---

## Summary

### **Current Status:**

✅ **ALL analysis scripts now available for BOTH configs!**

**Universal (works for both):**
- `analyze_tensorboard.py` - TensorBoard log analysis
- `test_policy_smoothness_auto.py` - Auto-detects architecture

**Patch-based specific:**
- `test_policy_smoothness.py`
- `analyze_smoothness_offline.py`
- `analyze_smoothness_custom.py`

**Point-based specific:**
- `test_policy_smoothness_point.py`
- `analyze_smoothness_offline_point.py`

### **Recommended Workflow (BOTH configs):**

#### Download from HPC
```bash
scp -r hpc:~/project/logs* ./
scp -r hpc:~/project/checkpoints* ./
```

#### Quick Analysis (Auto-Detect)
```bash
cd run0

# Step 1: Analyze training logs
python analyze_tensorboard.py ../logs_*/run_TIMESTAMP

# Step 2: Test policy smoothness (auto-detects architecture!)
python test_policy_smoothness_auto.py \
  --checkpoint ../checkpoints*/model.pt \
  --config config_*.yaml \
  --num_episodes 5

# Step 3: Offline analysis of results
# For patch-based:
python analyze_smoothness_offline.py

# For point-based:
python analyze_smoothness_offline_point.py \
  --data_dir smoothness_analysis_point_TIMESTAMP \
  --episode 0
```

---

**Last updated:** 2026-02-09 (Point-based scripts added)
