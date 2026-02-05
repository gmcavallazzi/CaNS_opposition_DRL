# CLAUDE.md - Deep Reinforcement Learning for Opposition Control

This file provides guidance to Claude Code when working with the DRL-based opposition control implementation for turbulent flow control.

## Repository Overview

This repository combines:
- **CaNS (Canonical Navier-Stokes Solver)** - DNS solver in `src/` directory
- **Multi-Agent RL** - PettingZoo-based training environment with MADDPG
- **Opposition Control** - Learning wall-normal actuation patterns to reduce drag

## Code Structure

```
run0/
├── stwEnv_pettingzoo.py           # PettingZoo parallel environment (64 agents, 8×8 grid)
├── models_consistency.py          # MADDPG with consistency regularization + action memory
├── stwStart_consistency.py        # Training script
├── config_consistency.yaml        # Base config (2-channel obs)
├── config_consistency_with_memory.yaml  # Config with action memory (3-channel obs)
└── utils.py                       # Reward computation, curriculum learning

report/
├── action_memory_implementation.tex  # Full technical documentation (15 pages)
├── network_diagram.tex              # TikZ architecture diagram
└── ZERO_MEAN_FIX_SUMMARY.md        # Zero-mean constraint implementation details
```

## Critical Implementation Details

### 1. Action Memory (Bang-Bang Prevention)

**Problem:** Policy exhibited high-frequency oscillations (bang-bang control) due to:
- Closed-loop feedback instability
- Policy only seeing instantaneous observations without temporal context
- No memory of previous actions

**Solution:** State augmentation with previous action
```python
# OLD: 2-channel observations
obs = [u, w]  # Shape: (2, 8, 8)

# NEW: 3-channel observations (action memory enabled)
obs = [u, w, prev_action]  # Shape: (3, 8, 8)
```

**Configuration:**
```yaml
observation:
  include_prev_action: true   # Enable action memory
  prev_action_scale: 1.0      # Scaling factor for prev_action channel
```

**Implementation Notes:**
- Environment tracks `prev_action_field` (64×64) and extracts patches for each agent
- At episode reset, `prev_action_field` initialized to zeros
- Actor network automatically adapts to 2 or 3 input channels based on obs shape
- Critic receives 4 channels: `[u, w, prev_action, current_action]`
- **Backwards compatible**: Old configs (2-channel) still work without modification

### 2. Zero-Mean Constraint (Mass Conservation)

**Critical Requirements:**
1. Actions must be **exactly** zero-mean for CaNS solver stability
2. Constraint must be **differentiable** (part of computational graph)
3. Only **global** zero-mean is physically valid (not per-tile)

**Implementation Flow:**
```python
# During training:
actions_raw = actor(obs)  # From tanh ∈ [-1, 1]

# Differentiable zero-mean correction (gradients flow through!)
global_mean = actions_raw.mean()
actions = actions_raw - global_mean  # Still zero-mean

# Add exploration noise (zero-mean)
if noise_scale > 0:
    noise = np.random.normal(0, noise_scale, actions.shape)
    noise = noise - noise.mean()  # Ensure noise is zero-mean
    actions = actions + noise
    actions = np.clip(actions, -1, 1)  # May break zero-mean!

# CRITICAL: Re-apply zero-mean after clipping
actions = actions - actions.mean()  # Guarantees exact zero-mean
```

**Key Points:**
- **Clipping only needed when noise > 0** (late training with no noise: skip clipping)
- **Zero-mean correction is differentiable** during training → policy learns to output near-zero mean
- **Final correction after clipping** guarantees exact zero-mean for solver
- **No external post-processing** in environment (breaks gradient flow)

**Configuration:**
```yaml
model:
  smoothness:
    lambda_zero: 0.0          # DEPRECATED: Per-tile constraint (physically invalid)
    lambda_global_mean: 0.05  # NEW: Global zero-mean penalty (valid)
    lambda_spatial: 0.5       # Spatial smoothness
```

**Why Per-Tile is Invalid:**
- Different flow regions require different average actuation
- Upstream may need net blowing (positive mean)
- Downstream may need net suction (negative mean)
- Forcing each 8×8 tile to zero-mean is an artificial constraint

### 3. Replay Buffer (Variable Observation Channels)

**Issue:** Buffer must support both 2-channel and 3-channel observations for backwards compatibility.

**Implementation:**
```python
class BatchedReplayBuffer:
    def __init__(self, capacity, n_agents, agent_ids, obs_channels=2):
        self.obs_channels = obs_channels  # 2 or 3
        self.obs_buf = np.zeros((capacity, n_agents, obs_channels, 8, 8))
        self.next_obs_buf = np.zeros((capacity, n_agents, obs_channels, 8, 8))
```

**Backwards Compatibility in `load()`:**
- Loading 2-channel buffer into 3-channel buffer → pads with zeros for prev_action
- Loading 3-channel buffer into 2-channel buffer → drops prev_action channel

**Initialization:**
```python
# Training script auto-detects channels from environment
input_channels = obs_shape[0]  # 2 or 3
replay_buffer = BatchedReplayBuffer(buffer_size, num_agents, agents,
                                    obs_channels=input_channels)
```

### 4. TensorBoard Logging (Per-Episode Aggregation)

**Philosophy:** All metrics logged at **episode end**, not per-step, for clarity and performance.

**Implementation Pattern:**
```python
# During episode: collect metrics
episode_actor_grad_norms = []
episode_critic_grad_norms = []

# During training update:
episode_actor_grad_norms.append(actor_metrics['global_norm'])
episode_critic_grad_norms.append(critic_metrics['global_norm'])

# At episode end: log averages
writer.add_scalar('Episode/actor_grad_norm', np.mean(episode_actor_grad_norms), episode)
writer.add_scalar('Episode/critic_grad_norm', np.mean(episode_critic_grad_norms), episode)

# After logging: reset lists
episode_actor_grad_norms = []
```

**Logged Metrics (per episode):**
- **Performance**: reward, dpdx_mean/min/max (drag)
- **Losses**: actor_loss, critic_loss, spatial_loss, global_mean_loss, consistency_loss
- **Gradients**: actor_grad_norm, critic_grad_norm, exploding/vanishing indicators
- **Training**: actor_lr, critic_lr, gradient_clip, training_step, consistency_active
- **Actions**: action_mean (should → 0), action_std, noise_scale
- **Histograms**: actions, q_values (every 5 episodes)

**Rare/Debug Metrics (step-based):**
- Layer-wise gradients (every 200 steps, if enabled)
- Activation health (every 500 steps)

### 5. Training Warmup

**Critical:** Training does **not** start immediately. Need to fill replay buffer first.

**Configuration:**
```yaml
model:
  batch_size: 32
  train_freq: 300         # Train every N steps
  learning_starts: 5400   # Buffer warmup: ~3 episodes before training
```

**Training Condition:**
```python
if replay_buffer.size > learning_starts and total_steps % train_freq == 0:
    # Train networks
    maddpg.update_batched(batch)
```

**Implications:**
- First ~3 episodes: only basic metrics logged (reward, dpdx, action stats)
- After buffer fills: all loss/gradient metrics appear
- `training_step` counter only increments during actual training
- Consistency loss activates after additional warmup (default 5000 training steps)

### 6. Network Architecture

**Actor (CNNActorWithSimilarity):**
- **Input**: (2 or 3, 8, 8) observation patch per agent
- **Encoder**: Conv layers → features
- **Similarity branch**: Global average pooling → similarity embedding
- **Decoder**: Transposed conv → action map
- **Output**: (8, 8) action via `tanh` ∈ [-1, 1]

**Critic (CNNCriticCircular):**
- **Input**: Full field (2 or 3, 64, 64) obs + (1, 64, 64) action = (3 or 4, 64, 64)
- **Architecture**: Circular convolutions (periodic boundaries) + MLP
- **Output**: Scalar Q-value

**Key Point:** Critic needs both `prev_action` and `current_action` for context-dependent evaluation:
```python
# Critic input with action memory
critic_input = [u, w, prev_action, current_action]  # 4 channels
Q = critic(critic_input)  # Evaluates: "how good is current_action given prev_action?"
```

### 7. Common Pitfalls & Solutions

**Pitfall 1: "Why are metrics missing after 1 episode?"**
- **Cause**: Buffer hasn't filled to `learning_starts` threshold
- **Solution**: Wait 3-4 episodes or reduce `learning_starts` in config

**Pitfall 2: "Actions have non-zero mean!"**
- **Cause**: Post-processing correction outside computational graph
- **Solution**: Use differentiable correction in training script, not environment

**Pitfall 3: "Bang-bang behavior persists"**
- **Cause**: No action memory, or per-tile zero-mean constraint
- **Solution**: Enable `include_prev_action: true`, set `lambda_zero: 0.0`

**Pitfall 4: "Replay buffer shape mismatch"**
- **Cause**: Buffer initialized with wrong number of channels
- **Solution**: Pass `obs_channels` parameter during initialization

**Pitfall 5: "Clipping breaks zero-mean"**
- **Cause**: `np.clip()` can make zero-mean arrays non-zero-mean
- **Solution**: Always apply `actions = actions - actions.mean()` after clipping

**Pitfall 6: "Training is too slow"**
- **Cause**: Logging every training step instead of every episode
- **Solution**: Accumulate metrics during episode, log averages at end

## Configuration Files

### Base Configuration (2-channel, no action memory)
`config_consistency.yaml` - Standard training without temporal context

### Action Memory Configuration (3-channel)
`config_consistency_with_memory.yaml` - **Recommended** for smooth control

Key differences:
```yaml
observation:
  include_prev_action: true    # Enable action memory

model:
  smoothness:
    lambda_zero: 0.0           # Disable per-tile constraint
    lambda_global_mean: 0.05   # Enable global constraint
    lambda_spatial: 0.5        # Spatial smoothness
    lambda_temporal: 0.0       # Not needed with action memory
```

## Training Workflow

### 1. Start Training
```bash
cd run0
python stwStart_consistency.py --config config_consistency_with_memory.yaml
```

### 2. Monitor TensorBoard
```bash
tensorboard --logdir=./logs_consistency --port=6006
```

### 3. Key Metrics to Watch

**Episode 1-3 (warmup):**
- Only basic metrics appear
- `training_step` stays at 0
- Collecting data for replay buffer

**After Episode 4+ (training active):**
- All loss metrics appear
- `Episode/global_mean_loss` should decrease (policy learning constraint)
- `Episode/action_mean` should approach 0
- `Episode/actor_grad_norm` should be stable (<10)
- `Episode/dpdx_mean` should decrease (drag reduction)

**Convergence indicators:**
- `global_mean_loss` < 0.01 (policy outputs near-zero mean)
- `action_mean` < 0.05 (corrections are small)
- `actor_exploding` = 0 (stable training)
- Smooth action trajectories (view histograms)

## Documentation

- **Full Technical Report**: `report/action_memory_implementation.pdf` (15 pages)
- **Network Diagram**: `report/network_diagram.pdf`
- **Zero-Mean Details**: `report/ZERO_MEAN_FIX_SUMMARY.md`
- **CaNS Solver**: `src/` directory

## Important Reminders

1. **Always use action memory** (`include_prev_action: true`) for smooth control
2. **Zero-mean must be differentiable** - apply in training script, not environment
3. **Global constraint only** (`lambda_zero: 0.0`, `lambda_global_mean > 0`)
4. **Training starts after warmup** - need ~3 episodes before metrics appear
5. **Clipping only when noise > 0** - optimization for inference
6. **All metrics logged per episode** - not per training step
7. **Replay buffer needs obs_channels** - for backwards compatibility
8. **Critic needs all 4 channels** - [u, w, prev_action, current_action]

## Next Steps for Development

- Monitor training convergence (5-10 episodes)
- Analyze smoothness metrics (temporal gradient RMS, FFT spectrum)
- Compare with/without action memory
- Test generalization to different flow conditions
- Deploy optimal policy for validation runs
