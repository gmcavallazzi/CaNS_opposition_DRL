"""
Offline analysis of smoothness data with detailed visualizations.
Focus on timesteps 1000-1250 of the last episode.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Load last episode data
print("Loading episode 4 data...")
data = np.load('smoothness_check/episode_4_data.npz')

# Extract data
observations = data['observations']  # [T, 64, 2, 8, 8]
actions = data['actions']            # [T, 64, 8, 8]
rewards = data['rewards']
dpdx = data['dpdx']

T, N, H, W = actions.shape
print(f"Data shape: T={T}, N_agents={N}, H={H}, W={W}")
print(f"Time range: 0 to {T-1}")

# Physical time information
dt_sim = 0.0085  # Simulation timestep
action_every = 6  # Actions computed every 6 simulation steps
dt_action = action_every * dt_sim  # = 0.051
print(f"\nTiming:")
print(f"  Simulation dt: {dt_sim}")
print(f"  Action computed every {action_every} sim steps")
print(f"  Effective action dt: {dt_action}")

# Focus on timesteps 0-250
t_start = 0
t_end = 250
physical_time_start = t_start * dt_action
physical_time_end = t_end * dt_action
print(f"\nFocusing on timesteps {t_start} to {t_end}")
print(f"  Physical time: {physical_time_start:.2f} to {physical_time_end:.2f} (duration: {physical_time_end - physical_time_start:.2f})")

actions_focused = actions[t_start:t_end]  # [250, 64, 8, 8]
obs_focused = observations[t_start:t_end]  # [250, 64, 2, 8, 8]

# ============================================================================
# VISUALIZATION 1: Action field evolution over time
# ============================================================================
print("\nCreating action field evolution plot...")

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

# Select 12 timesteps evenly spaced in the range
time_indices = np.linspace(t_start, t_end-1, 12, dtype=int)

for idx, t in enumerate(time_indices):
    # Reconstruct full 64x64 action field from 64 agents
    action_field = np.zeros((64, 64))
    agent_idx = 0
    for i in range(8):
        for j in range(8):
            action_field[i*8:(i+1)*8, j*8:(j+1)*8] = actions[t, agent_idx]
            agent_idx += 1

    im = axes[idx].imshow(action_field, cmap='seismic', vmin=-1, vmax=1,
                          origin='lower', aspect='auto')
    axes[idx].set_title(f't = {t}', fontsize=10)
    axes[idx].set_xlabel('j (spanwise)', fontsize=8)
    axes[idx].set_ylabel('i (streamwise)', fontsize=8)

    # Add grid lines for agent patches
    for k in range(0, 65, 8):
        axes[idx].axhline(k - 0.5, color='black', linewidth=0.5, alpha=0.2)
        axes[idx].axvline(k - 0.5, color='black', linewidth=0.5, alpha=0.2)

fig.colorbar(im, ax=axes, orientation='horizontal',
             fraction=0.05, pad=0.05, label='Action value')
plt.suptitle(f'Action Field Evolution (t={t_start} to {t_end})', fontsize=14)
plt.tight_layout()
plt.savefig('smoothness_check/action_evolution.png', dpi=300, bbox_inches='tight')
print("Saved: smoothness_check/action_evolution.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Temporal gradients (action changes between consecutive steps)
# ============================================================================
print("\nCreating temporal gradient visualization...")

# Compute temporal gradients
temporal_grad = np.diff(actions_focused, axis=0)  # [249, 64, 8, 8]

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Select 6 timesteps
time_indices = np.linspace(0, len(temporal_grad)-1, 6, dtype=int)

for idx, t_local in enumerate(time_indices):
    t_global = t_start + t_local

    # Reconstruct temporal gradient field
    grad_field = np.zeros((64, 64))
    agent_idx = 0
    for i in range(8):
        for j in range(8):
            grad_field[i*8:(i+1)*8, j*8:(j+1)*8] = temporal_grad[t_local, agent_idx]
            agent_idx += 1

    vmax = max(0.1, np.abs(grad_field).max())
    im = axes[idx].imshow(grad_field, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                          origin='lower', aspect='auto')
    axes[idx].set_title(f'Δa(t={t_global} → {t_global+1})\n'
                       f'RMS={np.sqrt(np.mean(grad_field**2)):.4f}', fontsize=10)
    axes[idx].set_xlabel('j', fontsize=8)
    axes[idx].set_ylabel('i', fontsize=8)
    plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

plt.suptitle('Temporal Gradients (Action Changes)', fontsize=14)
plt.tight_layout()
plt.savefig('smoothness_check/temporal_gradients.png', dpi=300, bbox_inches='tight')
print("Saved: smoothness_check/temporal_gradients.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Spatial gradients (differences between adjacent agents)
# ============================================================================
print("\nCreating spatial gradient visualization...")

# Reshape to agent grid
actions_grid = actions_focused.reshape(-1, 8, 8, 8, 8)  # [250, 8_i, 8_j, 8, 8]

# Compute spatial gradients in i and j directions
grad_i = np.diff(actions_grid, axis=1)  # [250, 7, 8, 8, 8]
grad_j = np.diff(actions_grid, axis=2)  # [250, 8, 7, 8, 8]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Show gradient magnitude at 4 timesteps for each direction
time_indices = np.linspace(0, len(actions_focused)-1, 4, dtype=int)

for idx, t_local in enumerate(time_indices):
    t_global = t_start + t_local

    # Gradient in i-direction (between agent rows)
    grad_i_field = grad_i[t_local].mean(axis=(2, 3))  # [7, 8] - average over patch dims
    im0 = axes[0, idx].imshow(grad_i_field, cmap='viridis', origin='lower')
    axes[0, idx].set_title(f't={t_global}\nRMS={np.sqrt(np.mean(grad_i[t_local]**2)):.4f}', fontsize=10)
    axes[0, idx].set_ylabel('Agent gradient i', fontsize=8)
    plt.colorbar(im0, ax=axes[0, idx], fraction=0.046, pad=0.04)

    # Gradient in j-direction (between agent columns)
    grad_j_field = grad_j[t_local].mean(axis=(2, 3))  # [8, 7] - average over patch dims
    im1 = axes[1, idx].imshow(grad_j_field, cmap='viridis', origin='lower')
    axes[1, idx].set_title(f't={t_global}\nRMS={np.sqrt(np.mean(grad_j[t_local]**2)):.4f}', fontsize=10)
    axes[1, idx].set_ylabel('Agent gradient j', fontsize=8)
    plt.colorbar(im1, ax=axes[1, idx], fraction=0.046, pad=0.04)

plt.suptitle('Spatial Gradients Between Adjacent Agents', fontsize=14)
plt.tight_layout()
plt.savefig('smoothness_check/spatial_gradients.png', dpi=300, bbox_inches='tight')
print("Saved: smoothness_check/spatial_gradients.png")
plt.close()

# ============================================================================
# VISUALIZATION 4: Time series of smoothness metrics
# ============================================================================
print("\nCreating smoothness metrics time series...")

# Compute metrics over focused time range
temporal_grad_full = np.diff(actions_focused, axis=0)  # [249, 64, 8, 8]
temporal_grad_rms = np.sqrt(np.mean(temporal_grad_full**2, axis=(1, 2, 3)))  # [249]

spatial_var = np.var(actions_focused, axis=1).mean(axis=(1, 2))  # [250]

# Spatial gradients RMS over time
actions_grid = actions_focused.reshape(-1, 8, 8, 8, 8)
grad_i = np.diff(actions_grid, axis=1)
grad_j = np.diff(actions_grid, axis=2)
spatial_grad_rms = np.sqrt(np.mean(grad_i**2, axis=(1, 2, 3, 4)) +
                           np.mean(grad_j**2, axis=(1, 2, 3, 4)))  # [250]

fig, axes = plt.subplots(4, 1, figsize=(15, 16))

# Plot 1: Temporal gradient RMS
t_vals = np.arange(t_start, t_end-1)
axes[0].plot(t_vals, temporal_grad_rms, linewidth=1, alpha=0.8)
axes[0].set_xlabel('Time step')
axes[0].set_ylabel('Temporal gradient RMS')
axes[0].set_title('Temporal Smoothness: Action Change Rate')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=np.mean(temporal_grad_rms), color='r', linestyle='--',
               label=f'Mean: {np.mean(temporal_grad_rms):.4f}')
axes[0].legend()

# Plot 2: Spatial variance
t_vals = np.arange(t_start, t_end)
axes[1].plot(t_vals, spatial_var, linewidth=1, alpha=0.8, color='green')
axes[1].set_xlabel('Time step')
axes[1].set_ylabel('Spatial variance')
axes[1].set_title('Spatial Variance Across Agents')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=np.mean(spatial_var), color='r', linestyle='--',
               label=f'Mean: {np.mean(spatial_var):.4f}')
axes[1].legend()

# Plot 3: Spatial gradient RMS
axes[2].plot(t_vals, spatial_grad_rms, linewidth=1, alpha=0.8, color='orange')
axes[2].set_xlabel('Time step')
axes[2].set_ylabel('Spatial gradient RMS')
axes[2].set_title('Spatial Smoothness: Gradient Between Adjacent Agents')
axes[2].grid(True, alpha=0.3)
axes[2].axhline(y=np.mean(spatial_grad_rms), color='r', linestyle='--',
               label=f'Mean: {np.mean(spatial_grad_rms):.4f}')
axes[2].legend()

# Plot 4: Action statistics
action_mean = actions_focused.mean(axis=(1, 2, 3))  # [250]
action_std = actions_focused.std(axis=(1, 2, 3))    # [250]

axes[3].plot(t_vals, action_mean, label='Mean', linewidth=1.5, alpha=0.8)
axes[3].fill_between(t_vals, action_mean - action_std, action_mean + action_std,
                     alpha=0.3, label='±1 std')
axes[3].set_xlabel('Time step')
axes[3].set_ylabel('Action value')
axes[3].set_title('Action Statistics Over Time')
axes[3].grid(True, alpha=0.3)
axes[3].legend()

plt.tight_layout()
plt.savefig('smoothness_check/smoothness_metrics_timeseries.png', dpi=300, bbox_inches='tight')
print("Saved: smoothness_check/smoothness_metrics_timeseries.png")
plt.close()

# ============================================================================
# VISUALIZATION 5: Input-output relationship at tracked point (32, 32)
# ============================================================================
print("\nCreating input-output comparison at point (32,32)...")

point_u = data['point_u'][t_start:t_end]
point_w = data['point_w'][t_start:t_end]
point_action = data['point_action'][t_start:t_end]

fig, axes = plt.subplots(4, 1, figsize=(15, 16))

t_vals = np.arange(t_start, t_end)

# U velocity
axes[0].plot(t_vals, point_u, linewidth=1, alpha=0.8)
axes[0].set_ylabel('U velocity')
axes[0].set_title(f'Input: U-velocity at point (32,32)\n'
                 f'Mean={np.mean(point_u):.4f}, Std={np.std(point_u):.4f}')
axes[0].grid(True, alpha=0.3)

# W velocity
axes[1].plot(t_vals, point_w, linewidth=1, alpha=0.8, color='green')
axes[1].set_ylabel('W velocity')
axes[1].set_title(f'Input: W-velocity at point (32,32)\n'
                 f'Mean={np.mean(point_w):.4f}, Std={np.std(point_w):.4f}')
axes[1].grid(True, alpha=0.3)

# Action
axes[2].plot(t_vals, point_action, linewidth=1, alpha=0.8, color='red')
axes[2].set_ylabel('Action')
axes[2].set_title(f'Output: Action at point (32,32)\n'
                 f'Mean={np.mean(point_action):.4f}, Std={np.std(point_action):.4f}')
axes[2].grid(True, alpha=0.3)

# Correlation plot: action vs (u, w)
axes[3].scatter(point_u, point_action, alpha=0.3, s=10, label='U vs Action')
axes[3].scatter(point_w, point_action, alpha=0.3, s=10, label='W vs Action')
axes[3].set_xlabel('Velocity')
axes[3].set_ylabel('Action')
axes[3].set_title('Input-Output Correlation at point (32,32)')
axes[3].legend()
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('smoothness_check/input_output_point32_32.png', dpi=300, bbox_inches='tight')
print("Saved: smoothness_check/input_output_point32_32.png")
plt.close()

# ============================================================================
# VISUALIZATION 6: FFT analysis on focused window
# ============================================================================
print("\nCreating FFT analysis...")

# Temporal FFT (average action over space)
# dt = 6 * 0.0085 = 0.051 (actions computed every 6 simulation timesteps)
dt_action = 6 * 0.0085  # = 0.051
actions_time_avg = actions_focused.mean(axis=(1, 2, 3))  # [250]
temporal_fft = np.abs(np.fft.rfft(actions_time_avg))
temporal_freqs = np.fft.rfftfreq(len(actions_time_avg), d=dt_action)

# Spatial FFT (average over time)
actions_grid = actions_focused.reshape(-1, 8, 8, 8, 8)
actions_spatial_avg = actions_grid.mean(axis=(0, 3, 4))  # [8, 8]
spatial_fft_2d = np.abs(np.fft.rfft2(actions_spatial_avg))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Temporal FFT
axes[0].semilogy(temporal_freqs[1:], temporal_fft[1:])  # Exclude DC
top_idx = np.argsort(temporal_fft[1:])[-5:][::-1] + 1
axes[0].scatter(temporal_freqs[top_idx], temporal_fft[top_idx],
               c='red', s=100, zorder=5, label='Top 5 frequencies')
axes[0].set_xlabel('Frequency (Hz)')
axes[0].set_ylabel('FFT Magnitude')
dominant_freqs_str = ', '.join([f'{f:.2f}' for f in temporal_freqs[top_idx][:3]])
axes[0].set_title(f'Temporal Frequency Spectrum (dt={dt_action:.4f})\n'
                 f't={t_start}-{t_end}, Dominant: {dominant_freqs_str} Hz')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Spatial FFT 2D
im = axes[1].imshow(spatial_fft_2d, cmap='viridis', aspect='auto', origin='lower')
axes[1].set_xlabel('Wavenumber j')
axes[1].set_ylabel('Wavenumber i')
axes[1].set_title(f'Spatial Frequency Spectrum 2D (t={t_start}-{t_end})')
plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('smoothness_check/fft_analysis.png', dpi=300, bbox_inches='tight')
print("Saved: smoothness_check/fft_analysis.png")
plt.close()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY STATISTICS (timesteps 1000-1250)")
print("="*80)

print(f"\nTemporal smoothness:")
print(f"  Temporal gradient RMS: {np.mean(temporal_grad_rms):.6f} ± {np.std(temporal_grad_rms):.6f}")
print(f"  Min: {np.min(temporal_grad_rms):.6f}, Max: {np.max(temporal_grad_rms):.6f}")

print(f"\nSpatial smoothness:")
print(f"  Spatial variance: {np.mean(spatial_var):.6f} ± {np.std(spatial_var):.6f}")
print(f"  Spatial gradient RMS: {np.mean(spatial_grad_rms):.6f} ± {np.std(spatial_grad_rms):.6f}")

print(f"\nAction statistics:")
print(f"  Mean: {np.mean(actions_focused):.4f}")
print(f"  Std: {np.std(actions_focused):.4f}")
print(f"  Min: {np.min(actions_focused):.4f}")
print(f"  Max: {np.max(actions_focused):.4f}")

print(f"\nPoint (32,32) statistics:")
print(f"  U velocity: mean={np.mean(point_u):.4f}, std={np.std(point_u):.4f}")
print(f"  W velocity: mean={np.mean(point_w):.4f}, std={np.std(point_w):.4f}")
print(f"  Action: mean={np.mean(point_action):.4f}, std={np.std(point_action):.4f}")

print(f"\nRewards (mean over focused window): {np.mean(rewards[t_start:t_end]):.2f}")
print(f"dpdx (mean over focused window): {np.mean(dpdx[t_start:t_end]):.6f}")

# Recompute dominant frequencies for summary
actions_time_avg = actions_focused.mean(axis=(1, 2, 3))
temporal_fft = np.abs(np.fft.rfft(actions_time_avg))
temporal_freqs = np.fft.rfftfreq(len(actions_time_avg), d=dt_action)
top_idx = np.argsort(temporal_fft[1:])[-5:][::-1] + 1

print(f"\nFrequency analysis (dt={dt_action:.4f}):")
print(f"  Top 5 temporal frequencies (Hz): {temporal_freqs[top_idx]}")
print(f"  Corresponding periods (s): {1/temporal_freqs[top_idx]}")
print(f"  Periods in action steps: {1/(temporal_freqs[top_idx] * dt_action)}")

print("\n" + "="*80)
print("All visualizations saved to smoothness_check/")
print("="*80)
