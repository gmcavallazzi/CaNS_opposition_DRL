"""
Offline analysis of smoothness data for POINT-BASED system with detailed visualizations.

Key differences from patch-based version:
- Actions are [T, 4096] scalars instead of [T, 64, 8, 8] spatial patches
- Direct reshape to 64×64 field (no patch assembly)
- Direct spatial gradients on 64×64 field (no agent-grid gradients)

Usage:
    python analyze_smoothness_offline_point.py --data_dir smoothness_analysis_point_TIMESTAMP --episode 0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Offline smoothness analysis for point-based system')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing episode_N_data.npz files')
    parser.add_argument('--episode', type=int, default=0,
                       help='Episode number to analyze (default: 0)')
    parser.add_argument('--t_start', type=int, default=0,
                       help='Start timestep for analysis')
    parser.add_argument('--t_end', type=int, default=250,
                       help='End timestep for analysis (0 for all)')

    args = parser.parse_args()

    # Load data
    data_path = os.path.join(args.data_dir, f'episode_{args.episode}_data.npz')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Episode file not found: {data_path}")

    print(f"Loading {data_path}...")
    data = np.load(data_path)

    # Extract data
    observations = data['observations']  # [T, 4096, 2 or 3]
    actions = data['actions']            # [T, 4096]
    rewards = data['rewards']
    dpdx = data['dpdx']

    T, N = actions.shape
    C = observations.shape[2]  # 2 or 3 channels
    print(f"Data shape: T={T}, N_agents={N}")
    print(f"Observation channels: {C} ({'with' if C==3 else 'without'} action memory)")
    print(f"Time range: 0 to {T-1}")

    # Physical time information
    dt_sim = 0.0085  # Simulation timestep
    action_every = 6  # Actions computed every 6 simulation steps
    dt_action = action_every * dt_sim  # = 0.051
    print(f"\nTiming:")
    print(f"  Simulation dt: {dt_sim}")
    print(f"  Action computed every {action_every} sim steps")
    print(f"  Effective action dt: {dt_action}")

    # Focus on timesteps
    t_start = args.t_start
    t_end = args.t_end if args.t_end > 0 else T
    t_end = min(t_end, T)
    physical_time_start = t_start * dt_action
    physical_time_end = t_end * dt_action
    print(f"\nFocusing on timesteps {t_start} to {t_end}")
    print(f"  Physical time: {physical_time_start:.2f} to {physical_time_end:.2f} (duration: {physical_time_end - physical_time_start:.2f})")

    actions_focused = actions[t_start:t_end]  # [T_focus, 4096]
    obs_focused = observations[t_start:t_end]  # [T_focus, 4096, 2 or 3]

    # Create output directory
    output_dir = args.data_dir
    print(f"\nResults will be saved to: {output_dir}")

    # ============================================================================
    # VISUALIZATION 1: Action field evolution over time (consecutive pairs)
    # ============================================================================
    print("\n[1/6] Creating action field evolution plot (consecutive pairs)...")

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    # Select 6 evenly spaced timesteps, then show each with its consecutive timestep
    # This gives us 12 plots showing 6 pairs: (t, t+1)
    n_pairs = 6
    base_time_indices = np.linspace(0, len(actions_focused)-2, n_pairs, dtype=int)

    for pair_idx, t_base in enumerate(base_time_indices):
        for offset in [0, 1]:  # Show t and t+1
            plot_idx = pair_idx * 2 + offset
            t_local = t_base + offset
            t_global = t_start + t_local

            # Reconstruct full 64×64 action field (simple reshape!)
            action_field = actions_focused[t_local].reshape(64, 64)

            im = axes[plot_idx].imshow(action_field, cmap='seismic', vmin=-1, vmax=1,
                                       origin='lower', aspect='auto', interpolation='bilinear')
            axes[plot_idx].set_title(f't = {t_global}', fontsize=10)
            axes[plot_idx].set_xlabel('j (spanwise)', fontsize=8)
            axes[plot_idx].set_ylabel('i (streamwise)', fontsize=8)

    # Add colorbar on the right side
    fig.colorbar(im, ax=axes, orientation='vertical',
                 fraction=0.02, pad=0.02, label='Action value')
    plt.suptitle(f'Action Field Evolution - Consecutive Pairs (Point-Based, t={t_start} to {t_end})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'action_evolution.png'), dpi=300, bbox_inches='tight')
    print("Saved: action_evolution.png")
    plt.close()

    # ============================================================================
    # VISUALIZATION 2: Temporal gradients (action changes between consecutive steps)
    # ============================================================================
    print("\n[2/6] Creating temporal gradient visualization...")

    # Compute temporal gradients
    temporal_grad = np.diff(actions_focused, axis=0)  # [T_focus-1, 4096]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Select 6 timesteps
    time_indices = np.linspace(0, len(temporal_grad)-1, 6, dtype=int)

    # Compute global colormap limits (same for all subplots)
    grad_fields = [temporal_grad[t].reshape(64, 64) for t in time_indices]
    global_vmax = max(0.1, max(np.abs(gf).max() for gf in grad_fields))

    for idx, t_local in enumerate(time_indices):
        t_global = t_start + t_local

        # Reconstruct temporal gradient field
        grad_field = grad_fields[idx]

        # Use interpolation='bilinear' for smoothing
        im = axes[idx].imshow(grad_field, cmap='RdBu_r', vmin=-global_vmax, vmax=global_vmax,
                              origin='lower', aspect='auto', interpolation='bilinear')
        axes[idx].set_title(f'Δa(t={t_global} → {t_global+1})\n'
                           f'RMS={np.sqrt(np.mean(grad_field**2)):.4f}', fontsize=10)
        axes[idx].set_xlabel('j', fontsize=8)
        axes[idx].set_ylabel('i', fontsize=8)
        plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    plt.suptitle(f'Temporal Gradients (Action Changes) - Global vmax={global_vmax:.4f}', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temporal_gradients.png'), dpi=300, bbox_inches='tight')
    print("Saved: temporal_gradients.png")
    plt.close()

    # ============================================================================
    # VISUALIZATION 3: Spatial gradients (differences between adjacent points)
    # ============================================================================
    print("\n[3/6] Creating spatial gradient visualization...")

    # Reshape to 64×64 field
    actions_field = actions_focused.reshape(-1, 64, 64)  # [T_focus, 64, 64]

    # Compute spatial gradients in i and j directions
    grad_i = np.diff(actions_field, axis=1)  # [T_focus, 63, 64]
    grad_j = np.diff(actions_field, axis=2)  # [T_focus, 64, 63]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Show gradient magnitude at 4 timesteps for each direction
    time_indices = np.linspace(0, len(actions_field)-1, 4, dtype=int)

    for idx, t_local in enumerate(time_indices):
        t_global = t_start + t_local

        # Gradient in i-direction
        grad_i_field = grad_i[t_local]  # [63, 64]
        im0 = axes[0, idx].imshow(grad_i_field, cmap='viridis', origin='lower')
        axes[0, idx].set_title(f't={t_global}\nRMS={np.sqrt(np.mean(grad_i[t_local]**2)):.4f}', fontsize=10)
        axes[0, idx].set_ylabel('Spatial gradient i', fontsize=8)
        plt.colorbar(im0, ax=axes[0, idx], fraction=0.046, pad=0.04)

        # Gradient in j-direction
        grad_j_field = grad_j[t_local]  # [64, 63]
        im1 = axes[1, idx].imshow(grad_j_field, cmap='viridis', origin='lower')
        axes[1, idx].set_title(f't={t_global}\nRMS={np.sqrt(np.mean(grad_j[t_local]**2)):.4f}', fontsize=10)
        axes[1, idx].set_ylabel('Spatial gradient j', fontsize=8)
        plt.colorbar(im1, ax=axes[1, idx], fraction=0.046, pad=0.04)

    plt.suptitle('Spatial Gradients Between Adjacent Points', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_gradients.png'), dpi=300, bbox_inches='tight')
    print("Saved: spatial_gradients.png")
    plt.close()

    # ============================================================================
    # VISUALIZATION 4: Time series of smoothness metrics
    # ============================================================================
    print("\n[4/6] Creating smoothness metrics time series...")

    # Compute metrics over focused time range
    temporal_grad_full = np.diff(actions_focused, axis=0)  # [T_focus-1, 4096]
    temporal_grad_rms = np.sqrt(np.mean(temporal_grad_full**2, axis=1))  # [T_focus-1]

    spatial_var = np.var(actions_focused, axis=1)  # [T_focus]

    # Spatial gradients RMS over time
    actions_field = actions_focused.reshape(-1, 64, 64)
    grad_i = np.diff(actions_field, axis=1)
    grad_j = np.diff(actions_field, axis=2)
    spatial_grad_rms = np.sqrt(np.mean(grad_i**2, axis=(1, 2)) +
                               np.mean(grad_j**2, axis=(1, 2)))  # [T_focus]

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
    axes[1].set_title('Spatial Variance Across Points')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=np.mean(spatial_var), color='r', linestyle='--',
                   label=f'Mean: {np.mean(spatial_var):.4f}')
    axes[1].legend()

    # Plot 3: Spatial gradient RMS
    axes[2].plot(t_vals, spatial_grad_rms, linewidth=1, alpha=0.8, color='orange')
    axes[2].set_xlabel('Time step')
    axes[2].set_ylabel('Spatial gradient RMS')
    axes[2].set_title('Spatial Smoothness: Gradient Between Adjacent Points')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=np.mean(spatial_grad_rms), color='r', linestyle='--',
                   label=f'Mean: {np.mean(spatial_grad_rms):.4f}')
    axes[2].legend()

    # Plot 4: Action statistics
    action_mean = actions_focused.mean(axis=1)  # [T_focus]
    action_std = actions_focused.std(axis=1)    # [T_focus]

    axes[3].plot(t_vals, action_mean, label='Mean', linewidth=1.5, alpha=0.8)
    axes[3].fill_between(t_vals, action_mean - action_std, action_mean + action_std,
                         alpha=0.3, label='±1 std')
    axes[3].set_xlabel('Time step')
    axes[3].set_ylabel('Action value')
    axes[3].set_title('Action Statistics Over Time')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'smoothness_metrics_timeseries.png'), dpi=300, bbox_inches='tight')
    print("Saved: smoothness_metrics_timeseries.png")
    plt.close()

    # ============================================================================
    # VISUALIZATION 5: Input-output relationship at tracked point (32, 32) + dpdx
    # ============================================================================
    print("\n[5/6] Creating input-output comparison at point (32,32) with dpdx...")

    point_u = data['point_u'][t_start:t_end]
    point_w = data['point_w'][t_start:t_end]
    point_action = data['point_action'][t_start:t_end]
    dpdx_focused = dpdx[t_start:t_end]

    fig, axes = plt.subplots(5, 1, figsize=(15, 20))

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

    # dpdx (drag)
    axes[3].plot(t_vals, dpdx_focused, linewidth=1, alpha=0.8, color='purple')
    axes[3].set_ylabel('dpdx')
    axes[3].set_title(f'Performance: Pressure Gradient (dpdx)\n'
                     f'Mean={np.mean(dpdx_focused):.6f}, Std={np.std(dpdx_focused):.6f}')
    axes[3].axhline(y=-0.0042, color='k', linestyle='--', alpha=0.5, label='Uncontrolled')
    axes[3].axhline(y=-0.002, color='r', linestyle='--', alpha=0.5, label='Target')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    # Correlation plot: action vs (u, w)
    axes[4].scatter(point_u, point_action, alpha=0.3, s=10, label='U vs Action')
    axes[4].scatter(point_w, point_action, alpha=0.3, s=10, label='W vs Action')
    axes[4].set_xlabel('Velocity')
    axes[4].set_ylabel('Action')
    axes[4].set_title('Input-Output Correlation at point (32,32)')
    axes[4].legend()
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'input_output_point32_32.png'), dpi=300, bbox_inches='tight')
    print("Saved: input_output_point32_32.png")
    plt.close()

    # ============================================================================
    # VISUALIZATION 6: FFT analysis
    # ============================================================================
    print("\n[6/6] Creating FFT analysis...")

    # Temporal FFT (average action over space)
    actions_time_avg = actions_focused.mean(axis=1)  # [T_focus]
    temporal_fft = np.abs(np.fft.rfft(actions_time_avg))
    temporal_freqs = np.fft.rfftfreq(len(actions_time_avg), d=dt_action)

    # Spatial FFT (average over time)
    actions_field = actions_focused.reshape(-1, 64, 64)
    actions_spatial_avg = actions_field.mean(axis=0)  # [64, 64]
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
    plt.savefig(os.path.join(output_dir, 'fft_analysis.png'), dpi=300, bbox_inches='tight')
    print("Saved: fft_analysis.png")
    plt.close()

    # ============================================================================
    # SUMMARY STATISTICS
    # ============================================================================
    print("\n" + "="*80)
    print(f"SUMMARY STATISTICS (timesteps {t_start}-{t_end})")
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

    print(f"\nPerformance metrics:")
    print(f"  Rewards (mean): {np.mean(rewards[t_start:t_end]):.2f}")
    print(f"  dpdx (mean): {np.mean(dpdx[t_start:t_end]):.6f}")
    print(f"  dpdx (min): {np.min(dpdx[t_start:t_end]):.6f}")
    print(f"  dpdx (max): {np.max(dpdx[t_start:t_end]):.6f}")
    uncontrolled = -0.0042
    target = -0.002
    dpdx_mean = np.mean(dpdx[t_start:t_end])
    improvement = (uncontrolled - dpdx_mean) / (uncontrolled - target) * 100
    print(f"  Drag reduction progress: {improvement:.1f}% (uncontrolled: {uncontrolled}, target: {target})")

    # Recompute dominant frequencies for summary
    top_idx = np.argsort(temporal_fft[1:])[-5:][::-1] + 1

    print(f"\nFrequency analysis (dt={dt_action:.4f}):")
    print(f"  Top 5 temporal frequencies (Hz): {temporal_freqs[top_idx]}")
    print(f"  Corresponding periods (s): {1/temporal_freqs[top_idx]}")
    print(f"  Periods in action steps: {1/(temporal_freqs[top_idx] * dt_action)}")

    print("\n" + "="*80)
    print(f"All visualizations saved to {output_dir}/")
    print("="*80)


if __name__ == "__main__":
    main()
