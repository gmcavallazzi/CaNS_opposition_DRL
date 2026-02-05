"""
Offline analysis of smoothness data with detailed visualizations.
Customizable version for different data directories and episodes.

Usage:
    python analyze_smoothness_custom.py --data_dir smoothness_check1 --episode 2
    python analyze_smoothness_custom.py --data_dir smoothness_check1 --episode 1 --t_start 500 --t_end 1000
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Offline smoothness analysis')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing episode_N_data.npz files')
    parser.add_argument('--episode', type=int, default=-1,
                       help='Episode number to analyze (-1 for last episode)')
    parser.add_argument('--t_start', type=int, default=0,
                       help='Start timestep for analysis')
    parser.add_argument('--t_end', type=int, default=250,
                       help='End timestep for analysis (0 for all)')

    args = parser.parse_args()

    # Find available episodes
    episode_files = sorted([f for f in os.listdir(args.data_dir)
                           if f.startswith('episode_') and f.endswith('_data.npz')])

    if not episode_files:
        raise ValueError(f"No episode data files found in {args.data_dir}")

    # Determine which episode to analyze
    if args.episode == -1:
        episode_num = len(episode_files) - 1
        print(f"Auto-selected last episode: {episode_num}")
    else:
        episode_num = args.episode

    data_path = os.path.join(args.data_dir, f'episode_{episode_num}_data.npz')

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Episode file not found: {data_path}")

    print(f"Loading {data_path}...")
    data = np.load(data_path)

    # Extract data
    observations = data['observations']  # [T, 64, C, 8, 8] where C=2 or 3
    actions = data['actions']            # [T, 64, 8, 8]
    rewards = data['rewards']
    dpdx = data['dpdx']

    T, N, H, W = actions.shape
    C = observations.shape[2]  # 2 or 3 channels
    print(f"Data shape: T={T}, N_agents={N}, H={H}, W={W}")
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

    # Set time range
    t_start = args.t_start
    t_end = args.t_end if args.t_end > 0 else T
    t_end = min(t_end, T)

    physical_time_start = t_start * dt_action
    physical_time_end = t_end * dt_action
    print(f"\nAnalyzing timesteps {t_start} to {t_end}")
    print(f"  Physical time: {physical_time_start:.2f} to {physical_time_end:.2f} "
          f"(duration: {physical_time_end - physical_time_start:.2f})")

    actions_focused = actions[t_start:t_end]
    obs_focused = observations[t_start:t_end]

    # Create output directory
    output_dir = os.path.join(args.data_dir, f'detailed_analysis_ep{episode_num}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nResults will be saved to: {output_dir}")

    # ============================================================================
    # VISUALIZATION 1: Action field evolution over time
    # ============================================================================
    print("\n[1/6] Creating action field evolution plot...")

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    # Select 12 timesteps evenly spaced
    time_indices = np.linspace(0, len(actions_focused)-1, 12, dtype=int)

    for idx, t_local in enumerate(time_indices):
        t_global = t_start + t_local

        # Reconstruct full 64x64 action field from 64 agents
        action_field = np.zeros((64, 64))
        agent_idx = 0
        for i in range(8):
            for j in range(8):
                action_field[i*8:(i+1)*8, j*8:(j+1)*8] = actions_focused[t_local, agent_idx]
                agent_idx += 1

        im = axes[idx].imshow(action_field, cmap='seismic', vmin=-1, vmax=1,
                              origin='lower', aspect='auto')
        axes[idx].set_title(f't = {t_global} ({t_global*dt_action:.2f}s)', fontsize=10)
        axes[idx].set_xlabel('j (spanwise)', fontsize=8)
        axes[idx].set_ylabel('i (streamwise)', fontsize=8)

        # Add grid lines for agent patches
        for k in range(0, 65, 8):
            axes[idx].axhline(k - 0.5, color='black', linewidth=0.5, alpha=0.2)
            axes[idx].axvline(k - 0.5, color='black', linewidth=0.5, alpha=0.2)

    fig.colorbar(im, ax=axes, orientation='horizontal',
                 fraction=0.05, pad=0.05, label='Action value')
    plt.suptitle(f'Action Field Evolution (Episode {episode_num}, t={t_start} to {t_end})',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_action_evolution.png'),
                dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/01_action_evolution.png")
    plt.close()

    # ============================================================================
    # VISUALIZATION 2: Temporal gradients
    # ============================================================================
    print("[2/6] Creating temporal gradient visualization...")

    temporal_grad = np.diff(actions_focused, axis=0)  # [T-1, 64, 8, 8]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

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

    plt.suptitle(f'Temporal Gradients - Action Changes (Episode {episode_num})', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_temporal_gradients.png'),
                dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/02_temporal_gradients.png")
    plt.close()

    # ============================================================================
    # VISUALIZATION 3: Spatial gradients
    # ============================================================================
    print("[3/6] Creating spatial gradient visualization...")

    # Reshape to agent grid
    actions_grid = actions_focused.reshape(-1, 8, 8, 8, 8)  # [T, 8_i, 8_j, 8, 8]

    # Compute spatial gradients in i and j directions
    grad_i = np.diff(actions_grid, axis=1)  # [T, 7, 8, 8, 8]
    grad_j = np.diff(actions_grid, axis=2)  # [T, 8, 7, 8, 8]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    time_indices = np.linspace(0, len(actions_focused)-1, 4, dtype=int)

    for idx, t_local in enumerate(time_indices):
        t_global = t_start + t_local

        # Gradient in i-direction
        grad_i_field = grad_i[t_local].mean(axis=(2, 3))  # [7, 8]
        im0 = axes[0, idx].imshow(grad_i_field, cmap='viridis', origin='lower')
        axes[0, idx].set_title(f't={t_global}\nRMS={np.sqrt(np.mean(grad_i[t_local]**2)):.4f}',
                               fontsize=10)
        axes[0, idx].set_ylabel('Agent gradient i', fontsize=8)
        plt.colorbar(im0, ax=axes[0, idx], fraction=0.046, pad=0.04)

        # Gradient in j-direction
        grad_j_field = grad_j[t_local].mean(axis=(2, 3))  # [8, 7]
        im1 = axes[1, idx].imshow(grad_j_field, cmap='viridis', origin='lower')
        axes[1, idx].set_title(f't={t_global}\nRMS={np.sqrt(np.mean(grad_j[t_local]**2)):.4f}',
                               fontsize=10)
        axes[1, idx].set_ylabel('Agent gradient j', fontsize=8)
        plt.colorbar(im1, ax=axes[1, idx], fraction=0.046, pad=0.04)

    plt.suptitle(f'Spatial Gradients Between Adjacent Agents (Episode {episode_num})',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '03_spatial_gradients.png'),
                dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/03_spatial_gradients.png")
    plt.close()

    # ============================================================================
    # VISUALIZATION 4: Smoothness metrics time series
    # ============================================================================
    print("[4/6] Creating smoothness metrics time series...")

    # Compute metrics
    temporal_grad_full = np.diff(actions_focused, axis=0)
    temporal_grad_rms = np.sqrt(np.mean(temporal_grad_full**2, axis=(1, 2, 3)))

    spatial_var = np.var(actions_focused, axis=1).mean(axis=(1, 2))

    actions_grid = actions_focused.reshape(-1, 8, 8, 8, 8)
    grad_i = np.diff(actions_grid, axis=1)
    grad_j = np.diff(actions_grid, axis=2)
    spatial_grad_rms = np.sqrt(np.mean(grad_i**2, axis=(1, 2, 3, 4)) +
                               np.mean(grad_j**2, axis=(1, 2, 3, 4)))

    fig, axes = plt.subplots(4, 1, figsize=(15, 16))

    # Temporal gradient RMS
    t_vals = np.arange(t_start, t_end-1)
    axes[0].plot(t_vals, temporal_grad_rms, linewidth=1, alpha=0.8)
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('Temporal gradient RMS')
    axes[0].set_title('Temporal Smoothness: Action Change Rate')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=np.mean(temporal_grad_rms), color='r', linestyle='--',
                   label=f'Mean: {np.mean(temporal_grad_rms):.4f}')
    axes[0].legend()

    # Spatial variance
    t_vals = np.arange(t_start, t_end)
    axes[1].plot(t_vals, spatial_var, linewidth=1, alpha=0.8, color='green')
    axes[1].set_xlabel('Time step')
    axes[1].set_ylabel('Spatial variance')
    axes[1].set_title('Spatial Variance Across Agents')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=np.mean(spatial_var), color='r', linestyle='--',
                   label=f'Mean: {np.mean(spatial_var):.4f}')
    axes[1].legend()

    # Spatial gradient RMS
    axes[2].plot(t_vals, spatial_grad_rms, linewidth=1, alpha=0.8, color='orange')
    axes[2].set_xlabel('Time step')
    axes[2].set_ylabel('Spatial gradient RMS')
    axes[2].set_title('Spatial Smoothness: Gradient Between Adjacent Agents')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=np.mean(spatial_grad_rms), color='r', linestyle='--',
                   label=f'Mean: {np.mean(spatial_grad_rms):.4f}')
    axes[2].legend()

    # Action statistics
    action_mean = actions_focused.mean(axis=(1, 2, 3))
    action_std = actions_focused.std(axis=(1, 2, 3))

    axes[3].plot(t_vals, action_mean, label='Mean', linewidth=1.5, alpha=0.8)
    axes[3].fill_between(t_vals, action_mean - action_std, action_mean + action_std,
                         alpha=0.3, label='±1 std')
    axes[3].set_xlabel('Time step')
    axes[3].set_ylabel('Action value')
    axes[3].set_title('Action Statistics Over Time')
    axes[3].grid(True, alpha=0.3)
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Zero')
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_smoothness_metrics_timeseries.png'),
                dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/04_smoothness_metrics_timeseries.png")
    plt.close()

    # ============================================================================
    # VISUALIZATION 5: Input-output relationship at point (32, 32)
    # ============================================================================
    print("[5/6] Creating input-output comparison at point (32,32)...")

    if 'point_u' in data:
        point_u = data['point_u'][t_start:t_end]
        point_w = data['point_w'][t_start:t_end]
        point_action = data['point_action'][t_start:t_end]

        # Get dpdx data (full episode, not just focused window)
        dpdx_vals = data['dpdx'][t_start:t_end] if 'dpdx' in data else None

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
        if dpdx_vals is not None:
            axes[3].plot(t_vals, dpdx_vals, linewidth=1, alpha=0.8, color='purple')
            axes[3].axhline(y=-0.002, color='g', linestyle='--', alpha=0.7, label='Target')
            axes[3].axhline(y=-0.0042, color='orange', linestyle='--', alpha=0.7, label='Uncontrolled')
            axes[3].set_ylabel('dpdx')
            axes[3].set_title(f'Performance: Pressure Gradient (Drag)\n'
                             f'Mean={np.mean(dpdx_vals):.6f}, Std={np.std(dpdx_vals):.6f}')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        else:
            axes[3].text(0.5, 0.5, 'dpdx data not available',
                        transform=axes[3].transAxes, ha='center', va='center')

        # Correlation plot
        axes[4].scatter(point_u, point_action, alpha=0.3, s=10, label='U vs Action')
        axes[4].scatter(point_w, point_action, alpha=0.3, s=10, label='W vs Action')
        axes[4].set_xlabel('Velocity')
        axes[4].set_ylabel('Action')
        axes[4].set_title('Input-Output Correlation at point (32,32)')
        axes[4].legend()
        axes[4].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '05_input_output_point32_32.png'),
                    dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_dir}/05_input_output_point32_32.png")
        plt.close()
    else:
        print("  Point tracking data not found, skipping...")

    # ============================================================================
    # VISUALIZATION 6: FFT analysis (ALWAYS uses FULL episode for best resolution)
    # ============================================================================
    print("[6/6] Creating FFT analysis...")
    print(f"  Note: FFT computed on FULL episode (all {T} timesteps) for best frequency resolution")

    # Temporal FFT - use FULL episode data for maximum frequency resolution
    actions_time_avg_full = actions.mean(axis=(1, 2, 3))  # [T] - full episode
    temporal_fft = np.abs(np.fft.rfft(actions_time_avg_full))
    temporal_freqs = np.fft.rfftfreq(len(actions_time_avg_full), d=dt_action)

    freq_resolution = 1.0 / (len(actions_time_avg_full) * dt_action)
    nyquist_freq = 1.0 / (2 * dt_action)
    print(f"  Frequency resolution: {freq_resolution:.4f} Hz")
    print(f"  Nyquist frequency: {nyquist_freq:.2f} Hz")

    # Spatial FFT - use FULL episode for time averaging
    actions_grid_full = actions.reshape(-1, 8, 8, 8, 8)  # [T, 8_i, 8_j, 8, 8]
    actions_spatial_avg = actions_grid_full.mean(axis=(0, 3, 4))  # [8, 8]
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
                     f'Full episode (T={T}, Δf={freq_resolution:.4f} Hz)\n'
                     f'Dominant: {dominant_freqs_str} Hz')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Spatial FFT 2D
    im = axes[1].imshow(spatial_fft_2d, cmap='viridis', aspect='auto', origin='lower')
    axes[1].set_xlabel('Wavenumber j')
    axes[1].set_ylabel('Wavenumber i')
    axes[1].set_title(f'Spatial Frequency Spectrum 2D\n(Full episode average)')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '06_fft_analysis.png'),
                dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/06_fft_analysis.png")
    plt.close()

    # ============================================================================
    # SUMMARY STATISTICS
    # ============================================================================
    print("\n" + "="*80)
    print(f"SUMMARY STATISTICS - Episode {episode_num} (timesteps {t_start}-{t_end})")
    print("="*80)

    print(f"\nTemporal smoothness:")
    print(f"  Temporal gradient RMS: {np.mean(temporal_grad_rms):.6f} ± {np.std(temporal_grad_rms):.6f}")
    print(f"  Min: {np.min(temporal_grad_rms):.6f}, Max: {np.max(temporal_grad_rms):.6f}")

    print(f"\nSpatial smoothness:")
    print(f"  Spatial variance: {np.mean(spatial_var):.6f} ± {np.std(spatial_var):.6f}")
    print(f"  Spatial gradient RMS: {np.mean(spatial_grad_rms):.6f} ± {np.std(spatial_grad_rms):.6f}")

    print(f"\nAction statistics:")
    print(f"  Mean: {np.mean(actions_focused):.6f} (should be ~0 for zero-mean constraint)")
    print(f"  Std: {np.std(actions_focused):.4f}")
    print(f"  Min: {np.min(actions_focused):.4f}")
    print(f"  Max: {np.max(actions_focused):.4f}")

    if 'point_u' in data:
        print(f"\nPoint (32,32) statistics:")
        print(f"  U velocity: mean={np.mean(point_u):.4f}, std={np.std(point_u):.4f}")
        print(f"  W velocity: mean={np.mean(point_w):.4f}, std={np.std(point_w):.4f}")
        print(f"  Action: mean={np.mean(point_action):.4f}, std={np.std(point_action):.4f}")

    print(f"\nFFT Analysis (full episode, T={T}):")
    print(f"  Frequency resolution: {freq_resolution:.4f} Hz")
    print(f"  Nyquist frequency: {nyquist_freq:.2f} Hz")
    print(f"  Dominant temporal frequencies (top 3): {dominant_freqs_str} Hz")

    # Save summary to file
    summary_file = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"SUMMARY STATISTICS - Episode {episode_num} (timesteps {t_start}-{t_end})\n")
        f.write("="*80 + "\n\n")
        f.write(f"Temporal smoothness:\n")
        f.write(f"  Temporal gradient RMS: {np.mean(temporal_grad_rms):.6f} ± {np.std(temporal_grad_rms):.6f}\n")
        f.write(f"  Min: {np.min(temporal_grad_rms):.6f}, Max: {np.max(temporal_grad_rms):.6f}\n\n")
        f.write(f"Spatial smoothness:\n")
        f.write(f"  Spatial variance: {np.mean(spatial_var):.6f} ± {np.std(spatial_var):.6f}\n")
        f.write(f"  Spatial gradient RMS: {np.mean(spatial_grad_rms):.6f} ± {np.std(spatial_grad_rms):.6f}\n\n")
        f.write(f"Action statistics:\n")
        f.write(f"  Mean: {np.mean(actions_focused):.6f}\n")
        f.write(f"  Std: {np.std(actions_focused):.4f}\n")
        f.write(f"  Min: {np.min(actions_focused):.4f}\n")
        f.write(f"  Max: {np.max(actions_focused):.4f}\n\n")
        f.write(f"FFT Analysis (full episode, T={T}):\n")
        f.write(f"  Frequency resolution: {freq_resolution:.4f} Hz\n")
        f.write(f"  Nyquist frequency: {nyquist_freq:.2f} Hz\n")
        f.write(f"  Dominant temporal frequencies (top 3): {dominant_freqs_str} Hz\n")

    print(f"\nSummary saved to: {summary_file}")
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
