"""
Create a 2D action policy map showing the relationship between u, w velocities
and action values at point (32, 32).

This reveals what the neural network learned: for any given (u, w) observation,
what action does the policy output?

Usage:
    python create_action_map_custom.py --data_dir smoothness_check1
    python create_action_map_custom.py --data_dir smoothness_check1 --n_bins 100
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import binned_statistic_2d
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Create action policy map')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing episode_N_data.npz files')
    parser.add_argument('--n_bins', type=int, default=50,
                       help='Number of bins in each direction (default: 50)')
    parser.add_argument('--min_samples', type=int, default=5,
                       help='Minimum samples per bin to display (default: 5)')

    args = parser.parse_args()

    # Find available episodes
    episode_files = sorted([f for f in os.listdir(args.data_dir)
                           if f.startswith('episode_') and f.endswith('_data.npz')])

    if not episode_files:
        raise ValueError(f"No episode data files found in {args.data_dir}")

    n_episodes = len(episode_files)
    print(f"Found {n_episodes} episode(s) in {args.data_dir}")

    # Load data from all episodes
    print("\nLoading episode data...")
    all_u = []
    all_w = []
    all_actions = []

    for ep in range(n_episodes):
        print(f"  Loading episode {ep}...")
        data_path = os.path.join(args.data_dir, f'episode_{ep}_data.npz')
        data = np.load(data_path)

        # Check if point tracking data exists
        if 'point_u' not in data:
            raise ValueError(f"Episode {ep} data missing 'point_u'. "
                           "Make sure test was run with point tracking enabled.")

        # Extract point data
        point_u = data['point_u']
        point_w = data['point_w']
        point_action = data['point_action']

        all_u.append(point_u)
        all_w.append(point_w)
        all_actions.append(point_action)

    # Concatenate all episodes
    u_vals = np.concatenate(all_u)
    w_vals = np.concatenate(all_w)
    action_vals = np.concatenate(all_actions)

    print(f"\nTotal data points: {len(u_vals)}")
    print(f"U range: [{u_vals.min():.4f}, {u_vals.max():.4f}]")
    print(f"W range: [{w_vals.min():.4f}, {w_vals.max():.4f}]")
    print(f"Action range: [{action_vals.min():.4f}, {action_vals.max():.4f}]")

    # Create output directory
    output_dir = args.data_dir
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================================
    # VISUALIZATION 1: Multi-panel overview
    # ============================================================================
    print("\nCreating action policy map (4-panel overview)...")

    n_bins = args.n_bins

    # Calculate mean action value in each bin
    statistic, x_edges, y_edges, binnumber = binned_statistic_2d(
        u_vals, w_vals, action_vals,
        statistic='mean',
        bins=n_bins
    )

    # Count number of samples in each bin
    counts, _, _, _ = binned_statistic_2d(
        u_vals, w_vals, action_vals,
        statistic='count',
        bins=n_bins
    )

    # Mask bins with too few samples
    min_samples = args.min_samples
    statistic_masked = np.ma.masked_where(counts < min_samples, statistic)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))

    # Main plot: 2D map
    ax1 = plt.subplot(2, 2, 1)
    im1 = ax1.imshow(statistic_masked.T, origin='lower', aspect='auto',
                     extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                     cmap='seismic', vmin=-1, vmax=1)
    ax1.set_xlabel('U velocity', fontsize=12)
    ax1.set_ylabel('W velocity', fontsize=12)
    ax1.set_title(f'Mean Action Value at Point (32,32)\n(bins with ≥{min_samples} samples)',
                  fontsize=14)
    plt.colorbar(im1, ax=ax1, label='Mean Action')
    ax1.grid(True, alpha=0.3)

    # Sample count distribution
    ax2 = plt.subplot(2, 2, 2)
    im2 = ax2.imshow(np.log10(counts.T + 1), origin='lower', aspect='auto',
                     extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                     cmap='viridis')
    ax2.set_xlabel('U velocity', fontsize=12)
    ax2.set_ylabel('W velocity', fontsize=12)
    ax2.set_title('Sample Count Distribution (log10)', fontsize=14)
    plt.colorbar(im2, ax=ax2, label='log10(count + 1)')
    ax2.grid(True, alpha=0.3)

    # Scatter plot with all points (downsampled for visibility)
    ax3 = plt.subplot(2, 2, 3)
    n_plot = min(10000, len(u_vals))
    indices = np.random.choice(len(u_vals), n_plot, replace=False)
    scatter = ax3.scatter(u_vals[indices], w_vals[indices],
                         c=action_vals[indices], s=5, alpha=0.3,
                         cmap='seismic', vmin=-1, vmax=1)
    ax3.set_xlabel('U velocity', fontsize=12)
    ax3.set_ylabel('W velocity', fontsize=12)
    ax3.set_title(f'Scatter Plot (random {n_plot} points)', fontsize=14)
    plt.colorbar(scatter, ax=ax3, label='Action')
    ax3.grid(True, alpha=0.3)

    # Standard deviation of action in each bin
    std_statistic, _, _, _ = binned_statistic_2d(
        u_vals, w_vals, action_vals,
        statistic='std',
        bins=n_bins
    )
    std_masked = np.ma.masked_where(counts < min_samples, std_statistic)

    ax4 = plt.subplot(2, 2, 4)
    im4 = ax4.imshow(std_masked.T, origin='lower', aspect='auto',
                     extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                     cmap='plasma', vmin=0)
    ax4.set_xlabel('U velocity', fontsize=12)
    ax4.set_ylabel('W velocity', fontsize=12)
    ax4.set_title('Action Std Dev in Each Bin\n(High std = inconsistent policy)', fontsize=14)
    plt.colorbar(im4, ax=ax4, label='Std Dev')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'action_map_u_w.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # ============================================================================
    # VISUALIZATION 2: High-resolution main map
    # ============================================================================
    print("Creating high-resolution action map...")

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Use finer bins for higher resolution
    n_bins_fine = 80
    statistic_fine, x_edges_fine, y_edges_fine, _ = binned_statistic_2d(
        u_vals, w_vals, action_vals,
        statistic='mean',
        bins=n_bins_fine
    )
    counts_fine, _, _, _ = binned_statistic_2d(
        u_vals, w_vals, action_vals,
        statistic='count',
        bins=n_bins_fine
    )
    statistic_fine_masked = np.ma.masked_where(counts_fine < min_samples, statistic_fine)

    im = ax.imshow(statistic_fine_masked.T, origin='lower', aspect='auto',
                   extent=[x_edges_fine[0], x_edges_fine[-1],
                          y_edges_fine[0], y_edges_fine[-1]],
                   cmap='seismic', vmin=-1, vmax=1)
    ax.set_xlabel('U velocity', fontsize=14)
    ax.set_ylabel('W velocity', fontsize=14)
    ax.set_title(f'Action Policy Map at Point (32,32)\nMean Action as Function of (u, w)\n'
                 f'({len(u_vals)} total samples, bins with ≥{min_samples} samples shown)',
                 fontsize=16)
    cbar = plt.colorbar(im, ax=ax, label='Mean Action Value')
    cbar.ax.tick_params(labelsize=12)
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5)

    # Add contour lines
    X, Y = np.meshgrid(
        (x_edges_fine[:-1] + x_edges_fine[1:]) / 2,
        (y_edges_fine[:-1] + y_edges_fine[1:]) / 2
    )
    contour_levels = [-0.8, -0.4, 0, 0.4, 0.8]
    contours = ax.contour(X, Y, statistic_fine_masked.T,
                          levels=contour_levels,
                          colors='black', linewidths=1, alpha=0.5)
    ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'action_map_u_w_highres.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # ============================================================================
    # VISUALIZATION 3: 1D slices through the map
    # ============================================================================
    print("Creating 1D slice plots...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Slice 1: Action vs U (at different W values)
    w_percentiles = [10, 25, 50, 75, 90]
    w_values = np.percentile(w_vals, w_percentiles)

    for w_val, percentile in zip(w_values, w_percentiles):
        # Find points near this W value
        w_tolerance = 0.1 * w_vals.std()
        mask = np.abs(w_vals - w_val) < w_tolerance

        if mask.sum() > 10:  # Need enough points
            # Bin by U value
            u_bins = 30
            u_stat, u_edges, _ = binned_statistic(
                u_vals[mask], action_vals[mask],
                statistic='mean', bins=u_bins
            )
            u_centers = (u_edges[:-1] + u_edges[1:]) / 2

            axes[0].plot(u_centers, u_stat,
                        label=f'W ≈ {w_val:.3f} (p{percentile})',
                        marker='o', markersize=4, alpha=0.7)

    axes[0].set_xlabel('U velocity', fontsize=12)
    axes[0].set_ylabel('Mean Action', fontsize=12)
    axes[0].set_title('Action vs U (at different W values)', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    # Slice 2: Action vs W (at different U values)
    u_percentiles = [10, 25, 50, 75, 90]
    u_values = np.percentile(u_vals, u_percentiles)

    for u_val, percentile in zip(u_values, u_percentiles):
        # Find points near this U value
        u_tolerance = 0.1 * u_vals.std()
        mask = np.abs(u_vals - u_val) < u_tolerance

        if mask.sum() > 10:
            # Bin by W value
            w_bins = 30
            w_stat, w_edges, _ = binned_statistic(
                w_vals[mask], action_vals[mask],
                statistic='mean', bins=w_bins
            )
            w_centers = (w_edges[:-1] + w_edges[1:]) / 2

            axes[1].plot(w_centers, w_stat,
                        label=f'U ≈ {u_val:.3f} (p{percentile})',
                        marker='o', markersize=4, alpha=0.7)

    axes[1].set_xlabel('W velocity', fontsize=12)
    axes[1].set_ylabel('Mean Action', fontsize=12)
    axes[1].set_title('Action vs W (at different U values)', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'action_map_slices.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()

    # ============================================================================
    # STATISTICS
    # ============================================================================
    print("\n" + "="*80)
    print("POLICY STATISTICS")
    print("="*80)
    print(f"\nData summary:")
    print(f"  Total samples: {len(u_vals)}")
    print(f"  Episodes used: {n_episodes}")
    print(f"  U velocity: mean={u_vals.mean():.4f}, std={u_vals.std():.4f}")
    print(f"  W velocity: mean={w_vals.mean():.4f}, std={w_vals.std():.4f}")
    print(f"  Action: mean={action_vals.mean():.6f}, std={action_vals.std():.4f}")

    # Compute correlation
    corr_u = np.corrcoef(u_vals, action_vals)[0, 1]
    corr_w = np.corrcoef(w_vals, action_vals)[0, 1]
    print(f"\nCorrelations:")
    print(f"  U vs Action: {corr_u:.4f}")
    print(f"  W vs Action: {corr_w:.4f}")

    if abs(corr_u) > abs(corr_w):
        print(f"  → Policy is more sensitive to U velocity")
    else:
        print(f"  → Policy is more sensitive to W velocity")

    # Find regions with extreme actions
    high_action_mask = action_vals > 0.5
    low_action_mask = action_vals < -0.5

    if high_action_mask.sum() > 0:
        print(f"\nHigh action regions (action > 0.5): {high_action_mask.sum()} samples "
              f"({100*high_action_mask.sum()/len(action_vals):.1f}%)")
        print(f"  Mean U: {u_vals[high_action_mask].mean():.4f}")
        print(f"  Mean W: {w_vals[high_action_mask].mean():.4f}")

    if low_action_mask.sum() > 0:
        print(f"\nLow action regions (action < -0.5): {low_action_mask.sum()} samples "
              f"({100*low_action_mask.sum()/len(action_vals):.1f}%)")
        print(f"  Mean U: {u_vals[low_action_mask].mean():.4f}")
        print(f"  Mean W: {w_vals[low_action_mask].mean():.4f}")

    # Check policy consistency (std in bins)
    mean_std = np.ma.mean(std_masked)
    print(f"\nPolicy consistency:")
    print(f"  Mean std within bins: {mean_std:.4f}")
    if mean_std < 0.1:
        print(f"  → Policy is deterministic (low variance within bins)")
    elif mean_std < 0.3:
        print(f"  → Policy is moderately consistent")
    else:
        print(f"  → Policy is noisy (high variance within bins)")

    # Save statistics to file
    summary_file = os.path.join(output_dir, 'action_map_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("ACTION POLICY MAP STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data summary:\n")
        f.write(f"  Total samples: {len(u_vals)}\n")
        f.write(f"  Episodes: {n_episodes}\n")
        f.write(f"  U velocity: mean={u_vals.mean():.4f}, std={u_vals.std():.4f}\n")
        f.write(f"  W velocity: mean={w_vals.mean():.4f}, std={w_vals.std():.4f}\n")
        f.write(f"  Action: mean={action_vals.mean():.6f}, std={action_vals.std():.4f}\n\n")
        f.write(f"Correlations:\n")
        f.write(f"  U vs Action: {corr_u:.4f}\n")
        f.write(f"  W vs Action: {corr_w:.4f}\n\n")
        f.write(f"Policy consistency: mean std within bins = {mean_std:.4f}\n")

    print(f"\nSummary saved to: {summary_file}")
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nAction map creation complete!")


if __name__ == "__main__":
    from scipy.stats import binned_statistic  # Import for 1D binning
    main()
