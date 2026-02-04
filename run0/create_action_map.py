"""
Create a 2D map showing the relationship between u, w velocities
and action values at point (32, 32).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from scipy.stats import binned_statistic_2d

# Load data from all episodes
print("Loading episode data...")
all_u = []
all_w = []
all_actions = []

for ep in range(5):
    print(f"  Loading episode {ep}...")
    data = np.load(f'smoothness_check/episode_{ep}_data.npz')

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

# Create 2D binned statistics
n_bins = 50  # Number of bins in each direction

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
min_samples = 5
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
ax4.set_title('Action Std Dev in Each Bin', fontsize=14)
plt.colorbar(im4, ax=ax4, label='Std Dev')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('smoothness_check/action_map_u_w.png', dpi=300, bbox_inches='tight')
print("\nSaved: smoothness_check/action_map_u_w.png")
plt.close()

# Create a higher resolution version focused on the main plot
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

plt.tight_layout()
plt.savefig('smoothness_check/action_map_u_w_highres.png', dpi=300, bbox_inches='tight')
print("Saved: smoothness_check/action_map_u_w_highres.png")
plt.close()

# Print statistics about the policy
print("\n" + "="*80)
print("POLICY STATISTICS")
print("="*80)
print(f"\nData summary:")
print(f"  Total samples: {len(u_vals)}")
print(f"  U velocity: mean={u_vals.mean():.4f}, std={u_vals.std():.4f}")
print(f"  W velocity: mean={w_vals.mean():.4f}, std={w_vals.std():.4f}")
print(f"  Action: mean={action_vals.mean():.4f}, std={action_vals.std():.4f}")

# Compute correlation
corr_u = np.corrcoef(u_vals, action_vals)[0, 1]
corr_w = np.corrcoef(w_vals, action_vals)[0, 1]
print(f"\nCorrelations:")
print(f"  U vs Action: {corr_u:.4f}")
print(f"  W vs Action: {corr_w:.4f}")

# Find regions with extreme actions
high_action_mask = action_vals > 0.5
low_action_mask = action_vals < -0.5

if high_action_mask.sum() > 0:
    print(f"\nHigh action regions (action > 0.5): {high_action_mask.sum()} samples")
    print(f"  Mean U: {u_vals[high_action_mask].mean():.4f}")
    print(f"  Mean W: {w_vals[high_action_mask].mean():.4f}")

if low_action_mask.sum() > 0:
    print(f"\nLow action regions (action < -0.5): {low_action_mask.sum()} samples")
    print(f"  Mean U: {u_vals[low_action_mask].mean():.4f}")
    print(f"  Mean W: {w_vals[low_action_mask].mean():.4f}")

print("\n" + "="*80)
