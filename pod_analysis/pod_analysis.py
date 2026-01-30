"""
Proper Orthogonal Decomposition (POD) Analysis for Vertical Velocity (vez)

Performs POD on vertical velocity field (vez),
selects modes to recover specified energy threshold, and projects all snapshots
onto the selected modes for compressed representation.

Usage:
    python pod_analysis.py --data-dir /path/to/data [options]

Arguments:
    --data-dir PATH         Path to directory containing binary field data (required)
    --output-dir PATH       Path to output directory (default: ./pod_results_vez)
    --n-snapshots INT       Number of snapshots to load (default: 20000)
    --start-snapshot INT    Starting snapshot index (default: 1)
    --energy-threshold FLOAT Energy threshold for mode selection (default: 0.99)
    --n-modes-plot INT      Number of modes to plot (default: 6)
    --n-components INT      Max components for randomized SVD (default: 80)
    --no-normalize          Disable field normalization
    --full-svd              Use full SVD instead of randomized

Output:
    - POD modes (spatial patterns)
    - POD coefficients (temporal evolution)
    - Comprehensive visualizations
    - Energy distribution plots
    - Temporal evolution plots
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for HPC
import matplotlib.pyplot as plt
import glob
import argparse
import sys
from pathlib import Path
from sklearn.utils.extmath import randomized_svd
from tqdm import tqdm
from joblib import Parallel, delayed

# Import read function from parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))
from read_2d_slice import read_binary_2d as _read_binary_2d

# Wrapper to suppress verbose output from read_binary_2d
def read_binary_2d(filename, nx, ny, dtype=np.float64):
    """Silent wrapper for read_binary_2d"""
    import os
    from io import StringIO
    import contextlib

    # Suppress stdout during read
    with contextlib.redirect_stdout(StringIO()):
        return _read_binary_2d(filename, nx, ny, dtype)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='POD Analysis for vertical velocity (vez) field')
parser.add_argument('--data-dir', type=str, required=True,
                    help='Path to directory containing binary field data')
parser.add_argument('--output-dir', type=str, default='./pod_results_vez',
                    help='Path to output directory (default: ./pod_results_vez)')
parser.add_argument('--n-snapshots', type=int, default=None,
                    help='Number of snapshots to load (default: all available files)')
parser.add_argument('--start-snapshot', type=int, default=1,
                    help='Starting snapshot index (default: 1, skip first snapshot)')
parser.add_argument('--energy-threshold', type=float, default=0.99,
                    help='Energy threshold for mode selection (default: 0.99)')
parser.add_argument('--n-modes-plot', type=int, default=6,
                    help='Number of modes to plot (default: 6)')
parser.add_argument('--n-components', type=int, default=100,
                    help='Max components for randomized SVD (default: 20)')
parser.add_argument('--no-normalize', action='store_true',
                    help='Disable field normalization')
parser.add_argument('--full-svd', action='store_true',
                    help='Use full SVD instead of randomized SVD')
parser.add_argument('--n-jobs', type=int, default=1,
                    help='Number of parallel jobs for data loading (default: 1)')
parser.add_argument('--slice-id', type=str, default='*',
                    help='Slice ID to load (default: * for auto-detect, or specify like "19")')
parser.add_argument('--nx', type=int, default=192,
                    help='Grid size in x direction (default: 192)')
parser.add_argument('--ny', type=int, default=192,
                    help='Grid size in y direction (default: 192)')

args = parser.parse_args()

# Set parameters from arguments
data_dir = Path(args.data_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

nx, ny = args.nx, args.ny
n_snapshots = args.n_snapshots
start_snapshot = args.start_snapshot
energy_threshold = args.energy_threshold
use_randomized_svd = not args.full_svd
n_components = args.n_components
normalize_fields = not args.no_normalize
n_modes_to_plot = args.n_modes_plot
n_jobs = args.n_jobs
slice_id = args.slice_id

print("="*80)
print("POD ANALYSIS - VERTICAL VELOCITY (VEZ)")
print("="*80)
print(f"\nConfiguration:")
print(f"  Data directory: {data_dir}")
print(f"  Output directory: {output_dir}")
print(f"  Slice ID pattern: {slice_id}")
print(f"  Start snapshot: {start_snapshot} (skipping first snapshot)")
print(f"  Requested snapshots: {'all available' if n_snapshots is None else n_snapshots}")
print(f"  Grid: {nx} x {ny}")
print(f"  Field: Vertical velocity (vez)")
print(f"  Energy threshold: {energy_threshold*100}%")
print(f"  Modes to plot: {n_modes_to_plot}")
print(f"  Field normalization: {normalize_fields}")
print(f"  SVD method: {'Randomized' if use_randomized_svd else 'Full'}")
if use_randomized_svd:
    print(f"  Max components: {n_components}")
if n_jobs > 1:
    print(f"  Parallel loading: {n_jobs} jobs")

# ============================================================================
# 1. Load all data
# ============================================================================
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)


# Get snapshot files
pattern = f'vez_slice_{slice_id}_fld_*.bin'
vez_files = sorted(glob.glob(str(data_dir / pattern)))

if len(vez_files) == 0:
    print(f"\nERROR: No vez files found in {data_dir}")
    print(f"Looking for pattern: {pattern}")
    print("\nAvailable files in directory:")
    all_files = sorted(list(data_dir.glob('*.bin')))[:10]  # Show first 10 files
    for f in all_files:
        print(f"  {f.name}")
    if len(all_files) == 0:
        print("  (no .bin files found)")

    # Try to auto-detect slice ID
    print("\nTrying to auto-detect slice ID...")
    test_pattern = 'vez_slice_*_fld_*.bin'
    test_files = list(data_dir.glob(test_pattern))
    if test_files:
        sample_file = test_files[0].name
        # Extract slice ID: vez_slice_19_fld_1273700.bin -> 19
        parts = sample_file.split('_')
        if len(parts) >= 4:
            detected_slice = parts[2]
            print(f"  Detected slice ID: {detected_slice}")
            print(f"  Retry with: --slice-id {detected_slice}")
    exit(1)

# Extract snapshot IDs from filenames
# Format: vez_slice_19_fld_1273700.bin -> 1273700
snapshot_ids = []
for f in vez_files:
    filename = Path(f).name
    # Split by underscore and get the part after 'fld_'
    parts = filename.split('_fld_')
    if len(parts) == 2:
        snap_id = parts[1].replace('.bin', '')
        snapshot_ids.append(snap_id)

# Determine actual number of snapshots to load
n_available = len(snapshot_ids) - start_snapshot

# Detect actual slice ID
first_file = Path(vez_files[0]).name
actual_slice = first_file.split('_')[2]  # vez_slice_19_fld_... -> 19
print(f"Detected slice ID: {actual_slice}")

if n_snapshots is None:
    # Use all available files
    n_snapshots_to_load = n_available
    print(f"Found {len(snapshot_ids)} total snapshot files")
    print(f"Loading all {n_snapshots_to_load} available snapshots starting from index {start_snapshot}...")
elif n_snapshots > n_available:
    print(f"\nWARNING: Requested {n_snapshots} snapshots, but only {n_available} available")
    print(f"         (after skipping first {start_snapshot})")
    n_snapshots_to_load = n_available
    print(f"Found {len(snapshot_ids)} total snapshot files")
    print(f"Loading {n_snapshots_to_load} snapshots starting from index {start_snapshot}...")
else:
    n_snapshots_to_load = n_snapshots
    print(f"Found {len(snapshot_ids)} total snapshot files")
    print(f"Loading {n_snapshots_to_load} snapshots starting from index {start_snapshot}...")

# Load data - single channel: vez
field_data = np.zeros((n_snapshots_to_load, nx, ny), dtype=np.float32)

if n_jobs > 1:
    print(f"  Using parallel loading with {n_jobs} jobs...")

    def load_snapshot(i):
        snap_id = snapshot_ids[i + start_snapshot]
        filepath = str(data_dir / f'vez_slice_{actual_slice}_fld_{snap_id}.bin')
        return read_binary_2d(filepath, nx, ny, dtype=np.float64)

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(load_snapshot)(i) for i in range(n_snapshots_to_load)
    )

    for i, vez in enumerate(results):
        field_data[i] = vez
else:
    for i in tqdm(range(n_snapshots_to_load), desc="Loading snapshots"):
        snap_id = snapshot_ids[i + start_snapshot]
        filepath = str(data_dir / f'vez_slice_{actual_slice}_fld_{snap_id}.bin')
        vez = read_binary_2d(filepath, nx, ny, dtype=np.float64)
        field_data[i] = vez

print(f"\nData loaded successfully!")
print(f"  Shape: {field_data.shape} (n_snapshots, nx, ny)")
print(f"\nData statistics:")
print(f"  vez: Mean={field_data.mean():.6e}, Std={field_data.std():.6e}")

# ============================================================================
# 2. Perform POD
# ============================================================================
print("\n" + "="*80)
print("PERFORMING POD")
print("="*80)

n_snapshots_actual, nx, ny = field_data.shape
n_spatial = nx * ny

# Reshape to matrix form
data_matrix = field_data.reshape(n_snapshots_actual, -1)
print(f"Data matrix: {data_matrix.shape}")

# Compute temporal mean field (average each spatial point across all time)
print("\nComputing temporal mean field...")
mean_field = data_matrix.mean(axis=0)
data_centered = data_matrix - mean_field

print(f"  Temporal mean field computed")
print(f"    Mean: {mean_field.mean():.6e}")
print(f"    Std: {mean_field.std():.6e}")
print(f"    Range: [{mean_field.min():.6e}, {mean_field.max():.6e}]")

# Normalize field by std
if normalize_fields:
    print("\nNormalizing field by standard deviation...")
    std_field_value = np.std(data_centered)

    print(f"  Field standard deviation: {std_field_value:.6e}")

    if std_field_value > 0:
        data_centered = data_centered / std_field_value
        std_field = std_field_value
    else:
        std_field = 1.0

    print("  Field normalized")
else:
    # When not normalizing, set std field to 1 for reconstruction compatibility
    print("\nSkipping field normalization...")
    std_field = 1.0

# Perform SVD
print("\nPerforming SVD...")
if use_randomized_svd:
    print(f"  Using randomized SVD...")
    n_components_actual = min(n_components, n_snapshots_actual - 1, n_spatial - 1)
    print(f"  Computing {n_components_actual} components...")

    U, S, Vt = randomized_svd(
        data_centered.T,
        n_components=n_components_actual,
        n_iter=5,
        random_state=42
    )
else:
    print(f"  Using full SVD...")
    U, S, Vt = np.linalg.svd(data_centered.T, full_matrices=False)

print(f"\nSVD complete!")
print(f"  U shape: {U.shape}")
print(f"  S shape: {S.shape}")
print(f"  Vt shape: {Vt.shape}")

# Compute energy
energy = S**2
total_energy = energy.sum()
energy_ratio = np.cumsum(energy) / total_energy

# Find modes for threshold
n_modes_threshold = np.searchsorted(energy_ratio, energy_threshold) + 1

print(f"\nEnergy analysis:")
print(f"  Total energy: {total_energy:.6e}")
print(f"  Energy threshold: {energy_threshold*100}%")
print(f"  Modes needed: {n_modes_threshold}")
print(f"  Actual energy recovered: {energy_ratio[n_modes_threshold-1]*100:.2f}%")

# Extract modes and coefficients
spatial_modes = U[:, :n_modes_threshold].T
spatial_modes = spatial_modes.reshape(n_modes_threshold, nx, ny)

temporal_coefficients = np.diag(S[:n_modes_threshold]) @ Vt[:n_modes_threshold, :]

print(f"\nPOD modes extracted:")
print(f"  Spatial modes: {spatial_modes.shape} (n_modes, nx, ny)")
print(f"  Temporal coefficients: {temporal_coefficients.shape} (n_modes, n_snapshots)")

# ============================================================================
# 3. Visualize modes
# ============================================================================
print("\n" + "="*80)
print("VISUALIZING MODES")
print("="*80)

n_modes_to_visualize = min(n_modes_to_plot, n_modes_threshold)

# Plot spatial modes for vez
# Determine grid layout
n_cols = 3
n_rows = int(np.ceil(n_modes_to_visualize / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
if n_rows == 1:
    axes = axes.reshape(1, -1)

for mode_idx in range(n_modes_to_visualize):
    row = mode_idx // n_cols
    col = mode_idx % n_cols
    ax = axes[row, col]

    mode_field = spatial_modes[mode_idx, :, :]

    # Symmetric colormap for velocity
    vmax = np.abs(mode_field).max()
    vmin = -vmax

    im = ax.imshow(
        mode_field.T,
        cmap='RdBu_r',
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
        origin='lower'
    )

    energy_pct = energy[mode_idx] / total_energy * 100
    ax.set_title(f'Mode {mode_idx+1} ({energy_pct:.2f}%)', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax)

# Hide empty subplots
for mode_idx in range(n_modes_to_visualize, n_rows * n_cols):
    row = mode_idx // n_cols
    col = mode_idx % n_cols
    axes[row, col].axis('off')

plt.suptitle('POD Modes: Vertical Velocity (vez)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'pod_modes_vez.png', dpi=150, bbox_inches='tight')
print(f"  Saved: pod_modes_vez.png")
plt.close()

# Plot mean field
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
mean_field_2d = mean_field.reshape(nx, ny)

vmax = np.abs(mean_field_2d).max()
vmin = -vmax

im = ax.imshow(
    mean_field_2d.T,
    cmap='RdBu_r',
    vmin=vmin,
    vmax=vmax,
    aspect='auto',
    origin='lower'
)
ax.set_title('Mean vez Field', fontsize=14, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig(output_dir / 'pod_mean_field.png', dpi=150)
print(f"  Saved: pod_mean_field.png")
plt.close()

# Energy distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

modes_to_show = min(50, len(S))
energy_pct = energy[:modes_to_show] / total_energy * 100

axes[0].bar(range(1, modes_to_show+1), energy_pct, alpha=0.7)
axes[0].set_xlabel('Mode number', fontsize=12)
axes[0].set_ylabel('Energy contribution (%)', fontsize=12)
axes[0].set_title(f'Energy Distribution (first {modes_to_show} modes)', fontsize=14, fontweight='bold')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axvline(n_modes_threshold, color='red', linestyle='--',
               linewidth=2, label=f"{energy_threshold*100}% energy ({n_modes_threshold} modes)")
axes[0].legend()

cumulative_energy = energy_ratio[:modes_to_show] * 100
axes[1].plot(range(1, modes_to_show+1), cumulative_energy, 'b-', linewidth=2)
axes[1].axhline(energy_threshold*100, color='red', linestyle='--', linewidth=2,
                label=f'{energy_threshold*100}% threshold')
axes[1].axvline(n_modes_threshold, color='red', linestyle='--', linewidth=2,
                label=f'{n_modes_threshold} modes')
axes[1].set_xlabel('Number of modes', fontsize=12)
axes[1].set_ylabel('Cumulative energy (%)', fontsize=12)
axes[1].set_title('Cumulative Energy Recovery', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([0, 105])
axes[1].legend()

plt.tight_layout()
plt.savefig(output_dir / 'pod_energy_distribution.png', dpi=150)
print(f"  Saved: pod_energy_distribution.png")
plt.close()

# ============================================================================
# 4. Additional useful visualizations
# ============================================================================
print("\nCreating additional visualizations...")

# Plot temporal coefficients for first modes
n_coeff_to_plot = min(6, n_modes_threshold)
n_samples_to_show = min(1000, n_snapshots_actual)

fig, axes = plt.subplots(n_coeff_to_plot, 1, figsize=(14, 2*n_coeff_to_plot))
if n_coeff_to_plot == 1:
    axes = [axes]

for i in range(n_coeff_to_plot):
    axes[i].plot(temporal_coefficients[i, :n_samples_to_show], linewidth=0.5, alpha=0.8)
    axes[i].set_ylabel(f'Mode {i+1}\nCoeff', fontsize=10)
    axes[i].grid(True, alpha=0.3)
    energy_pct = energy[i] / total_energy * 100
    axes[i].set_title(f'Mode {i+1} Temporal Evolution ({energy_pct:.2f}% energy)',
                     fontsize=11, loc='right')

axes[-1].set_xlabel(f'Snapshot (showing first {n_samples_to_show})', fontsize=12)
plt.suptitle('Temporal Coefficients Evolution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir / 'pod_temporal_coefficients.png', dpi=150)
print(f"  Saved: pod_temporal_coefficients.png")
plt.close()

# Plot RMS amplitude of temporal coefficients
rms_amplitudes = np.sqrt(np.mean(temporal_coefficients**2, axis=1))
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
modes_range = range(1, min(30, n_modes_threshold) + 1)
ax.bar(modes_range, rms_amplitudes[:len(modes_range)], alpha=0.7, color='steelblue')
ax.set_xlabel('Mode number', fontsize=12)
ax.set_ylabel('RMS Amplitude', fontsize=12)
ax.set_title('RMS Amplitude of Temporal Coefficients', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(output_dir / 'pod_rms_amplitudes.png', dpi=150)
print(f"  Saved: pod_rms_amplitudes.png")
plt.close()

# Energy spectrum with more detail
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Linear scale
modes_detail = min(20, len(S))
energy_pct_detail = energy[:modes_detail] / total_energy * 100
axes[0].bar(range(1, modes_detail+1), energy_pct_detail, alpha=0.7, color='steelblue')
axes[0].set_xlabel('Mode number', fontsize=12)
axes[0].set_ylabel('Energy contribution (%)', fontsize=12)
axes[0].set_title(f'Energy Spectrum (first {modes_detail} modes, linear scale)',
                  fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')

# Log-log scale for full spectrum
all_modes = range(1, len(S) + 1)
all_energy_pct = energy / total_energy * 100
axes[1].loglog(all_modes, all_energy_pct, 'o-', markersize=3, alpha=0.7, color='steelblue')
axes[1].axvline(n_modes_threshold, color='red', linestyle='--', linewidth=2,
                label=f'{n_modes_threshold} modes ({energy_threshold*100}%)')
axes[1].set_xlabel('Mode number', fontsize=12)
axes[1].set_ylabel('Energy contribution (%)', fontsize=12)
axes[1].set_title('Energy Spectrum (log-log scale)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, which='both')
axes[1].legend()

plt.tight_layout()
plt.savefig(output_dir / 'pod_energy_spectrum.png', dpi=150)
print(f"  Saved: pod_energy_spectrum.png")
plt.close()

# Mode correlation matrix (for first few modes)
n_modes_corr = min(10, n_modes_threshold)
temporal_subset = temporal_coefficients[:n_modes_corr, :]
correlation_matrix = np.corrcoef(temporal_subset)

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(n_modes_corr))
ax.set_yticks(range(n_modes_corr))
ax.set_xticklabels([f'M{i+1}' for i in range(n_modes_corr)])
ax.set_yticklabels([f'M{i+1}' for i in range(n_modes_corr)])
ax.set_xlabel('Mode', fontsize=12)
ax.set_ylabel('Mode', fontsize=12)
ax.set_title(f'Temporal Correlation Matrix (first {n_modes_corr} modes)',
             fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Correlation coefficient')

# Add correlation values as text
for i in range(n_modes_corr):
    for j in range(n_modes_corr):
        text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=8)

plt.tight_layout()
plt.savefig(output_dir / 'pod_mode_correlation.png', dpi=150)
print(f"  Saved: pod_mode_correlation.png")
plt.close()

# Singular value decay
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.semilogy(range(1, len(S)+1), S, 'o-', markersize=4, alpha=0.7, color='steelblue')
ax.axvline(n_modes_threshold, color='red', linestyle='--', linewidth=2,
           label=f'{n_modes_threshold} modes ({energy_threshold*100}%)')
ax.set_xlabel('Mode number', fontsize=12)
ax.set_ylabel('Singular value', fontsize=12)
ax.set_title('Singular Value Decay', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend()
plt.tight_layout()
plt.savefig(output_dir / 'pod_singular_values.png', dpi=150)
print(f"  Saved: pod_singular_values.png")
plt.close()

# ============================================================================
# 5. Save results
# ============================================================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

# Save coefficients
np.savez(output_dir / 'pod_coefficients.npz', temporal_coefficients)
print(f"  Saved: pod_coefficients.npz")
print(f"    Shape: {temporal_coefficients.shape} (n_modes, n_snapshots)")

# Save spatial modes
np.save(output_dir / 'pod_modes.npy', spatial_modes)
print(f"  Saved: pod_modes.npy")

# Save mean field
mean_field_reshaped = mean_field.reshape(nx, ny)
np.save(output_dir / 'pod_mean_field.npy', mean_field_reshaped)
print(f"  Saved: pod_mean_field.npy")

# Save std field (always save, even if it's 1.0)
np.save(output_dir / 'pod_std_field.npy', std_field)
print(f"  Saved: pod_std_field.npy")

# Save metadata
metadata = {
    'n_modes': n_modes_threshold,
    'total_modes': len(S),
    'energy_threshold': energy_threshold,
    'energy_recovered': float(energy_ratio[n_modes_threshold-1]),
    'total_energy': float(total_energy),
    'normalize_fields': normalize_fields,
    'singular_values': S[:n_modes_threshold],
    'energy_per_mode': energy[:n_modes_threshold],
    'cumulative_energy_ratio': energy_ratio[:n_modes_threshold],
    'rms_amplitudes': rms_amplitudes
}

np.savez(output_dir / 'pod_metadata.npz', **metadata)
print(f"  Saved: pod_metadata.npz")

# Save summary
with open(output_dir / 'pod_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("POD ANALYSIS SUMMARY - VERTICAL VELOCITY (VEZ)\n")
    f.write("="*80 + "\n\n")
    f.write(f"Data directory: {data_dir}\n")
    f.write(f"Output directory: {output_dir}\n")
    f.write(f"Field: Vertical velocity (vez)\n")
    f.write(f"Grid size: {nx} x {ny}\n")
    f.write(f"Field normalization: {'Enabled' if normalize_fields else 'Disabled'}\n")
    f.write(f"Energy threshold: {energy_threshold*100}%\n")
    f.write(f"Modes selected: {n_modes_threshold}\n")
    f.write(f"Energy recovered: {energy_ratio[n_modes_threshold-1]*100:.2f}%\n")
    f.write(f"Total energy: {total_energy:.6e}\n\n")
    f.write("Top 20 modes by energy contribution:\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Mode':<8} {'Energy':<15} {'% Total':<12} {'Cumulative %':<15} {'RMS Amp':<12}\n")
    f.write("-"*80 + "\n")
    for i in range(min(20, n_modes_threshold)):
        e = energy[i]
        pct = e / total_energy * 100
        cum_pct = energy_ratio[i] * 100
        rms = rms_amplitudes[i]
        f.write(f"{i+1:<8} {e:<15.6e} {pct:<12.3f} {cum_pct:<15.2f} {rms:<12.6e}\n")
    f.write("="*80 + "\n")

print(f"  Saved: pod_summary.txt")

print("\n" + "="*80)
print("POD ANALYSIS COMPLETE!")
print("="*80)
print(f"\nResults saved to: {output_dir}")
print(f"\nData files:")
print(f"  - Coefficients: pod_coefficients.npz ({n_modes_threshold} modes, {n_snapshots_actual} snapshots)")
print(f"  - Modes: pod_modes.npy ({n_modes_threshold}, {nx}, {ny})")
print(f"  - Mean field: pod_mean_field.npy ({nx}, {ny})")
print(f"  - Std field: pod_std_field.npy")
print(f"  - Metadata: pod_metadata.npz")
print(f"  - Summary: pod_summary.txt")
print(f"\nVisualizations:")
print(f"  - pod_modes_vez.png - First {n_modes_to_visualize} vez modes")
print(f"  - pod_mean_field.png - Mean vez field")
print(f"  - pod_energy_distribution.png - Energy distribution and cumulative sum")
print(f"  - pod_energy_spectrum.png - Detailed energy spectrum")
print(f"  - pod_temporal_coefficients.png - Temporal evolution of first modes")
print(f"  - pod_rms_amplitudes.png - RMS amplitudes of temporal coefficients")
print(f"  - pod_mode_correlation.png - Correlation between modes")
print(f"  - pod_singular_values.png - Singular value decay")
