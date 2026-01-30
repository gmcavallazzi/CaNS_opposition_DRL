#!/usr/bin/env python3
"""
Verify if dataset has coherent structures (independent of POD)
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import glob
import argparse
from pathlib import Path
from io import StringIO
import contextlib

# Import read function
sys.path.insert(0, str(Path(__file__).parent.parent))
from read_2d_slice import read_binary_2d

def read_quiet(filepath, nx, ny):
    with contextlib.redirect_stdout(StringIO()):
        return read_binary_2d(filepath, nx, ny, dtype=np.float64)

# Parse arguments
parser = argparse.ArgumentParser(description='Verify if dataset has coherent structures')
parser.add_argument('data_dir', nargs='?', default='.', help='Path to data directory')
parser.add_argument('--nx', type=int, default=192, help='Grid size in x direction (default: 192)')
parser.add_argument('--ny', type=int, default=192, help='Grid size in y direction (default: 192)')
args = parser.parse_args()

nx, ny = args.nx, args.ny
data_dir = Path(args.data_dir)

# Find files
files = sorted(glob.glob(str(data_dir / 'vez_slice_*_fld_*.bin')))
print(f"Found {len(files)} files\n")

if len(files) < 10:
    print("ERROR: Need at least 10 files")
    sys.exit(1)

# ============================================================================
# Test 1: Visual inspection of individual snapshots
# ============================================================================
print("="*70)
print("TEST 1: Visual inspection of raw snapshots")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

sample_indices = [0, 100, 500, 1000, 5000, 10000]
for i, idx in enumerate(sample_indices):
    if idx >= len(files):
        idx = len(files) - 1

    data = read_quiet(files[idx], nx, ny)

    vmax = np.abs(data).max()
    im = axes[i].imshow(data.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        origin='lower', aspect='auto')
    axes[i].set_title(f'Snapshot {idx}\nstd={data.std():.4f}')
    plt.colorbar(im, ax=axes[i])

plt.suptitle('Raw Snapshots - Look for spatial patterns/structures', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('verify_snapshots.png', dpi=150)
print("Saved: verify_snapshots.png")
print("LOOK FOR: Streaks, vortices, or any repeating spatial patterns")
print("If you see random noise → no coherent structures")
print("If you see organized patterns → coherent structures exist\n")

# ============================================================================
# Test 2: Spatial variance map
# ============================================================================
print("="*70)
print("TEST 2: Spatial variance map")
print("="*70)

print("Loading first 1000 snapshots...")
snapshots = []
for i in range(min(1000, len(files))):
    snapshots.append(read_quiet(files[i], nx, ny))
data = np.array(snapshots)

# Compute variance at each spatial point across time
variance_map = data.var(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im0 = axes[0].imshow(variance_map.T, cmap='hot', origin='lower', aspect='auto')
axes[0].set_title('Variance Map')
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
plt.colorbar(im0, ax=axes[0], label='Variance')

axes[1].hist(variance_map.flatten(), bins=50, alpha=0.7)
axes[1].set_xlabel('Variance')
axes[1].set_ylabel('Count')
axes[1].set_title('Variance Distribution')
axes[1].axvline(variance_map.mean(), color='r', linestyle='--',
                label=f'Mean={variance_map.mean():.6f}')
axes[1].legend()

plt.tight_layout()
plt.savefig('verify_variance.png', dpi=150)
print("Saved: verify_variance.png")

# Analysis
var_std = variance_map.std()
var_mean = variance_map.mean()
var_ratio = var_std / var_mean
print(f"Variance statistics:")
print(f"  Mean variance: {var_mean:.6e}")
print(f"  Std of variance: {var_std:.6e}")
print(f"  Ratio (std/mean): {var_ratio:.4f}")
print()
print("INTERPRETATION:")
print(f"  Ratio < 0.1: Variance is nearly uniform → no preferred locations → likely flat POD")
print(f"  Ratio > 0.3: Variance varies spatially → structures at specific locations")
print(f"  Your ratio: {var_ratio:.4f}")
print()

# ============================================================================
# Test 3: Two-point spatial correlation
# ============================================================================
print("="*70)
print("TEST 3: Spatial correlation length")
print("="*70)

# Take middle row of first snapshot
snapshot = data[0, :, :]
middle_row = snapshot[:, ny//2]

# Compute autocorrelation
mean = middle_row.mean()
centered = middle_row - mean
autocorr = np.correlate(centered, centered, mode='full')
autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
autocorr = autocorr / autocorr[0]  # Normalize

# Find correlation length (where correlation drops to 1/e)
try:
    corr_length = np.where(autocorr < 1/np.e)[0][0]
except:
    corr_length = len(autocorr)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(autocorr[:50], 'o-', markersize=3)
ax.axhline(1/np.e, color='r', linestyle='--', label='1/e')
ax.axvline(corr_length, color='r', linestyle='--',
           label=f'Correlation length = {corr_length} points')
ax.set_xlabel('Spatial lag (grid points)')
ax.set_ylabel('Autocorrelation')
ax.set_title('Spatial Autocorrelation (middle row)')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('verify_spatial_correlation.png', dpi=150)
print("Saved: verify_spatial_correlation.png")
print(f"Correlation length: {corr_length} grid points ({corr_length/nx*100:.1f}% of domain)")
print()
print("INTERPRETATION:")
print(f"  Length < 5 points: Very small structures, nearly random")
print(f"  Length > 20 points: Large coherent structures exist")
print(f"  Your length: {corr_length} points")
print()

# ============================================================================
# Test 4: Snapshot-to-snapshot difference
# ============================================================================
print("="*70)
print("TEST 4: Temporal evolution (snapshot-to-snapshot change)")
print("="*70)

differences = []
for i in range(min(100, len(data)-1)):
    diff = np.abs(data[i+1] - data[i]).mean()
    differences.append(diff)

mean_diff = np.mean(differences)
snapshot_std = data[0].std()
relative_change = mean_diff / snapshot_std

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(differences, alpha=0.7)
ax.axhline(mean_diff, color='r', linestyle='--',
           label=f'Mean change={mean_diff:.6f}')
ax.set_xlabel('Snapshot pair')
ax.set_ylabel('Mean absolute difference')
ax.set_title('Consecutive Snapshot Differences')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('verify_temporal_change.png', dpi=150)
print("Saved: verify_temporal_change.png")
print(f"Mean change between snapshots: {mean_diff:.6e}")
print(f"Snapshot std: {snapshot_std:.6e}")
print(f"Relative change: {relative_change:.4f}")
print()
print("INTERPRETATION:")
print(f"  Relative change > 0.5: Snapshots very different → poorly correlated")
print(f"  Relative change < 0.1: Snapshots very similar → well correlated")
print(f"  Your relative change: {relative_change:.4f}")
print()

# ============================================================================
# Summary
# ============================================================================
print("="*70)
print("SUMMARY")
print("="*70)
print("\nCheck the 4 generated images:")
print("  1. verify_snapshots.png - Do you see patterns?")
print("  2. verify_variance.png - Is variance uniform or patchy?")
print("  3. verify_spatial_correlation.png - How far do correlations extend?")
print("  4. verify_temporal_change.png - How smooth is temporal evolution?")
print()
print("If all tests suggest randomness/uniformity, then your flat POD spectrum is REAL.")
print("="*70)
