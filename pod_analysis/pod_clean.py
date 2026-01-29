#!/usr/bin/env python3
"""
Clean POD implementation for 2D velocity slices
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import glob

# Import read function
sys.path.insert(0, str(Path(__file__).parent.parent))
from read_2d_slice import read_binary_2d

# Suppress read function output
from io import StringIO
import contextlib
def read_quiet(filepath, nx, ny):
    with contextlib.redirect_stdout(StringIO()):
        return read_binary_2d(filepath, nx, ny, dtype=np.float64)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True)
parser.add_argument('--output-dir', default='./pod_clean_results')
parser.add_argument('--n-jobs', type=int, default=1)
parser.add_argument('--n-modes', type=int, default=20, help='Number of modes to extract')
args = parser.parse_args()

data_dir = Path(args.data_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True)
nx, ny = 192, 192

print("="*80)
print("CLEAN POD IMPLEMENTATION")
print("="*80)

# ============================================================================
# 1. Load data
# ============================================================================
print("\n1. LOADING DATA")

# Find files
pattern = 'vez_slice_*_fld_*.bin'
files = sorted(glob.glob(str(data_dir / pattern)))
print(f"   Found {len(files)} files")

if len(files) == 0:
    print(f"ERROR: No files matching {pattern}")
    sys.exit(1)

# Load in parallel
print(f"   Loading with {args.n_jobs} jobs...")
if args.n_jobs > 1:
    snapshots = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(read_quiet)(f, nx, ny) for f in files
    )
else:
    snapshots = [read_quiet(f, nx, ny) for f in tqdm(files)]

# Stack into array: (n_snapshots, nx, ny)
data = np.array(snapshots, dtype=np.float32)
n_snapshots = data.shape[0]
print(f"   Loaded shape: {data.shape}")
print(f"   Data mean: {data.mean():.6e}, std: {data.std():.6e}")

# ============================================================================
# 2. Standard POD
# ============================================================================
print("\n2. PERFORMING POD")

# Reshape to matrix: (n_snapshots, n_points)
X = data.reshape(n_snapshots, -1)
print(f"   Matrix shape: {X.shape}")

# Remove temporal mean (standard POD)
X_mean = X.mean(axis=0)
X_centered = X - X_mean
print(f"   Mean field: mean={X_mean.mean():.6e}, std={X_mean.std():.6e}")

# SVD: X_centered = U @ diag(S) @ Vt
# U: temporal, Vt: spatial modes
print(f"   Computing SVD...")
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

print(f"   U shape: {U.shape}")  # (n_snapshots, n_snapshots)
print(f"   S shape: {S.shape}")  # (min(n_snapshots, n_points),)
print(f"   Vt shape: {Vt.shape}") # (min(...), n_points)

# Spatial modes are rows of Vt
spatial_modes = Vt[:args.n_modes].reshape(args.n_modes, nx, ny)

# Temporal coefficients are columns of U scaled by S
temporal_coeffs = U[:, :args.n_modes] * S[:args.n_modes]

# Energy
energy = S**2 / n_snapshots
total_energy = energy.sum()
energy_pct = 100 * energy / total_energy
cumulative = np.cumsum(energy_pct)

print(f"\n   Energy distribution (first 20 modes):")
for i in range(min(20, len(energy_pct))):
    print(f"   Mode {i+1:3d}: {energy_pct[i]:6.2f}%  (cumulative: {cumulative[i]:6.2f}%)")

# ============================================================================
# 3. Save results
# ============================================================================
print("\n3. SAVING RESULTS")

np.save(output_dir / 'modes.npy', spatial_modes)
np.save(output_dir / 'coefficients.npy', temporal_coeffs.T)
np.save(output_dir / 'mean_field.npy', X_mean.reshape(nx, ny))
np.save(output_dir / 'singular_values.npy', S)

print(f"   Saved to {output_dir}")

# ============================================================================
# 4. Plot
# ============================================================================
print("\n4. PLOTTING")

# Energy spectrum
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

modes_to_plot = min(50, len(S))
axes[0].bar(range(1, modes_to_plot+1), energy_pct[:modes_to_plot])
axes[0].set_xlabel('Mode')
axes[0].set_ylabel('Energy (%)')
axes[0].set_title('Energy Distribution')
axes[0].set_yscale('log')
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, modes_to_plot+1), cumulative[:modes_to_plot], 'o-')
axes[1].set_xlabel('Mode')
axes[1].set_ylabel('Cumulative Energy (%)')
axes[1].set_title('Cumulative Energy')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(90, color='r', linestyle='--', label='90%')
axes[1].axhline(99, color='r', linestyle=':', label='99%')
axes[1].legend()

plt.tight_layout()
plt.savefig(output_dir / 'energy_spectrum.png', dpi=150)
print(f"   Saved energy_spectrum.png")

# First 6 modes
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for i in range(6):
    mode = spatial_modes[i]
    vmax = np.abs(mode).max()

    im = axes[i].imshow(mode.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                        origin='lower', aspect='auto')
    axes[i].set_title(f'Mode {i+1} ({energy_pct[i]:.2f}%)')
    plt.colorbar(im, ax=axes[i])

plt.tight_layout()
plt.savefig(output_dir / 'modes.png', dpi=150)
print(f"   Saved modes.png")

print("\n" + "="*80)
print("DONE!")
print("="*80)
