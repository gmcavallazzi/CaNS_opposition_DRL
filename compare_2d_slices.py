#!/usr/bin/env python3
"""
Simple tool to compare two 2D binary slice files side by side
Usage: python compare_2d_slices.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

# Parameters - adjust these to match your simulation
NX = 192  # Global grid size in X
NY = 192  # Global grid size in Y
DTYPE = np.float64  # Use np.float32 for single precision

# File paths
file1 = 'data/test_w_original.bin'
file2 = 'data/test_w_readback.bin'

def read_binary_2d(filename, nx, ny, dtype=np.float64):
    """Read a 2D binary file written in Fortran column-major order"""
    data = np.fromfile(filename, dtype=dtype)
    expected_size = nx * ny

    if data.size != expected_size:
        print(f"WARNING: Expected {expected_size} values, got {data.size}")
        print(f"File: {filename}")

    # Fortran uses column-major order (F order)
    return data.reshape((nx, ny), order='F')

def compare_slices(file1, file2, nx, ny, dtype=np.float64):
    """Compare two 2D slice files and plot side by side"""

    # Read both files
    print(f"Reading {file1}...")
    data1 = read_binary_2d(file1, nx, ny, dtype)

    print(f"Reading {file2}...")
    data2 = read_binary_2d(file2, nx, ny, dtype)

    # Compute difference
    diff = data1 - data2
    max_diff = np.abs(diff).max()

    print(f"\n=== Comparison Results ===")
    print(f"Data1 range: [{data1.min():.6e}, {data1.max():.6e}]")
    print(f"Data2 range: [{data2.min():.6e}, {data2.max():.6e}]")
    print(f"Maximum absolute difference: {max_diff:.6e}")
    print(f"Mean absolute difference: {np.abs(diff).mean():.6e}")

    if max_diff == 0:
        print("\n✓ FILES ARE IDENTICAL!")
    else:
        print(f"\n✗ Files differ by up to {max_diff:.6e}")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Common colormap limits
    vmin = min(data1.min(), data2.min())
    vmax = max(data1.max(), data2.max())

    # Plot original
    im1 = axes[0].imshow(data1.T, origin='lower', aspect='auto',
                         cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0])

    # Plot readback
    im2 = axes[1].imshow(data2.T, origin='lower', aspect='auto',
                         cmap='RdBu_r', vmin=vmin, vmax=vmax)
    axes[1].set_title('Read-back')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1])

    # Plot difference
    if max_diff > 0:
        im3 = axes[2].imshow(diff.T, origin='lower', aspect='auto',
                             cmap='seismic', vmin=-max_diff, vmax=max_diff)
        axes[2].set_title(f'Difference (max={max_diff:.2e})')
    else:
        im3 = axes[2].imshow(diff.T, origin='lower', aspect='auto',
                             cmap='gray')
        axes[2].set_title('Difference (ZERO)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()

    # Save figure
    outfile = 'data/slice_comparison.png'
    plt.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {outfile}")

    plt.show()

if __name__ == '__main__':
    print("2D Slice Comparison Tool")
    print("=" * 50)
    print(f"Grid size: {NX} x {NY}")
    print(f"Data type: {DTYPE}")
    print()

    try:
        compare_slices(file1, file2, NX, NY, DTYPE)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nMake sure to run this from the run directory where data/ exists")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
