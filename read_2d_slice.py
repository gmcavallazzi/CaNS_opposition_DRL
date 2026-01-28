#!/usr/bin/env python3
"""
Read and visualize a single 2D binary slice file
Usage: python read_2d_slice.py <filename> [--nx NX] [--ny NY] [--float32]

Examples:
    python read_2d_slice.py data/test_w_original.bin
    python read_2d_slice.py data/test_w_original.bin --nx 192 --ny 192
    python read_2d_slice.py data/test_w_original.bin --nx 192 --ny 192 --float32
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

def read_binary_2d(filename, nx, ny, dtype=np.float64):
    """Read a 2D binary file written in Fortran column-major order"""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    data = np.fromfile(filename, dtype=dtype)
    expected_size = nx * ny

    print(f"File: {filename}")
    print(f"File size: {os.path.getsize(filename)} bytes")
    print(f"Expected elements: {expected_size}")
    print(f"Read elements: {data.size}")

    if data.size != expected_size:
        print(f"\nWARNING: Size mismatch!")
        print(f"  Expected: {expected_size} ({nx} x {ny})")
        print(f"  Got: {data.size}")
        print(f"  Bytes per element: {data.itemsize}")

        # Try to suggest correct grid size
        total = data.size
        # Find factor pairs
        factors = []
        for i in range(1, int(np.sqrt(total)) + 1):
            if total % i == 0:
                factors.append((i, total // i))

        if factors:
            print(f"\nPossible grid sizes for {total} elements:")
            for nx_try, ny_try in factors[-5:]:  # Show last 5 (likely square-ish)
                print(f"  {nx_try} x {ny_try}")

    # Fortran uses column-major order (F order)
    data_2d = data.reshape((nx, ny), order='F')
    return data_2d

def plot_slice(data, filename, save=True):
    """Plot a 2D slice"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot
    im = ax.imshow(data.T, origin='lower', aspect='auto', cmap='RdBu_r')
    ax.set_title(f'2D Slice: {os.path.basename(filename)}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value')

    # Add statistics as text
    stats_text = f'Min: {data.min():.6e}\n'
    stats_text += f'Max: {data.max():.6e}\n'
    stats_text += f'Mean: {data.mean():.6e}\n'
    stats_text += f'Std: {data.std():.6e}'

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            fontfamily='monospace',
            fontsize=9)

    plt.tight_layout()

    if save:
        outfile = filename.replace('.bin', '.png')
        if outfile == filename:  # Safety check
            outfile = filename + '.png'
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {outfile}")

    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Read and visualize a 2D binary slice file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s data/test_w_original.bin
    %(prog)s data/test_w_original.bin --nx 192 --ny 192
    %(prog)s data/test_w_original.bin --nx 192 --ny 192 --float32
        """
    )

    parser.add_argument('filename', help='Binary file to read')
    parser.add_argument('--nx', type=int, default=192, help='Grid size in X (default: 192)')
    parser.add_argument('--ny', type=int, default=192, help='Grid size in Y (default: 192)')
    parser.add_argument('--float32', action='store_true', help='Use single precision (default: double)')
    parser.add_argument('--no-plot', action='store_true', help='Do not show plot (only print stats)')
    parser.add_argument('--no-save', action='store_true', help='Do not save plot to file')

    args = parser.parse_args()

    # Set dtype
    dtype = np.float32 if args.float32 else np.float64

    print("=" * 60)
    print("2D Binary Slice Reader")
    print("=" * 60)
    print(f"Grid size: {args.nx} x {args.ny}")
    print(f"Data type: {dtype.__name__}")
    print()

    try:
        # Read the file
        data = read_binary_2d(args.filename, args.nx, args.ny, dtype)

        # Print statistics
        print("\n" + "=" * 60)
        print("Statistics:")
        print("=" * 60)
        print(f"Shape: {data.shape}")
        print(f"Min:   {data.min():.10e}")
        print(f"Max:   {data.max():.10e}")
        print(f"Mean:  {data.mean():.10e}")
        print(f"Std:   {data.std():.10e}")
        print()

        # Plot if requested
        if not args.no_plot:
            plot_slice(data, args.filename, save=not args.no_save)

        return 0

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
