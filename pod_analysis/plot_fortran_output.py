#!/usr/bin/env python3
"""
Plot the Fortran POD output binaries side-by-side with the originals.

Reads pod_mode1.bin and pod_modeN.bin produced by pod_fortran,
each containing vex then vez concatenated (192*192 doubles each).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

NX, NY = 192, 192
NPTS = NX * NY
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_slices')


def read_binary_2d(filename, nx, ny, dtype=np.float64):
    data = np.fromfile(filename, dtype=dtype)
    return data.reshape((nx, ny), order='F')


def read_two_fields(binfile):
    """Read a file containing two concatenated 192x192 fields."""
    raw = np.fromfile(binfile, dtype=np.float64)
    f1 = raw[:NPTS].reshape((NX, NY), order='F')
    f2 = raw[NPTS:2*NPTS].reshape((NX, NY), order='F')
    return f1, f2


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # originals
    vex_orig = read_binary_2d(os.path.join(DATA_DIR, 'vex_slice_19_fld_0449500.bin'), NX, NY)
    vez_orig = read_binary_2d(os.path.join(DATA_DIR, 'vez_slice_19_fld_0449500.bin'), NX, NY)

    # Fortran reconstructions
    mode1_vex, mode1_vez = read_two_fields(os.path.join(script_dir, 'pod_mode1.bin'))
    modeN_vex, modeN_vez = read_two_fields(os.path.join(script_dir, 'pod_modeN.bin'))

    # read summary to get n_keep
    n_keep = '?'
    sumfile = os.path.join(script_dir, 'pod_fortran_result.dat')
    if os.path.exists(sumfile):
        with open(sumfile) as f:
            for line in f:
                if 'Modes retained' in line:
                    n_keep = line.split(':')[1].strip()
                    break

    labels = ['vex', 'vez']
    rows = [[vex_orig, mode1_vex, modeN_vex],
            [vez_orig, mode1_vez, modeN_vez]]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        f"POD (Fortran) — {n_keep} modes retained (90% energy)\n"
        "Left: original | Centre: mode 1 (most energetic) | Right: least energetic retained",
        fontsize=11
    )

    for i in range(2):
        for j in range(3):
            im = axes[i, j].imshow(rows[i][j].T, origin='lower', aspect='auto', cmap='RdBu_r')
            axes[i, j].set_ylabel(labels[i] if j == 0 else '')
            plt.colorbar(im, ax=axes[i, j], label='value')

        axes[i, 0].set_title('original')
        axes[i, 1].set_title('mode 1 (most energetic)')
        axes[i, 2].set_title(f'mode {n_keep} (least energetic retained)')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    outpath = os.path.join(script_dir, 'pod_fortran_plot.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {outpath}")
    plt.show()


if __name__ == '__main__':
    main()
