#!/usr/bin/env python3
"""
POD (Proper Orthogonal Decomposition) of 2D velocity slice fields.

Reads vex and vez binary slices, stacks them into a snapshot matrix,
performs SVD-based POD, truncates at 90% energy, then plots the
reconstruction from only the most energetic mode and only the least
energetic (retained) mode.

Usage:
    python pod_python.py

Paths and grid size are set in the CONFIG section below.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data_slices')
FILES = [
    'vex_slice_19_fld_0449500.bin',
    'vez_slice_19_fld_0449500.bin',
]
NX, NY = 192, 192
DTYPE = np.float64
ENERGY_THRESHOLD = 0.90   # retain modes until cumulative energy >= 90 %
# ---------------------------------------------------------------------------


def read_binary_2d(filename, nx, ny, dtype=np.float64):
    """Read a 2D binary slice (Fortran column-major order).
    Reuses the convention from read_2d_slice.py."""
    data = np.fromfile(filename, dtype=dtype)
    return data.reshape((nx, ny), order='F')


def build_snapshot_matrix(fields):
    """Stack flattened 2D fields as columns of the snapshot matrix.
    Each column is one 'snapshot' (here each velocity component is a snapshot).
    Shape: (nx*ny, n_snapshots)."""
    return np.column_stack([f.ravel(order='F') for f in fields])


def pod_svd(snapshot_matrix):
    """Thin SVD-based POD.
    Returns:
        U  : left singular vectors  (modes)  shape (n_pts, n_modes)
        S  : singular values         shape (n_modes,)
        Vt : right singular vectors  shape (n_modes, n_snaps)
    """
    U, S, Vt = np.linalg.svd(snapshot_matrix, full_matrices=False)
    return U, S, Vt


def truncate_90(S):
    """Return number of modes to keep for >= ENERGY_THRESHOLD of total energy."""
    energy = S ** 2
    cumulative = np.cumsum(energy) / np.sum(energy)
    n_keep = int(np.argmax(cumulative >= ENERGY_THRESHOLD)) + 1
    return n_keep, cumulative


def reconstruct_single_mode(U, S, Vt, mode_index):
    """Reconstruct snapshot matrix using only one mode (0-indexed)."""
    return np.outer(U[:, mode_index], S[mode_index] * Vt[mode_index, :])


def main():
    # --- read fields ---
    fields = []
    for fname in FILES:
        path = os.path.join(DATA_DIR, fname)
        print(f"Reading {path}")
        fields.append(read_binary_2d(path, NX, NY, DTYPE))

    # --- build snapshot matrix and perform POD ---
    X = build_snapshot_matrix(fields)
    print(f"\nSnapshot matrix shape: {X.shape}  "
          f"({X.shape[0]} spatial points, {X.shape[1]} snapshots)")

    U, S, Vt = pod_svd(X)

    # --- energy & truncation ---
    n_keep, cumulative_energy = truncate_90(S)
    print(f"\nSingular values: {S}")
    print(f"Cumulative energy fraction: {cumulative_energy}")
    print(f"Modes retained for {ENERGY_THRESHOLD*100:.0f}% energy: {n_keep} / {len(S)}")

    # --- single-mode reconstructions ---
    # Most energetic retained mode = index 0
    # Least energetic retained mode = index n_keep-1
    X_mode0 = reconstruct_single_mode(U, S, Vt, 0)
    X_mode_last = reconstruct_single_mode(U, S, Vt, n_keep - 1)

    # --- reshape back to 2D for each field (snapshot/column) ---
    def to_fields(X_recon):
        return [X_recon[:, i].reshape((NX, NY), order='F')
                for i in range(X_recon.shape[1])]

    fields_mode0 = to_fields(X_mode0)
    fields_mode_last = to_fields(X_mode_last)

    field_labels = ['vex', 'vez']

    # --- plotting ---
    n_fields = len(FILES)
    fig, axes = plt.subplots(n_fields, 3, figsize=(14, 4 * n_fields))
    if n_fields == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        f"POD — {n_fields} snapshots | "
        f"{n_keep} modes retained ({ENERGY_THRESHOLD*100:.0f}% energy)\n"
        f"Left: most energetic mode (1st) | Right: least energetic retained mode ({n_keep}th)",
        fontsize=11, y=0.98
    )

    for i in range(n_fields):
        # original
        im0 = axes[i, 0].imshow(fields[i].T, origin='lower', aspect='auto',
                                 cmap='RdBu_r')
        axes[i, 0].set_title(f'{field_labels[i]} — original')
        plt.colorbar(im0, ax=axes[i, 0], label='value')

        # most energetic mode
        im1 = axes[i, 1].imshow(fields_mode0[i].T, origin='lower', aspect='auto',
                                 cmap='RdBu_r')
        axes[i, 1].set_title(f'{field_labels[i]} — mode 1 (most energetic)')
        plt.colorbar(im1, ax=axes[i, 1], label='value')

        # least energetic retained mode
        im2 = axes[i, 2].imshow(fields_mode_last[i].T, origin='lower', aspect='auto',
                                 cmap='RdBu_r')
        axes[i, 2].set_title(f'{field_labels[i]} — mode {n_keep} (least energetic retained)')
        plt.colorbar(im2, ax=axes[i, 2], label='value')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    outpath = os.path.join(os.path.dirname(__file__), 'pod_python_result.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {outpath}")
    plt.show()


if __name__ == '__main__':
    main()
