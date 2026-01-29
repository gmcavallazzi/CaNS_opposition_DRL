#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import sys

# Load mean field
mean_field = np.load(sys.argv[1] if len(sys.argv) > 1 else 'pod_clean_results/mean_field.npy')

print(f"Mean field shape: {mean_field.shape}")
print(f"Mean: {mean_field.mean():.6e}")
print(f"Std: {mean_field.std():.6e}")
print(f"Min: {mean_field.min():.6e}")
print(f"Max: {mean_field.max():.6e}")

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
vmax = max(abs(mean_field.min()), abs(mean_field.max()))
im = ax.imshow(mean_field.T, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
               origin='lower', aspect='auto')
ax.set_title(f'Time-Averaged Mean Field\nstd={mean_field.std():.6e}')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig('mean_field_plot.png', dpi=150)
print("\nSaved: mean_field_plot.png")
plt.show()
