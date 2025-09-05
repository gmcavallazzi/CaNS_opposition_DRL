import numpy as np
import matplotlib.pyplot as plt

# Load the saved actions
actions = np.load('deterministic_run_20250228_103101/actions.npy')
print(f"Actions mean = {np.mean(actions)}")
print(f"Actions std = {np.std(actions)}")
print(f"Actions max = {np.max(actions)}")
print(f"Actions min = {np.min(actions)}")

# Check the shape
print(f"Actions array shape: {actions.shape}")

# If it's a 3D array (steps, agents_i, agents_j)
if len(actions.shape) > 2:
    # Plot heatmap of actions at a specific timestep
    plt.figure(figsize=(10, 8))
    plt.imshow(actions[1000], cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Action value')
    plt.title('Action values at timestep 1000')
    plt.savefig('action_heatmap.png')
    
    # Plot time evolution of actions for a specific position
    center_i, center_j = actions.shape[1]//2, actions.shape[2]//2
    plt.figure(figsize=(10, 6))
    plt.plot(actions[:, center_i, center_j])
    plt.title(f'Action values over time at position ({center_i},{center_j})')
    plt.xlabel('Timestep')
    plt.ylabel('Action value')
    plt.grid(True)
    plt.savefig('action_time_series.png')