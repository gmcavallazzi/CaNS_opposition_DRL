import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from matplotlib.colors import Normalize

def analyze_buffer_observations(buffer_file, output_dir=None, sample_size=1000):
    """
    Analyze observation statistics from a replay buffer
    
    Args:
        buffer_file: Path to the buffer npz file
        output_dir: Directory to save analysis results
        sample_size: Number of observations to sample for analysis
    """
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.dirname(buffer_file)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading buffer from {buffer_file}...")
    buffer_data = np.load(buffer_file, allow_pickle=True)
    
    # Extract observations
    if 'obs' in buffer_data:
        observations = buffer_data['obs']
        buffer_size = len(observations)
        print(f"Buffer contains {buffer_size} observations with shape {observations.shape}")
        
        # If buffer is large, sample a subset
        if buffer_size > sample_size:
            indices = np.random.choice(buffer_size, sample_size, replace=False)
            sample_obs = observations[indices]
        else:
            sample_obs = observations
            
        # Determine observation type (flat or spatial)
        obs_shape = sample_obs.shape[1:]
        is_spatial = len(obs_shape) >= 2
        
        # Calculate basic statistics
        obs_mean = np.mean(sample_obs, axis=0)
        obs_std = np.std(sample_obs, axis=0)
        obs_min = np.min(sample_obs, axis=0)
        obs_max = np.max(sample_obs, axis=0)
        
        # Print overall statistics
        print("\nObservation Statistics:")
        print(f"Mean values range: [{np.min(obs_mean):.4f}, {np.max(obs_mean):.4f}]")
        print(f"Std dev values range: [{np.min(obs_std):.4f}, {np.max(obs_std):.4f}]")
        print(f"Min value: {np.min(obs_min):.4f}")
        print(f"Max value: {np.max(obs_max):.4f}")
        
        # Save statistics to file
        with open(os.path.join(output_dir, 'observation_stats.txt'), 'w') as f:
            f.write(f"Buffer file: {buffer_file}\n")
            f.write(f"Total observations: {buffer_size}\n")
            f.write(f"Observation shape: {obs_shape}\n\n")
            f.write("Overall Statistics:\n")
            f.write(f"Mean values range: [{np.min(obs_mean):.4f}, {np.max(obs_mean):.4f}]\n")
            f.write(f"Std dev values range: [{np.min(obs_std):.4f}, {np.max(obs_std):.4f}]\n")
            f.write(f"Min value: {np.min(obs_min):.4f}\n")
            f.write(f"Max value: {np.max(obs_max):.4f}\n")
        
        # Create visualizations
        if is_spatial:
            # Spatial observations (grid-based or strip-based)
            
            # Check if we have channels (depth dimension)
            if len(obs_shape) == 3:
                num_channels = obs_shape[2]
                
                # Create visualization for each channel
                for c in range(num_channels):
                    # Plot mean for this channel
                    plt.figure(figsize=(10, 8))
                    im = plt.imshow(obs_mean[:,:,c], cmap='viridis')
                    plt.colorbar(im, label=f'Mean Value (Channel {c})')
                    plt.title(f'Mean Observation Values - Channel {c}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'mean_obs_channel_{c}.png'), dpi=300)
                    plt.close()
                    
                    # Plot standard deviation for this channel
                    plt.figure(figsize=(10, 8))
                    im = plt.imshow(obs_std[:,:,c], cmap='plasma')
                    plt.colorbar(im, label=f'Std Dev (Channel {c})')
                    plt.title(f'Observation Standard Deviation - Channel {c}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'std_obs_channel_{c}.png'), dpi=300)
                    plt.close()
                
                # Sample and plot a few random observations
                num_samples = min(5, sample_obs.shape[0])
                for i in range(num_samples):
                    fig, axes = plt.subplots(1, num_channels, figsize=(5*num_channels, 4))
                    if num_channels == 1:
                        axes = [axes]  # Make iterable for single channel
                        
                    for c in range(num_channels):
                        im = axes[c].imshow(sample_obs[i,:,:,c], cmap='viridis')
                        plt.colorbar(im, ax=axes[c], label=f'Value')
                        axes[c].set_title(f'Channel {c}')
                    
                    plt.suptitle(f'Sample Observation {i+1}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'sample_obs_{i+1}.png'), dpi=300)
                    plt.close()
            else:
                # Single channel spatial observation
                plt.figure(figsize=(10, 8))
                im = plt.imshow(obs_mean, cmap='viridis')
                plt.colorbar(im, label='Mean Value')
                plt.title('Mean Observation Values')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'mean_obs.png'), dpi=300)
                plt.close()
                
                plt.figure(figsize=(10, 8))
                im = plt.imshow(obs_std, cmap='plasma')
                plt.colorbar(im, label='Std Dev')
                plt.title('Observation Standard Deviation')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'std_obs.png'), dpi=300)
                plt.close()
        else:
            # Non-spatial (flat) observations
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(obs_mean)), obs_mean)
            plt.xlabel('Observation Dimension')
            plt.ylabel('Mean Value')
            plt.title('Mean Observation Values')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'mean_obs.png'), dpi=300)
            plt.close()
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(obs_std)), obs_std)
            plt.xlabel('Observation Dimension')
            plt.ylabel('Standard Deviation')
            plt.title('Observation Standard Deviation')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'std_obs.png'), dpi=300)
            plt.close()
        
        # Create histogram of all observation values
        plt.figure(figsize=(10, 6))
        plt.hist(sample_obs.flatten(), bins=50, alpha=0.7, color='blue')
        plt.xlabel('Observation Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Observation Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'obs_histogram.png'), dpi=300)
        plt.close()
        
        # Check for inactive features (always the same value or near-zero variance)
        flat_std = obs_std.flatten()
        inactive_features = np.where(flat_std < 0.01)[0]
        if len(inactive_features) > 0:
            print(f"\nFound {len(inactive_features)} potentially inactive features (std dev < 0.01)")
            with open(os.path.join(output_dir, 'inactive_features.txt'), 'w') as f:
                f.write(f"Found {len(inactive_features)} potentially inactive features (std dev < 0.01)\n")
                for idx in inactive_features:
                    f.write(f"Feature index {idx}: mean={flat_std[idx]:.6f}, std={flat_std[idx]:.6f}\n")
        
        # Look at rewards and actions if available
        if 'rews' in buffer_data:
            rewards = buffer_data['rews']
            print(f"\nReward Statistics:")
            print(f"Mean reward: {np.mean(rewards):.4f}")
            print(f"Min reward: {np.min(rewards):.4f}")
            print(f"Max reward: {np.max(rewards):.4f}")
            
            plt.figure(figsize=(10, 6))
            plt.hist(rewards, bins=50, alpha=0.7, color='green')
            plt.xlabel('Reward Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of Rewards')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'reward_histogram.png'), dpi=300)
            plt.close()
        
        if 'acts' in buffer_data:
            actions = buffer_data['acts']
            print(f"\nAction Statistics:")
            print(f"Mean action: {np.mean(actions):.4f}")
            print(f"Min action: {np.min(actions):.4f}")
            print(f"Max action: {np.max(actions):.4f}")
            
            plt.figure(figsize=(10, 6))
            plt.hist(actions.flatten(), bins=50, alpha=0.7, color='red')
            plt.xlabel('Action Value')
            plt.ylabel('Frequency')
            plt.title('Distribution of Actions')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'action_histogram.png'), dpi=300)
            plt.close()
        
        print(f"\nAnalysis complete. Results saved to {output_dir}")
    else:
        print("Error: Could not find observations in buffer file")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze replay buffer observations')
    parser.add_argument('buffer_file', type=str, help='Path to buffer.npz file')
    parser.add_argument('--output', type=str, default=None, help='Output directory for analysis')
    parser.add_argument('--samples', type=int, default=1000, help='Number of observations to sample')
    
    args = parser.parse_args()
    analyze_buffer_observations(args.buffer_file, args.output, args.samples)