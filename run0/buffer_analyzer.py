import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob
import re

def extract_step_number(filename):
    """Extract step number from buffer filename (buffer_step_XXXXX.npz)"""
    match = re.search(r'buffer_step_(\d+)', filename)
    if match:
        return int(match.group(1))
    return 0

def load_buffer(buffer_path):
    """Load a saved replay buffer"""
    print(f"Loading buffer from: {buffer_path}")
    try:
        data = np.load(buffer_path, allow_pickle=True)
        return data
    except Exception as e:
        print(f"Error loading buffer: {e}")
        return None

def analyze_buffer(data, obs_type="grid", output_dir="buffer_analysis"):
    """Analyze buffer data and generate plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from buffer
    obs = data['obs']
    next_obs = data['next_obs'] 
    acts = data['acts']
    rews = data['rews']
    done = data['done']
    
    # Extract step number from buffer filename
    step_num = extract_step_number(args.buffer_path)
    
    print(f"Buffer contains {len(obs)} transitions")
    print(f"Observation shape: {obs.shape}")
    print(f"Action shape: {acts.shape}")
    
    # 1. Action histogram
    plt.figure(figsize=(10, 6))
    plt.hist(acts.flatten(), bins=50, alpha=0.75)
    plt.title(f'Action Distribution (Step {step_num})')
    plt.xlabel('Action Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'action_histogram_step_{step_num}.png'))
    plt.close()
    
    # 2. Reward distribution
    plt.figure(figsize=(10, 6))
    plt.hist(rews, bins=50, alpha=0.75)
    plt.title(f'Reward Distribution (Step {step_num})')
    plt.xlabel('Reward Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'reward_histogram_step_{step_num}.png'))
    plt.close()
    
    # 3. Extract velocity information from observations
    if obs_type == "grid":
        # For grid type, extract the central value (u component) of each observation
        # Assuming obs shape is (N, height, width, channels) and u is channel 0
        velocities = []
        for observation in obs:
            if len(observation.shape) == 3:  # (height, width, channels)
                center_h = observation.shape[0] // 2
                center_w = observation.shape[1] // 2
                velocity = observation[center_h, center_w, 0]  # u component
                velocities.append(velocity)
        velocities = np.array(velocities)
    elif obs_type == "strip":
        # For strip type, use average of u component across the strip
        # Assuming obs shape is (N, height, width, channels) and u is channel 0
        velocities = np.mean(obs[:, :, :, 0], axis=(1, 2))
    
    # 4. Velocity-Action relationship
    plt.figure(figsize=(10, 6))
    plt.scatter(velocities, acts.flatten(), alpha=0.1, s=5)
    
    # Add a trend line
    if len(velocities) > 1:
        z = np.polyfit(velocities, acts.flatten(), 1)
        p = np.poly1d(z)
        plt.plot(np.sort(velocities), p(np.sort(velocities)), "r--", 
                lw=2, label=f"Trend: y={z[0]:.4f}x+{z[1]:.4f}")
        
        # Calculate correlation coefficient
        corr_coef = np.corrcoef(velocities, acts.flatten())[0, 1]
        plt.title(f'Velocity vs Action (Step {step_num}, Correlation: {corr_coef:.4f})')
        
        # Highlight the trend - add an inverse relationship reference line
        vel_min, vel_max = velocities.min(), velocities.max()
        x_ref = np.linspace(vel_min, vel_max, 100)
        # Normalize velocity to [0,1] then map to [-1,1] with negative slope
        y_ref = -2.0 * ((x_ref - vel_min) / (vel_max - vel_min)) + 1.0
        plt.plot(x_ref, y_ref, "g--", lw=2, label="Ideal Inverse Relationship")
    else:
        plt.title(f'Velocity vs Action (Step {step_num})')
    
    plt.xlabel('Velocity (u component)')
    plt.ylabel('Action Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'velocity_action_relationship_step_{step_num}.png'))
    plt.close()
    
    # 5. Reward vs Action magnitude
    plt.figure(figsize=(10, 6))
    plt.scatter(np.abs(acts.flatten()), rews, alpha=0.1, s=5)
    plt.title(f'Action Magnitude vs Reward (Step {step_num})')
    plt.xlabel('|Action|')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'action_reward_relationship_step_{step_num}.png'))
    plt.close()
    
    # 6. Velocity distribution
    plt.figure(figsize=(10, 6))
    plt.hist(velocities, bins=50, alpha=0.75)
    plt.title(f'Velocity Distribution (Step {step_num})')
    plt.xlabel('Velocity Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'velocity_histogram_step_{step_num}.png'))
    plt.close()
    
    # 7. 2D histogram of velocity vs action
    plt.figure(figsize=(10, 8))
    
    # Create a custom colormap from white to blue
    cmap = LinearSegmentedColormap.from_list('WhiteToBlue', ['white', 'blue'])
    
    plt.hist2d(velocities, acts.flatten(), bins=50, cmap=cmap)
    plt.colorbar(label='Count')
    plt.title(f'Velocity vs Action Density (Step {step_num})')
    plt.xlabel('Velocity (u component)')
    plt.ylabel('Action Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'velocity_action_density_step_{step_num}.png'))
    plt.close()
    
    # 8. Summary statistics
    with open(os.path.join(output_dir, f'buffer_stats_step_{step_num}.txt'), 'w') as f:
        f.write(f"Buffer Analysis at Step {step_num}\n")
        f.write(f"================================\n\n")
        f.write(f"Transitions: {len(obs)}\n\n")
        
        f.write(f"Actions Statistics:\n")
        f.write(f"  Mean: {np.mean(acts):.4f}\n")
        f.write(f"  Std Dev: {np.std(acts):.4f}\n")
        f.write(f"  Min: {np.min(acts):.4f}\n")
        f.write(f"  Max: {np.max(acts):.4f}\n\n")
        
        f.write(f"Rewards Statistics:\n")
        f.write(f"  Mean: {np.mean(rews):.4f}\n")
        f.write(f"  Std Dev: {np.std(rews):.4f}\n")
        f.write(f"  Min: {np.min(rews):.4f}\n")
        f.write(f"  Max: {np.max(rews):.4f}\n\n")
        
        if len(velocities) > 0:
            f.write(f"Velocity Statistics:\n")
            f.write(f"  Mean: {np.mean(velocities):.4f}\n")
            f.write(f"  Std Dev: {np.std(velocities):.4f}\n")
            f.write(f"  Min: {np.min(velocities):.4f}\n")
            f.write(f"  Max: {np.max(velocities):.4f}\n\n")
            
            f.write(f"Velocity-Action Correlation: {corr_coef:.4f}\n")
            f.write(f"Velocity-Action Linear Fit: y = {z[0]:.4f}x + {z[1]:.4f}\n\n")
        
        f.write(f"Analysis completed at: {os.path.dirname(os.path.abspath(__file__))}\n")
    
    print(f"Analysis complete! Results saved to {output_dir}")

def analyze_buffer_progression(buffer_dir, pattern="buffer_step_*.npz", obs_type="grid", limit=None):
    """Analyze the progression of actions and rewards across multiple buffers"""
    # Find all buffer files matching the pattern
    buffer_files = glob.glob(os.path.join(buffer_dir, pattern))
    buffer_files.sort(key=extract_step_number)
    
    if limit and len(buffer_files) > limit:
        # Select buffers at regular intervals
        indices = np.linspace(0, len(buffer_files)-1, limit, dtype=int)
        buffer_files = [buffer_files[i] for i in indices]
    
    print(f"Found {len(buffer_files)} buffer files")
    
    # Create directory for progression analysis
    output_dir = "buffer_progression"
    os.makedirs(output_dir, exist_ok=True)
    
    # Track statistics across buffers
    steps = []
    action_means = []
    action_stds = []
    reward_means = []
    velocity_means = []
    correlations = []
    
    # Process each buffer
    for buffer_file in buffer_files:
        step = extract_step_number(buffer_file)
        print(f"Processing buffer at step {step}")
        steps.append(step)
        
        # Load buffer
        data = load_buffer(buffer_file)
        if data is None:
            continue
        
        # Extract data
        obs = data['obs']
        acts = data['acts']
        rews = data['rews']
        
        # Track action statistics
        action_means.append(np.mean(acts))
        action_stds.append(np.std(acts))
        
        # Track reward statistics
        reward_means.append(np.mean(rews))
        
        # Extract velocity information from observations
        if obs_type == "grid":
            # For grid type, extract the central value (u component) of each observation
            velocities = []
            for observation in obs:
                if len(observation.shape) == 3:  # (height, width, channels)
                    center_h = observation.shape[0] // 2
                    center_w = observation.shape[1] // 2
                    velocity = observation[center_h, center_w, 0]  # u component
                    velocities.append(velocity)
            velocities = np.array(velocities)
        elif obs_type == "strip":
            # For strip type, use average of u component across the strip
            velocities = np.mean(obs[:, :, :, 0], axis=(1, 2))
        
        velocity_means.append(np.mean(velocities))
        
        # Calculate correlation between velocity and action
        if len(velocities) > 1:
            corr_coef = np.corrcoef(velocities, acts.flatten())[0, 1]
            correlations.append(corr_coef)
        else:
            correlations.append(0)
    
    # Plot the progression of statistics
    
    # 1. Action mean and std
    plt.figure(figsize=(12, 6))
    plt.errorbar(steps, action_means, yerr=action_stds, fmt='o-', capsize=5)
    plt.title('Action Mean and Standard Deviation Over Time')
    plt.xlabel('Training Step')
    plt.ylabel('Action Value')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'action_statistics_progression.png'))
    plt.close()
    
    # 2. Reward mean
    plt.figure(figsize=(12, 6))
    plt.plot(steps, reward_means, 'o-')
    plt.title('Average Reward Over Time')
    plt.xlabel('Training Step')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'reward_progression.png'))
    plt.close()
    
    # 3. Velocity-Action correlation
    plt.figure(figsize=(12, 6))
    plt.plot(steps, correlations, 'o-')
    plt.title('Velocity-Action Correlation Over Time')
    plt.xlabel('Training Step')
    plt.ylabel('Correlation Coefficient')
    # Add horizontal line at -1 (perfect inverse correlation)
    plt.axhline(y=-1, color='r', linestyle='--', label='Perfect Inverse Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'correlation_progression.png'))
    plt.close()
    
    # 4. Velocity mean
    plt.figure(figsize=(12, 6))
    plt.plot(steps, velocity_means, 'o-')
    plt.title('Average Velocity Over Time')
    plt.xlabel('Training Step')
    plt.ylabel('Velocity')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'velocity_progression.png'))
    plt.close()
    
    print(f"Progression analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze MADDPG replay buffer data')
    parser.add_argument('--buffer_path', type=str, help='Path to the buffer file (.npz)')
    parser.add_argument('--buffer_dir', type=str, help='Directory containing multiple buffer files')
    parser.add_argument('--obs_type', type=str, default='grid', choices=['grid', 'strip'], 
                        help='Type of observation (grid or strip)')
    parser.add_argument('--output_dir', type=str, default='buffer_analysis', 
                        help='Directory to save analysis results')
    parser.add_argument('--progression', action='store_true', 
                        help='Analyze progression across multiple buffers')
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit number of buffers to analyze in progression mode')
    
    args = parser.parse_args()
    
    if args.progression and args.buffer_dir:
        analyze_buffer_progression(args.buffer_dir, obs_type=args.obs_type, limit=args.limit)
    elif args.buffer_path:
        data = load_buffer(args.buffer_path)
        if data is not None:
            analyze_buffer(data, obs_type=args.obs_type, output_dir=args.output_dir)
    else:
        parser.print_help()
        print("\nError: You must provide either --buffer_path or --buffer_dir with --progression")