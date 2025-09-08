import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
import yaml
import argparse
from models_pettingzoo import SharedPolicyMADDPG
from utils import load_config

def plot_actor_response_map(maddpg, config, resolution=50, save_path=None):
    """
    Create a 2D map showing the shared policy actor's response to different u and w values.
    
    Args:
        maddpg: The trained SharedPolicyMADDPG model
        config: Configuration dictionary
        resolution: Resolution of the plot (number of points along each axis)
        save_path: Path to save the figure (optional)
    """
    # Set the model to evaluation mode
    maddpg.actor.eval()
    device = next(maddpg.actor.parameters()).device
    
    # Get grid dimensions from config
    grid_i = config['grid']['target']['i']
    grid_j = config['grid']['target']['j']
    
    # Create a grid of u and w values to test
    u_values = np.linspace(-1.0, 1.0, resolution)
    w_values = np.linspace(-1.0, 1.0, resolution)
    
    # Create a meshgrid for the plot
    U, W = np.meshgrid(u_values, w_values)
    
    # Storage for actor outputs
    responses = np.zeros((resolution, resolution))
    
    # Determine observation shape based on halo value
    halo = config.get('halo', 0)
    obs_height = 2 * halo + 1 if halo > 0 else 1
    obs_width = 2 * halo + 1 if halo > 0 else 1
    
    # Get the full list of agents
    num_agents = grid_i * grid_j
    
    # Loop through all combinations of u and w values
    for i, u_val in enumerate(u_values):
        for j, w_val in enumerate(w_values):
            # Create a batch of identical observations for a single agent
            # Set both channels (u and w) to the test values
            if halo == 0:
                # Single point observation
                obs = np.zeros((1, 1, 1, 2), dtype=np.float32)
                obs[0, 0, 0, 0] = u_val * config['action']['om_max']  # Scale u by om_max
                obs[0, 0, 0, 1] = w_val * config['action']['om_max']  # Scale w by om_max
            else:
                # Observation with halo
                obs = np.zeros((1, obs_height, obs_width, 2), dtype=np.float32)
                
                # Set the center point to the test values
                # Fill the entire observation with the same value for simplicity
                obs[0, :, :, 0] = u_val * config['action']['om_max']  # Scale u by om_max
                obs[0, :, :, 1] = w_val * config['action']['om_max']  # Scale w by om_max
            
            # Flatten observation for the actor network
            flat_obs = obs.reshape(1, -1)
            
            # Convert to torch tensor
            obs_tensor = torch.FloatTensor(flat_obs).to(device)
            
            # Get the actor's response
            with torch.no_grad():
                action = maddpg.actor(obs_tensor).cpu().numpy()
            
            # Store the response
            responses[j, i] = action[0, 0]  # First element of the action
    
    # Create a figure for the 2D map
    plt.figure(figsize=(10, 8))
    
    # Use a diverging colormap centered at zero
    resp_min = responses.min()
    resp_max = responses.max()
    
    # Handle cases where min and max might be equal or not in ascending order
    if resp_min == resp_max:
        # If all responses are the same, create a small range around that value
        resp_min = resp_min - 0.01
        resp_max = resp_max + 0.01
        
    # Ensure values are in ascending order
    if resp_min < 0 and resp_max < 0:
        # All values are negative, vcenter should be between them
        vcenter = (resp_min + resp_max) / 2
    elif resp_min > 0 and resp_max > 0:
        # All values are positive, vcenter should be between them
        vcenter = (resp_min + resp_max) / 2
    else:
        # We have both positive and negative values, center at zero
        vcenter = 0
        
    # Ensure vmin < vcenter < vmax
    if vcenter <= resp_min:
        resp_min = vcenter - 0.01
    if vcenter >= resp_max:
        resp_max = vcenter + 0.01
        
    divnorm = TwoSlopeNorm(vmin=resp_min, vcenter=vcenter, vmax=resp_max)
    
    # Plot the 2D heatmap
    cmap = plt.cm.RdBu_r  # Red-Blue diverging colormap
    contour = plt.contourf(U, W, responses, levels=50, cmap=cmap, norm=divnorm)
    
    # Add color bar
    cbar = plt.colorbar(contour)
    cbar.set_label('Actor Response')
    
    # Add contour lines for clarity
    contour_lines = plt.contour(U, W, responses, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Add labels and title
    plt.xlabel('u (Normalized Streamwise Velocity)')
    plt.ylabel('w (Normalized Spanwise Velocity)')
    plt.title('Shared Policy Actor Network Response Map')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Show or save
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return responses

def load_and_visualize_shared_policy(checkpoint_path, config_path="config.yaml", save_dir="./visualizations"):
    """
    Load a trained MADDPG shared policy model and visualize its response map.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        config_path: Path to the configuration file
        save_dir: Directory to save visualizations
    """
    # Create directory for visualizations
    os.makedirs(save_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(config_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get grid dimensions
    grid_i = config['grid']['target']['i']
    grid_j = config['grid']['target']['j']
    
    # Generate the list of agents
    agents = [f"agent_{i}_{j}" for i in range(grid_i) for j in range(grid_j)]
    
    # Determine observation shape based on halo value
    halo = config.get('halo', 0)
    obs_height = 2 * halo + 1 if halo > 0 else 1
    obs_width = 2 * halo + 1 if halo > 0 else 1
    obs_shape = (obs_height, obs_width, 2)
    
    # Action shape is always (1,) for your framework
    act_shape = (1,)
    
    # Create MADDPG model
    maddpg = SharedPolicyMADDPG(
        agents=agents,
        obs_shape=obs_shape,
        act_shape=act_shape,
        device='cpu',
        pi_arch=config.get('net_arch', {}).get('pi', [64, 64]),
        qf_arch=config.get('net_arch', {}).get('qf', [64, 64, 64])
    )
    
    # Load weights
    if 'maddpg_state_dict' in checkpoint:
        maddpg.load_state_dict(checkpoint['maddpg_state_dict'])
    else:
        print("Warning: 'maddpg_state_dict' not found in checkpoint. Attempting to load state_dict directly.")
        maddpg.load_state_dict(checkpoint)
    
    # Extract checkpoint filename for output naming
    checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
    
    # Get timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate output path
    save_path = os.path.join(save_dir, f"{checkpoint_name}_response_map_{timestamp}.png")
    
    # Visualize the response map
    print(f"Generating response map for {checkpoint_path}...")
    print(f"Using configuration from {config_path}")
    print(f"Grid dimensions: {grid_i}x{grid_j}, Halo: {halo}")
    
    responses = plot_actor_response_map(maddpg, config, resolution=50, save_path=save_path)
    
    print(f"Response map saved to {save_path}")
    
    return responses

def plot_multi_checkpoint_comparison(checkpoint_paths, config_path="config.yaml", save_dir="./visualizations"):
    """
    Plot a comparison of multiple checkpoints to see how the policy evolves during training.
    
    Args:
        checkpoint_paths: List of paths to checkpoint files
        config_path: Path to the configuration file
        save_dir: Directory to save visualizations
    """
    # Create directory for visualizations
    os.makedirs(save_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(config_path)
    
    # Get timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a figure with subplots
    n_checkpoints = len(checkpoint_paths)
    n_cols = min(3, n_checkpoints)  # Max 3 columns
    n_rows = (n_checkpoints + n_cols - 1) // n_cols  # Calculate needed rows
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])  # Make it iterable
    axes = axes.flatten()
    
    # Generate a common colormap scale for all plots
    vmin, vmax, vcenter = None, None, 0
    responses_list = []
    
    # First pass: load all checkpoints and compute global min/max
    for checkpoint_path in checkpoint_paths:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get grid dimensions
        grid_i = config['grid']['target']['i']
        grid_j = config['grid']['target']['j']
        
        # Generate the list of agents
        agents = [f"agent_{i}_{j}" for i in range(grid_i) for j in range(grid_j)]
        
        # Determine observation shape based on halo value
        halo = config.get('halo', 0)
        obs_height = 2 * halo + 1 if halo > 0 else 1
        obs_width = 2 * halo + 1 if halo > 0 else 1
        obs_shape = (obs_height, obs_width, 2)
        
        # Action shape is always (1,) for your framework
        act_shape = (1,)
        
        # Create MADDPG model
        maddpg = SharedPolicyMADDPG(
            agents=agents,
            obs_shape=obs_shape,
            act_shape=act_shape,
            device='cpu',
            pi_arch=config.get('net_arch', {}).get('pi', [64, 64]),
            qf_arch=config.get('net_arch', {}).get('qf', [64, 64, 64])
        )
        
        # Load weights
        if 'maddpg_state_dict' in checkpoint:
            maddpg.load_state_dict(checkpoint['maddpg_state_dict'])
        else:
            print("Warning: 'maddpg_state_dict' not found in checkpoint. Attempting to load state_dict directly.")
            maddpg.load_state_dict(checkpoint)
        
        # Set up grid of u and w values
        resolution = 30  # Lower resolution for faster plotting
        u_values = np.linspace(-1.0, 1.0, resolution)
        w_values = np.linspace(-1.0, 1.0, resolution)
        
        # Get actor responses
        responses = np.zeros((resolution, resolution))
        
        # Determine observation shape
        if halo == 0:
            # Single point observation
            obs = np.zeros((1, 1, 1, 2), dtype=np.float32)
        else:
            # Observation with halo
            obs = np.zeros((1, obs_height, obs_width, 2), dtype=np.float32)
        
        # Loop through all combinations of u and w values
        for i, u_val in enumerate(u_values):
            for j, w_val in enumerate(w_values):
                # Set the center point to the test values
                # Fill the entire observation with the same value for simplicity
                if halo == 0:
                    obs[0, 0, 0, 0] = u_val * config['action']['om_max']
                    obs[0, 0, 0, 1] = w_val * config['action']['om_max']
                else:
                    obs[0, :, :, 0] = u_val * config['action']['om_max']
                    obs[0, :, :, 1] = w_val * config['action']['om_max']
                
                # Flatten observation for the actor network
                flat_obs = obs.reshape(1, -1)
                
                # Convert to torch tensor
                obs_tensor = torch.FloatTensor(flat_obs).to('cpu')
                
                # Get the actor's response
                with torch.no_grad():
                    action = maddpg.actor(obs_tensor).cpu().numpy()
                
                # Store the response
                responses[j, i] = action[0, 0]  # First element of the action
        
        responses_list.append(responses)
        
        # Update global min/max
        if vmin is None or np.min(responses) < vmin:
            vmin = np.min(responses)
        if vmax is None or np.max(responses) > vmax:
            vmax = np.max(responses)
    
    # Ensure vmin and vmax make sense
    if vmin >= vcenter:
        vmin = vcenter - 0.01
    if vmax <= vcenter:
        vmax = vcenter + 0.01
    
    divnorm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    
    # Second pass: create the plots with common scale
    for idx, (checkpoint_path, responses) in enumerate(zip(checkpoint_paths, responses_list)):
        # Extract step or episode number from filename
        checkpoint_name = os.path.basename(checkpoint_path)
        if "step_" in checkpoint_name:
            step_num = checkpoint_name.split("step_")[1].split(".")[0]
            title_text = f"Step {step_num}"
        elif "episode_" in checkpoint_name:
            episode_num = checkpoint_name.split("episode_")[1].split(".")[0]
            title_text = f"Episode {episode_num}"
        else:
            title_text = checkpoint_name.replace('.pt', '')
        
        # Get the axis for this plot
        ax = axes[idx]
        
        # Create a meshgrid for the plot
        resolution = responses.shape[0]
        u_values = np.linspace(-1.0, 1.0, resolution)
        w_values = np.linspace(-1.0, 1.0, resolution)
        U, W = np.meshgrid(u_values, w_values)
        
        # Plot the 2D heatmap
        cmap = plt.cm.RdBu_r  # Red-Blue diverging colormap
        contour = ax.contourf(U, W, responses, levels=50, cmap=cmap, norm=divnorm)
        
        # Add contour lines
        contour_lines = ax.contour(U, W, responses, levels=5, colors='k', linewidths=0.5, alpha=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
        
        # Add title
        ax.set_title(title_text)
        
        # Add labels on the edges
        if idx % n_cols == 0:  # Left edge
            ax.set_ylabel('w')
        if idx >= (n_rows - 1) * n_cols or idx == len(checkpoint_paths) - 1:  # Bottom edge
            ax.set_xlabel('u')
    
    # Hide empty subplots
    for idx in range(len(checkpoint_paths), len(axes)):
        axes[idx].set_visible(False)
    
    # Add a colorbar to the figure
    fig.colorbar(contour, ax=axes, shrink=0.8, label='Actor Response')
    
    # Add an overall title
    fig.suptitle('Policy Evolution During Training', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Save figure
    save_path = os.path.join(save_dir, f"policy_evolution_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"Policy evolution comparison saved to {save_path}")
    
    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize MADDPG shared policy response')
    parser.add_argument('--checkpoint', type=str, 
                      help='Path to single checkpoint file')
    parser.add_argument('--checkpoint_dir', type=str,
                      help='Directory containing multiple checkpoints for comparison')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default='./visualizations',
                      help='Directory to save visualizations')
    parser.add_argument('--mode', type=str, choices=['single', 'multi', 'best'], default='single',
                      help='Visualization mode: single checkpoint, multiple checkpoints, or best checkpoint')
    
    args = parser.parse_args()
    
    # Validate args
    if args.mode == 'single' and not args.checkpoint:
        parser.error("--checkpoint is required when mode is 'single'")
    if args.mode == 'multi' and not args.checkpoint_dir:
        parser.error("--checkpoint_dir is required when mode is 'multi'")
    
    # Mode-specific behavior
    if args.mode == 'single':
        # Visualize a single checkpoint
        load_and_visualize_shared_policy(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            save_dir=args.save_dir
        )
    elif args.mode == 'multi':
        # Find all checkpoints in the directory
        checkpoint_files = [os.path.join(args.checkpoint_dir, f) for f in os.listdir(args.checkpoint_dir) 
                           if f.startswith('checkpoint_step_') and f.endswith('.pt')]
        
        # Sort by step number
        checkpoint_files.sort(key=lambda x: int(os.path.basename(x).split('step_')[1].split('.')[0]))
        
        # Select a subset of checkpoints for visualization (e.g., every n-th checkpoint)
        n_checkpoints = min(9, len(checkpoint_files))  # Max 9 for a 3x3 grid
        if len(checkpoint_files) > n_checkpoints:
            indices = np.linspace(0, len(checkpoint_files)-1, n_checkpoints, dtype=int)
            checkpoint_files = [checkpoint_files[i] for i in indices]
        
        # Plot the comparison
        plot_multi_checkpoint_comparison(
            checkpoint_paths=checkpoint_files,
            config_path=args.config,
            save_dir=args.save_dir
        )
    elif args.mode == 'best':
        # Use the best model checkpoint
        best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
        
        if os.path.exists(best_model_path):
            load_and_visualize_shared_policy(
                checkpoint_path=best_model_path,
                config_path=args.config,
                save_dir=args.save_dir
            )
        else:
            print(f"Best model checkpoint not found at {best_model_path}")