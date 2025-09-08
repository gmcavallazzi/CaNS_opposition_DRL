import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from matplotlib.colors import TwoSlopeNorm
from models_pettingzoo import SharedPolicyMADDPG, MLPActor, MLPCritic
from utils import load_config
from matplotlib import rcParams

# Set up matplotlib for better visualizations
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})

# Define color scheme - teal and orange as primary colors
TEAL = '#008080'
ORANGE = '#FF8C00'
NAVY = '#000080'
CORAL = '#FF6F61'
FOREST = '#228B22'
COLORS = [TEAL, ORANGE, NAVY, CORAL, FOREST]

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
            flat_obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
            
            # Get the actor's response
            with torch.no_grad():
                action = maddpg.actor(flat_obs).cpu().numpy()
            
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
    cbar.set_label('Actor Response', fontsize=12)
    
    # Add contour lines for clarity
    contour_lines = plt.contour(U, W, responses, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    plt.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
    
    # Add labels and title
    plt.xlabel('u (Normalized Streamwise Velocity)', fontsize=12)
    plt.ylabel('w (Normalized Spanwise Velocity)', fontsize=12)
    plt.title('Shared Policy Actor Network Response Map', fontsize=14)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Show or save
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    return responses

def visualize_model_architecture(model, save_dir):
    """Visualize the model architecture with enhanced styling."""
    actor = model.actor
    critic = model.critic
    
    # Create a figure for architecture visualization
    plt.figure(figsize=(12, 10))
    
    # Get layer dimensions for actor
    actor_layers = []
    
    # Check if actor is MLP type
    if isinstance(actor, MLPActor):
        # Input layer
        actor_layers.append(("Input", actor.obs_dim, 0))
        
        # Get hidden layers by examining the network structure
        hidden_sizes = []
        for name, module in actor.named_modules():
            if isinstance(module, torch.nn.Linear):
                if hasattr(module, 'out_features'):
                    hidden_sizes.append(module.out_features)
        
        # Add hidden layers
        for i, size in enumerate(hidden_sizes[:-1]):  # Skip the output layer
            actor_layers.append((f"Hidden {i+1}", size, size))
        
        # Add output layer
        actor_layers.append(("Output", hidden_sizes[-1], actor.act_dim))
    
    # Normalize sizes for visualization
    max_size = max(size for _, size, _ in actor_layers)
    sizes = [0.3 + 0.7 * (size / max_size) for _, size, _ in actor_layers]
    
    # Draw network
    colors = plt.cm.viridis(np.linspace(0, 1, len(actor_layers)))
    for i, ((name, _, param_count), size, color) in enumerate(zip(actor_layers, sizes, colors)):
        # Layer box
        x = 0.1
        y = 0.9 - i * 0.12
        width = 0.8
        height = size * 0.1
        
        # Draw the rectangle with custom colors
        if i == 0:  # Input
            rect_color = TEAL
        elif i == len(actor_layers) - 1:  # Output
            rect_color = ORANGE
        else:  # Hidden layers
            rect_color = plt.cm.Blues(0.5 + 0.5 * i / (len(actor_layers)-2))
            
        rect = plt.Rectangle((x, y-height/2), width, height, facecolor=rect_color, edgecolor='black', alpha=0.8)
        plt.gca().add_patch(rect)
        
        # Add layer name
        plt.text(x + width/2, y, f"{name}", ha='center', va='center', fontsize=12, color='white')
        
        # Add connecting line
        if i > 0:
            plt.plot([x + width/2, x + width/2], [y + height/2 + 0.01, y - 0.12 + sizes[i-1]*0.1/2], 'k-')
    
    # Set plot parameters
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Actor MLP Architecture', fontsize=16)
    
    # Save figure
    plt.savefig(os.path.join(save_dir, "actor_architecture.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary figure with parameter counts
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    
    plt.figure(figsize=(8, 6))
    plt.bar(["Actor", "Critic", "Total"], 
            [actor_params, critic_params, actor_params + critic_params],
            color=[TEAL, ORANGE, NAVY])
    
    # Add parameter count labels
    for i, count in enumerate([actor_params, critic_params, actor_params + critic_params]):
        plt.text(i, count, f"{count:,}", ha='center', va='bottom', fontsize=12)
    
    plt.title("Model Parameter Counts", fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(save_dir, "parameter_counts.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Model architecture visualization saved to {save_dir}")

def visualize_weight_distributions(model, save_dir):
    """Visualize the distribution of weights."""
    actor = model.actor
    
    # Collect weights from different layer types
    actor_weights = []
    
    # Iterate through all named parameters
    for name, param in actor.named_parameters():
        if 'weight' in name:
            weights = param.data.cpu().numpy().flatten()
            layer_name = name.split('.')[-2] if '.' in name else name
            actor_weights.append((layer_name, weights))
    
    # Create histograms for layer weights
    plt.figure(figsize=(12, 8))
    
    for i, (name, weights) in enumerate(actor_weights):
        plt.subplot(len(actor_weights), 1, i+1)
        plt.hist(weights, bins=50, alpha=0.7, color=COLORS[i % len(COLORS)])
        plt.title(f'{name} Weight Distribution')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "weight_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a summary of weight statistics
    plt.figure(figsize=(12, 8))
    
    # Collect statistics for each layer
    layers = []
    means = []
    stds = []
    mins = []
    maxs = []
    
    for name, weights in actor_weights:
        layers.append(name)
        means.append(np.mean(weights))
        stds.append(np.std(weights))
        mins.append(np.min(weights))
        maxs.append(np.max(weights))
    
    # Create summary plots
    x = range(len(layers))
    
    # 1. Mean and std
    plt.subplot(2, 1, 1)
    plt.errorbar(x, means, yerr=stds, fmt='o-', capsize=5, label='Mean ± Std', 
                color=TEAL, ecolor=ORANGE, markersize=8, linewidth=2)
    plt.fill_between(x, np.array(means) - np.array(stds), np.array(means) + np.array(stds), 
                    alpha=0.2, color=TEAL)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xticks(x, layers, rotation=45, ha='right')
    plt.title('Weight Mean and Standard Deviation by Layer')
    plt.ylabel('Weight Value')
    plt.grid(alpha=0.3)
    plt.legend()
    
    # 2. Min and max
    plt.subplot(2, 1, 2)
    plt.plot(x, maxs, 'o-', label='Max', color=ORANGE, markersize=8, linewidth=2)
    plt.plot(x, mins, 'o-', label='Min', color=TEAL, markersize=8, linewidth=2)
    plt.fill_between(x, mins, maxs, alpha=0.2, color=NAVY)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xticks(x, layers, rotation=45, ha='right')
    plt.title('Weight Range by Layer')
    plt.ylabel('Weight Value')
    plt.grid(alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "weight_statistics.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Weight distribution visualizations saved to {save_dir}")

def plot_multi_resolution_response(maddpg, config, save_dir):
    """Create policy response maps at different resolutions"""
    resolutions = [20, 50, 100]
    fig, axes = plt.subplots(1, len(resolutions), figsize=(15, 5))
    
    for i, resolution in enumerate(resolutions):
        # Create a grid of u and w values to test
        u_values = np.linspace(-1.0, 1.0, resolution)
        w_values = np.linspace(-1.0, 1.0, resolution)
        
        # Get responses
        responses = calculate_responses(maddpg, config, u_values, w_values)
        
        # Plot on the corresponding axis
        ax = axes[i]
        U, W = np.meshgrid(u_values, w_values)
        
        # Use a diverging colormap centered at zero
        divnorm = get_diverging_norm(responses)
        
        # Plot the 2D heatmap
        cmap = plt.cm.RdBu_r  # Red-Blue diverging colormap
        contour = ax.contourf(U, W, responses, levels=30, cmap=cmap, norm=divnorm)
        
        # Add contour lines
        contour_lines = ax.contour(U, W, responses, levels=8, colors='k', linewidths=0.5, alpha=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
        
        # Add labels
        ax.set_xlabel('u velocity')
        ax.set_ylabel('w velocity')
        ax.set_title(f'Resolution: {resolution}x{resolution}')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "multi_resolution_response.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Multi-resolution response maps saved to {save_dir}")

def calculate_responses(maddpg, config, u_values, w_values):
    """Calculate actor responses for a grid of u and w values"""
    # Set the model to evaluation mode
    maddpg.actor.eval()
    device = next(maddpg.actor.parameters()).device
    
    # Determine observation shape based on halo value
    halo = config.get('halo', 0)
    obs_height = 2 * halo + 1 if halo > 0 else 1
    obs_width = 2 * halo + 1 if halo > 0 else 1
    
    # Storage for actor outputs
    resolution = len(u_values)
    responses = np.zeros((resolution, resolution))
    
    # Loop through all combinations of u and w values
    for i, u_val in enumerate(u_values):
        for j, w_val in enumerate(w_values):
            # Create observation with u and w values
            if halo == 0:
                # Single point observation
                obs = np.zeros((1, 1, 1, 2), dtype=np.float32)
                obs[0, 0, 0, 0] = u_val * config['action']['om_max']
                obs[0, 0, 0, 1] = w_val * config['action']['om_max']
            else:
                # Observation with halo
                obs = np.zeros((1, obs_height, obs_width, 2), dtype=np.float32)
                obs[0, :, :, 0] = u_val * config['action']['om_max']
                obs[0, :, :, 1] = w_val * config['action']['om_max']
            
            # Flatten observation for the actor network
            flat_obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
            
            # Get the actor's response
            with torch.no_grad():
                action = maddpg.actor(flat_obs).cpu().numpy()
            
            # Store the response
            responses[j, i] = action[0, 0]
    
    return responses

def get_diverging_norm(responses):
    """Get a TwoSlopeNorm for a diverging colormap centered appropriately"""
    resp_min = responses.min()
    resp_max = responses.max()
    
    # Handle cases where min and max might be equal or not in ascending order
    if resp_min == resp_max:
        resp_min = resp_min - 0.01
        resp_max = resp_max + 0.01
        
    # Determine center value for the colormap
    if resp_min < 0 and resp_max < 0:
        vcenter = (resp_min + resp_max) / 2
    elif resp_min > 0 and resp_max > 0:
        vcenter = (resp_min + resp_max) / 2
    else:
        vcenter = 0
        
    # Ensure vmin < vcenter < vmax
    if vcenter <= resp_min:
        resp_min = vcenter - 0.01
    if vcenter >= resp_max:
        resp_max = vcenter + 0.01
        
    return TwoSlopeNorm(vmin=resp_min, vcenter=vcenter, vmax=resp_max)

def plot_actor_activation_heatmap(maddpg, config, resolution=30, save_dir=None):
    """Create heatmap of actor network activations across different input values"""
    # Set the model to evaluation mode
    maddpg.actor.eval()
    device = next(maddpg.actor.parameters()).device
    
    # Create a grid of u and w values to test
    u_values = np.linspace(-1.0, 1.0, resolution)
    w_values = np.linspace(-1.0, 1.0, resolution)
    
    # Create a meshgrid for the plot
    U, W = np.meshgrid(u_values, w_values)
    
    # Determine observation shape based on halo value
    halo = config.get('halo', 0)
    obs_height = 2 * halo + 1 if halo > 0 else 1
    obs_width = 2 * halo + 1 if halo > 0 else 1
    
    # Different regions of parameter space
    regions = [
        ("Positive U, Positive W", (0.5, 0.5)),
        ("Negative U, Positive W", (-0.5, 0.5)),
        ("Positive U, Negative W", (0.5, -0.5)),
        ("Negative U, Negative W", (-0.5, -0.5)),
        ("Zero U, Zero W", (0.0, 0.0))
    ]
    
    # Create a figure for all regions
    plt.figure(figsize=(15, 8))
    
    for idx, (region_name, (u_val, w_val)) in enumerate(regions):
        # Create observation
        if halo == 0:
            obs = np.zeros((1, 1, 1, 2), dtype=np.float32)
            obs[0, 0, 0, 0] = u_val * config['action']['om_max']
            obs[0, 0, 0, 1] = w_val * config['action']['om_max']
        else:
            obs = np.zeros((1, obs_height, obs_width, 2), dtype=np.float32)
            obs[0, :, :, 0] = u_val * config['action']['om_max']
            obs[0, :, :, 1] = w_val * config['action']['om_max']
        
        # Flatten observation
        flat_obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        
        # Plot on a subplot
        plt.subplot(1, len(regions), idx+1)
        
        # Get actor response
        with torch.no_grad():
            action = maddpg.actor(flat_obs).cpu().numpy()
        
        # Create a title with region and response
        plt.title(f"{region_name}\nResponse: {action[0, 0]:.4f}")
        
        # Just a colored rectangle to represent the activation
        color = plt.cm.RdBu_r((action[0, 0] + 1) / 2)  # Map from [-1,1] to [0,1]
        plt.fill_between([0, 1], [0, 0], [1, 1], color=color)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "actor_activations.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Actor activation heatmap saved to {save_dir}")

def visualize_maddpg_model(checkpoint_path, config_path="config.yaml", save_dir="./model_visualizations"):
    """
    Visualize a trained MADDPG model with various analyses
    
    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to configuration file
        save_dir: Directory to save visualizations
    """
    # Create directory for visualizations
    os.makedirs(save_dir, exist_ok=True)
    
    # Load configuration
    config = load_config(config_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get grid dimensions from config
    grid_i = config['grid']['target']['i']
    grid_j = config['grid']['target']['j']
    
    # Generate the list of agents
    agents = [f"agent_{i}_{j}" for i in range(grid_i) for j in range(grid_j)]
    
    # Determine observation shape based on halo value
    halo = config.get('halo', 0)
    obs_height = 2 * halo + 1 if halo > 0 else 1
    obs_width = 2 * halo + 1 if halo > 0 else 1
    obs_shape = (obs_height, obs_width, 2)
    
    # Action shape is always (1,) for the framework
    act_shape = (1,)
    
    # Create MADDPG model
    pi_arch = config.get('net_arch', {}).get('pi', [64, 64])
    qf_arch = config.get('net_arch', {}).get('qf', [64, 64, 64])
    
    print(f"Creating model with observation shape {obs_shape}, action shape {act_shape}")
    print(f"Actor architecture: {pi_arch}, Critic architecture: {qf_arch}")
    
    maddpg = SharedPolicyMADDPG(
        agents=agents,
        obs_shape=obs_shape,
        act_shape=act_shape,
        device='cpu',
        pi_arch=pi_arch,
        qf_arch=qf_arch
    )
    
    # Load weights
    if 'maddpg_state_dict' in checkpoint:
        maddpg.load_state_dict(checkpoint['maddpg_state_dict'])
    else:
        print("Warning: 'maddpg_state_dict' not found in checkpoint. Attempting to load state_dict directly.")
        maddpg.load_state_dict(checkpoint)
    
    # Extract checkpoint filename for output naming
    checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
    
    # 1. Visualize model architecture
    print("Visualizing model architecture...")
    visualize_model_architecture(maddpg, save_dir)
    
    # 2. Visualize weight distributions
    print("Visualizing weight distributions...")
    visualize_weight_distributions(maddpg, save_dir)
    
    # 3. Plot actor response map
    print("Creating actor response map...")
    response_map_path = os.path.join(save_dir, f"{checkpoint_name}_response_map.png")
    plot_actor_response_map(maddpg, config, resolution=50, save_path=response_map_path)
    
    # 4. Plot multi-resolution response
    print("Creating multi-resolution response maps...")
    plot_multi_resolution_response(maddpg, config, save_dir)
    
    # 5. Plot actor activations for different regions
    print("Creating actor activation heatmaps...")
    plot_actor_activation_heatmap(maddpg, config, save_dir=save_dir)
    
    print(f"All visualizations saved to {save_dir}")
    return save_dir

def main():
    parser = argparse.ArgumentParser(description='Enhanced MADDPG Model Visualizer')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default='./model_visualizations',
                      help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Visualize the MADDPG model
    visualize_maddpg_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        save_dir=args.save_dir
    )

if __name__ == "__main__":
    main()