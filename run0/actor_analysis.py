import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, Normalize
import os
from typing import Dict, Tuple, Optional
from models_pettingzoo import MLPActor
from utils import load_config


def create_neuron_activation_plot(actor, config: Dict, resolution: int = 100, 
                                save_path: Optional[str] = None, 
                                colormap: str = 'viridis',
                                normalize_globally: bool = True,
                                show_weights: bool = True,
                                figsize: Tuple[int, int] = (20, 10)):
    """
    Create improved visualization of hidden neuron activation maps.
    
    Args:
        actor: Trained actor network
        config: Configuration dictionary
        resolution: Grid resolution for the plot
        save_path: Path to save the plot (optional)
        colormap: Colormap to use ('viridis', 'plasma', 'inferno', 'magma', 'cividis')
        normalize_globally: If True, use same color scale for all neurons for comparison
        show_weights: Whether to show weight values in titles
        figsize: Figure size (width, height)
    """
    
    actor.eval()
    device = next(actor.parameters()).device
    
    # Extract network parameters
    params = dict(actor.named_parameters())
    
    # Find the input->hidden weights (should be shape (8, 2))
    W1 = None
    b1 = None
    
    for key in params.keys():
        if 'weight' in key:
            weight = params[key].detach().cpu().numpy()
            if weight.shape == (8, 2):  # Input to hidden
                W1 = weight
                print(f"Found input->hidden weights: {key}")
        if 'bias' in key:
            bias = params[key].detach().cpu().numpy()
            if len(bias) == 8:  # Hidden layer bias
                b1 = bias
                print(f"Found hidden bias: {key}")
    
    if W1 is None:
        raise ValueError("Could not find input->hidden weights with shape (8, 2)")
    
    # Get action bounds and double the span
    om_max = config['action']['om_max']
    
    # Create grid of u and w values with double the span
    u_vals = np.linspace(-4*om_max, 4*om_max, resolution)
    w_vals = np.linspace(-3*om_max, 3*om_max, resolution)
    U, W = np.meshgrid(u_vals, w_vals)
    
    # Calculate activation for each neuron
    neuron_activations = np.zeros((8, resolution, resolution))
    
    print("Computing neuron activations...")
    for i in range(resolution):
        if i % 20 == 0:
            print(f"  Progress: {i}/{resolution}")
        for j in range(resolution):
            u, w = u_vals[j], w_vals[i]
            inputs = np.array([u, w])
            
            # Forward pass through first layer
            hidden_pre = np.dot(W1, inputs) + (b1 if b1 is not None else 0)
            hidden_post = np.maximum(0, hidden_pre)  # ReLU activation
            
            neuron_activations[:, i, j] = hidden_post
    
    # Determine normalization strategy
    if normalize_globally:
        # Use the same color scale for all neurons
        vmax_global = np.max(neuron_activations)
        vmin_global = 0  # ReLU ensures no negative values
        print(f"Global normalization: min={vmin_global:.4f}, max={vmax_global:.4f}")
    
    # Create the plot
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.flatten()
    
    # Choose colormap
    cmap = plt.cm.get_cmap(colormap)
    
    for neuron_idx in range(8):
        ax = axes[neuron_idx]
        activations = neuron_activations[neuron_idx]
        
        # Set normalization
        if normalize_globally:
            norm = Normalize(vmin=vmin_global, vmax=vmax_global)
        else:
            # Individual normalization for each neuron
            vmax_local = np.max(activations)
            vmin_local = 0
            norm = Normalize(vmin=vmin_local, vmax=vmax_local)
        
        # Create the activation map (smooth, no contour lines)
        im = ax.contourf(U, W, activations, levels=50, cmap=cmap, norm=norm)
        
        # Create title with neuron info
        u_weight = W1[neuron_idx, 0]
        w_weight = W1[neuron_idx, 1]
        bias_val = b1[neuron_idx] if b1 is not None else 0.0
        
        if show_weights:
            title = f'Neuron {neuron_idx}\n(u: {u_weight:.3f}, w: {w_weight:.3f})\nbias: {bias_val:.3f}'
        else:
            title = f'Neuron {neuron_idx}'
        
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel('u-velocity', fontsize=10)
        ax.set_ylabel('w-velocity', fontsize=10)
        
        # Add colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Activation', fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        
        # Mark the origin
        ax.plot(0, 0, 'r+', markersize=8, markeredgewidth=2)
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')
    
    # Add overall title
    if normalize_globally:
        suptitle = f'Hidden Neuron Activation Maps (Global Normalization)\nColormap: {colormap}, Resolution: {resolution}x{resolution}'
    else:
        suptitle = f'Hidden Neuron Activation Maps (Individual Normalization)\nColormap: {colormap}, Resolution: {resolution}x{resolution}'
    
    fig.suptitle(suptitle, fontsize=14, y=0.98)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.3, wspace=0.3)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print detailed neuron analysis
    print("\nDetailed Neuron Analysis:")
    print("="*80)
    
    for i in range(8):
        activations = neuron_activations[i]
        max_act = np.max(activations)
        mean_act = np.mean(activations)
        min_act = np.min(activations)
        std_act = np.std(activations)
        active_pct = np.mean(activations > 0.01 * max_act) * 100  # Consider > 1% of max as "active"
        
        u_w = W1[i, 0]
        w_w = W1[i, 1]
        bias_val = b1[i] if b1 is not None else 0.0
        
        # Calculate activation ranges in different quadrants
        mid_u = resolution // 2
        mid_w = resolution // 2
        
        # Quadrant analysis: [u<0,w<0], [u>0,w<0], [u<0,w>0], [u>0,w>0]
        q1 = activations[:mid_w, :mid_u]  # u<0, w<0 (upstream, wall-ward)
        q2 = activations[:mid_w, mid_u:]  # u>0, w<0 (downstream, wall-ward) 
        q3 = activations[mid_w:, :mid_u]  # u<0, w>0 (upstream, away-from-wall)
        q4 = activations[mid_w:, mid_u:]  # u>0, w>0 (downstream, away-from-wall)
        
        q1_mean = np.mean(q1)
        q2_mean = np.mean(q2)
        q3_mean = np.mean(q3)
        q4_mean = np.mean(q4)
        
        # Calculate wall-normal preference (negative w vs positive w)
        wall_ward_mean = np.mean(activations[:mid_w, :])  # w < 0
        away_wall_mean = np.mean(activations[mid_w:, :])  # w > 0
        wall_preference = wall_ward_mean - away_wall_mean
        
        # Calculate streamwise preference  
        upstream_mean = np.mean(activations[:, :mid_u])   # u < 0
        downstream_mean = np.mean(activations[:, mid_u:]) # u > 0
        stream_preference = downstream_mean - upstream_mean
        
        # Activation uniformity (coefficient of variation)
        uniformity = std_act / mean_act if mean_act > 0 else 0
        
        print(f"\nNeuron {i}:")
        print(f"  Weights: u={u_w:7.3f}, w={w_w:7.3f}, bias={bias_val:7.3f}")
        print(f"  Activation: max={max_act:6.3f}, mean={mean_act:6.3f}, std={std_act:6.3f}")
        print(f"  Active area: {active_pct:5.1f}%,  Uniformity: {uniformity:5.3f} (lower=more uniform)")
        print(f"  Wall-normal preference: {wall_preference:+7.4f} (+favors wall-ward, -favors away)")
        print(f"  Streamwise preference:  {stream_preference:+7.4f} (+favors downstream, -favors upstream)")
        print(f"  Quadrant means: Q1(u-,w-)={q1_mean:.3f}, Q2(u+,w-)={q2_mean:.3f}")
        print(f"                  Q3(u-,w+)={q3_mean:.3f}, Q4(u+,w+)={q4_mean:.3f}")
        
        # Interpretation helper
        if max_act < 0.01:
            print(f"  → DEAD NEURON: Essentially no activation")
        elif active_pct < 50:
            print(f"  → SPARSE ACTIVATION: Active in limited regions")
        elif uniformity < 0.1:
            print(f"  → BASELINE NEURON: Nearly uniform activation across input space")
        else:
            # Determine primary sensitivity
            if abs(wall_preference) > abs(stream_preference) * 2:
                if wall_preference > 0:
                    print(f"  → WALL-WARD DETECTOR: Primarily sensitive to flow toward wall")
                else:
                    print(f"  → AWAY-FROM-WALL DETECTOR: Primarily sensitive to flow away from wall")
            elif abs(stream_preference) > abs(wall_preference) * 2:
                if stream_preference > 0:
                    print(f"  → DOWNSTREAM DETECTOR: Primarily sensitive to downstream flow")
                else:
                    print(f"  → UPSTREAM DETECTOR: Primarily sensitive to upstream flow")
            else:
                print(f"  → MIXED RESPONSE: Sensitive to combination of u and w velocities")
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("- Uniformity < 0.1: Nearly uniform baseline activation")
    print("- Wall-normal preference: + = wall-ward bias, - = away-from-wall bias") 
    print("- Quadrants: Q1(u-,w-), Q2(u+,w-), Q3(u-,w+), Q4(u+,w+)")
    print("- Flow events: Q1≈ejection, Q2≈sweep, Q3≈outward, Q4≈inward interaction")


def load_and_visualize_neurons(checkpoint_path: str, config_path: str = "config.yaml", 
                              save_dir: str = "./neuron_analysis", **plot_kwargs):
    """
    Load model and create neuron activation visualization.
    """
    # Load config
    config = load_config(config_path)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create actor network
    obs_shape = (1, 1, 2)  # This will be flattened to 2 inputs
    act_shape = (1,)       # 1 output
    
    # Get architecture from config
    pi_arch = config.get('net_arch', {}).get('pi', [8])
    actor = MLPActor(obs_shape, act_shape, hidden_layers=pi_arch)
    
    # Load weights
    if 'maddpg_state_dict' in checkpoint:
        maddpg_state = checkpoint['maddpg_state_dict']
        actor_state = maddpg_state['actor']
        actor.load_state_dict(actor_state)
        print("Loaded actor weights successfully")
    else:
        raise ValueError("Could not find actor weights in checkpoint")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create visualization with specified colormap
    colormap_name = plot_kwargs.get('colormap', 'viridis')
    
    visualization_configs = [
        {
            'normalize_globally': True, 
            'save_path': os.path.join(save_dir, f'neurons_{colormap_name}_global.png')
        },
        {
            'normalize_globally': False, 
            'save_path': os.path.join(save_dir, f'neurons_{colormap_name}_individual.png')
        }
    ]
    
    for i, viz_config in enumerate(visualization_configs):
        print(f"\nCreating visualization {i+1}/2...")
        merged_kwargs = {**plot_kwargs, **viz_config}
        create_neuron_activation_plot(actor, config, **merged_kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize hidden neuron activations')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default='./neuron_analysis',
                       help='Directory to save plots')
    parser.add_argument('--resolution', type=int, default=100,
                       help='Grid resolution for the plot')
    parser.add_argument('--colormap', type=str, default='viridis',
                       choices=['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
                       help='Colormap to use')
    parser.add_argument('--normalize_globally', action='store_true',
                       help='Use global normalization across all neurons')
    parser.add_argument('--figsize', type=int, nargs=2, default=[20, 10],
                       help='Figure size (width height)')
    
    args = parser.parse_args()
    
    plot_kwargs = {
        'resolution': args.resolution,
        'colormap': args.colormap,
        'normalize_globally': args.normalize_globally,
        'figsize': tuple(args.figsize)
    }
    
    load_and_visualize_neurons(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        save_dir=args.save_dir,
        **plot_kwargs
    )