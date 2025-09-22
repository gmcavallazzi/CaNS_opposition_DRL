import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import TwoSlopeNorm, Normalize
import os
from typing import Dict, Tuple, Optional
from models_pettingzoo import MLPCritic, SharedPolicyMADDPG
from utils import load_config


def analyze_critic_network_structure(critic):
    """
    Analyze the structure and weight properties of the critic network.
    
    Args:
        critic: The trained critic network
        
    Returns:
        dict: Dictionary containing network analysis results
    """
    layer_stats = []
    layer_names = []
    
    # Analyze each linear layer
    for name, module in critic.named_modules():
        if isinstance(module, torch.nn.Linear):
            layer_names.append(name)
            
            # Get weights and biases
            weights = module.weight.data.cpu().numpy()
            bias = module.bias.data.cpu().numpy() if module.bias is not None else None
            
            # Weight statistics
            mean_abs_weight = np.mean(np.abs(weights))
            std_weight = np.std(weights)
            max_weight = np.max(np.abs(weights))
            
            # Effective parameters (weights with magnitude > threshold)
            threshold = 0.01
            effective_weights = np.sum(np.abs(weights) > threshold)
            total_weights = weights.size
            
            # Bias statistics
            mean_bias = np.mean(bias) if bias is not None else None
            
            stats = {
                'layer_size': total_weights,
                'mean_abs_weight': mean_abs_weight,
                'std_weight': std_weight,
                'max_weight': max_weight,
                'effective_parameters': effective_weights,
                'mean_bias': mean_bias,
                'weight_shape': weights.shape
            }
            
            layer_stats.append(stats)
    
    return {
        'layer_stats': (layer_names, layer_stats),
        'total_parameters': sum(stat['layer_size'] for stat in layer_stats)
    }


def create_critic_analysis(critic, actor, config: Dict, n_agents: int, resolution: int = 50,
                          save_path: Optional[str] = None, 
                          colormap: str = 'RdBu_r', figsize: Tuple[int, int] = (16, 12)):
    """
    Analyze critic network responses in the context of the full actor-critic system.
    This analyzes how the critic evaluates the actor's policy.
    
    Args:
        critic: Trained critic network
        actor: Trained actor network (needed to generate realistic actions)
        config: Configuration dictionary
        n_agents: Number of agents
        resolution: Grid resolution for analysis
        save_path: Path to save the plot (optional)
        colormap: Colormap to use
        figsize: Figure size (width, height)
    """
    
    critic.eval()
    actor.eval()
    device = next(critic.parameters()).device
    
    # Get action bounds - observations are already scaled by om_max in the environment
    om_max = config['action']['om_max']
    
    # Create grids for analysis with custom ranges
    # u: ±4*om_max, w: ±3*om_max
    u_vals = np.linspace(-4*om_max, 4*om_max, resolution)
    w_vals = np.linspace(-3*om_max, 3*om_max, resolution)
    
    # Create meshgrids
    U, W = np.meshgrid(u_vals, w_vals)
    
    print("Computing critic-actor system responses...")
    
    # Analysis 1: Q-values for (u,w) observations with actor-generated actions
    print(f"  Analysis 1: Q(obs, π(obs)) - Critic evaluating actor's policy")
    q_values_policy = np.zeros((resolution, resolution))
    actor_actions = np.zeros((resolution, resolution))
    
    for i in range(resolution):
        if i % 5 == 0:  # More frequent progress updates
            print(f"    Progress: {i}/{resolution}")
        for j in range(resolution):
            u, w = u_vals[j], w_vals[i]
            
            # Create observation for actor (single agent)
            obs_single = torch.tensor([[u, w]], dtype=torch.float32, device=device)
            
            # Get action from actor
            with torch.no_grad():
                action_single = actor(obs_single).cpu().numpy()[0, 0]
            
            actor_actions[i, j] = action_single
            
            # Create observation and action for all agents (critic input)
            obs_all = torch.zeros(1, n_agents * 2, device=device)
            for agent_idx in range(n_agents):
                obs_all[0, agent_idx*2:(agent_idx+1)*2] = torch.tensor([u, w], device=device)
            
            # All agents take the same action (shared policy)
            act_all = torch.full((1, n_agents), action_single, device=device)
            
            # Get Q-value from critic
            with torch.no_grad():
                q_value = critic(obs_all, act_all).cpu().numpy()[0, 0]
            
            q_values_policy[i, j] = q_value
    
    # Analysis 2: Action sensitivity - how Q-values change with action perturbations
    print(f"  Analysis 2: Action sensitivity analysis")
    action_sensitivity = np.zeros((resolution, resolution))
    
    perturbation = 0.1  # Small action perturbation
    
    for i in range(0, resolution, 5):  # Sample subset for efficiency
        for j in range(0, resolution, 5):
            u, w = u_vals[j], w_vals[i]
            
            # Get base action from actor
            obs_single = torch.tensor([[u, w]], dtype=torch.float32, device=device)
            with torch.no_grad():
                base_action = actor(obs_single).cpu().numpy()[0, 0]
            
            # Test perturbed actions
            perturbed_action_pos = np.clip(base_action + perturbation, -1.0, 1.0)
            perturbed_action_neg = np.clip(base_action - perturbation, -1.0, 1.0)
            
            # Create inputs for critic
            obs_all = torch.zeros(1, n_agents * 2, device=device)
            for agent_idx in range(n_agents):
                obs_all[0, agent_idx*2:(agent_idx+1)*2] = torch.tensor([u, w], device=device)
            
            # Get Q-values for base and perturbed actions
            with torch.no_grad():
                act_base = torch.full((1, n_agents), base_action, device=device)
                act_pos = torch.full((1, n_agents), perturbed_action_pos, device=device)
                act_neg = torch.full((1, n_agents), perturbed_action_neg, device=device)
                
                q_base = critic(obs_all, act_base).cpu().numpy()[0, 0]
                q_pos = critic(obs_all, act_pos).cpu().numpy()[0, 0]
                q_neg = critic(obs_all, act_neg).cpu().numpy()[0, 0]
            
            # Calculate sensitivity as max change in Q-value
            sensitivity = max(abs(q_pos - q_base), abs(q_neg - q_base))
            
            # Fill in local region (since we're subsampling)
            for di in range(5):
                for dj in range(5):
                    if i+di < resolution and j+dj < resolution:
                        action_sensitivity[i+di, j+dj] = sensitivity
    
    # Create comprehensive visualization  
    fig = plt.figure(figsize=figsize)
    
    # Plot 1: Q-values for actor's policy
    ax1 = plt.subplot(2, 3, 1)
    
    q_min, q_max = q_values_policy.min(), q_values_policy.max()
    if q_min < 0 and q_max > 0:
        norm1 = TwoSlopeNorm(vmin=q_min, vcenter=0, vmax=q_max)
    else:
        norm1 = Normalize(vmin=q_min, vmax=q_max)
    
    im1 = ax1.contourf(U, W, q_values_policy, levels=100, cmap=colormap, norm=norm1)  # Higher resolution
    ax1.set_xlabel('u-velocity (streamwise)')
    ax1.set_ylabel('w-velocity (wall-normal)')
    ax1.set_title('Q-values: Q(obs, π(obs))\n(Critic evaluating actor policy)')
    ax1.plot(0, 0, 'k+', markersize=10, markeredgewidth=2)
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # Plot 2: Actor's action map
    ax2 = plt.subplot(2, 3, 2)
    
    action_min, action_max = actor_actions.min(), actor_actions.max()
    
    # Handle case where all actions are the same (or very similar)
    if abs(action_max - action_min) < 1e-6:
        # All actions are essentially the same, use a simple colormap
        norm2 = Normalize(vmin=action_min-0.01, vmax=action_max+0.01)
        cmap2 = 'viridis'
    else:
        # Use diverging colormap if range allows
        if action_min < 0 and action_max > 0:
            norm2 = TwoSlopeNorm(vmin=action_min, vcenter=0, vmax=action_max)
        else:
            norm2 = Normalize(vmin=action_min, vmax=action_max)
        cmap2 = 'RdBu_r'
    
    im2 = ax2.contourf(U, W, actor_actions, levels=100, cmap=cmap2, norm=norm2)  # Higher resolution
    ax2.set_xlabel('u-velocity (streamwise)')
    ax2.set_ylabel('w-velocity (wall-normal)')
    ax2.set_title('Actor Actions: π(obs)\n(What actions actor chooses)')
    ax2.plot(0, 0, 'k+', markersize=10, markeredgewidth=2)
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # Plot 3: Action sensitivity
    ax3 = plt.subplot(2, 3, 3)
    
    im3 = ax3.contourf(U[::5, ::5], W[::5, ::5], action_sensitivity[::5, ::5], 
                       levels=100, cmap='Reds')  # Higher resolution
    ax3.set_xlabel('u-velocity (streamwise)')
    ax3.set_ylabel('w-velocity (wall-normal)')
    ax3.set_title('Action Sensitivity\n(How much Q changes with action)')
    ax3.plot(0, 0, 'k+', markersize=10, markeredgewidth=2)
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # Get critic network analysis
    critic_info = analyze_critic_network_structure(critic)
    
    # Plot 4: Layer weight distributions
    ax4 = plt.subplot(2, 3, 4)
    layer_names, weight_stats = critic_info['layer_stats']
    
    # Plot weight magnitude distributions for each layer
    positions = range(len(layer_names))
    weight_means = [stats['mean_abs_weight'] for stats in weight_stats]
    weight_stds = [stats['std_weight'] for stats in weight_stats]
    
    bars = ax4.bar(positions, weight_means, yerr=weight_stds, capsize=5, alpha=0.7)
    ax4.set_xlabel('Layer')
    ax4.set_ylabel('Mean |Weight|')
    ax4.set_title('Layer Weight Magnitudes')
    ax4.set_xticks(positions)
    ax4.set_xticklabels([f'L{i+1}' for i in positions], rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Color bars by layer depth
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(i / len(bars)))
    
    # Plot 5: Bias analysis
    ax5 = plt.subplot(2, 3, 5)
    bias_means = [stats['mean_bias'] for stats in weight_stats if stats['mean_bias'] is not None]
    bias_positions = [i for i, stats in enumerate(weight_stats) if stats['mean_bias'] is not None]
    
    if bias_means:
        bars = ax5.bar(bias_positions, bias_means, alpha=0.7, color='orange')
        ax5.set_xlabel('Layer')
        ax5.set_ylabel('Mean Bias')
        ax5.set_title('Layer Bias Values')
        ax5.set_xticks(bias_positions)
        ax5.set_xticklabels([f'L{i+1}' for i in bias_positions], rotation=45)
        ax5.grid(True, alpha=0.3)
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    else:
        ax5.text(0.5, 0.5, 'No bias parameters found', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Layer Bias Values')
    
    # Plot 6: Layer capacity utilization
    ax6 = plt.subplot(2, 3, 6)
    layer_sizes = [stats['layer_size'] for stats in weight_stats]
    effective_params = [stats['effective_parameters'] for stats in weight_stats]
    utilization = [(eff/size)*100 if size > 0 else 0 for eff, size in zip(effective_params, layer_sizes)]
    
    bars = ax6.bar(positions, utilization, alpha=0.7, color='green')
    ax6.set_xlabel('Layer')
    ax6.set_ylabel('Utilization %')
    ax6.set_title('Layer Parameter Utilization')
    ax6.set_xticks(positions)
    ax6.set_xticklabels([f'L{i+1}' for i in positions], rotation=45)
    ax6.set_ylim(0, 100)
    ax6.grid(True, alpha=0.3)
    
    # Color bars by utilization
    for i, (bar, util) in enumerate(zip(bars, utilization)):
        if util > 80:
            bar.set_color('green')
        elif util > 50:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Critic-actor analysis saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\nCritic-Actor System Analysis:")
    print("="*60)
    print(f"Q-value range: [{q_values_policy.min():.4f}, {q_values_policy.max():.4f}]")
    print(f"Action range: [{actor_actions.min():.4f}, {actor_actions.max():.4f}]")
    print(f"Action sensitivity range: [{action_sensitivity.min():.4f}, {action_sensitivity.max():.4f}]")
    
    # Find optimal and worst flow states according to critic
    best_state_idx = np.unravel_index(np.argmax(q_values_policy), q_values_policy.shape)
    worst_state_idx = np.unravel_index(np.argmin(q_values_policy), q_values_policy.shape)
    
    best_u, best_w = u_vals[best_state_idx[1]], w_vals[best_state_idx[0]]
    worst_u, worst_w = u_vals[worst_state_idx[1]], w_vals[worst_state_idx[0]]
    best_action = actor_actions[best_state_idx]
    worst_action = actor_actions[worst_state_idx]
    
    print(f"\nBest flow state (highest Q): (u={best_u:.4f}, w={best_w:.4f})")
    print(f"  Actor chooses action: {best_action:.4f}")
    print(f"  Q-value: {q_values_policy[best_state_idx]:.4f}")
    
    print(f"\nWorst flow state (lowest Q): (u={worst_u:.4f}, w={worst_w:.4f})")
    print(f"  Actor chooses action: {worst_action:.4f}")
    print(f"  Q-value: {q_values_policy[worst_state_idx]:.4f}")
    
    # Analyze correlation between Q-values and actions
    correlation = np.corrcoef(q_values_policy.flatten(), actor_actions.flatten())[0, 1]
    print(f"\nQ-value to Action correlation: {correlation:.4f}")
    if abs(correlation) > 0.5:
        print("  → Strong correlation: Critic and actor are aligned")
    elif abs(correlation) > 0.2:
        print("  → Moderate correlation: Some alignment")
    else:
        print("  → Weak correlation: Limited alignment")
    
    return {
        'q_values_policy': q_values_policy,
        'actor_actions': actor_actions,
        'action_sensitivity': action_sensitivity,
        'critic_info': critic_info,
        'u_vals': u_vals,
        'w_vals': w_vals
    }


def analyze_critic_layers(critic, config: Dict, n_agents: int, sample_size: int = 1000):
    """
    Analyze the activation patterns of individual layers in the critic network.
    
    Args:
        critic: Trained critic network
        config: Configuration dictionary  
        n_agents: Number of agents
        sample_size: Number of random samples to analyze
    """
    
    critic.eval()
    device = next(critic.parameters()).device
    
    print(f"Analyzing critic layer activations with {sample_size} random samples...")
    
    # Generate random samples
    om_max = config['action']['om_max']
    
    # Random observations and actions
    u_samples = np.random.uniform(-2*om_max, 2*om_max, sample_size)
    w_samples = np.random.uniform(-2*om_max, 2*om_max, sample_size)
    action_samples = np.random.uniform(-1.0, 1.0, sample_size)
    
    # Store layer activations
    layer_activations = {}
    
    # Hook functions to capture activations
    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor):
                layer_activations[name] = output.detach().cpu().numpy()
        return hook
    
    # Register hooks for each layer
    hooks = []
    layer_names = []
    
    for name, module in critic.named_modules():
        if isinstance(module, torch.nn.Linear):
            hook = module.register_forward_hook(get_activation(name))
            hooks.append(hook)
            layer_names.append(name)
    
    # Run samples through the network
    sample_activations = {name: [] for name in layer_names}
    
    for i in range(sample_size):
        if i % 100 == 0:
            print(f"  Processing sample {i}/{sample_size}")
        
        u, w, action = u_samples[i], w_samples[i], action_samples[i]
        
        # Create inputs
        obs_all = torch.zeros(1, n_agents * 2, device=device)
        for agent_idx in range(n_agents):
            obs_all[0, agent_idx*2:(agent_idx+1)*2] = torch.tensor([u, w], device=device)
        
        act_all = torch.full((1, n_agents), action, device=device)
        
        # Forward pass
        with torch.no_grad():
            _ = critic(obs_all, act_all)
        
        # Store activations
        for name in layer_names:
            if name in layer_activations:
                sample_activations[name].append(layer_activations[name].flatten())
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Analyze layer statistics
    print("\nCritic Layer Analysis:")
    print("="*80)
    
    for name in layer_names:
        if sample_activations[name]:
            # Convert to numpy array
            activations = np.array(sample_activations[name])
            
            # Calculate statistics
            mean_act = np.mean(activations)
            std_act = np.std(activations)
            min_act = np.min(activations)
            max_act = np.max(activations)
            
            # Calculate percentage of dead neurons (ReLU layers)
            dead_pct = np.mean(activations == 0) * 100
            
            # Calculate activation diversity (how many neurons are consistently active)
            neuron_means = np.mean(activations, axis=0)
            active_neurons = np.sum(neuron_means > 0.01)
            total_neurons = len(neuron_means)
            
            print(f"\nLayer: {name}")
            print(f"  Shape: {activations.shape}")
            print(f"  Activation range: [{min_act:.4f}, {max_act:.4f}]")
            print(f"  Mean ± Std: {mean_act:.4f} ± {std_act:.4f}")
            print(f"  Dead neurons: {dead_pct:.1f}%")
            print(f"  Active neurons: {active_neurons}/{total_neurons} ({100*active_neurons/total_neurons:.1f}%)")
            
            if 'output' in name.lower():
                print(f"  → OUTPUT LAYER: Produces Q-values")
            elif dead_pct > 50:
                print(f"  → SPARSE LAYER: Many inactive neurons")
            elif std_act < 0.1:
                print(f"  → SATURATED LAYER: Low activation diversity")
            else:
                print(f"  → ACTIVE LAYER: Good activation diversity")


def load_and_analyze_critic(checkpoint_path: str, config_path: str = "config.yaml", 
                           save_dir: str = "./critic_analysis", **plot_kwargs):
    """
    Load model and create comprehensive critic analysis.
    """
    # Load config
    config = load_config(config_path)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get grid dimensions to determine number of agents
    grid_i = config['grid']['target']['i']
    grid_j = config['grid']['target']['j']
    n_agents = grid_i * grid_j
    
    # Create MADDPG model (need both actor and critic)
    agents = [f"agent_{i}_{j}" for i in range(grid_i) for j in range(grid_j)]
    
    # Determine observation shape based on halo value
    halo = config.get('halo', 0)
    obs_height = 2 * halo + 1 if halo > 0 else 1
    obs_width = 2 * halo + 1 if halo > 0 else 1
    obs_shape = (obs_height, obs_width, 2)
    act_shape = (1,)
    
    # Get architecture from config
    pi_arch = config.get('net_arch', {}).get('pi', [8])
    qf_arch = config.get('net_arch', {}).get('qf', [16, 64, 64])
    
    maddpg = SharedPolicyMADDPG(
        agents=agents,
        obs_shape=obs_shape,
        act_shape=act_shape,
        pi_arch=pi_arch,
        qf_arch=qf_arch
    )
    
    # Load weights
    if 'maddpg_state_dict' in checkpoint:
        maddpg.load_state_dict(checkpoint['maddpg_state_dict'])
        print("Loaded MADDPG weights successfully")
    else:
        raise ValueError("Could not find MADDPG weights in checkpoint")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Analyzing critic with {n_agents} agents")
    print(f"Critic architecture: {qf_arch}")
    print(f"Input dimensions: obs={np.prod(obs_shape)} * {n_agents} = {np.prod(obs_shape) * n_agents}")
    print(f"                  act=1 * {n_agents} = {n_agents}")
    print(f"                  total={np.prod(obs_shape) * n_agents + n_agents}")
    
    # Analysis 1: Response maps
    print("\nCreating critic-actor system analysis...")
    results = create_critic_analysis(
        maddpg.critic, maddpg.actor, config, n_agents, 
        save_path=os.path.join(save_dir, 'critic_analysis.png'),
        **plot_kwargs
    )
    
    # Analysis 2: Layer activations
    print("\nAnalyzing layer activations...")
    analyze_critic_layers(maddpg.critic, config, n_agents, sample_size=500)
    
    print(f"\nCritic analysis complete! Results saved in {save_dir}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze critic network')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--save_dir', type=str, default='./critic_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--resolution', type=int, default=50,
                       help='Grid resolution for analysis')
    
    args = parser.parse_args()
    
    plot_kwargs = {
        'resolution': args.resolution
    }
    
    load_and_analyze_critic(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        save_dir=args.save_dir,
        **plot_kwargs
    )