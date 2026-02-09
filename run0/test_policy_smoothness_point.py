"""
Test script for analyzing spatial and temporal smoothness of trained point-based model.

This script evaluates whether the point-based system achieved smooth actuations by:
1. Running N test episodes with the trained policy (no exploration noise)
2. Computing spatial and temporal variance/gradients on the 64×64 field
3. Performing FFT analysis to identify dominant frequencies in space and time
4. Comparing observations (inputs) and actions (outputs) at specific points

Key differences from patch-based version:
- 4096 agents (one per point) instead of 64 agents (one per 8×8 patch)
- Scalar observations [4096, 3] instead of spatial [64, 3, 8, 8]
- Scalar actions [4096, 1] instead of spatial [64, 8, 8]
- Direct field reconstruction (simple reshape) instead of patch assembly

Usage:
    cd run0
    python test_policy_smoothness_point.py \
      --checkpoint ../checkpoints_point/latest_model.pt \
      --config config_point.yaml \
      --num_episodes 5
"""

import os
import sys
import torch
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

from stwEnv_pettingzoo_point import STWParallelEnvPoint
from models_point import SharedPolicyMADDPGPoint
from utils import load_config


def analyze_policy_smoothness(checkpoint_path, config_path, num_episodes=5,
                              save_results=True, point_idx=(32, 32)):
    """
    Analyze spatial and temporal smoothness of a trained point-based model.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        num_episodes: Number of episodes to run
        save_results: Whether to save metrics and visualizations
        point_idx: (i, j) position in 64x64 domain to track for input/output comparison
    """

    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)

    # Set device
    device = 'cpu'  # Use CPU for testing to avoid GPU memory issues
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create agents list (4096 agents, one per point)
    agents = [f"agent_{i}_{j}" for i in range(64) for j in range(64)]
    num_agents = len(agents)
    print(f"Number of agents: {num_agents} (64×64 grid)")

    # Determine input channels from config
    include_prev_action = config.get('observation', {}).get('include_prev_action', False)
    input_channels = 3 if include_prev_action else 2
    print(f"Input channels: {input_channels} ({'with' if include_prev_action else 'without'} action memory)")

    # Initialize model
    print("Initializing point-based model...")
    maddpg = SharedPolicyMADDPGPoint(
        agents=agents,
        device=device,
        input_dim=input_channels,
        gamma=config['model']['gamma'],
        tau=config['model']['tau'],
        lr=config['model']['learning_rate'],
        critic_lr=config['model']['critic_learning_rate'],
        dropout_rate=0.0,  # No dropout for testing
        weight_decay=config['model']['weight_decay'],
        actor_hidden_dim=config['net_arch']['actor_hidden_dim'],
        critic_conv_channels=config['net_arch']['critic_conv'],
        critic_mlp_layers=config['net_arch']['critic_mlp'],
        gradient_clip=config['training']['gradient_clip'],
        lambda_temporal=config['model']['smoothness']['lambda_temporal'],
        lambda_global_mean=config['model']['smoothness']['lambda_global_mean'],
        consistency_enable=config['model']['smoothness']['consistency'].get('enable', False),
        consistency_lambda=config['model']['smoothness']['consistency'].get('lambda', 0.1),
        consistency_tau=config['model']['smoothness']['consistency'].get('tau_similarity', 0.9),
        consistency_margin=config['model']['smoothness']['consistency'].get('margin', 0.1),
        consistency_sample_size=config['model']['smoothness']['consistency'].get('sample_size', 512),
        consistency_warmup_steps=config['model']['smoothness']['consistency'].get('warmup_steps', 5000),
        similarity_dim=config['net_arch']['similarity_dim']
    )

    # Load weights
    maddpg.load_state_dict(checkpoint['maddpg_state_dict'])
    maddpg.actor.eval()
    print("Model loaded successfully!")

    # Print checkpoint info
    if 'episode' in checkpoint:
        print(f"Checkpoint from episode: {checkpoint['episode']}")
    if 'best_reward' in checkpoint:
        print(f"Best reward: {checkpoint['best_reward']:.4f}")

    # Create environment
    print("\nInitializing environment...")
    env = STWParallelEnvPoint(config)

    # Get episode length
    episode_length = config['training']['start_episode_length']
    print(f"Episode length: {episode_length} steps")
    print(f"Number of episodes: {num_episodes}")

    # Get dt for temporal frequency analysis
    dt = config.get('field_shift', {}).get('dt', 1.0)

    # Create results directory
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"smoothness_analysis_point_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        print(f"Results will be saved to: {results_dir}")

    # Convert point_idx to agent coordinates
    point_i, point_j = point_idx
    track_agent_name = f"agent_{point_i}_{point_j}"
    track_agent_idx = point_i * 64 + point_j

    print(f"\nTracking point ({point_i}, {point_j}): agent {track_agent_name} (idx {track_agent_idx})")

    # Storage for all episodes
    all_episodes_data = []

    # Run episodes
    print("\n" + "="*80)
    print("RUNNING TEST EPISODES")
    print("="*80)

    for episode in range(num_episodes):
        print(f"\nStarting episode {episode+1}/{num_episodes}...")

        # Storage for this episode
        episode_data = {
            'observations': [],  # Will store [T, 4096, 2 or 3]
            'actions': [],       # Will store [T, 4096]
            'rewards': [],
            'dpdx': [],
            # Point tracking
            'point_u': [],
            'point_w': [],
            'point_action': []
        }

        obs, info = env.reset()

        for step in tqdm(range(episode_length), desc=f"Episode {episode+1}"):
            # Convert obs dict to array [4096, 2 or 3]
            obs_array = np.stack([obs[agent] for agent in agents])
            obs_torch = torch.FloatTensor(obs_array).to(device)

            # Get deterministic actions (NO noise)
            with torch.no_grad():
                actions = maddpg.select_actions_batched(obs_torch)  # [4096]

            # Store data
            episode_data['observations'].append(obs_array.copy())
            episode_data['actions'].append(actions.copy())

            # Track specific point
            u_val = obs[track_agent_name][0]  # Scalar
            w_val = obs[track_agent_name][1]  # Scalar
            action_val = actions[track_agent_idx]
            episode_data['point_u'].append(float(u_val))
            episode_data['point_w'].append(float(w_val))
            episode_data['point_action'].append(float(action_val))

            # Convert to dict for environment
            action_dict = {agent: actions[i] for i, agent in enumerate(agents)}

            # Step environment
            obs, rewards, dones, truncated, infos = env.step(action_dict)

            # Store metrics
            episode_data['rewards'].append(np.mean(list(rewards.values())))
            episode_data['dpdx'].append(float(infos[agents[0]].get('dpdx', 0)))

        # Convert to arrays
        episode_data['observations'] = np.array(episode_data['observations'])  # [T, 4096, 2 or 3]
        episode_data['actions'] = np.array(episode_data['actions'])            # [T, 4096]
        episode_data['rewards'] = np.array(episode_data['rewards'])
        episode_data['dpdx'] = np.array(episode_data['dpdx'])
        episode_data['point_u'] = np.array(episode_data['point_u'])
        episode_data['point_w'] = np.array(episode_data['point_w'])
        episode_data['point_action'] = np.array(episode_data['point_action'])

        all_episodes_data.append(episode_data)

        print(f"Episode {episode+1} completed: "
              f"mean reward={np.mean(episode_data['rewards']):.2f}, "
              f"mean dpdx={np.mean(episode_data['dpdx']):.6f}")

    # ============================================================================
    # ANALYSIS 1: SPATIAL STATISTICS
    # ============================================================================
    print("\n" + "="*80)
    print("SPATIAL SMOOTHNESS ANALYSIS")
    print("="*80)

    spatial_stats = []
    for ep_idx, ep_data in enumerate(all_episodes_data):
        actions = ep_data['actions']  # [T, 4096]
        T, N = actions.shape

        # Reshape to 64×64 field
        actions_field = actions.reshape(T, 64, 64)  # [T, 64, 64]

        # Spatial variance at each timestep
        spatial_var_time = np.var(actions_field, axis=(1, 2))  # [T]

        # Spatial gradients (finite differences on 64×64 field)
        grad_i = np.diff(actions_field, axis=1)  # [T, 63, 64]
        grad_j = np.diff(actions_field, axis=2)  # [T, 64, 63]
        spatial_grad_rms = np.sqrt(np.mean(grad_i**2) + np.mean(grad_j**2))

        stats = {
            'variance_time_series': spatial_var_time,
            'mean_variance': np.mean(spatial_var_time),
            'std_variance': np.std(spatial_var_time),
            'spatial_gradient_rms': spatial_grad_rms,
        }
        spatial_stats.append(stats)

        print(f"\nEpisode {ep_idx+1}:")
        print(f"  Spatial variance (mean±std): {stats['mean_variance']:.6f} ± {stats['std_variance']:.6f}")
        print(f"  Spatial gradient RMS: {stats['spatial_gradient_rms']:.6f}")

    # Aggregate
    print(f"\nAGGREGATE ({num_episodes} episodes):")
    print(f"  Spatial variance: {np.mean([s['mean_variance'] for s in spatial_stats]):.6f} "
          f"± {np.std([s['mean_variance'] for s in spatial_stats]):.6f}")
    print(f"  Spatial gradient RMS: {np.mean([s['spatial_gradient_rms'] for s in spatial_stats]):.6f} "
          f"± {np.std([s['spatial_gradient_rms'] for s in spatial_stats]):.6f}")

    # ============================================================================
    # ANALYSIS 2: TEMPORAL STATISTICS
    # ============================================================================
    print("\n" + "="*80)
    print("TEMPORAL SMOOTHNESS ANALYSIS")
    print("="*80)

    temporal_stats = []
    for ep_idx, ep_data in enumerate(all_episodes_data):
        actions = ep_data['actions']  # [T, 4096]
        T, N = actions.shape

        # Temporal variance (variance over time for each point)
        temporal_var = np.var(actions, axis=0).mean()  # Scalar

        # Temporal gradient (first derivative)
        temporal_grad = np.diff(actions, axis=0)  # [T-1, 4096]
        temporal_grad_rms = np.sqrt(np.mean(temporal_grad**2))

        # Temporal acceleration (second derivative)
        temporal_accel = np.diff(temporal_grad, axis=0)  # [T-2, 4096]
        temporal_accel_rms = np.sqrt(np.mean(temporal_accel**2))

        stats = {
            'variance': temporal_var,
            'gradient_rms': temporal_grad_rms,
            'acceleration_rms': temporal_accel_rms,
        }
        temporal_stats.append(stats)

        print(f"\nEpisode {ep_idx+1}:")
        print(f"  Temporal variance: {stats['variance']:.6f}")
        print(f"  Temporal gradient RMS: {stats['gradient_rms']:.6f}")
        print(f"  Temporal acceleration RMS: {stats['acceleration_rms']:.6f}")

    # Aggregate
    print(f"\nAGGREGATE ({num_episodes} episodes):")
    print(f"  Temporal variance: {np.mean([s['variance'] for s in temporal_stats]):.6f} "
          f"± {np.std([s['variance'] for s in temporal_stats]):.6f}")
    print(f"  Temporal gradient RMS: {np.mean([s['gradient_rms'] for s in temporal_stats]):.6f} "
          f"± {np.std([s['gradient_rms'] for s in temporal_stats]):.6f}")
    print(f"  Temporal acceleration RMS: {np.mean([s['acceleration_rms'] for s in temporal_stats]):.6f} "
          f"± {np.std([s['acceleration_rms'] for s in temporal_stats]):.6f}")

    # ============================================================================
    # ANALYSIS 3: FREQUENCY ANALYSIS
    # ============================================================================
    print("\n" + "="*80)
    print("FREQUENCY ANALYSIS")
    print("="*80)
    print(f"Using dt = {dt} for temporal frequencies\n")

    frequency_stats = []
    for ep_idx, ep_data in enumerate(all_episodes_data):
        actions = ep_data['actions']  # [T, 4096]
        T, N = actions.shape

        # TEMPORAL FFT (average over all points)
        actions_time_avg = actions.mean(axis=1)  # [T]
        temporal_fft = np.abs(np.fft.rfft(actions_time_avg))
        temporal_freqs = np.fft.rfftfreq(T, d=dt)

        # Find dominant frequencies (excluding DC)
        top_temporal_idx = np.argsort(temporal_fft[1:])[-5:][::-1] + 1
        dominant_temporal_freqs = temporal_freqs[top_temporal_idx]
        dominant_temporal_powers = temporal_fft[top_temporal_idx]

        # SPATIAL FFT (average over time)
        actions_field = actions.reshape(T, 64, 64)  # [T, 64, 64]
        actions_spatial_avg = actions_field.mean(axis=0)  # [64, 64]
        spatial_fft_2d = np.abs(np.fft.rfft2(actions_spatial_avg))

        # Find dominant spatial modes
        flat_fft = spatial_fft_2d.flatten()
        top_spatial_idx = np.argsort(flat_fft[1:])[-5:][::-1] + 1
        dominant_spatial_powers = flat_fft[top_spatial_idx]

        stats = {
            'temporal_freqs': temporal_freqs,
            'temporal_fft': temporal_fft,
            'dominant_temporal_freqs': dominant_temporal_freqs,
            'dominant_temporal_powers': dominant_temporal_powers,
            'spatial_fft_2d': spatial_fft_2d,
            'dominant_spatial_powers': dominant_spatial_powers,
        }
        frequency_stats.append(stats)

        print(f"Episode {ep_idx+1}:")
        print(f"  Dominant temporal frequencies (Hz): {dominant_temporal_freqs}")
        print(f"  Dominant temporal powers: {dominant_temporal_powers}")

    # ============================================================================
    # ANALYSIS 4: INPUT-OUTPUT COMPARISON AT POINT
    # ============================================================================
    print("\n" + "="*80)
    print(f"INPUT-OUTPUT COMPARISON AT POINT {point_idx}")
    print("="*80)

    for ep_idx, ep_data in enumerate(all_episodes_data):
        u_vals = ep_data['point_u']
        w_vals = ep_data['point_w']
        action_vals = ep_data['point_action']

        print(f"\nEpisode {ep_idx+1}:")
        print(f"  U velocity: mean={np.mean(u_vals):.4f}, std={np.std(u_vals):.4f}")
        print(f"  W velocity: mean={np.mean(w_vals):.4f}, std={np.std(w_vals):.4f}")
        print(f"  Action:     mean={np.mean(action_vals):.4f}, std={np.std(action_vals):.4f}")

    # ============================================================================
    # SAVE RESULTS
    # ============================================================================
    if save_results:
        print("\n" + "="*80)
        print("SAVING RESULTS")
        print("="*80)

        # Save episode data as NPZ files
        for ep_idx, ep_data in enumerate(all_episodes_data):
            npz_path = os.path.join(results_dir, f'episode_{ep_idx}_data.npz')
            np.savez(npz_path, **ep_data)
            print(f"Saved episode {ep_idx} data: {npz_path}")

        # Save summary JSON
        summary = {
            'num_episodes': num_episodes,
            'episode_length': episode_length,
            'dt': dt,
            'tracked_point': point_idx,
            'num_agents': num_agents,
            'architecture': 'point-based',
            'spatial_variance_mean': float(np.mean([s['mean_variance'] for s in spatial_stats])),
            'spatial_variance_std': float(np.std([s['mean_variance'] for s in spatial_stats])),
            'spatial_gradient_rms_mean': float(np.mean([s['spatial_gradient_rms'] for s in spatial_stats])),
            'temporal_variance_mean': float(np.mean([s['variance'] for s in temporal_stats])),
            'temporal_variance_std': float(np.std([s['variance'] for s in temporal_stats])),
            'temporal_gradient_rms_mean': float(np.mean([s['gradient_rms'] for s in temporal_stats])),
            'temporal_acceleration_rms_mean': float(np.mean([s['acceleration_rms'] for s in temporal_stats])),
        }

        json_path = os.path.join(results_dir, 'summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved JSON: {json_path}")

        # ========================================================================
        # GENERATE PLOTS
        # ========================================================================
        print("\nGenerating plots...")

        # Plot 1: Spatial and temporal variance
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Spatial variance over time
        for ep_idx, stats in enumerate(spatial_stats):
            axes[0, 0].plot(stats['variance_time_series'], alpha=0.7, label=f'Ep {ep_idx+1}')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Spatial Variance')
        axes[0, 0].set_title('Spatial Variance Over Time (Point-Based)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Temporal variance bar plot
        ep_labels = [f"Ep{i+1}" for i in range(num_episodes)]
        temporal_vars = [s['variance'] for s in temporal_stats]
        axes[0, 1].bar(ep_labels, temporal_vars, alpha=0.7, color='steelblue')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Temporal Variance')
        axes[0, 1].set_title('Temporal Variance per Episode')
        axes[0, 1].grid(True, alpha=0.3, axis='y')

        # Temporal FFT
        stats = frequency_stats[0]
        axes[1, 0].semilogy(stats['temporal_freqs'], stats['temporal_fft'])
        axes[1, 0].scatter(stats['dominant_temporal_freqs'], stats['dominant_temporal_powers'],
                          c='red', s=100, zorder=5, label='Dominant')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('FFT Magnitude')
        axes[1, 0].set_title('Temporal Frequency Spectrum (Episode 1)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Spatial FFT 2D
        im = axes[1, 1].imshow(stats['spatial_fft_2d'], cmap='viridis', aspect='auto', origin='lower')
        axes[1, 1].set_xlabel('Frequency j')
        axes[1, 1].set_ylabel('Frequency i')
        axes[1, 1].set_title('Spatial Frequency Spectrum 2D (Episode 1)')
        plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig_path = os.path.join(results_dir, 'smoothness_analysis.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {fig_path}")
        plt.close()

        # Plot 2: Input-output comparison at tracked point
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        for ep_idx, ep_data in enumerate(all_episodes_data):
            t = np.arange(len(ep_data['point_u']))
            axes[0].plot(t, ep_data['point_u'], alpha=0.7, label=f'Ep {ep_idx+1}')
            axes[1].plot(t, ep_data['point_w'], alpha=0.7, label=f'Ep {ep_idx+1}')
            axes[2].plot(t, ep_data['point_action'], alpha=0.7, label=f'Ep {ep_idx+1}')

        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('U velocity')
        axes[0].set_title(f'Input: U-velocity at point {point_idx}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('W velocity')
        axes[1].set_title(f'Input: W-velocity at point {point_idx}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].set_xlabel('Time Step')
        axes[2].set_ylabel('Action')
        axes[2].set_title(f'Output: Action at point {point_idx}')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(results_dir, 'input_output_comparison.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {fig_path}")
        plt.close()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    if save_results:
        print(f"Results saved to: {results_dir}")

    return all_episodes_data, spatial_stats, temporal_stats, frequency_stats


def main():
    parser = argparse.ArgumentParser(
        description='Analyze spatial and temporal smoothness of trained point-based model'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--num_episodes', type=int, default=5,
                       help='Number of episodes to run (default: 5)')
    parser.add_argument('--point_i', type=int, default=32,
                       help='i-coordinate of point to track (default: 32)')
    parser.add_argument('--point_j', type=int, default=32,
                       help='j-coordinate of point to track (default: 32)')
    parser.add_argument('--no_save', action='store_true',
                       help='Do not save results')

    args = parser.parse_args()

    # Run analysis
    analyze_policy_smoothness(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.num_episodes,
        save_results=not args.no_save,
        point_idx=(args.point_i, args.point_j)
    )


if __name__ == "__main__":
    main()
