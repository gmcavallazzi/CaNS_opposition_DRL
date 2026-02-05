"""
Test script for evaluating a trained consistency model on episodes.

Features:
- Runs multiple episodes with the trained model
- Tracks metrics (dpdx, e_ks, uw, rewards, actions)
- Creates point tracking at position (32, 32) for u, w, and action fields
- Generates separate plots for each episode showing u, w, and action time series at (32,32)
- Saves all metrics and visualizations

Usage:
    python test_consistency_model.py --checkpoint checkpoints_consistency/best_model.pt --config config_consistency.yaml
    python test_consistency_model.py --checkpoint checkpoints_consistency/best_model.pt --config config_consistency.yaml --num_episodes 5
"""

import os
import sys
import torch
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt

from stwEnv_pettingzoo import STWParallelEnv
from models_consistency import SharedPolicyMADDPGConsistency
from utils import load_config


def test_consistency_model(checkpoint_path, config_path='config_consistency.yaml',
                           num_episodes=1, save_results=True, visualize=True):
    """
    Test a trained consistency model on new episodes.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        num_episodes: Number of episodes to run
        save_results: Whether to save metrics and visualizations
        visualize: Whether to create visualizations
    """

    # Load configuration
    print("Loading configuration...")
    config = load_config(config_path)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create agents list (64 agents in 8x8 grid)
    agents = [f"agent_{i}_{j}" for i in range(8) for j in range(8)]

    # Determine input channels from config
    include_prev_action = config.get('observation', {}).get('include_prev_action', False)
    input_channels = 3 if include_prev_action else 2
    print(f"Input channels: {input_channels} ({'with' if include_prev_action else 'without'} action memory)")

    # Initialize model with same parameters as training
    print("Initializing model...")
    maddpg = SharedPolicyMADDPGConsistency(
        agents=agents,
        device=device,
        input_channels=input_channels,
        gamma=config['model']['gamma'],
        tau=config['model']['tau'],
        lr=config['model']['learning_rate'],
        critic_lr=config['model']['critic_learning_rate'],
        dropout_rate=0.05,
        weight_decay=config['model']['weight_decay'],
        actor_channels=config['net_arch']['actor_channels'],
        critic_conv_channels=config['net_arch']['critic_conv'],
        critic_mlp_layers=config['net_arch']['critic_mlp'],
        gradient_clip=config['training']['gradient_clip'],
        lambda_temporal=config['model']['smoothness']['lambda_temporal'],
        lambda_spatial=config['model']['smoothness']['lambda_spatial'],
        lambda_zero=config['model']['smoothness']['lambda_zero'],
        consistency_enable=config['model']['smoothness']['consistency']['enable'],
        consistency_lambda=config['model']['smoothness']['consistency']['lambda'],
        consistency_tau=config['model']['smoothness']['consistency']['tau_similarity'],
        consistency_margin=config['model']['smoothness']['consistency']['margin'],
        consistency_boundary_only=config['model']['smoothness']['consistency']['boundary_only'],
        consistency_warmup_steps=config['model']['smoothness']['consistency']['warmup_steps'],
        similarity_dim=config['net_arch']['similarity_dim']
    )

    # Load weights
    maddpg.load_state_dict(checkpoint['maddpg_state_dict'])
    maddpg.actor.eval()
    print("Model loaded successfully!")

    # Print checkpoint info
    if 'episode' in checkpoint:
        print(f"Checkpoint from episode: {checkpoint['episode']}")
    if 'total_timesteps' in checkpoint:
        print(f"Total timesteps: {checkpoint['total_timesteps']}")
    if 'best_reward' in checkpoint:
        print(f"Best reward: {checkpoint['best_reward']:.4f}")

    # Create environment
    print("\nInitializing environment...")
    env = STWParallelEnv(config)

    # Get episode length from config
    episode_length = config['training']['start_episode_length']
    print(f"Episode length: {episode_length} steps")
    print(f"Number of episodes: {num_episodes}")

    # Create results directory
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"evaluation_consistency_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        print(f"Results will be saved to: {results_dir}")

    # Storage for all episodes
    all_episode_metrics = []
    all_episode_rewards = []

    # Run episodes
    for episode in range(num_episodes):
        print("\n" + "="*60)
        print(f"Starting evaluation episode {episode+1}/{num_episodes}...")
        print("="*60)

        # Storage for this episode's metrics
        metrics = {
            'rewards': [],
            'dpdx': [],
            'e_ks': [],
            'uw': [],
            'action_mean': [],
            'action_std': [],
            'action_min': [],
            'action_max': []
        }

        # Storage for point (32, 32) tracking
        # Position (32, 32) is in agent at grid position (4, 4)
        # Agent name: "agent_4_4", index in flat list: 4*8 + 4 = 36
        # Local position within patch: (0, 0)
        point_tracking = {
            'u_32_32': [],
            'w_32_32': [],
            'action_32_32': []
        }
        track_agent_name = "agent_4_4"
        track_agent_idx = 4 * 8 + 4  # 36
        track_local_i = 0  # 32 % 8 = 0
        track_local_j = 0  # 32 % 8 = 0

        # Storage for visualizations (only first episode)
        action_history = []

        obs, info = env.reset()
        episode_reward = 0

        for step in tqdm(range(episode_length), desc=f"Episode {episode+1}/{num_episodes}"):
            # Convert obs dict to tensor [64, 2, 8, 8]
            obs_array = np.stack([obs[agent] for agent in agents])
            obs_torch = torch.FloatTensor(obs_array).to(device)

            # Track point (32, 32) values before action
            u_val = obs[track_agent_name][0, track_local_i, track_local_j]  # u channel
            w_val = obs[track_agent_name][1, track_local_i, track_local_j]  # w channel
            point_tracking['u_32_32'].append(float(u_val))
            point_tracking['w_32_32'].append(float(w_val))

            # Get deterministic actions (no exploration noise)
            with torch.no_grad():
                actions = maddpg.select_actions_batched(obs_torch)  # [64, 8, 8]

            # Track action at point (32, 32)
            action_val = actions[track_agent_idx, track_local_i, track_local_j]
            point_tracking['action_32_32'].append(float(action_val))

            # Store actions for visualization (only first episode)
            if visualize and episode == 0 and (step % 100 == 0 or step < 10):
                action_history.append((step, actions.copy()))

            # Convert to dict for environment
            action_dict = {agent: actions[i] for i, agent in enumerate(agents)}

            # Step environment
            obs, rewards, dones, truncated, infos = env.step(action_dict)

            # Collect metrics
            step_reward = np.mean(list(rewards.values()))
            episode_reward += step_reward

            metrics['rewards'].append(step_reward)
            metrics['dpdx'].append(float(infos['dpdx']))
            metrics['e_ks'].append(float(infos['e_ks']))
            metrics['uw'].append(float(infos['uw']))

            # Action statistics
            metrics['action_mean'].append(np.mean(actions))
            metrics['action_std'].append(np.std(actions))
            metrics['action_min'].append(np.min(actions))
            metrics['action_max'].append(np.max(actions))

            # Print progress
            if step % 100 == 0:
                print(f"Step {step:4d}: dpdx={infos['dpdx']:.6f}, "
                      f"e_ks={infos['e_ks']:.6f}, "
                      f"reward={step_reward:.4f}, "
                      f"action_mean={np.mean(actions):.4f}")

        # Store episode results
        all_episode_metrics.append(metrics)
        all_episode_rewards.append(episode_reward)

        # Create point tracking plot for this episode
        if save_results:
            fig, ax = plt.subplots(figsize=(12, 6))
            steps = np.arange(episode_length)

            ax.plot(steps, point_tracking['u_32_32'], label='u at (32,32)', linewidth=1.5, alpha=0.8)
            ax.plot(steps, point_tracking['w_32_32'], label='w at (32,32)', linewidth=1.5, alpha=0.8)
            ax.plot(steps, point_tracking['action_32_32'], label='action at (32,32)', linewidth=1.5, alpha=0.8)

            ax.set_xlabel('Step', fontsize=12)
            ax.set_ylabel('Value', fontsize=12)
            ax.set_title(f'Time Series at Point (32, 32) - Episode {episode+1}', fontsize=14)
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save to results directory
            point_plot_path = os.path.join(results_dir, f'episode_{episode+1}_point_32_32.png')
            plt.savefig(point_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            # Also save the raw data
            point_data_path = os.path.join(results_dir, f'episode_{episode+1}_point_32_32.npz')
            np.savez(point_data_path, **point_tracking, steps=steps)

        # Print episode summary
        print(f"\nEpisode {episode+1} finished!")
        print(f"  Total episode reward: {episode_reward:.2f}")
        print(f"  Average reward per step: {episode_reward/episode_length:.4f}")
        print(f"  Average dpdx: {np.mean(metrics['dpdx']):.6f} (target: -0.002)")
        print(f"  Average e_ks: {np.mean(metrics['e_ks']):.6f}")
        print(f"  Average uw: {np.mean(metrics['uw']):.6f}")
        print(f"\n  Point (32,32) statistics:")
        print(f"    u:      mean={np.mean(point_tracking['u_32_32']):.4f}, std={np.std(point_tracking['u_32_32']):.4f}")
        print(f"    w:      mean={np.mean(point_tracking['w_32_32']):.4f}, std={np.std(point_tracking['w_32_32']):.4f}")
        print(f"    action: mean={np.mean(point_tracking['action_32_32']):.4f}, std={np.std(point_tracking['action_32_32']):.4f}")

    # Print final results across all episodes
    print("\n" + "="*60)
    print("All episodes completed!")
    print("="*60)
    print(f"\nResults across {num_episodes} episode(s):")
    print(f"  Mean episode reward: {np.mean(all_episode_rewards):.2f} ± {np.std(all_episode_rewards):.2f}")

    # Compute average metrics across all episodes
    avg_dpdx = np.mean([np.mean(ep['dpdx']) for ep in all_episode_metrics])
    avg_e_ks = np.mean([np.mean(ep['e_ks']) for ep in all_episode_metrics])
    avg_uw = np.mean([np.mean(ep['uw']) for ep in all_episode_metrics])

    print(f"\nAverage metrics across all episodes:")
    print(f"  dpdx:  {avg_dpdx:.6f} (target: -0.002)")
    print(f"  e_ks:  {avg_e_ks:.6f}")
    print(f"  uw:    {avg_uw:.6f}")

    if num_episodes > 1:
        std_dpdx = np.std([np.mean(ep['dpdx']) for ep in all_episode_metrics])
        std_e_ks = np.std([np.mean(ep['e_ks']) for ep in all_episode_metrics])
        std_uw = np.std([np.mean(ep['uw']) for ep in all_episode_metrics])
        print(f"\nStandard deviation across episodes:")
        print(f"  dpdx:  {std_dpdx:.6f}")
        print(f"  e_ks:  {std_e_ks:.6f}")
        print(f"  uw:    {std_uw:.6f}")

    # Save results
    if save_results:
        # Save metrics for all episodes
        for ep_idx, ep_metrics in enumerate(all_episode_metrics):
            metrics_path = os.path.join(results_dir, f'episode_{ep_idx+1}_metrics.npz')
            np.savez(metrics_path, **ep_metrics)
        print(f"\nMetrics for {num_episodes} episode(s) saved to: {results_dir}")

        # Save summary as text
        summary_path = os.path.join(results_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Consistency Model Evaluation\n")
            f.write(f"={'='*60}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Config: {config_path}\n")
            f.write(f"Number of episodes: {num_episodes}\n")
            f.write(f"Episode length: {episode_length} steps\n")
            f.write(f"Date: {datetime.now()}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Mean episode reward: {np.mean(all_episode_rewards):.2f} ± {np.std(all_episode_rewards):.2f}\n")
            f.write(f"  Average dpdx: {avg_dpdx:.6f}\n")
            f.write(f"  Average e_ks: {avg_e_ks:.6f}\n")
            f.write(f"  Average uw: {avg_uw:.6f}\n")
            if num_episodes > 1:
                f.write(f"\n  Standard deviations:\n")
                f.write(f"    dpdx: {std_dpdx:.6f}\n")
                f.write(f"    e_ks: {std_e_ks:.6f}\n")
                f.write(f"    uw: {std_uw:.6f}\n")
            f.write(f"\nPer-episode rewards:\n")
            for ep_idx, ep_reward in enumerate(all_episode_rewards):
                f.write(f"  Episode {ep_idx+1}: {ep_reward:.2f}\n")
            f.write(f"\nPoint Tracking:\n")
            f.write(f"  Tracked position: (32, 32) in full 64x64 domain\n")
            f.write(f"  Corresponding to agent: agent_4_4, local position (0, 0)\n")
            f.write(f"  Files saved: episode_N_point_32_32.png and episode_N_point_32_32.npz for each episode\n")
        print(f"Summary saved to: {summary_path}")

    # Create visualizations (using first episode)
    if visualize:
        print("\nCreating visualizations...")

        # Use first episode's metrics for visualization
        first_ep_metrics = all_episode_metrics[0]

        # 1. Metrics over time
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        steps = np.arange(episode_length)

        # dpdx
        axes[0, 0].plot(steps, first_ep_metrics['dpdx'], linewidth=0.8)
        axes[0, 0].axhline(y=-0.002, color='r', linestyle='--', label='Target', alpha=0.7)
        axes[0, 0].axhline(y=-0.0042, color='orange', linestyle='--', label='Uncontrolled', alpha=0.7)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('dpdx')
        axes[0, 0].set_title('Pressure Gradient (Episode 1)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # e_ks
        axes[0, 1].plot(steps, first_ep_metrics['e_ks'], linewidth=0.8, color='green')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('e_ks')
        axes[0, 1].set_title('Turbulent Kinetic Energy (Episode 1)')
        axes[0, 1].grid(True, alpha=0.3)

        # Rewards
        axes[1, 0].plot(steps, first_ep_metrics['rewards'], linewidth=0.8, color='purple')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Instantaneous Reward (Episode 1)')
        axes[1, 0].grid(True, alpha=0.3)

        # Action statistics
        axes[1, 1].plot(steps, first_ep_metrics['action_mean'], label='Mean', linewidth=0.8)
        axes[1, 1].fill_between(steps,
                                np.array(first_ep_metrics['action_mean']) - np.array(first_ep_metrics['action_std']),
                                np.array(first_ep_metrics['action_mean']) + np.array(first_ep_metrics['action_std']),
                                alpha=0.3, label='±1 std')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Action Value')
        axes[1, 1].set_title('Action Statistics (Episode 1)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_results:
            metrics_plot_path = os.path.join(results_dir, 'metrics_over_time.png')
            plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to: {metrics_plot_path}")
        plt.close()

        # 2. Action field snapshots
        if action_history:
            n_snapshots = min(6, len(action_history))
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            axes = axes.flatten()

            for idx in range(n_snapshots):
                step, actions_snap = action_history[idx]

                # Reconstruct 64x64 action field from 64 agents × 8×8 patches
                action_field = np.zeros((64, 64))
                agent_idx = 0
                for i in range(8):
                    for j in range(8):
                        action_field[i*8:(i+1)*8, j*8:(j+1)*8] = actions_snap[agent_idx]
                        agent_idx += 1

                # Plot
                im = axes[idx].imshow(action_field, cmap='seismic', vmin=-1, vmax=1,
                                     origin='lower', aspect='auto')
                axes[idx].set_title(f'Step {step}')
                axes[idx].set_xlabel('j (spanwise)')
                axes[idx].set_ylabel('i (streamwise)')

                # Add grid lines to show 8x8 patches
                for k in range(0, 65, 8):
                    axes[idx].axhline(k - 0.5, color='black', linewidth=0.5, alpha=0.3)
                    axes[idx].axvline(k - 0.5, color='black', linewidth=0.5, alpha=0.3)

            # Hide unused subplots
            for idx in range(n_snapshots, 6):
                axes[idx].axis('off')

            # Add colorbar
            fig.colorbar(im, ax=axes, orientation='horizontal',
                        fraction=0.05, pad=0.05, label='Action value')

            plt.suptitle('Action Field Snapshots', fontsize=14)
            plt.tight_layout()

            if save_results:
                actions_plot_path = os.path.join(results_dir, 'action_snapshots.png')
                plt.savefig(actions_plot_path, dpi=300, bbox_inches='tight')
                print(f"Action snapshots saved to: {actions_plot_path}")
            plt.close()

    print("\nEvaluation complete!")
    if save_results:
        print(f"\nAll results saved to: {results_dir}")
        print(f"  - Episode metrics: episode_N_metrics.npz")
        print(f"  - Point (32,32) tracking: episode_N_point_32_32.png and .npz")
        print(f"  - Summary: summary.txt")
        if visualize:
            print(f"  - Visualizations: metrics_over_time.png, action_snapshots.png")

    return all_episode_metrics, all_episode_rewards


def main():
    parser = argparse.ArgumentParser(
        description='Test consistency model on episodes'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=1,
        help='Number of episodes to run (default: 1)'
    )
    parser.add_argument(
        '--no_save',
        action='store_true',
        help='Do not save results'
    )
    parser.add_argument(
        '--no_visualize',
        action='store_true',
        help='Do not create visualizations'
    )

    args = parser.parse_args()

    # Run evaluation
    test_consistency_model(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.num_episodes,
        save_results=not args.no_save,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    main()
