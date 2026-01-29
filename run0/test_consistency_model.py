"""
Test script for evaluating a trained consistency model on a single episode.

Usage:
    python test_consistency_model.py --checkpoint path/to/best_model.pt
    python test_consistency_model.py  # Uses default best_model.pt location
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
                           episode_length=None, save_results=True, visualize=True):
    """
    Test a trained consistency model on one new episode.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        episode_length: Length of episode (uses config default if None)
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

    # Initialize model with same parameters as training
    print("Initializing model...")
    maddpg = SharedPolicyMADDPGConsistency(
        agents=agents,
        device=device,
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

    # Set episode length
    if episode_length is None:
        episode_length = config['training']['start_episode_length']
    print(f"Episode length: {episode_length} steps")

    # Create results directory
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"evaluation_consistency_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        print(f"Results will be saved to: {results_dir}")

    # Storage for metrics
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

    # Storage for visualizations
    action_history = []

    # Run episode
    print("\n" + "="*60)
    print("Starting evaluation episode...")
    print("="*60)

    obs, info = env.reset()
    episode_reward = 0

    for step in tqdm(range(episode_length), desc="Running episode"):
        # Convert obs dict to tensor [64, 2, 8, 8]
        obs_array = np.stack([obs[agent] for agent in agents])
        obs_torch = torch.FloatTensor(obs_array).to(device)

        # Get deterministic actions (no exploration noise)
        with torch.no_grad():
            actions = maddpg.select_actions_batched(obs_torch)  # [64, 8, 8]

        # Store actions for visualization
        if visualize and (step % 100 == 0 or step < 10):
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

    # Print final results
    print("\n" + "="*60)
    print("Episode finished!")
    print("="*60)
    print(f"Total episode reward: {episode_reward:.2f}")
    print(f"Average reward per step: {episode_reward/episode_length:.4f}")
    print(f"\nFinal metrics:")
    print(f"  dpdx:  {metrics['dpdx'][-1]:.6f} (target: -0.002)")
    print(f"  e_ks:  {metrics['e_ks'][-1]:.6f}")
    print(f"  uw:    {metrics['uw'][-1]:.6f}")
    print(f"\nAction statistics (final step):")
    print(f"  Mean: {metrics['action_mean'][-1]:.4f}")
    print(f"  Std:  {metrics['action_std'][-1]:.4f}")
    print(f"  Min:  {metrics['action_min'][-1]:.4f}")
    print(f"  Max:  {metrics['action_max'][-1]:.4f}")

    # Episode-wide statistics
    print(f"\nEpisode-wide averages:")
    print(f"  dpdx:  {np.mean(metrics['dpdx']):.6f}")
    print(f"  e_ks:  {np.mean(metrics['e_ks']):.6f}")
    print(f"  uw:    {np.mean(metrics['uw']):.6f}")

    # Save results
    if save_results:
        # Save metrics as numpy arrays
        metrics_path = os.path.join(results_dir, 'metrics.npz')
        np.savez(metrics_path, **metrics)
        print(f"\nMetrics saved to: {metrics_path}")

        # Save summary as text
        summary_path = os.path.join(results_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Consistency Model Evaluation\n")
            f.write(f"={'='*60}\n")
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Config: {config_path}\n")
            f.write(f"Episode length: {episode_length}\n")
            f.write(f"Date: {datetime.now()}\n\n")
            f.write(f"Results:\n")
            f.write(f"  Total episode reward: {episode_reward:.2f}\n")
            f.write(f"  Average reward: {episode_reward/episode_length:.4f}\n")
            f.write(f"  Average dpdx: {np.mean(metrics['dpdx']):.6f}\n")
            f.write(f"  Average e_ks: {np.mean(metrics['e_ks']):.6f}\n")
            f.write(f"  Average uw: {np.mean(metrics['uw']):.6f}\n")
        print(f"Summary saved to: {summary_path}")

    # Create visualizations
    if visualize:
        print("\nCreating visualizations...")

        # 1. Metrics over time
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        steps = np.arange(episode_length)

        # dpdx
        axes[0, 0].plot(steps, metrics['dpdx'], linewidth=0.8)
        axes[0, 0].axhline(y=-0.002, color='r', linestyle='--', label='Target', alpha=0.7)
        axes[0, 0].axhline(y=-0.0042, color='orange', linestyle='--', label='Uncontrolled', alpha=0.7)
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('dpdx')
        axes[0, 0].set_title('Pressure Gradient')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # e_ks
        axes[0, 1].plot(steps, metrics['e_ks'], linewidth=0.8, color='green')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('e_ks')
        axes[0, 1].set_title('Turbulent Kinetic Energy')
        axes[0, 1].grid(True, alpha=0.3)

        # Rewards
        axes[1, 0].plot(steps, metrics['rewards'], linewidth=0.8, color='purple')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Reward')
        axes[1, 0].set_title('Instantaneous Reward')
        axes[1, 0].grid(True, alpha=0.3)

        # Action statistics
        axes[1, 1].plot(steps, metrics['action_mean'], label='Mean', linewidth=0.8)
        axes[1, 1].fill_between(steps,
                                np.array(metrics['action_mean']) - np.array(metrics['action_std']),
                                np.array(metrics['action_mean']) + np.array(metrics['action_std']),
                                alpha=0.3, label='±1 std')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Action Value')
        axes[1, 1].set_title('Action Statistics')
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
        print(f"All results saved to: {results_dir}")

    return metrics, episode_reward


def main():
    parser = argparse.ArgumentParser(
        description='Test consistency model on a single episode'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints_consistency/best_model.pt',
        help='Path to model checkpoint (default: checkpoints_consistency/best_model.pt)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config_consistency.yaml',
        help='Path to configuration file (default: config_consistency.yaml)'
    )
    parser.add_argument(
        '--episode_length',
        type=int,
        default=None,
        help='Episode length (if not specified, uses config default)'
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
        episode_length=args.episode_length,
        save_results=not args.no_save,
        visualize=not args.no_visualize
    )


if __name__ == "__main__":
    main()
