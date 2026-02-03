"""
Test trained MADDPG policy for spatial and temporal smoothness analysis.

This script:
1. Loads a trained policy checkpoint
2. Runs N test episodes without exploration noise
3. Collects spatial and temporal action statistics
4. Performs FFT analysis to identify dominant frequencies
5. Compares observations (inputs) and actions (outputs) at specific indices

Usage:
    python test_policy_smoothness.py --checkpoint logs_consistency/maddpg_consistency_agents64_20260130_061535/checkpoint_165.pt --n_episodes 5
"""

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import fft
from scipy.stats import describe
import h5py
import json
import argparse
from pathlib import Path

# Add run0 directory to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'run0'))

# Import your models and environment
from models_consistency import CNNActorWithSimilarity, SharedPolicyMADDPGConsistency
from stwEnv_pettingzoo import STWParallelEnv


class PolicySmoothnessAnalyzer:
    def __init__(self, checkpoint_path, config_path=None, n_episodes=5, output_dir='smoothness_analysis'):
        """
        Initialize the policy smoothness analyzer.

        Args:
            checkpoint_path: Path to the trained model checkpoint
            config_path: Path to configuration JSON file (if None, will try to find it)
            n_episodes: Number of test episodes to run
            output_dir: Directory to save analysis results
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.n_episodes = n_episodes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Find config file if not provided
        if config_path is None:
            # Try to find config file in same directory as checkpoint
            checkpoint_dir = self.checkpoint_path.parent
            config_files = list(checkpoint_dir.glob('*_config.json'))
            if config_files:
                self.config_path = config_files[0]
            else:
                self.config_path = None
        else:
            self.config_path = Path(config_path)

        # Storage for collected data
        self.episodes_data = []
        self.spatial_stats = {}
        self.temporal_stats = {}
        self.frequency_analysis = {}

        # Model components
        self.actor = None
        self.config = None

    def load_policy(self):
        """Load the trained policy from checkpoint."""
        print(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')

        # Extract relevant info
        self.config = checkpoint.get('config', {})
        self.actor_state_dict = checkpoint['actor_state_dict']

        # TODO: Initialize your actor network here with the config
        # self.actor = YourActorNetwork(self.config)
        # self.actor.load_state_dict(self.actor_state_dict)
        # self.actor.eval()

        print(f"Policy loaded successfully")
        return checkpoint

    def run_test_episodes(self, env):
        """
        Run test episodes and collect action/observation data.

        Args:
            env: Your environment instance
        """
        print(f"\nRunning {self.n_episodes} test episodes...")

        for episode in range(self.n_episodes):
            obs = env.reset()
            done = False
            step = 0

            episode_data = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'timesteps': []
            }

            while not done:
                # Get action from policy (no exploration noise)
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    action = self.actor(obs_tensor).cpu().numpy().squeeze()

                # Store data
                episode_data['observations'].append(obs.copy())
                episode_data['actions'].append(action.copy())
                episode_data['timesteps'].append(step)

                # Step environment
                obs, reward, done, info = env.step(action)
                episode_data['rewards'].append(reward)

                step += 1

            # Convert lists to arrays
            for key in ['observations', 'actions', 'rewards']:
                episode_data[key] = np.array(episode_data[key])

            self.episodes_data.append(episode_data)
            print(f"  Episode {episode+1}/{self.n_episodes} completed: "
                  f"{len(episode_data['actions'])} steps, "
                  f"mean reward: {np.mean(episode_data['rewards']):.2f}")

    def compute_spatial_statistics(self):
        """
        Compute spatial variance and smoothness metrics.
        Actions shape: [time_steps, n_agents, action_dim]
        """
        print("\nComputing spatial statistics...")

        for ep_idx, ep_data in enumerate(self.episodes_data):
            actions = ep_data['actions']  # [T, N, A]
            T, N, A = actions.shape

            # Spatial variance at each timestep (variance across agents)
            spatial_var_time = np.var(actions, axis=1)  # [T, A]
            spatial_std_time = np.std(actions, axis=1)  # [T, A]

            # Spatial gradients (finite differences between neighboring agents)
            # Assuming agents are ordered spatially
            spatial_gradients = np.diff(actions, axis=1)  # [T, N-1, A]
            spatial_grad_magnitude = np.linalg.norm(spatial_gradients, axis=-1)  # [T, N-1]

            # Statistics
            self.spatial_stats[f'episode_{ep_idx}'] = {
                'variance_over_time': spatial_var_time,
                'std_over_time': spatial_std_time,
                'mean_variance': np.mean(spatial_var_time),
                'max_variance': np.max(spatial_var_time),
                'min_variance': np.min(spatial_var_time),
                'spatial_gradient_magnitude': spatial_grad_magnitude,
                'mean_spatial_gradient': np.mean(spatial_grad_magnitude),
                'max_spatial_gradient': np.max(spatial_grad_magnitude),
            }

        # Aggregate across episodes
        all_mean_vars = [stats['mean_variance'] for stats in self.spatial_stats.values()]
        all_mean_grads = [stats['mean_spatial_gradient'] for stats in self.spatial_stats.values()]

        print(f"  Spatial variance (mean across episodes): {np.mean(all_mean_vars):.6f} ± {np.std(all_mean_vars):.6f}")
        print(f"  Spatial gradient (mean across episodes): {np.mean(all_mean_grads):.6f} ± {np.std(all_mean_grads):.6f}")

        return self.spatial_stats

    def compute_temporal_statistics(self):
        """
        Compute temporal variance and smoothness metrics.
        Actions shape: [time_steps, n_agents, action_dim]
        """
        print("\nComputing temporal statistics...")

        for ep_idx, ep_data in enumerate(self.episodes_data):
            actions = ep_data['actions']  # [T, N, A]
            T, N, A = actions.shape

            # Temporal variance for each agent (variance across time)
            temporal_var_agents = np.var(actions, axis=0)  # [N, A]
            temporal_std_agents = np.std(actions, axis=0)  # [N, A]

            # Temporal gradients (finite differences between consecutive timesteps)
            temporal_gradients = np.diff(actions, axis=0)  # [T-1, N, A]
            temporal_grad_magnitude = np.linalg.norm(temporal_gradients, axis=-1)  # [T-1, N]

            # Acceleration (second derivative in time)
            temporal_acceleration = np.diff(temporal_gradients, axis=0)  # [T-2, N, A]
            temporal_accel_magnitude = np.linalg.norm(temporal_acceleration, axis=-1)  # [T-2, N]

            # Statistics
            self.temporal_stats[f'episode_{ep_idx}'] = {
                'variance_per_agent': temporal_var_agents,
                'std_per_agent': temporal_std_agents,
                'mean_variance': np.mean(temporal_var_agents),
                'max_variance': np.max(temporal_var_agents),
                'min_variance': np.min(temporal_var_agents),
                'temporal_gradient_magnitude': temporal_grad_magnitude,
                'mean_temporal_gradient': np.mean(temporal_grad_magnitude),
                'max_temporal_gradient': np.max(temporal_grad_magnitude),
                'temporal_acceleration_magnitude': temporal_accel_magnitude,
                'mean_temporal_acceleration': np.mean(temporal_accel_magnitude),
            }

        # Aggregate across episodes
        all_mean_vars = [stats['mean_variance'] for stats in self.temporal_stats.values()]
        all_mean_grads = [stats['mean_temporal_gradient'] for stats in self.temporal_stats.values()]
        all_mean_accels = [stats['mean_temporal_acceleration'] for stats in self.temporal_stats.values()]

        print(f"  Temporal variance (mean across episodes): {np.mean(all_mean_vars):.6f} ± {np.std(all_mean_vars):.6f}")
        print(f"  Temporal gradient (mean across episodes): {np.mean(all_mean_grads):.6f} ± {np.std(all_mean_grads):.6f}")
        print(f"  Temporal acceleration (mean across episodes): {np.mean(all_mean_accels):.6f} ± {np.std(all_mean_accels):.6f}")

        return self.temporal_stats

    def compute_frequency_analysis(self, dt=1.0):
        """
        Perform FFT analysis to identify dominant frequencies in space and time.

        Args:
            dt: Time step for temporal FFT (in seconds or simulation units)
        """
        print("\nPerforming frequency analysis...")

        for ep_idx, ep_data in enumerate(self.episodes_data):
            actions = ep_data['actions']  # [T, N, A]
            T, N, A = actions.shape

            analysis = {}

            # Temporal FFT (for each agent, averaged across action dimensions)
            temporal_fft_results = []
            for agent_idx in range(N):
                agent_actions = actions[:, agent_idx, :].mean(axis=-1)  # [T]
                fft_result = np.abs(fft.rfft(agent_actions))
                freqs = fft.rfftfreq(T, d=dt)
                temporal_fft_results.append(fft_result)

            temporal_fft_mean = np.mean(temporal_fft_results, axis=0)
            temporal_freqs = fft.rfftfreq(T, d=dt)

            # Find dominant temporal frequencies (top 5)
            top_temporal_indices = np.argsort(temporal_fft_mean)[-5:][::-1]
            dominant_temporal_freqs = temporal_freqs[top_temporal_indices]
            dominant_temporal_powers = temporal_fft_mean[top_temporal_indices]

            analysis['temporal'] = {
                'frequencies': temporal_freqs,
                'fft_magnitude': temporal_fft_mean,
                'dominant_frequencies': dominant_temporal_freqs,
                'dominant_powers': dominant_temporal_powers,
            }

            # Spatial FFT (at each timestep, averaged across action dimensions)
            spatial_fft_results = []
            for t in range(T):
                time_actions = actions[t, :, :].mean(axis=-1)  # [N]
                fft_result = np.abs(fft.rfft(time_actions))
                spatial_fft_results.append(fft_result)

            spatial_fft_mean = np.mean(spatial_fft_results, axis=0)
            spatial_freqs = fft.rfftfreq(N)  # Normalized spatial frequency

            # Find dominant spatial frequencies (top 5)
            top_spatial_indices = np.argsort(spatial_fft_mean)[-5:][::-1]
            dominant_spatial_freqs = spatial_freqs[top_spatial_indices]
            dominant_spatial_powers = spatial_fft_mean[top_spatial_indices]

            analysis['spatial'] = {
                'frequencies': spatial_freqs,
                'fft_magnitude': spatial_fft_mean,
                'dominant_frequencies': dominant_spatial_freqs,
                'dominant_powers': dominant_spatial_powers,
            }

            self.frequency_analysis[f'episode_{ep_idx}'] = analysis

            print(f"\n  Episode {ep_idx}:")
            print(f"    Dominant temporal frequencies: {dominant_temporal_freqs}")
            print(f"    Dominant spatial wavenumbers: {dominant_spatial_freqs}")

        return self.frequency_analysis

    def compare_input_output_at_index(self, agent_idx=0):
        """
        Compare observations (inputs) and actions (outputs) at a specific agent index.

        Args:
            agent_idx: Index of the agent to analyze
        """
        print(f"\nComparing input/output at agent index {agent_idx}...")

        comparisons = []

        for ep_idx, ep_data in enumerate(self.episodes_data):
            observations = ep_data['observations']  # [T, N, obs_dim]
            actions = ep_data['actions']  # [T, N, action_dim]

            # Extract data for specific agent
            agent_obs = observations[:, agent_idx, :]  # [T, obs_dim]
            agent_actions = actions[:, agent_idx, :]  # [T, action_dim]

            # Compute statistics
            comparison = {
                'episode': ep_idx,
                'agent_idx': agent_idx,
                'observation_mean': np.mean(agent_obs, axis=0),
                'observation_std': np.std(agent_obs, axis=0),
                'action_mean': np.mean(agent_actions, axis=0),
                'action_std': np.std(agent_actions, axis=0),
                'obs_temporal_gradient': np.mean(np.abs(np.diff(agent_obs, axis=0)), axis=0),
                'action_temporal_gradient': np.mean(np.abs(np.diff(agent_actions, axis=0)), axis=0),
                'observations': agent_obs,
                'actions': agent_actions,
            }

            comparisons.append(comparison)

            print(f"\n  Episode {ep_idx}:")
            print(f"    Observation mean: {comparison['observation_mean']}")
            print(f"    Action mean: {comparison['action_mean']}")
            print(f"    Observation temporal gradient: {comparison['obs_temporal_gradient']}")
            print(f"    Action temporal gradient: {comparison['action_temporal_gradient']}")

        return comparisons

    def plot_results(self):
        """Generate comprehensive visualization plots."""
        print("\nGenerating plots...")

        # 1. Spatial variance over time
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        for ep_idx, stats in self.spatial_stats.items():
            axes[0, 0].plot(stats['variance_over_time'], alpha=0.7, label=ep_idx)
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Spatial Variance')
        axes[0, 0].set_title('Spatial Variance Over Time')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Temporal variance per agent (averaged across episodes)
        all_temporal_vars = []
        for stats in self.temporal_stats.values():
            all_temporal_vars.append(stats['variance_per_agent'].mean(axis=-1))  # Average across action dims
        mean_temporal_var = np.mean(all_temporal_vars, axis=0)
        std_temporal_var = np.std(all_temporal_vars, axis=0)

        agents = np.arange(len(mean_temporal_var))
        axes[0, 1].plot(agents, mean_temporal_var, 'b-', linewidth=2)
        axes[0, 1].fill_between(agents,
                                mean_temporal_var - std_temporal_var,
                                mean_temporal_var + std_temporal_var,
                                alpha=0.3)
        axes[0, 1].set_xlabel('Agent Index')
        axes[0, 1].set_ylabel('Temporal Variance')
        axes[0, 1].set_title('Temporal Variance per Agent')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Frequency analysis (temporal)
        ep_key = list(self.frequency_analysis.keys())[0]
        temporal_analysis = self.frequency_analysis[ep_key]['temporal']
        axes[1, 0].semilogy(temporal_analysis['frequencies'],
                           temporal_analysis['fft_magnitude'])
        axes[1, 0].scatter(temporal_analysis['dominant_frequencies'],
                          temporal_analysis['dominant_powers'],
                          c='red', s=100, zorder=5, label='Dominant')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('FFT Magnitude')
        axes[1, 0].set_title('Temporal Frequency Spectrum')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Frequency analysis (spatial)
        spatial_analysis = self.frequency_analysis[ep_key]['spatial']
        axes[1, 1].semilogy(spatial_analysis['frequencies'],
                           spatial_analysis['fft_magnitude'])
        axes[1, 1].scatter(spatial_analysis['dominant_frequencies'],
                          spatial_analysis['dominant_powers'],
                          c='red', s=100, zorder=5, label='Dominant')
        axes[1, 1].set_xlabel('Wavenumber (normalized)')
        axes[1, 1].set_ylabel('FFT Magnitude')
        axes[1, 1].set_title('Spatial Frequency Spectrum')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'smoothness_analysis.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {self.output_dir / 'smoothness_analysis.png'}")

        # 5. Input-Output comparison at specific agent
        comparison_data = self.compare_input_output_at_index(agent_idx=0)

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        for comp in comparison_data:
            ep_idx = comp['episode']
            t = np.arange(len(comp['observations']))

            # Plot first observation dimension
            axes[0].plot(t, comp['observations'][:, 0], alpha=0.7, label=f'Ep {ep_idx}')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Observation (dim 0)')
        axes[0].set_title('Input: Observation at Agent 0 (first dimension)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for comp in comparison_data:
            ep_idx = comp['episode']
            t = np.arange(len(comp['actions']))

            # Plot first action dimension
            axes[1].plot(t, comp['actions'][:, 0], alpha=0.7, label=f'Ep {ep_idx}')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Action (dim 0)')
        axes[1].set_title('Output: Action at Agent 0 (first dimension)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'input_output_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  Saved: {self.output_dir / 'input_output_comparison.png'}")

        plt.close('all')

    def save_results(self):
        """Save analysis results to files."""
        print("\nSaving results...")

        # Save as HDF5 for easy loading later
        h5_path = self.output_dir / 'smoothness_analysis.h5'
        with h5py.File(h5_path, 'w') as f:
            # Episodes data
            for ep_idx, ep_data in enumerate(self.episodes_data):
                grp = f.create_group(f'episode_{ep_idx}')
                grp.create_dataset('observations', data=ep_data['observations'])
                grp.create_dataset('actions', data=ep_data['actions'])
                grp.create_dataset('rewards', data=ep_data['rewards'])

            # Spatial stats
            spatial_grp = f.create_group('spatial_stats')
            for ep_key, stats in self.spatial_stats.items():
                ep_grp = spatial_grp.create_group(ep_key)
                for key, value in stats.items():
                    if isinstance(value, np.ndarray):
                        ep_grp.create_dataset(key, data=value)
                    else:
                        ep_grp.attrs[key] = value

            # Temporal stats
            temporal_grp = f.create_group('temporal_stats')
            for ep_key, stats in self.temporal_stats.items():
                ep_grp = temporal_grp.create_group(ep_key)
                for key, value in stats.items():
                    if isinstance(value, np.ndarray):
                        ep_grp.create_dataset(key, data=value)
                    else:
                        ep_grp.attrs[key] = value

            # Frequency analysis
            freq_grp = f.create_group('frequency_analysis')
            for ep_key, analysis in self.frequency_analysis.items():
                ep_grp = freq_grp.create_group(ep_key)
                for domain in ['spatial', 'temporal']:
                    domain_grp = ep_grp.create_group(domain)
                    for key, value in analysis[domain].items():
                        domain_grp.create_dataset(key, data=value)

        print(f"  Saved: {h5_path}")

        # Save summary as JSON
        summary = {
            'n_episodes': self.n_episodes,
            'spatial_variance_mean': float(np.mean([s['mean_variance'] for s in self.spatial_stats.values()])),
            'spatial_variance_std': float(np.std([s['mean_variance'] for s in self.spatial_stats.values()])),
            'temporal_variance_mean': float(np.mean([s['mean_variance'] for s in self.temporal_stats.values()])),
            'temporal_variance_std': float(np.std([s['mean_variance'] for s in self.temporal_stats.values()])),
            'spatial_gradient_mean': float(np.mean([s['mean_spatial_gradient'] for s in self.spatial_stats.values()])),
            'temporal_gradient_mean': float(np.mean([s['mean_temporal_gradient'] for s in self.temporal_stats.values()])),
        }

        json_path = self.output_dir / 'summary.json'
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"  Saved: {json_path}")

    def run_full_analysis(self, env):
        """Run complete analysis pipeline."""
        print("="*80)
        print("POLICY SMOOTHNESS ANALYSIS")
        print("="*80)

        # Load policy
        self.load_policy()

        # Run test episodes
        self.run_test_episodes(env)

        # Compute statistics
        self.compute_spatial_statistics()
        self.compute_temporal_statistics()
        self.compute_frequency_analysis()

        # Compare input/output
        self.compare_input_output_at_index(agent_idx=0)

        # Visualize
        self.plot_results()

        # Save results
        self.save_results()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Test policy smoothness')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to policy checkpoint')
    parser.add_argument('--n_episodes', type=int, default=5,
                       help='Number of test episodes')
    parser.add_argument('--output_dir', type=str, default='smoothness_analysis',
                       help='Output directory for results')
    parser.add_argument('--dt', type=float, default=1.0,
                       help='Time step for temporal frequency analysis')

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = PolicySmoothnessAnalyzer(
        checkpoint_path=args.checkpoint,
        n_episodes=args.n_episodes,
        output_dir=args.output_dir
    )

    # TODO: Initialize your environment here
    # env = YourEnvironment(config)

    # Run analysis
    # analyzer.run_full_analysis(env)

    print("\nNOTE: You need to implement:")
    print("1. Your environment initialization")
    print("2. Your actor network architecture")
    print("3. Loading the actor properly from checkpoint")


if __name__ == '__main__':
    main()
