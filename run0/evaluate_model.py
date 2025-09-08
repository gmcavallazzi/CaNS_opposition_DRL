import os
import torch
import numpy as np
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import csv
from tqdm import tqdm

from models import Actor
from utils import load_config
from stwEnv import MultiAgentSTW

class ModelEvaluator:
    def __init__(self, model_path, config_path='config.yaml', device=None):
        """
        Evaluator for running a trained model deterministically
        
        Args:
            model_path: Path to the saved model (best_model.pt or final_model.pt)
            config_path: Path to configuration file
            device: Device to run evaluation on
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize environment
        self.env = MultiAgentSTW(self.config)
        self.grid_i = self.config['grid']['target']['i']
        self.grid_j = self.config['grid']['target']['j']
        
        # Load model
        self.load_model()
        
        # Set up results directory
        self.results_dir = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_model(self):
        """Load the trained actor model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract observation shape if available in checkpoint
        if 'obs_shape' in checkpoint:
            self.obs_shape = checkpoint['obs_shape']
            print(f"Using observation shape from checkpoint: {self.obs_shape}")
        else:
            # Default observation shape based on halo parameter
            halo = self.config.get('halo', 1)
            obs_height = 2 * halo + 1 if halo > 0 else 1
            obs_width = 2 * halo + 1 if halo > 0 else 1
            self.obs_shape = (obs_height, obs_width, 2)
            print(f"Using default observation shape: {self.obs_shape}")
        
        # Initialize actor model
        self.actor = Actor(obs_shape=self.obs_shape).to(self.device)
        
        # Load state dict
        if 'actor_state_dict' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            print(f"Loaded actor weights from checkpoint")
        else:
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model weights from checkpoint")
        
        # Set to evaluation mode
        self.actor.eval()
        
    def run_evaluation(self, num_episodes=1, episode_length=None, save_data=True, render=False):
        """
        Run deterministic evaluation of the trained policy
        
        Args:
            num_episodes: Number of episodes to run
            episode_length: Length of each episode (if None, uses config value)
            save_data: Whether to save evaluation data
            render: Whether to render/visualize during evaluation
        """
        # Use specified episode length or from config
        episode_length = episode_length or self.config['training']['end_episode_length']
        
        # Prepare CSV file for saving results
        if save_data:
            csv_path = os.path.join(self.results_dir, 'evaluation_metrics.csv')
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                'Episode', 'Step', 'Reward', 'dpdx', 'e_ks', 'uw', 
                'Action_Mean', 'Action_Std', 'Action_Min', 'Action_Max'
            ])
        
        # Track metrics for all episodes
        all_rewards = []
        all_dpdx = []
        all_eks = []
        all_uw = []
        
        # Run evaluation episodes
        try:
            for episode in range(num_episodes):
                print(f"\nRunning evaluation episode {episode+1}/{num_episodes}")
                
                # Reset environment
                obs, info = self.env.reset()
                episode_rewards = []
                episode_dpdx = []
                episode_eks = []
                episode_uw = []
                episode_actions = []
                
                # Run episode
                for step in tqdm(range(episode_length)):
                    # Convert observation to tensor
                    obs_torch = torch.FloatTensor(obs).to(self.device)
                    
                    # Get deterministic actions (no exploration noise)
                    with torch.no_grad():
                        actions_torch = self.actor(obs_torch)
                    actions = actions_torch.cpu().numpy()
                    
                    # Log action statistics
                    action_mean = np.mean(actions)
                    action_std = np.std(actions)
                    action_min = np.min(actions)
                    action_max = np.max(actions)
                    episode_actions.append(actions)
                    
                    # Step environment
                    next_obs, rewards, done, truncated, info = self.env.step(actions.flatten())
                    
                    # Update observation
                    obs = next_obs
                    
                    # Record metrics
                    mean_reward = np.mean(rewards)
                    episode_rewards.append(mean_reward)
                    episode_dpdx.append(float(info['dpdx']))
                    episode_eks.append(float(info['e_ks']))
                    episode_uw.append(float(info['uw']))
                    
                    # Log step results to CSV
                    if save_data:
                        csv_writer.writerow([
                            episode, step, mean_reward, 
                            float(info['dpdx']), float(info['e_ks']), float(info['uw']),
                            action_mean, action_std, action_min, action_max
                        ])
                        csv_file.flush()
                    
                    # Optional rendering logic here if needed
                    if render:
                        self.render_step(step, actions, info)
                    
                    # Break if done
                    if done:
                        break
                
                # Save episode data arrays
                if save_data:
                    episode_path = os.path.join(self.results_dir, f'episode_{episode}')
                    os.makedirs(episode_path, exist_ok=True)
                    
                    np.save(os.path.join(episode_path, 'rewards.npy'), np.array(episode_rewards))
                    np.save(os.path.join(episode_path, 'dpdx.npy'), np.array(episode_dpdx))
                    np.save(os.path.join(episode_path, 'e_ks.npy'), np.array(episode_eks))
                    np.save(os.path.join(episode_path, 'uw.npy'), np.array(episode_uw))
                    np.save(os.path.join(episode_path, 'actions.npy'), np.array(episode_actions))
                
                # Plot episode metrics
                self.plot_episode_metrics(episode, episode_rewards, episode_dpdx, episode_eks, episode_uw)
                
                # Store episode metrics for overall statistics
                all_rewards.extend(episode_rewards)
                all_dpdx.extend(episode_dpdx)
                all_eks.extend(episode_eks)
                all_uw.extend(episode_uw)
                
                # Print episode summary
                print(f"Episode {episode+1} Summary:")
                print(f"  Mean Reward: {np.mean(episode_rewards):.4f}")
                print(f"  Mean dpdx: {np.mean(episode_dpdx):.6f}")
                print(f"  Mean e_ks: {np.mean(episode_eks):.2f}")
                print(f"  Mean uw: {np.mean(episode_uw):.6f}")
        
        finally:
            # Close CSV file if open
            if save_data and 'csv_file' in locals():
                csv_file.close()
            
            # Close environment
            self.env.close()
        
        # Print overall statistics
        print("\nOverall Evaluation Results:")
        print(f"  Mean Reward: {np.mean(all_rewards):.4f}")
        print(f"  Mean dpdx: {np.mean(all_dpdx):.6f}")
        print(f"  Mean e_ks: {np.mean(all_eks):.2f}")
        print(f"  Mean uw: {np.mean(all_uw):.6f}")
        
        # Save overall metrics plot
        self.plot_overall_metrics(all_rewards, all_dpdx, all_eks, all_uw)
        
        return {
            'rewards': np.array(all_rewards),
            'dpdx': np.array(all_dpdx),
            'e_ks': np.array(all_eks),
            'uw': np.array(all_uw)
        }
    
    def plot_episode_metrics(self, episode, rewards, dpdx, e_ks, uw):
        """Plot metrics for a single episode"""
        episode_path = os.path.join(self.results_dir, f'episode_{episode}')
        os.makedirs(episode_path, exist_ok=True)
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axs[0, 0].plot(rewards)
        axs[0, 0].set_title('Rewards')
        axs[0, 0].set_xlabel('Step')
        axs[0, 0].set_ylabel('Reward')
        
        # Plot dpdx
        axs[0, 1].plot(dpdx)
        axs[0, 1].set_title('Pressure Gradient (dpdx)')
        axs[0, 1].set_xlabel('Step')
        axs[0, 1].set_ylabel('dpdx')
        
        # Plot e_ks
        axs[1, 0].plot(e_ks)
        axs[1, 0].set_title('Turbulence Kinetic Energy (e_ks)')
        axs[1, 0].set_xlabel('Step')
        axs[1, 0].set_ylabel('e_ks')
        
        # Plot uw
        axs[1, 1].plot(uw)
        axs[1, 1].set_title('Reynolds Stress (uw)')
        axs[1, 1].set_xlabel('Step')
        axs[1, 1].set_ylabel('uw')
        
        plt.tight_layout()
        plt.savefig(os.path.join(episode_path, 'metrics.png'))
        plt.close()
    
    def plot_overall_metrics(self, rewards, dpdx, e_ks, uw):
        """Plot overall metrics across all episodes"""
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axs[0, 0].plot(rewards)
        axs[0, 0].set_title('Rewards (All Episodes)')
        axs[0, 0].set_xlabel('Step')
        axs[0, 0].set_ylabel('Reward')
        
        # Plot dpdx
        axs[0, 1].plot(dpdx)
        axs[0, 1].set_title('Pressure Gradient (All Episodes)')
        axs[0, 1].set_xlabel('Step')
        axs[0, 1].set_ylabel('dpdx')
        
        # Plot e_ks
        axs[1, 0].plot(e_ks)
        axs[1, 0].set_title('Turbulence Kinetic Energy (All Episodes)')
        axs[1, 0].set_xlabel('Step')
        axs[1, 0].set_ylabel('e_ks')
        
        # Plot uw
        axs[1, 1].plot(uw)
        axs[1, 1].set_title('Reynolds Stress (All Episodes)')
        axs[1, 1].set_xlabel('Step')
        axs[1, 1].set_ylabel('uw')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'overall_metrics.png'))
        plt.close()
        
    def render_step(self, step, actions, info):
        """Render/visualize current step (optional implementation)"""
        # This is a placeholder for visualization logic if needed
        # For example, you could create heatmaps of actions or flow fields
        pass

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained STW policy')
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pt',
                      help='Path to the model checkpoint (default: ./checkpoints/best_model.pt)')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--episodes', type=int, default=1,
                      help='Number of episodes to run (default: 1)')
    parser.add_argument('--length', type=int, default=None,
                      help='Episode length (default: use config value)')
    parser.add_argument('--no-save', action='store_true',
                      help='Do not save evaluation data')
    parser.add_argument('--render', action='store_true',
                      help='Render visualization during evaluation')
    
    args = parser.parse_args()
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model, args.config)
    
    # Run evaluation
    results = evaluator.run_evaluation(
        num_episodes=args.episodes,
        episode_length=args.length,
        save_data=not args.no_save,
        render=args.render
    )
    
    print("Evaluation complete!")

if __name__ == "__main__":
    main()