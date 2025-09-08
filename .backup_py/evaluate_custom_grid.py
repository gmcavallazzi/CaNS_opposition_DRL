#!/usr/bin/env python3
"""
Clean evaluation script with image saving capability
"""

import os
import torch
import numpy as np
import yaml
import argparse
from typing import Dict, Tuple
from datetime import datetime

from stwEnv_pettingzoo import STWParallelEnv  # Your modified environment
from models_pettingzoo import SharedPolicyMADDPG


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def evaluate_with_image_saving(
    checkpoint_path: str,
    config_path: str = "config.yaml", 
    validation_grid: Tuple[int, int] = (192, 192),
    num_episodes: int = 2,  # Fewer episodes when saving images
    device: str = "cpu",
    save_images: bool = True,
    base_save_dir: str = "./evaluation_images"
):
    """Evaluate model and save images of actions and observations"""
    
    # Create timestamped save directory
    if save_images:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_save_dir = f"{base_save_dir}_{validation_grid[0]}x{validation_grid[1]}_{timestamp}"
        print(f"Images will be saved to: {image_save_dir}")
    else:
        image_save_dir = None
    
    print("="*60)
    print(f"EVALUATING WITH IMAGE SAVING")
    print(f"Grid: {validation_grid[0]}x{validation_grid[1]}")
    print("="*60)
    
    # Load config
    config = load_config(config_path)
    
    # Extract key info from config
    pi_arch = config['net_arch']['pi']  # [8]
    training_grid = (config['grid']['target']['i'], config['grid']['target']['j'])
    action_max = config['action']['om_max']
    
    print(f"Training grid: {training_grid[0]}x{training_grid[1]}")
    print(f"Actor architecture: {pi_arch}")
    print(f"Action range: ±{action_max}")
    
    # Update config for validation
    config['grid']['target']['i'] = validation_grid[0]
    config['grid']['target']['j'] = validation_grid[1]
    
    # Create environment WITH image saving
    env = STWParallelEnv(
        config, 
        save_images=save_images,
        image_save_dir=image_save_dir
    )
    agents = env.possible_agents
    obs_shape = env.observation_spaces[agents[0]].shape
    act_shape = env.action_spaces[agents[0]].shape
    
    print(f"Environment: {len(agents)} agents")
    print(f"Obs shape: {obs_shape}, Act shape: {act_shape}")
    
    # Create policy
    policy = SharedPolicyMADDPG(
        agents=agents,
        obs_shape=obs_shape,
        act_shape=act_shape,
        device=device,
        pi_arch=pi_arch,
        qf_arch=[16, 64, 64]  # From config
    )
    
    # Load ONLY actor weights
    print(f"Loading: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'maddpg_state_dict' in checkpoint and 'actor' in checkpoint['maddpg_state_dict']:
        actor_weights = checkpoint['maddpg_state_dict']['actor']
        policy.actor.load_state_dict(actor_weights)
        print("✓ Actor loaded (critic skipped)")
    else:
        raise ValueError("Could not find actor weights in checkpoint")
    
    # Set to evaluation mode
    policy.actor.eval()
    
    # Run evaluation
    all_rewards = []
    all_lengths = []
    all_dpdx = []
    all_actions = []
    
    for ep in range(num_episodes):
        print(f"\nEpisode {ep+1}/{num_episodes}")
        
        # Reset
        observations, _ = env.reset()
        
        episode_reward = 0.0
        episode_steps = 0
        episode_dpdx_vals = []
        episode_action_vals = []
        
        done = False
        while not done:
            # Get actions for all agents
            actions = {}
            
            with torch.no_grad():
                for agent in agents:
                    if agent in observations:
                        obs_tensor = torch.FloatTensor(observations[agent]).unsqueeze(0).to(device)
                        action = policy.select_action(agent, obs_tensor)
                        actions[agent] = action
            
            # Extract single action value for statistics
            sample_action = next(iter(actions.values()))
            if isinstance(sample_action, np.ndarray):
                if sample_action.shape == (1, 1):
                    action_value = sample_action[0, 0]
                elif sample_action.shape == (1,):
                    action_value = sample_action[0]
                else:
                    action_value = float(sample_action.flatten()[0])
            else:
                action_value = float(sample_action)
            
            episode_action_vals.append(action_value)
            
            # Step environment (this will save images automatically)
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Collect metrics
            step_reward = np.mean(list(rewards.values()))
            episode_reward += step_reward
            episode_steps += 1
            
            # Get physics info
            if agents[0] in infos:
                episode_dpdx_vals.append(infos[agents[0]].get('dpdx', 0.0))
            
            # Check done
            done = any(terminations.values()) or any(truncations.values())
            
            # Print progress for first few steps
            if episode_steps <= 5:
                print(f"  Step {episode_steps}: Reward={step_reward:.4f}, "
                      f"Action={action_value:.4f}, dpdx={episode_dpdx_vals[-1]:.6f}")
        
        # Store episode results
        all_rewards.append(episode_reward)
        all_lengths.append(episode_steps)
        all_dpdx.append(np.mean(episode_dpdx_vals) if episode_dpdx_vals else 0.0)
        all_actions.append(np.mean(episode_action_vals))
        
        print(f"  Episode {ep+1} Summary:")
        print(f"    Total Reward: {episode_reward:.4f}")
        print(f"    Length: {episode_steps}")
        print(f"    Mean dpdx: {all_dpdx[-1]:.6f}")
        print(f"    Mean action: {all_actions[-1]:.4f}")
    
    env.close()
    
    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Grid: {validation_grid[0]}x{validation_grid[1]} ({len(agents)} agents)")
    print(f"Episodes: {num_episodes}")
    print(f"Images saved to: {image_save_dir}")
    print()
    print(f"Rewards:  {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    print(f"Lengths:  {np.mean(all_lengths):.1f} ± {np.std(all_lengths):.1f}")
    print(f"dpdx:     {np.mean(all_dpdx):.6f} ± {np.std(all_dpdx):.6f}")
    print(f"Actions:  {np.mean(all_actions):.4f} ± {np.std(all_actions):.4f}")
    
    if save_images:
        print(f"\n✓ Images saved successfully!")
        print(f"  Actions to CaNS: actions_to_cans_ep*.png")
        print(f"  U observations: observations_from_cans_ep*.png (left panel)")
        print(f"  W observations: observations_from_cans_ep*.png (right panel)")
    
    return {
        'rewards': all_rewards,
        'lengths': all_lengths,
        'dpdx': all_dpdx,
        'actions': all_actions,
        'mean_reward': np.mean(all_rewards),
        'mean_length': np.mean(all_lengths),
        'mean_dpdx': np.mean(all_dpdx),
        'mean_action': np.mean(all_actions),
        'image_save_dir': image_save_dir
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policy and save images")
    parser.add_argument("checkpoint_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--grid_i", type=int, default=192, help="Grid dimension i")
    parser.add_argument("--grid_j", type=int, default=192, help="Grid dimension j")
    parser.add_argument("--episodes", type=int, default=2, help="Number of evaluation episodes")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--no_save", action="store_true", help="Disable image saving")
    parser.add_argument("--save_dir", type=str, default="./evaluation_images", help="Base directory to save images")
    
    args = parser.parse_args()

    results = evaluate_with_image_saving(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config,
        validation_grid=(args.grid_i, args.grid_j),
        num_episodes=args.episodes,
        device=args.device,
        save_images=not args.no_save,
        base_save_dir=args.save_dir
    )

    print("\n✓ Evaluation complete.")


if __name__ == "__main__":
    main()

