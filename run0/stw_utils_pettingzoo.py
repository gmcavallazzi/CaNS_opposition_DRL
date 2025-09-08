import time
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import torch
import yaml
from stwEnv_pettingzoo import STWParallelEnv
from models_pettingzoo import SharedPolicyMADDPG

def profile_environment(config: Dict, num_steps: int = 100):
    """
    Profile the environment to measure step time.
    
    Args:
        config: Configuration dictionary
        num_steps: Number of steps to profile
    """
    print(f"Profiling agents for {num_steps} steps...")
    env = STWParallelEnv(config)
    
    # Get agent list
    agents = env.possible_agents
    
    # Reset environment
    observations, _ = env.reset()
    
    # Take random actions
    step_times = []
    for _ in range(num_steps):
        # Create random actions for all agents
        actions = {agent: env.action_spaces[agent].sample() for agent in agents}
        
        # Measure step time
        start_time = time.time()
        observations, rewards, _, _, _ = env.step(actions)
        end_time = time.time()
        
        step_times.append(end_time - start_time)
    
    # Close environment
    env.close()
    
    # Print statistics
    print(f"Step time statistics:")
    print(f"  Mean: {np.mean(step_times):.6f} seconds")
    print(f"  Min: {np.min(step_times):.6f} seconds")
    print(f"  Max: {np.max(step_times):.6f} seconds")
    print(f"  Std: {np.std(step_times):.6f} seconds")
    print(f"  Throughput: {1.0 / np.mean(step_times):.2f} steps/second")
    
    return step_times

def evaluate_policy(
    config: Dict,
    policy_path: str,
    num_episodes: int = 5,
    device: str = "cpu"
):
    """
    Evaluate a trained policy.
    
    Args:
        config: Configuration dictionary
        policy_path: Path to saved policy checkpoint
        num_episodes: Number of episodes to evaluate
        device: Device to run evaluation on
    """
    print(f"Evaluating shared policy for {num_episodes} episodes...")
    
    # Create environment
    env = STWParallelEnv(config)
    
    # Get agent list and observation/action shapes
    agents = env.possible_agents
    obs_shape = env.observation_spaces[agents[0]].shape
    act_shape = env.action_spaces[agents[0]].shape
    
    # Extract network architecture from config
    pi_arch = config.get('net_arch', {}).get('pi', [64, 64])
    qf_arch = config.get('net_arch', {}).get('qf', [64, 64, 64])
    
    print(f"Using network architecture - Actor: {pi_arch}, Critic: {qf_arch}")
    
    # Load policy with explicit architecture parameters
    policy = SharedPolicyMADDPG(
        agents=agents,
        obs_shape=obs_shape,
        act_shape=act_shape,
        device=device,
        pi_arch=pi_arch,
        qf_arch=qf_arch
    )
    
    # Load checkpoint
    checkpoint = torch.load(policy_path, map_location=device)
    policy.load_state_dict(checkpoint['maddpg_state_dict'])
    
    # Initialize metrics
    episode_rewards = []
    episode_dpdx = []
    
    # Evaluate for multiple episodes
    for episode in range(num_episodes):
        # Reset environment
        observations, _ = env.reset()
        
        # Initialize episode metrics
        total_rewards = []
        step_dpdx = []
        
        # Run episode
        done = False
        while not done:
            # Select actions for all agents
            actions = {}
            for agent in agents:
                obs = torch.FloatTensor(observations[agent]).unsqueeze(0).to(device)
                
                # Get deterministic action (no exploration noise during evaluation)
                with torch.no_grad():
                    action = policy.select_action(agent, obs)
                
                actions[agent] = action
            
            # Step environment
            observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Store metrics
            total_rewards.append(np.mean(list(rewards.values())))
            
            # Store physics metrics from any agent (all agents have same physics info)
            first_agent = agents[0]
            info = infos[first_agent]
            step_dpdx.append(info['dpdx'])
            
            # Check if episode is done
            done = any(terminations.values()) or any(truncations.values())
        
        # Store episode metrics
        episode_rewards.append(np.mean(total_rewards))
        episode_dpdx.append(np.mean(step_dpdx))
        
        print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_rewards[-1]:.3f}, "
              f"dpdx: {episode_dpdx[-1]:.6f}")
    
    # Close environment
    env.close()
    
    # Print statistics
    print("\nEvaluation Results:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"  Mean dpdx: {np.mean(episode_dpdx):.6f} ± {np.std(episode_dpdx):.6f}")
    
    return {
        'rewards': episode_rewards,
        'dpdx': episode_dpdx
    }

def visualize_observations(config: Dict):
    """
    Visualize observations for different agent configurations.
    
    Args:
        config: Configuration dictionary
    """
    # Create environment
    env = STWParallelEnv(config, render_mode="rgb_array")
    
    # Get agent list
    agents = env.possible_agents
    
    # Reset environment
    observations, _ = env.reset()
    
    # Take a random action to get real data
    actions = {agent: env.action_spaces[agent].sample() for agent in agents}
    observations, _, _, _, _ = env.step(actions)
    
    # Visualize observations for a few agents
    sample_agents = agents[:min(3, len(agents))]
    
    plt.figure(figsize=(15, 5 * len(sample_agents)))
    
    for i, agent in enumerate(sample_agents):
        obs = observations[agent]
        
        # Grid agent has shape (H, W, 2)
        h, w, c = obs.shape
            
        for j in range(c):
            plt.subplot(len(sample_agents), c, i * c + j + 1)
            plt.imshow(obs[:, :, j], cmap='viridis')
            plt.title(f"{agent} - Channel {j}")
    
    plt.tight_layout()
    
    # Also visualize the environment's render output
    rgb_array = env.render()
    
    if rgb_array is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb_array)
        plt.title("Environment Render")
        plt.tight_layout()
    
    # Close environment
    env.close()
    
    plt.show()

def compare_configurations(base_config: Dict, variations: List[Dict], num_steps: int = 100):
    """
    Compare different environment configurations in terms of speed.
    
    Args:
        base_config: Base configuration dictionary
        variations: List of configuration variations to compare
        num_steps: Number of steps to profile for each variation
    """
    results = []
    
    # Profile base configuration
    base_time = profile_environment(base_config, num_steps=num_steps)
    results.append(("Base", np.mean(base_time), np.std(base_time)))
    
    # Profile variations
    for i, variation in enumerate(variations):
        # Create merged config
        merged_config = {**base_config, **variation}
        
        # Profile
        var_time = profile_environment(merged_config, num_steps=num_steps)
        results.append((f"Variation {i+1}", np.mean(var_time), np.std(var_time)))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    names = [result[0] for result in results]
    means = [result[1] for result in results]
    stds = [result[2] for result in results]
    
    plt.bar(names, means, yerr=stds, capsize=10)
    plt.xlabel("Configuration")
    plt.ylabel("Step Time (seconds)")
    plt.title("Environment Step Time Comparison")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config: Dict, config_path: str):
    """Save configuration to YAML file."""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="STW PettingZoo Utilities")
    parser.add_argument("command", choices=["profile", "evaluate", "visualize"], 
                      help="Command to execute")
    parser.add_argument("--config", type=str, default="config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--num_steps", type=int, default=100,
                      help="Number of steps for profiling")
    parser.add_argument("--num_episodes", type=int, default=5,
                      help="Number of episodes for evaluation")
    parser.add_argument("--policy_path", type=str, default=None,
                      help="Path to saved policy for evaluation")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Execute command
    if args.command == "profile":
        profile_environment(config, num_steps=args.num_steps)
    elif args.command == "evaluate":
        if args.policy_path is None:
            print("Error: --policy_path is required for evaluation")
        else:
            evaluate_policy(
                config, 
                args.policy_path,
                num_episodes=args.num_episodes
            )
    elif args.command == "visualize":
        visualize_observations(config)