import time
import os
import numpy as np
import torch
import yaml
from stwEnv_pettingzoo import STWParallelEnv
from models_pettingzoo import SharedPolicyMADDPG

def evaluate_policy(
    config_path: str,
    policy_path: str,
    num_episodes: int = 5,
    device: str = "cpu"
):
    """
    Evaluate a trained policy.

    Args:
        config_path: Path to configuration YAML file
        policy_path: Path to saved policy checkpoint
        num_episodes: Number of episodes to evaluate
        device: Device to run evaluation on
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate STW Policy")
    parser.add_argument("--config", type=str, default="config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--policy_path", type=str, required=True,
                      help="Path to saved policy checkpoint")
    parser.add_argument("--num_episodes", type=int, default=5,
                      help="Number of episodes for evaluation")
    parser.add_argument("--device", type=str, default="cpu",
                      help="Device to run evaluation on")

    args = parser.parse_args()

    evaluate_policy(
        config_path=args.config,
        policy_path=args.policy_path,
        num_episodes=args.num_episodes,
        device=args.device
    )