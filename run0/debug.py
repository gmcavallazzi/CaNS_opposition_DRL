import numpy as np
import torch
from stwEnv_pettingzoo import STWParallelEnv
from utils import load_config, compute_reward

def detailed_environment_debug():
    print("Starting detailed environment debugging...")
    
    # Load configuration
    try:
        config = load_config('config.yaml')
        print("✓ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return
    
    # Create environment
    try:
        env = STWParallelEnv(config)
        print("✓ Environment created successfully")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        return
    
    # Reset environment
    try:
        observations, infos = env.reset()
        print("✓ Environment reset successful")
    except Exception as e:
        print(f"❌ Error during reset: {e}")
        return
    
    # Detailed observation check
    print("\nDetailed Observation Check:")
    for agent in env.possible_agents[:5]:
        obs = observations[agent]
        print(f"{agent} Observation:")
        print(f"  Shape: {obs.shape}")
        print(f"  Type: {obs.dtype}")
        print(f"  Min: {obs.min()}")
        print(f"  Max: {obs.max()}")
        print(f"  First few values:\n{obs[0:3, 0:3, :]}")
    
    # Prepare detailed actions
    print("\nPreparing Actions:")
    actions = {}
    for agent in env.possible_agents:
        # Carefully sample action to ensure it's within space
        action_space = env.action_spaces[agent]
        try:
            action = action_space.sample()
            actions[agent] = action
            print(f"{agent} Action: {action}, Shape: {action.shape}")
        except Exception as e:
            print(f"❌ Error sampling action for {agent}: {e}")
            return
    
    # Detailed step debugging
    print("\nDetailed Step Debugging:")
    try:
        # Add extensive print statements inside the step method
        next_observations, rewards, terminations, truncations, infos = env.step(actions)
        
        print("\nStep Results:")
        print("Rewards:")
        for agent, reward in rewards.items():
            print(f"  {agent}: {reward}")
        
        print("\nInfos:")
        for agent, info in infos.items():
            print(f"  {agent}: {info}")
        
        # Verify reward computation
        print("\nReward Computation Debug:")
        try:
            test_dpdx = infos[env.possible_agents[0]]['dpdx']
            print(f"dpdx value: {test_dpdx}")
            
            # Explicitly show reward computation
            try:
                computed_reward = compute_reward(test_dpdx, config)
                print(f"Computed Reward: {computed_reward}")
            except Exception as e:
                print(f"❌ Error in reward computation: {e}")
                # Print out full config for reward section
                print("\nReward Configuration:")
                print(config.get('reward', {}))
        except Exception as e:
            print(f"❌ Error extracting dpdx: {e}")
        
    except Exception as e:
        print(f"❌ Error during environment step: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\nDebugging complete.")

if __name__ == "__main__":
    detailed_environment_debug()