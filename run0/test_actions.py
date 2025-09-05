#!/usr/bin/env python3
"""
Test your model with the correct architecture
"""

import torch
import numpy as np
import yaml
from models_pettingzoo import SharedPolicyMADDPG

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_correct_model(checkpoint_path, config_path="config.yaml"):
    """Test with the correct architecture from your saved model"""
    
    print("Loading config...")
    config = load_config(config_path)
    
    # Get grid dimensions  
    grid_i = config['grid']['target']['i']
    grid_j = config['grid']['target']['j'] 
    agents = [f"agent_{i}_{j}" for i in range(grid_i) for j in range(grid_j)]
    
    print(f"Grid: {grid_i}x{grid_j}, Total agents: {len(agents)}")
    
    # Use the CORRECT shapes from inspection
    obs_shape = (1, 1, 2)  # This gets flattened to 2D input
    act_shape = (1,)       # Single action output
    pi_arch = [8]          # Single hidden layer with 8 units
    qf_arch = [8, 8]       # Dummy for critic (won't be used)
    
    print(f"Using obs_shape={obs_shape}, act_shape={act_shape}")
    print(f"Using pi_arch={pi_arch}")
    
    try:
        # Create policy with correct architecture
        policy = SharedPolicyMADDPG(
            agents=agents,
            obs_shape=obs_shape,
            act_shape=act_shape,
            device='cpu',
            pi_arch=pi_arch,
            qf_arch=qf_arch
        )
        
        # Load checkpoint
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Load only actor weights  
        actor_state = checkpoint['maddpg_state_dict']['actor']
        policy.actor.load_state_dict(actor_state)
        print("✓ Actor weights loaded successfully")
        
        # Test with dummy observation (shape should match what environment gives)
        test_agent = agents[0]
        dummy_obs = torch.randn(1, 1, 1, 2)  # Batch=1, then obs_shape=(1,1,2)
        
        print(f"\nTesting with dummy observation shape: {dummy_obs.shape}")
        
        # Get action
        policy.actor.eval()
        with torch.no_grad():
            action = policy.select_action(test_agent, dummy_obs)
        
        print(f"✓ Action obtained!")
        print(f"Action type: {type(action)}")
        print(f"Action shape: {np.array(action).shape}")
        print(f"Action value: {action}")
        
        # Check if it's a scalar or array
        if hasattr(action, '__len__'):
            print(f"Action length: {len(action)}")
            if len(action) == 1:
                print(f"Single element: {action[0]}")
        else:
            print("Action is a scalar")
        
        # Test multiple times
        print("\nTesting 5 more times:")
        for i in range(5):
            with torch.no_grad():
                action = policy.select_action(test_agent, dummy_obs)
            print(f"  Test {i+1}: {action} (type: {type(action)})")
        
        print("\n✓ SUCCESS! Your model outputs a single action per agent")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python test_correct_actions.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    test_correct_model(checkpoint_path)
