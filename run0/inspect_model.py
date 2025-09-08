#!/usr/bin/env python3
"""
Inspect the actual architecture of your saved model
"""

import torch
import yaml

def inspect_checkpoint(checkpoint_path, config_path="config.yaml"):
    """Inspect what's actually in your checkpoint"""
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    print("\nCheckpoint info:")
    print(f"  Total steps: {checkpoint.get('total_steps', 'N/A')}")
    print(f"  Episode: {checkpoint.get('episode', 'N/A')}")
    print(f"  Best reward: {checkpoint.get('best_reward', 'N/A')}")
    print(f"  Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    
    if 'maddpg_state_dict' in checkpoint:
        state_dict = checkpoint['maddpg_state_dict']
        print(f"\nmaddpg_state_dict keys: {list(state_dict.keys())}")
        
        if 'actor' in state_dict:
            actor_state = state_dict['actor']
            print(f"\nActor state_dict keys: {list(actor_state.keys())}")
            
            # Analyze the architecture from the weights
            print("\nActor layer analysis:")
            for key, tensor in actor_state.items():
                print(f"  {key}: {tensor.shape}")
            
            # Try to infer the architecture
            print("\nInferred architecture:")
            
            # Look for input dimension (first layer)
            if 'net.2.weight' in actor_state:
                input_dim = actor_state['net.2.weight'].shape[1]
                first_hidden = actor_state['net.2.weight'].shape[0]
                print(f"  Input dim: {input_dim}")
                print(f"  First hidden: {first_hidden}")
            
            # Look for output dimension
            if 'output_layer.weight' in actor_state:
                output_dim = actor_state['output_layer.weight'].shape[0]
                last_hidden = actor_state['output_layer.weight'].shape[1]
                print(f"  Last hidden: {last_hidden}")
                print(f"  Output dim: {output_dim}")
            
            # Try to find all hidden layers
            hidden_layers = []
            i = 2
            while f'net.{i}.weight' in actor_state:
                layer_size = actor_state[f'net.{i}.weight'].shape[0]
                hidden_layers.append(layer_size)
                i += 2  # Skip bias layers
                if i > 20:  # Safety break
                    break
            
            print(f"  Hidden layers: {hidden_layers}")
            
            # Look at config to see what was intended
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                intended_pi_arch = config.get('net_arch', {}).get('pi', [])
                print(f"  Config pi_arch: {intended_pi_arch}")
                
                # Check if they match
                if hidden_layers == intended_pi_arch:
                    print("  ✓ Architecture matches config")
                else:
                    print(f"  ✗ Architecture mismatch! Saved: {hidden_layers}, Config: {intended_pi_arch}")
                    
            except Exception as e:
                print(f"  Could not load config: {e}")
        
        if 'critic' in state_dict:
            critic_state = state_dict['critic']
            print(f"\nCritic state_dict keys: {list(critic_state.keys())}")
    
    else:
        print("No maddpg_state_dict found")
        print("Available keys:", list(checkpoint.keys()))

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python inspect_model.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    inspect_checkpoint(checkpoint_path)
