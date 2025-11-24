import os
import torch
import numpy as np
import argparse
from stwEnv_pettingzoo import STWParallelEnv
from models_pettingzoo import SharedPolicyMADDPG
from utils import load_config

def compute_features(u_patch, w_patch, dx=1.0, dy=1.0):
    """
    Compute physical features from 8x8 patches.
    
    Args:
        u_patch: (8, 8) u-velocity patch
        w_patch: (8, 8) w-velocity patch
        dx, dy: Grid spacing (normalized or physical)
        
    Returns:
        features: (N_features,) array containing:
                  [u_mean, w_mean, u_std, w_std, du/dy_mean, dw/dx_mean]
    """
    # 1. Basic statistics
    u_mean = np.mean(u_patch)
    w_mean = np.mean(w_patch)
    u_std = np.std(u_patch)
    w_std = np.std(w_patch)
    
    # 2. Gradients
    # Gradient returns list: [d/dx (axis 0), d/dy (axis 1)]
    # Note: In our grid (i, j), i is streamwise (x), j is spanwise (y)
    grad_u = np.gradient(u_patch, dx, edge_order=2)
    grad_w = np.gradient(w_patch, dy, edge_order=2)
    
    du_dx = grad_u[0]
    du_dy = grad_u[1]
    dw_dx = grad_w[0]
    dw_dy = grad_w[1]
    
    # Mean gradients
    du_dy_mean = np.mean(du_dy)
    dw_dx_mean = np.mean(dw_dx)
    
    # 3. Center values (often most relevant for local control)
    center_idx = 3  # 4th pixel (0-indexed)
    u_center = u_patch[center_idx, center_idx]
    w_center = w_patch[center_idx, center_idx]
    
    return np.array([
        u_mean, w_mean, 
        u_std, w_std, 
        du_dy_mean, dw_dx_mean,
        u_center, w_center
    ], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description='Generate dataset for PySR from trained agent')
    parser.add_argument('--steps', type=int, default=10000, help='Number of time steps to record')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='pysr_dataset.npz', help='Output file path')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    
    # Initialize environment
    env = STWParallelEnv(config)
    agents = env.possible_agents
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    maddpg = SharedPolicyMADDPG(
        agents=agents,
        device=device,
        actor_channels=config.get('net_arch', {}).get('actor_channels', [16, 32]),
        critic_conv_channels=config.get('net_arch', {}).get('critic_conv', [32, 64, 32]),
        critic_mlp_layers=config.get('net_arch', {}).get('critic_mlp', [256, 128])
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    maddpg.load_state_dict(checkpoint['maddpg_state_dict'])
    
    # Data collection buffers
    feature_list = []
    action_list = []
    
    print(f"Starting data collection for {args.steps} steps...")
    
    obs, _ = env.reset()
    
    try:
        for step in range(args.steps):
            # 1. Get actions from model
            all_obs = np.stack([obs[agent] for agent in agents]).astype(np.float32)
            all_obs_tensor = torch.FloatTensor(all_obs).to(device)
            
            with torch.no_grad():
                all_actions = maddpg.select_actions_batched(all_obs_tensor) # [64, 8, 8]
            
            # 2. Store Data (Features -> Action)
            # We process each agent independently
            for i, agent in enumerate(agents):
                # Get patch data
                u_patch = all_obs[i, 0] # Normalized u
                w_patch = all_obs[i, 1] # Normalized w
                action_patch = all_actions[i]
                
                # Compute input features
                feats = compute_features(u_patch, w_patch)
                
                # Compute output target (e.g., mean action or center action)
                # For symbolic regression, we usually want a scalar output per patch
                # Let's try to predict the MEAN action for the patch, 
                # or the CENTER action if the policy is local.
                # Given the 8x8 output, let's save the MEAN action as the primary target.
                target_action = np.mean(action_patch)
                
                feature_list.append(feats)
                action_list.append(target_action)
            
            # 3. Step environment
            actions_dict = {agent: all_actions[i] for i, agent in enumerate(agents)}
            next_obs, _, _, _, _ = env.step(actions_dict)
            obs = next_obs
            
            if step % 100 == 0:
                print(f"Step {step}/{args.steps} collected. Total samples: {len(feature_list)}")
                
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        # Save data
        X = np.array(feature_list)
        y = np.array(action_list)
        
        feature_names = [
            'u_mean', 'w_mean', 
            'u_std', 'w_std', 
            'du_dy_mean', 'dw_dx_mean',
            'u_center', 'w_center'
        ]
        
        print(f"Saving dataset to {args.output}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        
        np.savez(
            args.output,
            X=X,
            y=y,
            feature_names=feature_names
        )
        
        env.close()

if __name__ == "__main__":
    main()
