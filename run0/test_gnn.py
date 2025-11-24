import torch
import numpy as np
from models_pettingzoo import GridGNNActor

def test_gnn_actor():
    print("Testing GridGNNActor...")
    
    # 1. Initialize Actor
    actor = GridGNNActor(conv_channels=[16, 32])
    print("Actor initialized successfully.")
    
    # 2. Create Dummy Input
    # Shape: [Batch, N_Agents, Channels, Height, Width]
    # Batch=2, Agents=64, Channels=2, H=8, W=8
    batch_size = 2
    n_agents = 64
    obs_batch = torch.randn(batch_size, n_agents, 2, 8, 8)
    
    print(f"Input shape: {obs_batch.shape}")
    
    # 3. Forward Pass
    actions = actor(obs_batch)
    
    print(f"Output shape: {actions.shape}")
    
    # 4. Check Shape
    expected_shape = (batch_size, n_agents, 8, 8)
    assert actions.shape == expected_shape, f"Expected {expected_shape}, got {actions.shape}"
    print("Shape check passed.")
    
    # 5. Check Gradients
    loss = actions.sum()
    loss.backward()
    
    # Check if gradients exist for GNN layer
    gnn_grad = actor.gnn_conv.weight.grad
    assert gnn_grad is not None, "GNN layer has no gradients!"
    print(f"GNN gradient norm: {gnn_grad.norm().item()}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_gnn_actor()
