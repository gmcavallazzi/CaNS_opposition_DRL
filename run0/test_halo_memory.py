#!/usr/bin/env python3
"""
Test script to verify that halo=1 implementation doesn't cause memory allocation issues.
This script tests the initialization of components without running full training.
"""

import os
import sys
import json
import torch
import numpy as np

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stwEnv_pettingzoo import STWParallelEnv
from models_pettingzoo import SharedPolicyMADDPG, BatchedReplayBuffer

def test_halo_memory_allocation():
    """Test that halo=1 doesn't cause memory allocation issues."""

    # Create a minimal config with halo=1
    config = {
        'grid': {
            'target': {'i': 64, 'j': 64}  # 64x64 grid = 4096 agents
        },
        'halo': 1,  # This was causing the memory issue
        'model': {
            'buffer_size': 10000,  # Much smaller buffer for testing
            'batch_size': 32,
            'gradient_steps': 1,
            'train_freq': 1,
            'gamma': 0.95,
            'tau': 0.001,
            'learning_rate': 3e-4,
            'weight_decay': 1e-5
        },
        'training': {
            'gradient_clip': 1.0,
            'xavier_init_gain': 1.0,
            'save_freq': 1000
        },
        'net_arch': {
            'pi': [64, 64],
            'qf': [64, 64, 64],
            'qf_conv': [32, 64],
            'qf_mlp': [128, 128]
        },
        'total_timesteps': 1000,
        'env': {
            'dt': 0.1,
            'om_max': 10.0,
            'grid_j': 64,
            'grid_i': 64,
            'rank': 0,
            'size': 1
        }
    }

    print("Testing halo=1 memory allocation fix...")
    print(f"Grid size: {config['grid']['target']['i']}x{config['grid']['target']['j']}")
    print(f"Halo: {config['halo']}")
    print(f"Buffer size: {config['model']['buffer_size']}")

    try:
        # Test 1: Environment creation
        print("\n1. Creating environment...")
        env = STWParallelEnv(config)
        agents = env.possible_agents
        actor_obs_shape = env.observation_spaces[agents[0]].shape
        act_shape = env.action_spaces[agents[0]].shape
        num_agents = len(agents)

        # For critics, use full grid observation
        grid_size = config['grid']['target']['i']
        critic_obs_shape = (grid_size, grid_size, 2)

        print(f"   ✓ Environment created successfully")
        print(f"   - Number of agents: {num_agents}")
        print(f"   - Actor observation shape: {actor_obs_shape}")
        print(f"   - Critic observation shape: {critic_obs_shape}")
        print(f"   - Action shape: {act_shape}")

        # Calculate memory usage
        actor_obs_size = np.prod(actor_obs_shape)
        print(f"   - Actor obs size: {actor_obs_size} values per agent")

        # Test 2: MADDPG creation
        print("\n2. Creating MADDPG trainer...")
        device = torch.device('cpu')  # Use CPU to avoid GPU memory issues

        maddpg = SharedPolicyMADDPG(
            agents=agents,
            obs_shape=actor_obs_shape,  # Use actor observation shape
            act_shape=act_shape,
            gamma=config['model']['gamma'],
            tau=config['model']['tau'],
            lr=config['model']['learning_rate'],
            weight_decay=config['model']['weight_decay'],
            device=device,
            pi_arch=config['net_arch']['pi'],
            qf_arch=config['net_arch']['qf'],
            qf_conv=config['net_arch']['qf_conv'],
            qf_mlp=config['net_arch']['qf_mlp'],
            grid_size=grid_size,  # Critic uses full grid size
            gradient_clip=config['training']['gradient_clip'],
            xavier_init_gain=config['training']['xavier_init_gain']
        )
        print(f"   ✓ MADDPG trainer created successfully")

        # Test 3: Replay buffer creation
        print("\n3. Creating replay buffer...")
        buffer_size = config['model']['buffer_size']

        # This was the line causing the 1.34 TiB allocation error
        replay_buffer = BatchedReplayBuffer(buffer_size, actor_obs_shape, act_shape, num_agents, agents)

        print(f"   ✓ Replay buffer created successfully")

        # Calculate actual memory usage
        obs_memory = buffer_size * num_agents * actor_obs_size * 4  # float32 = 4 bytes
        print(f"   - Buffer memory usage: {obs_memory / (1024**3):.2f} GB")

        # Test 4: Test spatial data preparation with halo observations
        print("\n4. Testing spatial data preparation...")

        # Create dummy batch data with halo observations
        batch_size = 4
        obs_batch = torch.randn(batch_size, num_agents * actor_obs_size)
        act_batch = torch.randn(batch_size, num_agents)

        # Test the _prepare_spatial_data method
        obs_fields, act_field = maddpg._prepare_spatial_data(obs_batch, act_batch)

        print(f"   ✓ Spatial data preparation successful")
        print(f"   - Input obs batch shape: {obs_batch.shape}")
        print(f"   - Output obs fields shape: {obs_fields.shape}")
        print(f"   - Output act field shape: {act_field.shape}")

        # Verify dimensions
        expected_obs_shape = (batch_size, 2, grid_size, grid_size)
        expected_act_shape = (batch_size, 1, grid_size, grid_size)

        assert obs_fields.shape == expected_obs_shape, f"Wrong obs shape: {obs_fields.shape} != {expected_obs_shape}"
        assert act_field.shape == expected_act_shape, f"Wrong act shape: {act_field.shape} != {expected_act_shape}"

        print(f"   ✓ Output dimensions verified correctly")

        print("\n🎉 ALL TESTS PASSED! Halo=1 memory allocation issue is fixed.")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_halo_memory_allocation()
    sys.exit(0 if success else 1)