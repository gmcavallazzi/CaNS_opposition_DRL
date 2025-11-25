import sys
import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
import os

# Mock the env_pettingzoo module before importing stwStart_gnn
mock_env_module = MagicMock()
sys.modules['env_pettingzoo'] = mock_env_module

# Mock STWParallelEnv
class MockSTWParallelEnv:
    def __init__(self, config):
        self.possible_agents = [f"agent_{i}_{j}" for i in range(8) for j in range(8)]
        self.observation_spaces = {agent: MagicMock() for agent in self.possible_agents}
        self.action_spaces = {agent: MagicMock() for agent in self.possible_agents}
        for agent in self.possible_agents:
            self.observation_spaces[agent].shape = (2, 8, 8)
            self.action_spaces[agent].shape = (8, 8)
        
        self.u_obs_field = np.zeros((64, 64))
        self.w_obs_field = np.zeros((64, 64))
        self.om_max = 1.0

    def reset(self):
        obs = {agent: np.zeros((2, 8, 8), dtype=np.float32) for agent in self.possible_agents}
        infos = {agent: {'dpdx': -1.0} for agent in self.possible_agents}
        return obs, infos

    def step(self, actions):
        obs = {agent: np.zeros((2, 8, 8), dtype=np.float32) for agent in self.possible_agents}
        rewards = {agent: 0.0 for agent in self.possible_agents}
        terminations = {agent: False for agent in self.possible_agents}
        truncations = {agent: False for agent in self.possible_agents}
        infos = {agent: {'dpdx': -1.0} for agent in self.possible_agents}
        return obs, rewards, terminations, truncations, infos

    def close(self):
        pass

mock_env_module.STWParallelEnv = MockSTWParallelEnv

# Now import the module under test
# We need to make sure we are in the correct directory for imports to work
sys.path.append(os.getcwd() + "/run0")
from run0.stwStart_gnn import train_maddpg

class TestGNNFix(unittest.TestCase):
    def test_gnn_training_loop(self):
        config = {
            'total_timesteps': 600, # Run enough to trigger the check at 500
            'model': {
                'batch_size': 32,
                'gradient_steps': 1,
                'train_freq': 100,
                'buffer_size': 1000,
                'gamma': 0.99,
                'tau': 0.01,
                'learning_rate': 1e-3,
                'weight_decay': 1e-5,
                'smoothness': {}
            },
            'training': {
                'save_freq': 1000,
                'gradient_clip': 1.0,
                'start_episode_length': 100
            },
            'net_arch': {}
        }
        
        # Run training
        # This should pass the check at step 500 without error
        try:
            train_maddpg(
                config=config,
                checkpoint_dir="./tmp_checkpoints",
                logs_dir="./tmp_logs",
                device="cpu"
            )
        except Exception as e:
            self.fail(f"Training failed with error: {e}")

if __name__ == '__main__':
    unittest.main()
