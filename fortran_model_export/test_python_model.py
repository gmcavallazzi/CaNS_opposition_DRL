import torch
import numpy as np

# Load the model
model = torch.load('../best_model.pt', map_location='cpu', weights_only=False)

# Extract actor network
actor = model['maddpg_state_dict']['actor']

# Create a dummy actor network to load the state dict
import sys
sys.path.append('../run0')
from models_pettingzoo import MLPActor

# Create actor with same architecture
obs_shape = (2,)  # 2 input dimensions
act_shape = (1,)  # 1 output dimension
hidden_layers = [8]  # From config pi: [8]

actor_net = MLPActor(obs_shape, act_shape, hidden_layers=hidden_layers)
actor_net.load_state_dict(actor)
actor_net.eval()

# Get input from user
print("Enter two input values:")
input_values = input().split()
val1, val2 = float(input_values[0]), float(input_values[1])

test_input = torch.tensor([[val1, val2]], dtype=torch.float32)

with torch.no_grad():
    output = actor_net(test_input)

print(f"Input: {test_input.numpy()[0]}")
print(f"Output: {output.numpy()[0][0]}")
