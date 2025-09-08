import torch
import torch.nn.functional as F
import numpy as np

# Load the model
model = torch.load('../best_model.pt', map_location='cpu', weights_only=False)

# Extract actor network
actor_state = model['maddpg_state_dict']['actor']

# Create a dummy actor network to load the state dict
import sys
sys.path.append('../run0')
from models_pettingzoo import MLPActor

# Create actor with same architecture
obs_shape = (2,)  # 2 input dimensions
act_shape = (1,)  # 1 output dimension
hidden_layers = [8]  # From config pi: [8]

actor_net = MLPActor(obs_shape, act_shape, hidden_layers=hidden_layers)
actor_net.load_state_dict(actor_state)
actor_net.eval()

# Get input from user
print("Enter two input values:")
input_values = input().split()
val1, val2 = float(input_values[0]), float(input_values[1])

# Manual forward pass with debug prints
input_tensor = torch.tensor([[val1, val2]], dtype=torch.float32)
print(f"Input: {input_tensor.numpy()[0]}")

# Step by step through the network
x = input_tensor
print(f"\nStep 0: Initial input: {x}")

# Apply each layer manually
for i, layer in enumerate(actor_net.net):
    x_before = x.clone()
    x = layer(x)
    print(f"Step {i+1}: After {type(layer).__name__}: {x}")
    
    # Print layer parameters if they exist
    if hasattr(layer, 'weight') and layer.weight is not None:
        print(f"  Layer weight shape: {layer.weight.shape}")
        print(f"  Layer weight: {layer.weight}")
    if hasattr(layer, 'bias') and layer.bias is not None:
        print(f"  Layer bias shape: {layer.bias.shape}")
        print(f"  Layer bias: {layer.bias}")

# Apply output layer
print(f"\nBefore output layer: {x}")
x = actor_net.output_layer(x)
print(f"After output layer: {x}")
print(f"Output layer weight: {actor_net.output_layer.weight}")
print(f"Output layer bias: {actor_net.output_layer.bias}")

# Apply final tanh
print(f"\nBefore tanh: {x}")
x = torch.tanh(x)
print(f"After tanh (final): {x}")

print(f"\nFinal output: {x.detach().numpy()[0][0]}")

# Also run the complete forward pass for comparison
with torch.no_grad():
    complete_output = actor_net(input_tensor)
print(f"Complete forward pass output: {complete_output.numpy()[0][0]}")