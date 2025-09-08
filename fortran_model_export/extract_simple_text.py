import torch
import numpy as np

# Load the model
model = torch.load('../best_model.pt', map_location='cpu', weights_only=False)

# Extract actor weights and biases
actor = model['maddpg_state_dict']['actor']

# Extract parameters
ln1_weight = actor['net.1.weight'].detach().numpy().astype(np.float64)  # (2,)
ln1_bias = actor['net.1.bias'].detach().numpy().astype(np.float64)      # (2,)
linear_weight = actor['net.2.weight'].detach().numpy().astype(np.float64)  # (8, 2)
linear_bias = actor['net.2.bias'].detach().numpy().astype(np.float64)      # (8,)
ln2_weight = actor['net.3.weight'].detach().numpy().astype(np.float64)  # (8,)
ln2_bias = actor['net.3.bias'].detach().numpy().astype(np.float64)      # (8,)
output_weight = actor['output_layer.weight'].detach().numpy().astype(np.float64)  # (1, 8)
output_bias = actor['output_layer.bias'].detach().numpy().astype(np.float64)      # (1,)

# Save just numbers to text file
with open('weights_simple.txt', 'w') as f:
    # Dimensions
    f.write(f"{ln1_weight.shape[0]} {linear_weight.shape[0]} {output_weight.shape[0]}\n")
    
    # Input LayerNorm weights
    for w in ln1_weight:
        f.write(f"{w:.16e}\n")
    
    # Input LayerNorm biases
    for b in ln1_bias:
        f.write(f"{b:.16e}\n")
    
    # Linear layer weights (8x2 matrix, row by row)
    for i in range(linear_weight.shape[0]):  # 8 rows
        for j in range(linear_weight.shape[1]):  # 2 cols
            f.write(f"{linear_weight[i,j]:.16e}\n")
    
    # Linear layer biases
    for b in linear_bias:
        f.write(f"{b:.16e}\n")
    
    # Hidden LayerNorm weights
    for w in ln2_weight:
        f.write(f"{w:.16e}\n")
    
    # Hidden LayerNorm biases
    for b in ln2_bias:
        f.write(f"{b:.16e}\n")
    
    # Output layer weights (1x8 matrix)
    for j in range(output_weight.shape[1]):  # 8 cols
        f.write(f"{output_weight[0,j]:.16e}\n")
    
    # Output layer bias
    f.write(f"{output_bias[0]:.16e}\n")

print("Simple weights saved to 'weights_simple.txt'")
print(f"First linear weight row: {linear_weight[0,:]} bias: {linear_bias[0]}")
print(f"Expected output for [-0.4787, -0.5567]: {(-0.4787)*linear_weight[0,0] + (-0.5567)*linear_weight[0,1] + linear_bias[0]}")