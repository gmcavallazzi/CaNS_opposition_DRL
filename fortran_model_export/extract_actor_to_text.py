import torch
import numpy as np

# Load the model
model = torch.load('../best_model.pt', map_location='cpu', weights_only=False)

# Extract actor weights and biases
actor = model['maddpg_state_dict']['actor']

print("Available parameters in actor:")
for key, value in actor.items():
    print(f"  {key}: {value.shape}")

# Extract parameters according to actual MLPActor structure:
# net.1 = LayerNorm(input_dim)  
# net.2 = Linear(input_dim -> hidden_dim)
# net.3 = LayerNorm(hidden_dim)
# output_layer = Linear(hidden_dim -> output_dim)

# LayerNorm parameters for input (net.1)
ln1_weight = actor['net.1.weight'].detach().numpy().astype(np.float64)  # (2,)
ln1_bias = actor['net.1.bias'].detach().numpy().astype(np.float64)      # (2,)

# Linear layer parameters (net.2)  
linear_weight = actor['net.2.weight'].detach().numpy().astype(np.float64)  # (8, 2)
linear_bias = actor['net.2.bias'].detach().numpy().astype(np.float64)      # (8,)

# LayerNorm parameters for hidden (net.3)
ln2_weight = actor['net.3.weight'].detach().numpy().astype(np.float64)  # (8,)
ln2_bias = actor['net.3.bias'].detach().numpy().astype(np.float64)      # (8,)

# Output layer parameters
output_weight = actor['output_layer.weight'].detach().numpy().astype(np.float64)  # (1, 8)
output_bias = actor['output_layer.bias'].detach().numpy().astype(np.float64)      # (1,)

print(f"\nExtracted parameters:")
print(f"Input LayerNorm:  weight{ln1_weight.shape}, bias{ln1_bias.shape}")
print(f"Linear layer:     weight{linear_weight.shape}, bias{linear_bias.shape}")
print(f"Hidden LayerNorm: weight{ln2_weight.shape}, bias{ln2_bias.shape}")  
print(f"Output layer:     weight{output_weight.shape}, bias{output_bias.shape}")

# Save to text file
with open('actor_weights.txt', 'w') as f:
    f.write("# Network dimensions\n")
    f.write(f"{ln1_weight.shape[0]} {linear_weight.shape[0]} {output_weight.shape[0]}\n")
    
    f.write("\n# Input LayerNorm weights\n")
    for w in ln1_weight:
        f.write(f"{w:.16e}\n")
    
    f.write("\n# Input LayerNorm biases\n")
    for b in ln1_bias:
        f.write(f"{b:.16e}\n")
    
    f.write("\n# Linear layer weights (8x2 matrix, row by row)\n")
    for i in range(linear_weight.shape[0]):  # 8 rows
        for j in range(linear_weight.shape[1]):  # 2 cols
            f.write(f"{linear_weight[i,j]:.16e}\n")
    
    f.write("\n# Linear layer biases\n")
    for b in linear_bias:
        f.write(f"{b:.16e}\n")
    
    f.write("\n# Hidden LayerNorm weights\n")
    for w in ln2_weight:
        f.write(f"{w:.16e}\n")
    
    f.write("\n# Hidden LayerNorm biases\n")
    for b in ln2_bias:
        f.write(f"{b:.16e}\n")
    
    f.write("\n# Output layer weights (1x8 matrix)\n")
    for j in range(output_weight.shape[1]):  # 8 cols
        f.write(f"{output_weight[0,j]:.16e}\n")
    
    f.write("\n# Output layer bias\n")
    f.write(f"{output_bias[0]:.16e}\n")

print("Actor weights saved to 'actor_weights.txt'")

# Also print first few values for verification
print(f"\nVerification - first linear weight row: {linear_weight[0,:]}")
print(f"First linear bias: {linear_bias[0]}")
print(f"Expected first output for input [-0.4787, -0.5567]:")
expected = (-0.4787) * linear_weight[0,0] + (-0.5567) * linear_weight[0,1] + linear_bias[0]
print(f"  {expected:.6f}")