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

# Save to binary file (Fortran-readable)
with open('actor_weights.bin', 'wb') as f:
    # Write dimensions first
    input_dim = ln1_weight.shape[0]      # 2
    hidden_dim = linear_weight.shape[0]  # 8  
    output_dim = output_weight.shape[0]  # 1
    
    np.array([input_dim], dtype=np.int32).tofile(f)
    np.array([hidden_dim], dtype=np.int32).tofile(f) 
    np.array([output_dim], dtype=np.int32).tofile(f)
    
    # Write parameters in order they'll be used in forward pass
    # 1. Input LayerNorm
    ln1_weight.tofile(f)
    ln1_bias.tofile(f)
    
    # 2. Linear layer (no transpose - PyTorch weight is (out_features, in_features))
    linear_weight.tofile(f)  
    linear_bias.tofile(f)
    
    # 3. Hidden LayerNorm  
    ln2_weight.tofile(f)
    ln2_bias.tofile(f)
    
    # 4. Output layer
    output_weight.tofile(f)  # Already (1, 8) - no transpose needed
    output_bias.tofile(f)

print("Actor weights saved to 'actor_weights.bin'")