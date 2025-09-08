import torch

# Load the model
model = torch.load('best_model.pt', map_location='cpu', weights_only=False)

# Print structure
print("Model keys:", list(model.keys()))

# Print MADDPG state dict structure
if 'maddpg_state_dict' in model:
    maddpg = model['maddpg_state_dict']
    print("\nMADDPG keys:", list(maddpg.keys()))
    
    # Look for actor networks
    for key in maddpg.keys():
        if 'actor' in key.lower() and 'optimizer' not in key.lower():
            print(f"\n{key} structure:")
            for param_name, param_value in maddpg[key].items():
                if hasattr(param_value, 'shape'):
                    print(f"  {param_name}: {param_value.shape}")
                    print(f"  Values: {param_value}")
                    print()
                else:
                    print(f"  {param_name}: {type(param_value)}")
                    print()