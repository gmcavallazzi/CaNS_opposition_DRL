import matplotlib.pyplot as plt
import os
import numpy as np
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def compute_reward_alt(dpdx, config):
    # Normalize dpdx
    dpdx_min = config['reward']['dpdx']['min']
    dpdx_max = config['reward']['dpdx']['max']
    norm_factor = dpdx_max - dpdx_min
    
    # Compute normalized dpdx
    normalized_dpdx = (dpdx - dpdx_min) / norm_factor
    
    # Compute tanh reward
    dpavg = 0.5 * (dpdx_max + dpdx_min)
    sratio = 10 / (dpdx_max - dpdx_min)
    tanh_value = np.tanh(sratio * (dpdx - dpavg))
    
    w0 = config['reward']['weights'][0]
    w1 = config['reward']['weights'][1]
    tanh_reward = w0 + (w0 + w1) * (tanh_value + 1) / 2
    
    # The final reward is negative because we want to minimize the pressure gradient
    reward = -tanh_reward
    return reward/config['n_actuations']

def compute_reward(dpdx, config):
    # Reference uncontrolled dpdx value (most negative)
    dpdx_uncontrolled = config['reward']['dpdx']['min']
    
    # Calculate percentage reduction
    # As dpdx gets less negative (closer to zero), this value increases
    reduction = 1.0 - (dpdx / dpdx_uncontrolled)
    
    return reduction

def compute_local_reward(avg_tau, config):
    """
    Compute the local reward component based on the average wall shear stress.
    Normalizes the avg_tau value based on expected min/max range in config.
    
    Args:
        avg_tau: Average local wall shear stress
        config: Configuration dictionary with reward parameters
        
    Returns:
        local_reward: Normalized local reward value
    """
    # Get min and max values for tau from config
    tau_min = config['reward']['tau']['min']
    tau_max = config['reward']['tau']['max']
    norm_factor = tau_max - tau_min
    
    # Normalize (similar to compute_reward for dpdx)
    # Note: We invert the normalization because lower tau is better (1 = best, 0 = worst)
    norm_tau = 1.0 - (avg_tau - tau_min) / norm_factor
    
    return norm_tau

def rescale_action(action, config):
    return (action + 1) / 2 * (config['action']['max'] - config['action']['min']) + config['action']['min']

def rescale_amp(amp, config):
    norm_factor = config['action']['max'] - config['action']['min']
    return (amp - config['action']['min']) * (config['action']['om_max'] - config['action']['om_min']) / norm_factor + config['action']['om_min']

def img_rescale(mat, config):
    mat = np.clip(mat, config['observation']['min_expected'], config['observation']['max_expected'])
    res = (mat - config['observation']['min_expected']) / (config['observation']['max_expected'] - config['observation']['min_expected']) * 255
    return res.astype(np.uint8)

def comp_reward(dpdx, config):
    norm_factor = config['reward']['dpdx']['max'] - config['reward']['dpdx']['min']
    return (dpdx - config['reward']['dpdx']['min']) / norm_factor

def tanh_reward(x, config):
    dpavg = 0.5 * (config['reward']['dpdx']['max'] + config['reward']['dpdx']['min'])
    sratio = 10 / (config['reward']['dpdx']['max'] - config['reward']['dpdx']['min'])
    tanh_value = np.tanh(sratio * (x - dpavg))
    return config['reward']['weights'][0] + (config['reward']['weights'][0] + config['reward']['weights'][1]) * (tanh_value + 1) / 2

def comp_reward_derivative(dp, dpo, config):
    norm_factor = config['reward']['dpdx']['max'] - config['reward']['dpdx']['min']
    tanh2 = tanh_reward(dp, config)
    tanh1 = config['reward']['weights'][0] + config['reward']['weights'][1] - tanh2
    return (dp - config['reward']['dpdx']['min']) / norm_factor * tanh1 + (dp - dpo) / norm_factor * tanh2

def write_to_file(filename, content):
    with open(filename, 'a') as f:
        f.write(content)

def reshape_mpi_arr(size_box, size_mat, arr):
    if len(size_box) != 2 or len(size_mat) != 2:
        raise ValueError('Error, the box size and mat size must be arrays with two values')

    size_mat = np.array(size_mat, np.int64)
    size_box = np.array(size_box, np.int64)
    mat = np.zeros(size_mat)
    blocks = size_mat // size_box
    for k in range(1, int(np.prod(blocks)) + 1):
        row = int((k - 1) % blocks[0])
        col = int(np.ceil(k / blocks[0])) - 1
        ii = int(row * size_box[0])
        jj = int(col * size_box[1])
        temp = np.reshape(arr[(k-1)*np.prod(size_box):k*np.prod(size_box)], size_box)
        mat[ii:ii+size_box[0], jj:jj+size_box[1]] = temp

    return mat


def img_rescale(mat, config, min_expected=None, max_expected=None):
    """
    Rescale a matrix to [0, 255] range for visualization.
    
    Args:
        mat: Input matrix to rescale
        config: Configuration dictionary
        min_expected: Optional minimum value for rescaling. If None, uses config value.
        max_expected: Optional maximum value for rescaling. If None, uses config value.
        
    Returns:
        Rescaled matrix as uint8 (0-255)
    """
    # Use provided min/max values or fall back to config values
    min_val = min_expected if min_expected is not None else config['observation']['min_expected_u']
    max_val = max_expected if max_expected is not None else config['observation']['max_expected_u']
    
    # Clip values to the range [min_val, max_val]
    mat = np.clip(mat, min_val, max_val)
    
    # Rescale to [0, 255]
    res = (mat - min_val) / (max_val - min_val) * 255
    return res.astype(np.uint8)

def save_debug_visualization(step_num, raw_data, processed_data):
    """
    Save visualization of raw and processed data arrays as images.
    Useful for debugging data transfer and processing in RL training.
    
    Args:
        step_num (int): Current step number (used for filename)
        raw_data (np.ndarray): Raw data array (e.g., before normalization)
        processed_data (np.ndarray): Processed data array (e.g., after normalization)
    
    Usage:
        # In your training loop:
        save_debug_visualization(
            step_num=current_step,
            raw_data=raw_array,      # e.g., data received from simulation
            processed_data=proc_array # e.g., data after normalization/processing
        )
    
    Output:
        Creates two files in ./debug_vis/ for each call:
        - raw_step_XXXX.png: Visualization of raw_data
        - processed_step_XXXX.png: Visualization of processed_data
        Also prints basic statistics about the raw data
    """
    # Create debug directory if it doesn't exist
    os.makedirs('./debug_vis', exist_ok=True)
    
    # Save raw data visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(raw_data, cmap='viridis')
    plt.colorbar()
    plt.title(f'Raw Data - Step {step_num}')
    plt.savefig(f'./debug_vis/raw_step_{step_num:04d}.png')
    plt.close()
    
    # Print statistics about raw data
    print(f"\nStep {step_num} raw data stats:")
    print(f"Min: {np.min(raw_data):.6f}")
    print(f"Max: {np.max(raw_data):.6f}")
    print(f"Mean: {np.mean(raw_data):.6f}")
    print(f"Shape: {raw_data.shape}")
    print("Sample values:")
    print(raw_data[0:3, 0:3])
    print("-------------------------")
    
    # Save processed data visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(processed_data, cmap='viridis')
    plt.colorbar()
    plt.title(f'Processed Data - Step {step_num}')
    plt.savefig(f'./debug_vis/processed_step_{step_num:04d}.png')
    plt.close()
