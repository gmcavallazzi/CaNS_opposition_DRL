import matplotlib.pyplot as plt
import os
import numpy as np
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def compute_reward(dpdx, config):
    # Reference uncontrolled dpdx value (most negative)
    dpdx_uncontrolled = config['reward']['dpdx']['min']

    # Calculate percentage reduction
    # As dpdx gets less negative (closer to zero), this value increases
    reduction = 1.0 - (dpdx / dpdx_uncontrolled)

    return reduction

def compute_diversity_penalty(actions, config):
    """
    Compute diversity penalty to encourage action variation across agents.

    Args:
        actions: Dictionary of agent actions or numpy array of action values
        config: Configuration dictionary containing diversity parameters

    Returns:
        diversity_penalty: Penalty value (negative when diversity is low)
    """
    diversity_config = config.get('reward', {}).get('diversity', {})

    # Return 0 if diversity penalty is disabled
    if not diversity_config.get('enable', False):
        return 0.0

    # Convert actions to numpy array if needed
    if isinstance(actions, dict):
        action_values = np.array([action[0] if isinstance(action, (list, np.ndarray)) else action
                                for action in actions.values()])
    else:
        action_values = np.array(actions).flatten()

    # Calculate action standard deviation (measure of diversity)
    action_std = np.std(action_values)

    # Get parameters
    weight = diversity_config.get('weight', -0.1)
    min_std_threshold = diversity_config.get('min_std_threshold', 0.01)

    # Penalty is applied when std is below threshold
    # Penalty = weight * (1 - std/threshold) when std < threshold, 0 otherwise
    if action_std < min_std_threshold:
        penalty_factor = 1.0 - (action_std / min_std_threshold)
        diversity_penalty = weight * penalty_factor
    else:
        diversity_penalty = 0.0

    return diversity_penalty

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
