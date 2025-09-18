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