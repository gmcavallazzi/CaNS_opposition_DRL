import matplotlib.pyplot as plt
import os
import numpy as np
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def compute_reward(dpdx, config):
    # Reference uncontrolled dpdx value (natural flow without control)
    dpdx_uncontrolled = config['reward']['dpdx']['uncontrolled']

    # Calculate percentage reduction from uncontrolled baseline
    # As dpdx gets less negative (closer to zero), this value increases
    # When dpdx = dpdx_uncontrolled (no improvement): reduction = 0
    # When dpdx > dpdx_uncontrolled (worse): reduction < 0
    # When dpdx < dpdx_uncontrolled (better): reduction > 0
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

def compute_gradient_metrics(network) -> dict:
    """
    Compute gradient health metrics for a network.

    Args:
        network: PyTorch neural network

    Returns:
        Dictionary of gradient metrics
    """
    metrics = {}

    # Global gradient norm
    total_norm = 0.0
    param_count = 0
    grad_norms = []

    for name, param in network.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            grad_norms.append(param_norm)
            total_norm += param_norm ** 2
            param_count += 1

    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
        metrics['global_norm'] = total_norm
        metrics['mean_norm'] = sum(grad_norms) / len(grad_norms)
        metrics['max_norm'] = max(grad_norms) if grad_norms else 0.0
        metrics['min_norm'] = min(grad_norms) if grad_norms else 0.0
        metrics['std_norm'] = np.std(grad_norms) if len(grad_norms) > 1 else 0.0
    else:
        metrics = {k: 0.0 for k in ['global_norm', 'mean_norm', 'max_norm', 'min_norm', 'std_norm']}

    return metrics

def compute_layer_wise_gradients(network, network_name: str) -> dict:
    """
    Compute layer-wise gradient norms for detailed analysis.

    Args:
        network: PyTorch neural network
        network_name: Name prefix for logging ('actor' or 'critic')

    Returns:
        Dictionary mapping layer names to gradient norms
    """
    layer_metrics = {}

    for name, param in network.named_parameters():
        if param.grad is not None:
            layer_metrics[f"{network_name}/{name}"] = param.grad.data.norm(2).item()

    return layer_metrics

def check_gradient_health(metrics: dict, config: dict) -> tuple:
    """
    Check gradient health and determine if adjustments are needed.

    Args:
        metrics: Gradient metrics from compute_gradient_metrics
        config: Training configuration

    Returns:
        (is_exploding, is_vanishing, suggested_clip_value)
    """
    grad_config = config.get('training', {}).get('gradient_monitoring', {})
    explosion_threshold = float(grad_config.get('explosion_threshold', 10.0))
    vanishing_threshold = float(grad_config.get('vanishing_threshold', 1e-6))

    global_norm = metrics.get('global_norm', 0.0)

    is_exploding = global_norm > explosion_threshold
    is_vanishing = global_norm < vanishing_threshold

    # Suggest adaptive clipping value (slightly above current norm if exploding)
    if is_exploding:
        suggested_clip = explosion_threshold * 0.8
    else:
        suggested_clip = max(1.0, global_norm * 1.2)  # Allow some headroom

    return is_exploding, is_vanishing, suggested_clip

def adjust_learning_rates(optimizer, is_exploding: bool, config: dict):
    """
    Adjust learning rates when gradients explode.

    Args:
        optimizer: PyTorch optimizer
        is_exploding: Whether gradients are exploding
        config: Training configuration
    """
    if is_exploding:
        grad_config = config.get('training', {}).get('gradient_monitoring', {})
        reduction_factor = grad_config.get('lr_reduction_factor', 0.8)

        # Reduce learning rates
        for param_group in optimizer.param_groups:
            param_group['lr'] *= reduction_factor

def check_for_nan_inf(loss: float, metrics: dict, network_name: str) -> bool:
    """
    Check for NaN or Inf in loss and gradients.

    Args:
        loss: Loss value to check
        metrics: Gradient metrics dictionary
        network_name: Name of the network for logging

    Returns:
        True if NaN/Inf detected, False otherwise
    """
    import math

    # Check loss
    if math.isnan(loss) or math.isinf(loss):
        print(f"⚠️  NaN/Inf detected in {network_name} loss: {loss}")
        return True

    # Check gradients
    for key, value in metrics.items():
        if math.isnan(value) or math.isinf(value):
            print(f"⚠️  NaN/Inf detected in {network_name} gradient {key}: {value}")
            return True

    return False

def check_activation_health(network, sample_input: 'torch.Tensor', network_name: str) -> dict:
    """
    Check activation statistics to detect saturation or dead neurons.

    Args:
        network: PyTorch neural network
        sample_input: Sample input tensor
        network_name: Name for logging

    Returns:
        Dictionary of activation statistics
    """
    import torch

    activation_stats = {}

    # Hook to capture activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # Register hooks
    hooks = []
    for name, module in network.named_modules():
        if isinstance(module, (torch.nn.ReLU, torch.nn.Tanh)):
            hooks.append(module.register_forward_hook(get_activation(name)))

    # Forward pass
    with torch.no_grad():
        _ = network(sample_input)

    # Compute statistics
    for name, activation in activations.items():
        flat_act = activation.flatten()
        activation_stats[f"{network_name}/{name}/mean"] = flat_act.mean().item()
        activation_stats[f"{network_name}/{name}/std"] = flat_act.std().item()
        activation_stats[f"{network_name}/{name}/dead_ratio"] = (flat_act == 0).float().mean().item()

        # For tanh, check saturation
        if 'tanh' in name.lower():
            saturated = ((flat_act > 0.95) | (flat_act < -0.95)).float().mean().item()
            activation_stats[f"{network_name}/{name}/saturated_ratio"] = saturated

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activation_stats

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
