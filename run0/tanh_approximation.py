import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
from scipy.optimize import minimize
from matplotlib.colors import TwoSlopeNorm
from models_pettingzoo import SharedPolicyMADDPG
from utils import load_config

# Set up matplotlib for consistent styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.titlesize": 18
})

# Define color scheme
TEAL = '#008080'
ORANGE = '#FF8C00'
NAVY = '#000080'
PURPLE = '#800080'

def load_trained_model(checkpoint_path, config_path):
    """Load the trained MADDPG model"""
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    grid_i = config['grid']['target']['i']
    grid_j = config['grid']['target']['j']
    agents = [f"agent_{i}_{j}" for i in range(grid_i) for j in range(grid_j)]
    
    halo = config.get('halo', 0)
    obs_height = 2 * halo + 1 if halo > 0 else 1
    obs_width = 2 * halo + 1 if halo > 0 else 1
    obs_shape = (obs_height, obs_width, 2)
    act_shape = (1,)
    
    pi_arch = config.get('net_arch', {}).get('pi', [64, 64])
    qf_arch = config.get('net_arch', {}).get('qf', [64, 64, 64])
    
    maddpg = SharedPolicyMADDPG(
        agents=agents,
        obs_shape=obs_shape,
        act_shape=act_shape,
        device='cpu',
        pi_arch=pi_arch,
        qf_arch=qf_arch
    )
    
    if 'maddpg_state_dict' in checkpoint:
        maddpg.load_state_dict(checkpoint['maddpg_state_dict'])
    else:
        maddpg.load_state_dict(checkpoint)
    
    return maddpg, config

def calculate_neural_responses(maddpg, config, u_values, w_values):
    """Calculate the neural network responses for given u and w values"""
    maddpg.actor.eval()
    device = next(maddpg.actor.parameters()).device
    
    halo = config.get('halo', 0)
    obs_height = 2 * halo + 1 if halo > 0 else 1
    obs_width = 2 * halo + 1 if halo > 0 else 1
    
    resolution = len(u_values)
    responses = np.zeros((resolution, resolution))
    
    for i, u_val in enumerate(u_values):
        for j, w_val in enumerate(w_values):
            if halo == 0:
                obs = np.zeros((1, 1, 1, 2), dtype=np.float32)
                obs[0, 0, 0, 0] = u_val * config['action']['om_max']
                obs[0, 0, 0, 1] = w_val * config['action']['om_max']
            else:
                obs = np.zeros((1, obs_height, obs_width, 2), dtype=np.float32)
                obs[0, :, :, 0] = u_val * config['action']['om_max']
                obs[0, :, :, 1] = w_val * config['action']['om_max']
            
            flat_obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
            
            with torch.no_grad():
                action = maddpg.actor(flat_obs).cpu().numpy()
            
            responses[j, i] = action[0, 0]
    
    return responses

def asymmetric_tanh(u, w, params):
    """
    Asymmetric tanh with different scaling and steepness on each side
    params = [a1, a2, a3, A_pos, A_neg, B_pos, B_neg]
    """
    a1, a2, a3, A_pos, A_neg, B_pos, B_neg = params
    
    arg = a1 * u + a2 * w + a3
    
    result = np.where(arg >= 0, 
                     A_pos * np.tanh(B_pos * arg), 
                     A_neg * np.tanh(B_neg * arg))
    
    return result

def fit_asymmetric_tanh(u_grid, w_grid, responses):
    """Fit the asymmetric tanh function to the neural network responses"""
    
    u_flat = u_grid.flatten()
    w_flat = w_grid.flatten()
    response_flat = responses.flatten()
    
    def objective(params):
        predicted = asymmetric_tanh(u_flat, w_flat, params)
        return np.mean((predicted - response_flat) ** 2)
    
    resp_min = responses.min()
    resp_max = responses.max()
    
    initial_params = [
        1.0,           # a1
        1.0,           # a2
        0.0,           # a3
        abs(resp_max), # A_pos
        abs(resp_min), # A_neg
        3.0,           # B_pos
        3.0            # B_neg
    ]
    
    bounds = [
        (-3, 3),       # a1
        (-3, 3),       # a2  
        (-1, 1),       # a3
        (0.1, 2.0),    # A_pos
        (0.1, 2.0),    # A_neg
        (0.1, 40),     # B_pos
        (0.1, 40)      # B_neg
    ]
    
    best_result = None
    best_mse = float('inf')
    
    for attempt in range(5):
        perturbed_params = initial_params.copy()
        if attempt > 0:
            perturbed_params[0] += np.random.normal(0, 0.2)  # a1
            perturbed_params[1] += np.random.normal(0, 0.2)  # a2
            perturbed_params[5] += np.random.normal(0, 1.0)  # B_pos
            perturbed_params[6] += np.random.normal(0, 1.0)  # B_neg
        
        result = minimize(objective, perturbed_params, bounds=bounds, method='L-BFGS-B')
        
        if result.success and result.fun < best_mse:
            best_result = result
            best_mse = result.fun
    
    if best_result and best_result.success:
        print("Optimization successful!")
        print(f"Final MSE: {best_result.fun:.6f}")
        print(f"Optimized parameters: {best_result.x}")
    else:
        print("Optimization failed!")
        if best_result:
            print(f"Message: {best_result.message}")
        best_result = type('obj', (object,), {'x': initial_params, 'fun': objective(initial_params)})
    
    return best_result.x, best_result.fun

def plot_comparison(u_values, w_values, neural_responses, fitted_params, save_path=None):
    """Plot comparison between neural network and fitted tanh approximation"""
    
    U, W = np.meshgrid(u_values, w_values)
    fitted_responses = asymmetric_tanh(U, W, fitted_params)
    residuals = neural_responses - fitted_responses
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Common colormap settings
    resp_min = min(neural_responses.min(), fitted_responses.min())
    resp_max = max(neural_responses.max(), fitted_responses.max())
    
    if resp_min < 0 and resp_max > 0:
        vcenter = 0
    else:
        vcenter = (resp_min + resp_max) / 2
    
    if vcenter <= resp_min:
        resp_min = vcenter - 0.01
    if vcenter >= resp_max:
        resp_max = vcenter + 0.01
        
    divnorm = TwoSlopeNorm(vmin=resp_min, vcenter=vcenter, vmax=resp_max)
    cmap = plt.cm.RdBu_r
    
    # 1. Original neural network response
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(U, W, neural_responses, levels=50, cmap=cmap, norm=divnorm)
    contour_lines1 = ax1.contour(U, W, neural_responses, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    ax1.clabel(contour_lines1, inline=True, fontsize=8, fmt='%.2f')
    ax1.set_xlabel('u (Normalized Streamwise Velocity)')
    ax1.set_ylabel('w (Normalized Spanwise Velocity)')
    ax1.set_title('Original Neural Network Response')
    ax1.grid(True, linestyle='--', alpha=0.3)
    plt.colorbar(contour1, ax=ax1, label='Actor Response')
    
    # 2. Fitted tanh approximation
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(U, W, fitted_responses, levels=50, cmap=cmap, norm=divnorm)
    contour_lines2 = ax2.contour(U, W, fitted_responses, levels=10, colors='k', linewidths=0.5, alpha=0.5)
    ax2.clabel(contour_lines2, inline=True, fontsize=8, fmt='%.2f')
    ax2.set_xlabel('u (Normalized Streamwise Velocity)')
    ax2.set_ylabel('w (Normalized Spanwise Velocity)')
    ax2.set_title('Fitted Asymmetric Tanh Approximation')
    ax2.grid(True, linestyle='--', alpha=0.3)
    plt.colorbar(contour2, ax=ax2, label='Actor Response')
    
    # 3. Residuals
    ax3 = axes[1, 0]
    residual_max = max(abs(residuals.min()), abs(residuals.max()))
    residual_norm = TwoSlopeNorm(vmin=-residual_max, vcenter=0, vmax=residual_max)
    contour3 = ax3.contourf(U, W, residuals, levels=50, cmap=plt.cm.RdBu_r, norm=residual_norm)
    ax3.set_xlabel('u (Normalized Streamwise Velocity)')
    ax3.set_ylabel('w (Normalized Spanwise Velocity)')
    ax3.set_title('Residuals (Neural Network - Fitted)')
    ax3.grid(True, linestyle='--', alpha=0.3)
    plt.colorbar(contour3, ax=ax3, label='Residual')
    
    # 4. Diagonal line interpolation plot
    ax4 = axes[1, 1]
    
    # Create a parameterized line from (u=-1, w=1) to (u=1, w=-1)
    t_values = np.linspace(-1, 1, 100)
    u_line = t_values
    w_line = -t_values
    
    # Get responses along this line
    neural_line = []
    fitted_line = []
    
    for u_val, w_val in zip(u_line, w_line):
        u_idx = np.argmin(np.abs(u_values - u_val))
        w_idx = np.argmin(np.abs(w_values - w_val))
        
        neural_line.append(neural_responses[w_idx, u_idx])
        fitted_line.append(asymmetric_tanh(u_val, w_val, fitted_params))
    
    ax4.plot(t_values, neural_line, '-', color=TEAL, linewidth=3, 
            label='Neural Network', alpha=0.9)
    ax4.plot(t_values, fitted_line, '--', color=ORANGE, linewidth=3, 
            label='Fitted Tanh', alpha=0.9)
    
    ax4.set_xlabel('t (from (u=-1,w=1) at t=-1 to (u=1,w=-1) at t=1)')
    ax4.set_ylabel('Actor Response')
    ax4.set_title('Response Along Main Diagonal (u=t, w=-t)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add statistics
    mse = np.mean(residuals**2)
    mae = np.mean(np.abs(residuals))
    max_error = np.max(np.abs(residuals))
    r_squared = 1 - np.sum(residuals**2) / np.sum((neural_responses - np.mean(neural_responses))**2)
    
    stats_text = f'MSE: {mse:.6f}\nMAE: {mae:.6f}\nMax Error: {max_error:.6f}\nR²: {r_squared:.6f}'
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return mse, mae, max_error, r_squared

def benchmark_inference_speed(maddpg, config, fitted_params, num_trials=10):
    """Compare inference speed between neural network and tanh approximation"""
    
    print("\n" + "="*50)
    print("INFERENCE SPEED COMPARISON")
    print("="*50)
    
    test_points = [(np.random.uniform(-1, 1), np.random.uniform(-1, 1)) for _ in range(num_trials)]
    
    halo = config.get('halo', 0)
    obs_height = 2 * halo + 1 if halo > 0 else 1
    obs_width = 2 * halo + 1 if halo > 0 else 1
    device = next(maddpg.actor.parameters()).device
    
    nn_times = []
    tanh_times = []
    
    print(f"Testing {num_trials} random points...")
    
    for i, (test_u, test_w) in enumerate(test_points):
        print(f"Trial {i+1}/{num_trials}: u={test_u:.3f}, w={test_w:.3f}")
        
        if halo == 0:
            obs = np.zeros((1, 1, 1, 2), dtype=np.float32)
            obs[0, 0, 0, 0] = test_u * config['action']['om_max']
            obs[0, 0, 0, 1] = test_w * config['action']['om_max']
        else:
            obs = np.zeros((1, obs_height, obs_width, 2), dtype=np.float32)
            obs[0, :, :, 0] = test_u * config['action']['om_max']
            obs[0, :, :, 1] = test_w * config['action']['om_max']
        
        flat_obs = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        
        # Time neural network
        start_time = time.time()
        with torch.no_grad():
            nn_result = maddpg.actor(flat_obs).cpu().numpy()
        nn_time = time.time() - start_time
        nn_times.append(nn_time)
        
        # Time tanh approximation
        start_time = time.time()
        tanh_result = asymmetric_tanh(test_u, test_w, fitted_params)
        tanh_time = time.time() - start_time
        tanh_times.append(tanh_time)
        
        print(f"  NN: {nn_time*1e6:.1f} μs, Tanh: {tanh_time*1e6:.1f} μs, Speedup: {nn_time/tanh_time:.1f}x")
    
    avg_nn_time = np.mean(nn_times)
    avg_tanh_time = np.mean(tanh_times)
    avg_speedup = avg_nn_time / avg_tanh_time
    
    print("\n" + "-"*50)
    print("AVERAGE RESULTS:")
    print(f"Neural Network: {avg_nn_time*1e6:.1f} μs")
    print(f"Tanh Approximation: {avg_tanh_time*1e6:.1f} μs")
    print(f"Average Speedup: {avg_speedup:.1f}x")
    print("="*50)

def plot_uncertainty_analysis(u_values, w_values, fitted_params, save_path=None):
    """
    Plot the analytical function with uncertainty clouds for ±2% and ±5% uncertainty in U and W.
    This creates a separate plot showing uncertainty propagation along the main diagonal.
    """
    
    # Set up the same diagonal line as in the comparison plot
    t_values = np.linspace(-1, 1, 100)
    u_line = t_values
    w_line = -t_values
    
    # Calculate nominal response (no uncertainty)
    nominal_response = []
    for u_val, w_val in zip(u_line, w_line):
        response = asymmetric_tanh(u_val, w_val, fitted_params)
        nominal_response.append(response)
    
    nominal_response = np.array(nominal_response)
    
    # Function to calculate response statistics with uncertainty
    def calculate_uncertainty_bounds(u_line, w_line, fitted_params, uncertainty_percent, n_samples=1000):
        """Calculate response bounds given input uncertainty"""
        responses_samples = []
        
        for u_val, w_val in zip(u_line, w_line):
            # Generate samples with uncertainty
            u_uncertainty = abs(u_val) * uncertainty_percent / 100.0
            w_uncertainty = abs(w_val) * uncertainty_percent / 100.0
            
            u_samples = np.random.normal(u_val, u_uncertainty, n_samples)
            w_samples = np.random.normal(w_val, w_uncertainty, n_samples)
            
            # Clip to valid range [-1, 1] to maintain physical bounds
            u_samples = np.clip(u_samples, -1, 1)
            w_samples = np.clip(w_samples, -1, 1)
            
            # Calculate responses for all samples
            point_responses = []
            for u_s, w_s in zip(u_samples, w_samples):
                response = asymmetric_tanh(u_s, w_s, fitted_params)
                point_responses.append(response)
            
            responses_samples.append(point_responses)
        
        responses_samples = np.array(responses_samples)
        
        # Calculate statistics
        mean_response = np.mean(responses_samples, axis=1)
        std_response = np.std(responses_samples, axis=1)
        percentile_2_5 = np.percentile(responses_samples, 2.5, axis=1)
        percentile_97_5 = np.percentile(responses_samples, 97.5, axis=1)
        percentile_5 = np.percentile(responses_samples, 5, axis=1)
        percentile_95 = np.percentile(responses_samples, 95, axis=1)
        
        return {
            'mean': mean_response,
            'std': std_response,
            'lower_95': percentile_2_5,
            'upper_95': percentile_97_5,
            'lower_90': percentile_5,
            'upper_90': percentile_95
        }
    
    print("Calculating uncertainty bounds for ±2% uncertainty...")
    uncertainty_2pct = calculate_uncertainty_bounds(u_line, w_line, fitted_params, 2.0)
    
    print("Calculating uncertainty bounds for ±5% uncertainty...")
    uncertainty_5pct = calculate_uncertainty_bounds(u_line, w_line, fitted_params, 5.0)
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Define colors (can use existing color scheme)
    LIGHT_BLUE = '#87CEEB'
    LIGHT_ORANGE = '#FFE4B5'
    
    # Plot nominal response
    ax.plot(t_values, nominal_response, '-', color='black', linewidth=3, 
            label='Nominal (No Uncertainty)', alpha=0.9, zorder=10)
    
    # Plot ±5% uncertainty cloud (outer, lighter)
    ax.fill_between(t_values, uncertainty_5pct['lower_95'], uncertainty_5pct['upper_95'], 
                    color=LIGHT_ORANGE, alpha=0.4, label='±5% Uncertainty (95% CI)', zorder=2)
    ax.fill_between(t_values, uncertainty_5pct['lower_90'], uncertainty_5pct['upper_90'], 
                    color=ORANGE, alpha=0.3, label='±5% Uncertainty (90% CI)', zorder=3)
    
    # Plot ±2% uncertainty cloud (inner, darker)
    ax.fill_between(t_values, uncertainty_2pct['lower_95'], uncertainty_2pct['upper_95'], 
                    color=LIGHT_BLUE, alpha=0.6, label='±2% Uncertainty (95% CI)', zorder=4)
    ax.fill_between(t_values, uncertainty_2pct['lower_90'], uncertainty_2pct['upper_90'], 
                    color=TEAL, alpha=0.4, label='±2% Uncertainty (90% CI)', zorder=5)
    
    # Plot mean responses (should be close to nominal)
    ax.plot(t_values, uncertainty_2pct['mean'], '--', color=TEAL, linewidth=2, 
            label='±2% Mean', alpha=0.8, zorder=8)
    ax.plot(t_values, uncertainty_5pct['mean'], '--', color=ORANGE, linewidth=2, 
            label='±5% Mean', alpha=0.8, zorder=7)
    
    # Formatting
    ax.set_xlabel('t (from (u=-1,w=1) at t=-1 to (u=1,w=-1) at t=1)', fontsize=14)
    ax.set_ylabel('Actor Response', fontsize=14)
    ax.set_title('Analytical Function Response with Input Uncertainty\nAlong Main Diagonal (u=t, w=-t)', fontsize=16)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add statistics text box
    max_std_2pct = np.max(uncertainty_2pct['std'])
    max_std_5pct = np.max(uncertainty_5pct['std'])
    
    # Calculate relative uncertainties at key points
    mid_idx = len(t_values) // 2
    rel_unc_2pct_mid = uncertainty_2pct['std'][mid_idx] / abs(nominal_response[mid_idx]) * 100 if abs(nominal_response[mid_idx]) > 1e-6 else 0
    rel_unc_5pct_mid = uncertainty_5pct['std'][mid_idx] / abs(nominal_response[mid_idx]) * 100 if abs(nominal_response[mid_idx]) > 1e-6 else 0
    
    stats_text = (f'Max Std Dev:\n'
                 f'±2%: {max_std_2pct:.4f}\n'
                 f'±5%: {max_std_5pct:.4f}\n\n'
                 f'Relative Uncertainty at t=0:\n'
                 f'±2%: {rel_unc_2pct_mid:.1f}%\n'
                 f'±5%: {rel_unc_5pct_mid:.1f}%')
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Uncertainty analysis plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("UNCERTAINTY PROPAGATION ANALYSIS")
    print("="*60)
    
    print(f"\n±2% Input Uncertainty:")
    print(f"  Maximum standard deviation: {max_std_2pct:.6f}")
    print(f"  Average standard deviation: {np.mean(uncertainty_2pct['std']):.6f}")
    print(f"  95% CI width range: [{np.min(uncertainty_2pct['upper_95'] - uncertainty_2pct['lower_95']):.6f}, {np.max(uncertainty_2pct['upper_95'] - uncertainty_2pct['lower_95']):.6f}]")
    
    print(f"\n±5% Input Uncertainty:")
    print(f"  Maximum standard deviation: {max_std_5pct:.6f}")
    print(f"  Average standard deviation: {np.mean(uncertainty_5pct['std']):.6f}")
    print(f"  95% CI width range: [{np.min(uncertainty_5pct['upper_95'] - uncertainty_5pct['lower_95']):.6f}, {np.max(uncertainty_5pct['upper_95'] - uncertainty_5pct['lower_95']):.6f}]")
    
    print("\nUncertainty amplification factor (5% vs 2%):")
    amplification = np.mean(uncertainty_5pct['std']) / np.mean(uncertainty_2pct['std'])
    print(f"  Average: {amplification:.2f}x")
    print("="*60)
    
    return uncertainty_2pct, uncertainty_5pct

def main(checkpoint_path, config_path="config.yaml", resolution=50, save_path=None, uncertainty_save_path=None):
    """Main function to reproduce the response map and fit tanh approximation with uncertainty analysis"""
    
    print("Loading trained model...")
    maddpg, config = load_trained_model(checkpoint_path, config_path)
    
    print("Calculating neural network responses...")
    u_values = np.linspace(-1.0, 1.0, resolution)
    w_values = np.linspace(-1.0, 1.0, resolution)
    neural_responses = calculate_neural_responses(maddpg, config, u_values, w_values)
    
    print("Fitting asymmetric tanh approximation...")
    U, W = np.meshgrid(u_values, w_values)
    fitted_params, mse = fit_asymmetric_tanh(U, W, neural_responses)
    
    print(f"\nFitted parameters:")
    param_names = ['a1', 'a2', 'a3', 'A_pos', 'A_neg', 'B_pos', 'B_neg']
    for name, param in zip(param_names, fitted_params):
        print(f"{name}: {param:.4f}")
    
    print("\nCreating comparison plots...")
    stats = plot_comparison(u_values, w_values, neural_responses, fitted_params, save_path)
    
    print(f"\nFit quality:")
    print(f"MSE: {stats[0]:.6f}")
    print(f"MAE: {stats[1]:.6f}")
    print(f"Max Error: {stats[2]:.6f}")
    print(f"R²: {stats[3]:.6f}")
    
    # Add uncertainty analysis
    if uncertainty_save_path:
        print("\nCreating uncertainty analysis plot...")
        uncertainty_results = plot_uncertainty_analysis(u_values, w_values, fitted_params, uncertainty_save_path)
    else:
        uncertainty_results = None
    
    benchmark_inference_speed(maddpg, config, fitted_params, num_trials=10)

    # Save parameters with metadata
    param_names = ['a1', 'a2', 'a3', 'A_pos', 'A_neg', 'B_pos', 'B_neg']
    param_dict = {
        'parameters': fitted_params,
        'param_names': param_names,
        'comment': f'Asymmetric tanh parameters fitted to MADDPG actor. MSE: {mse:.6f}. Function: f(u,w) = A_pos*tanh(B_pos*(a1*u + a2*w + a3)) if arg>=0 else A_neg*tanh(B_neg*arg)',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    np.save('tanh_parameters.npy', param_dict)
    print(f"Parameters saved to tanh_parameters.npy")

    # To load the parameters
    #loaded = np.load('tanh_parameters.npy', allow_pickle=True).item()
    #fitted_params = loaded['parameters']
    #print(f"Comment: {loaded['comment']}")
    
    return fitted_params, stats, uncertainty_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit asymmetric tanh approximation to trained MADDPG actor response')
    parser.add_argument('--checkpoint', type=str, required=True,
                      help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--resolution', type=int, default=50,
                      help='Grid resolution for response map')
    parser.add_argument('--save_path', type=str, default='tanh_approximation_comparison.png',
                      help='Path to save comparison plot')
    parser.add_argument('--uncertainty_save_path', type=str, default='uncertainty_analysis.png',
                      help='Path to save uncertainty analysis plot')
    
    args = parser.parse_args()
    
    fitted_params, stats, uncertainty_results = main(args.checkpoint, args.config, args.resolution, args.save_path, None)