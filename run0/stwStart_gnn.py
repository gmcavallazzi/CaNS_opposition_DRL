import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, Dict, Any, List
import json
import argparse
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt

from stwEnv_pettingzoo import STWParallelEnv
from utils import (
    load_config,
    compute_gradient_metrics,
    compute_layer_wise_gradients,
    check_gradient_health,
    adjust_learning_rates,
    check_for_nan_inf,
    check_activation_health
)
from models_pettingzoo import (
    SharedPolicyMADDPG,
    BatchedReplayBuffer
)
import time

def compute_noise_scale(current_episode: int, config: Dict[str, Any]) -> float:
    """
    Compute noise scale based on episode progress and decay strategy.

    Args:
        current_episode: Current episode number
        config: Configuration dictionary containing action_noise parameters

    Returns:
        Current noise scale value
    """
    action_noise_config = config['model']['action_noise']
    initial_sigma = action_noise_config['initial_sigma']
    final_sigma = action_noise_config['final_sigma']
    decay_episodes = action_noise_config['decay_episodes']
    decay_type = action_noise_config['decay_type']

    # If we haven't reached decay episodes yet, compute decay
    if current_episode < decay_episodes:
        progress = current_episode / decay_episodes

        if decay_type == "linear":
            noise_scale = initial_sigma + (final_sigma - initial_sigma) * progress
        elif decay_type == "exponential":
            # Exponential decay: sigma = initial * exp(ln(final/initial) * progress)
            decay_factor = np.log(final_sigma / initial_sigma)
            noise_scale = initial_sigma * np.exp(decay_factor * progress)
        elif decay_type == "polynomial":
            # Polynomial decay (quadratic) for smoother transition
            noise_scale = initial_sigma + (final_sigma - initial_sigma) * (progress ** 2)
        else:
            raise ValueError(f"Unknown decay type: {decay_type}")
    else:
        # After decay period, use final sigma
        noise_scale = final_sigma

    return noise_scale

def create_field_visualization(u_field: np.ndarray, w_field: np.ndarray,
                               action_field: np.ndarray, episode: int = 0) -> np.ndarray:
    """
    Create a visualization of velocity and action fields.

    Args:
        u_field: [64, 64] u-velocity field
        w_field: [64, 64] w-velocity field
        action_field: [64, 64] action field
        episode: Episode number for title

    Returns:
        RGB image as numpy array [H, W, 3]
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Compute statistics
    u_stats = f"μ={np.mean(u_field):.4f}, σ={np.std(u_field):.4f}"
    w_stats = f"μ={np.mean(w_field):.4f}, σ={np.std(w_field):.4f}"
    action_stats = f"μ={np.mean(action_field):.4f}, σ={np.std(action_field):.4f}"

    # U-velocity field
    im0 = axes[0].imshow(u_field, cmap='RdBu_r', aspect='auto', origin='lower')
    axes[0].set_title(f'U-velocity field\n{u_stats}', fontsize=10)
    axes[0].set_xlabel('j (spanwise)', fontsize=9)
    axes[0].set_ylabel('i (streamwise)', fontsize=9)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # W-velocity field
    im1 = axes[1].imshow(w_field, cmap='RdBu_r', aspect='auto', origin='lower')
    axes[1].set_title(f'W-velocity field\n{w_stats}', fontsize=10)
    axes[1].set_xlabel('j (spanwise)', fontsize=9)
    axes[1].set_ylabel('i (streamwise)', fontsize=9)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Action field
    im2 = axes[2].imshow(action_field, cmap='seismic', aspect='auto',
                         origin='lower', vmin=-1, vmax=1)
    axes[2].set_title(f'Action field (Episode {episode})\n{action_stats}', fontsize=10)
    axes[2].set_xlabel('j (spanwise)', fontsize=9)
    axes[2].set_ylabel('i (streamwise)', fontsize=9)
    cbar = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Action value', fontsize=9)

    # Add grid lines to show 8x8 patches
    for ax in axes:
        for i in range(0, 65, 8):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # Convert to numpy array for TensorBoard
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return image

def create_action_change_visualization(action_t: np.ndarray, action_t_prev: np.ndarray,
                                       episode: int = 0) -> np.ndarray:
    """
    Visualize temporal action changes to detect boom-boom oscillations.

    Args:
        action_t: [64, 64] current action field
        action_t_prev: [64, 64] previous action field
        episode: Episode number

    Returns:
        RGB image as numpy array [H, W, 3]
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Current actions
    im0 = axes[0].imshow(action_t, cmap='seismic', aspect='auto',
                         origin='lower', vmin=-1, vmax=1)
    axes[0].set_title(f'Actions(t) - Episode {episode}', fontsize=10)
    axes[0].set_xlabel('j (spanwise)', fontsize=9)
    axes[0].set_ylabel('i (streamwise)', fontsize=9)
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Previous actions
    im1 = axes[1].imshow(action_t_prev, cmap='seismic', aspect='auto',
                        origin='lower', vmin=-1, vmax=1)
    axes[1].set_title(f'Actions(t-1)', fontsize=10)
    axes[1].set_xlabel('j (spanwise)', fontsize=9)
    axes[1].set_ylabel('i (streamwise)', fontsize=9)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Action change (temporal gradient)
    action_change = action_t - action_t_prev
    max_change = max(abs(np.min(action_change)), abs(np.max(action_change)), 0.1)

    im2 = axes[2].imshow(action_change, cmap='RdBu_r', aspect='auto',
                        origin='lower', vmin=-max_change, vmax=max_change)
    change_stats = f"μ={np.mean(action_change):.4f}, σ={np.std(action_change):.4f}\nmax|Δ|={np.max(np.abs(action_change)):.4f}"
    axes[2].set_title(f'Δ Actions (boom-boom indicator)\n{change_stats}', fontsize=10)
    axes[2].set_xlabel('j (spanwise)', fontsize=9)
    axes[2].set_ylabel('i (streamwise)', fontsize=9)
    cbar = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label('Action change', fontsize=9)

    # Add grid lines for 8x8 patches
    for ax in axes:
        for i in range(0, 65, 8):
            ax.axhline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(i - 0.5, color='black', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # Convert to numpy array
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return image

def train_maddpg(
    config: Dict[str, Any],
    checkpoint_dir: str = "./checkpoints_pettingzoo",
    logs_dir: str = "./logs_pettingzoo",
    device: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    resume_buffer: Optional[str] = None,
):
    """
    Train MADDPG agents with CNN policies for patch-based flow control.

    Args:
        config: Configuration dictionary
        checkpoint_dir: Directory for saving checkpoints
        logs_dir: Directory for tensorboard logs
        device: Device to run on ("cuda" or "cpu")
        resume_checkpoint: Path to checkpoint to resume from
        resume_buffer: Path to replay buffer to resume from
    """
    # Set device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Create environment
    env = STWParallelEnv(config)

    # Get agent list and observation/action spaces
    agents = env.possible_agents
    obs_shape = env.observation_spaces[agents[0]].shape  # (2, 8, 8)
    act_shape = env.action_spaces[agents[0]].shape  # (8, 8)
    num_agents = len(agents)  # 64 agents

    print(f"Environment created with {num_agents} agents")
    print(f"Observation shape per agent: {obs_shape}")
    print(f"Action shape per agent: {act_shape}")

    # Extract training parameters from config
    max_steps = config['total_timesteps']
    batch_size = config['model']['batch_size']
    gradient_steps = config['model']['gradient_steps']
    train_freq = config['model']['train_freq']
    save_freq = config['training']['save_freq']
    buffer_size = config['model']['buffer_size']

    # Memory estimation
    obs_size = np.prod(obs_shape)  # 2 * 8 * 8 = 128
    act_size = np.prod(act_shape)  # 8 * 8 = 64
    estimated_memory_gb = buffer_size * num_agents * (obs_size + act_size) * 4 / (1024**3)

    print(f"\nMemory usage estimation:")
    print(f"- Buffer size: {buffer_size:,}")
    print(f"- Observation size per agent: {obs_size}")
    print(f"- Action size per agent: {act_size}")
    print(f"- Estimated replay buffer memory: {estimated_memory_gb:.2f} GB")

    if estimated_memory_gb > 50:
        print(f"\n⚠️  WARNING: High memory usage detected!")
        print(f"   Estimated memory: {estimated_memory_gb:.1f} GB")
        print(f"   Consider reducing buffer_size in config.yaml")

    # Create shared policy MADDPG trainer with smoothness constraints
    maddpg = SharedPolicyMADDPG(
        agents=agents,
        gamma=config['model']['gamma'],
        tau=config['model']['tau'],
        lr=config['model']['learning_rate'],
        weight_decay=config['model']['weight_decay'],
        device=device,
        actor_channels=config.get('net_arch', {}).get('actor_channels', [16, 32]),
        critic_conv_channels=config.get('net_arch', {}).get('critic_conv', [32, 64, 32]),
        critic_mlp_layers=config.get('net_arch', {}).get('critic_mlp', [256, 128]),
        gradient_clip=config['training']['gradient_clip'],
        lambda_temporal=config['model']['smoothness'].get('lambda_temporal', 0.1),
        lambda_spatial=config['model']['smoothness'].get('lambda_spatial', 0.05),
        lambda_zero=config['model']['smoothness'].get('lambda_zero', 0.01),
        use_gnn=True
    )

    # Initialize replay buffer with previous action tracking
    replay_buffer = BatchedReplayBuffer(buffer_size, num_agents, agents)

    # Resume from checkpoint if provided
    total_steps = 0
    episode = 0
    best_reward = float('-inf')

    if resume_checkpoint:
        print(f"Loading checkpoint from {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        maddpg.load_state_dict(checkpoint['maddpg_state_dict'])
        total_steps = checkpoint.get('total_steps', 0)
        episode = checkpoint.get('episode', 0)
        best_reward = checkpoint.get('best_reward', float('-inf'))

        if resume_buffer:
            print(f"Loading replay buffer from {resume_buffer}")
            replay_buffer.load(resume_buffer)

    # Setup tensorboard writer
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"maddpg_cnn_agents{num_agents}_{current_time}"
    writer = SummaryWriter(f"{logs_dir}/{run_name}")

    # Save config
    with open(os.path.join(logs_dir, f"{run_name}_config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    # Initialize training metrics
    training_start_time = datetime.now()

    # Episode tracking metrics
    episode_rewards = {agent: 0.0 for agent in agents}
    episode_steps = 0
    episode_dpdx_values = []
    episode_q_values = []
    episode_actor_losses = []
    episode_critic_losses = []
    episode_temporal_losses = []
    episode_spatial_losses = []
    episode_zero_losses = []
    episode_actions = []

    # Action statistics tracking
    episode_reward_history = []
    episode_action_mean_history = []
    episode_action_std_history = []

    # Reset environment
    observations, infos = env.reset()

    # Initialize previous actions (all zeros for first step)
    prev_actions = {agent: np.zeros((8, 8), dtype=np.float32) for agent in agents}

    # Track action matrix for visualization
    prev_action_matrix = np.zeros((64, 64), dtype=np.float32)

    print("Starting training...")
    try:
        while total_steps < max_steps:
            # Select actions for all agents using batched processing
            # Stack all observations into a single tensor
            all_obs = np.stack([observations[agent] for agent in agents]).astype(np.float32)

            # Convert to tensor: [64, 2, 8, 8]
            all_obs_tensor = torch.FloatTensor(all_obs).to(device)

            # Get actions for all agents in a single forward pass
            with torch.no_grad():
                all_actions = maddpg.select_actions_batched(all_obs_tensor)  # [64, 8, 8]

                # Add exploration noise with decay schedule
                noise_scale = compute_noise_scale(episode, config)
                noise = np.random.normal(0, noise_scale, size=all_actions.shape)
                all_actions = np.clip(all_actions + noise, -1, 1)

            # Store raw actions for logging
            episode_actions.extend(all_actions.flatten())

            # Calculate action statistics for this step
            action_mean = np.mean(all_actions)
            action_std = np.std(all_actions)

            # Track action statistics
            episode_action_mean_history.append(action_mean)
            episode_action_std_history.append(action_std)

            # Convert to dictionary
            actions = {agent: all_actions[i] for i, agent in enumerate(agents)}

            # Step environment with actions
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            # Prepare batch data for replay buffer
            obs_batch = np.stack([observations[agent] for agent in agents])  # [64, 2, 8, 8]
            next_obs_batch = np.stack([next_observations[agent] for agent in agents])  # [64, 2, 8, 8]
            act_batch = np.stack([actions[agent] for agent in agents])  # [64, 8, 8]
            prev_act_batch = np.stack([prev_actions[agent] for agent in agents])  # [64, 8, 8]
            rew_batch = np.array([rewards[agent] for agent in agents])  # [64]
            done_batch = np.array([terminations[agent] for agent in agents])  # [64]

            # Add batch to buffer with previous actions
            replay_buffer.add_batch(
                obs_batch, act_batch, prev_act_batch, rew_batch, next_obs_batch, done_batch
            )

            # Update previous actions for next step
            prev_actions = actions.copy()

            # Update rewards
            for agent in agents:
                episode_rewards[agent] += rewards[agent]

            # Mean reward for this step
            step_mean_reward = np.mean(list(rewards.values()))
            episode_reward_history.append(step_mean_reward)

            # Store dpdx for the episode
            first_agent = agents[0]
            dpdx_value = infos[first_agent].get('dpdx', 0)
            episode_dpdx_values.append(dpdx_value)

            observations = next_observations
            episode_steps += 1
            total_steps += 1

            # Train networks
            if replay_buffer.size > batch_size and total_steps % train_freq == 0:
                batch_critic_losses = []
                batch_actor_losses = []
                batch_temporal_losses = []
                batch_spatial_losses = []
                batch_zero_losses = []
                batch_q_values = []

                for _ in range(gradient_steps):
                    # Sample batch from buffer
                    batch = replay_buffer.sample(batch_size, device)

                    # Batched update for shared policy
                    critic_loss, actor_loss, loss_breakdown = maddpg.update_batched(batch)

                    # ========== NaN/Inf Detection (CRITICAL) ==========
                    # Check critic
                    critic_metrics = compute_gradient_metrics(maddpg.critic)
                    if check_for_nan_inf(critic_loss, critic_metrics, "Critic"):
                        print("❌ Training failed: NaN/Inf detected in critic!")
                        print(f"   Critic loss: {critic_loss}")
                        print(f"   Last 5 critic losses: {batch_critic_losses[-5:]}")
                        raise ValueError("NaN/Inf detected in critic - training terminated")

                    # Check actor
                    actor_metrics = compute_gradient_metrics(maddpg.actor)
                    if check_for_nan_inf(actor_loss, actor_metrics, "Actor"):
                        print("❌ Training failed: NaN/Inf detected in actor!")
                        print(f"   Actor loss: {actor_loss}")
                        print(f"   Loss breakdown: {loss_breakdown}")
                        raise ValueError("NaN/Inf detected in actor - training terminated")

                    # Check if smoothness penalties are dominating (prevents learning)
                    q_loss_abs = abs(loss_breakdown['q_loss'])
                    total_smoothness = (loss_breakdown['temporal_loss'] +
                                       loss_breakdown['spatial_loss'] +
                                       loss_breakdown['zero_loss'])

                    if q_loss_abs > 0 and total_smoothness / q_loss_abs > 10.0:
                        print(f"⚠️  Warning: Smoothness losses dominating!")
                        print(f"   Q-loss: {q_loss_abs:.6f}")
                        print(f"   Smoothness total: {total_smoothness:.6f}")
                        print(f"   Ratio: {total_smoothness / q_loss_abs:.2f}x")
                        print(f"   Consider reducing lambda_temporal/spatial/zero")

                    # Record losses for episode tracking
                    batch_critic_losses.append(critic_loss)
                    batch_actor_losses.append(actor_loss)
                    batch_temporal_losses.append(loss_breakdown['temporal_loss'])
                    batch_spatial_losses.append(loss_breakdown['spatial_loss'])
                    batch_zero_losses.append(loss_breakdown['zero_loss'])

                    # Extract Q-values for tracking
                    with torch.no_grad():
                        obs_batch_sample = batch['obs']
                        act_batch_sample = batch['acts']

                        # Prepare spatial data for critic
                        obs_fields, act_field = maddpg._prepare_spatial_data(
                            obs_batch_sample, act_batch_sample
                        )
                        q_values = maddpg.critic(obs_fields, act_field)
                        batch_q_values.append(q_values.mean().item())

                # Gradient monitoring and adaptive control
                grad_config = config.get('training', {}).get('gradient_monitoring', {})
                if grad_config.get('enable', False) and total_steps % grad_config.get('log_frequency', 100) == 0:
                    # Compute gradient metrics for both networks
                    actor_metrics = compute_gradient_metrics(maddpg.actor)
                    critic_metrics = compute_gradient_metrics(maddpg.critic)

                    # Check gradient health and apply adaptive controls
                    actor_exploding, actor_vanishing, actor_clip = check_gradient_health(actor_metrics, config)
                    critic_exploding, critic_vanishing, critic_clip = check_gradient_health(critic_metrics, config)

                    # Apply adaptive learning rate adjustments
                    adjust_learning_rates(maddpg.actor_optimizer, actor_exploding, config)
                    adjust_learning_rates(maddpg.critic_optimizer, critic_exploding, config)

                    # Update gradient clipping if adaptive clipping is enabled
                    if grad_config.get('adaptive_clipping', False):
                        if actor_exploding:
                            maddpg.gradient_clip = min(actor_clip, maddpg.gradient_clip)
                        if critic_exploding:
                            maddpg.gradient_clip = min(critic_clip, maddpg.gradient_clip)

                    # Log gradient metrics to TensorBoard (Global Norm Only)
                    writer.add_scalar('Gradients/Actor/global_norm', actor_metrics['global_norm'], total_steps)
                    writer.add_scalar('Gradients/Critic/global_norm', critic_metrics['global_norm'], total_steps)

                    # Log GNN-specific metrics if applicable
                    if hasattr(maddpg.actor, 'gnn_conv'):
                        if maddpg.actor.gnn_conv.weight.grad is not None:
                            gnn_grad_norm = maddpg.actor.gnn_conv.weight.grad.norm().item()
                            writer.add_scalar('Gradients/Actor/gnn_grad_norm', gnn_grad_norm, total_steps)
                        
                        gnn_weight_norm = maddpg.actor.gnn_conv.weight.norm().item()
                        writer.add_scalar('Weights/Actor/gnn_weight_norm', gnn_weight_norm, total_steps)

                    # Log health indicators
                    writer.add_scalar('Gradients/Actor/is_exploding', float(actor_exploding), total_steps)
                    writer.add_scalar('Gradients/Actor/is_vanishing', float(actor_vanishing), total_steps)
                    writer.add_scalar('Gradients/Critic/is_exploding', float(critic_exploding), total_steps)
                    writer.add_scalar('Gradients/Critic/is_vanishing', float(critic_vanishing), total_steps)

                    # Log current learning rates
                    actor_lr = maddpg.actor_optimizer.param_groups[0]['lr']
                    critic_lr = maddpg.critic_optimizer.param_groups[0]['lr']
                    writer.add_scalar('Training/actor_lr', actor_lr, total_steps)
                    writer.add_scalar('Training/critic_lr', critic_lr, total_steps)
                    writer.add_scalar('Training/gradient_clip', maddpg.gradient_clip, total_steps)

                    # Layer-wise gradient tracking removed to reduce verbosity
                    # if grad_config.get('track_layer_wise', False):
                    #     ...

                    # Activation health monitoring (every 500 steps)
                    if total_steps % 500 == 0:
                        # Check actor activations
                        if maddpg.use_gnn:
                            # GNN expects [Batch, N_Agents, C, H, W]
                            # We treat the set of all agents as a single batch item
                            sample_obs = all_obs_tensor.unsqueeze(0)  # [1, 64, 2, 8, 8]
                        else:
                            # CNN expects [Batch, C, H, W]
                            sample_obs = all_obs_tensor[:8]  # Sample of 8 agents
                        actor_act_stats = check_activation_health(maddpg.actor, sample_obs, 'actor')
                        for stat_name, stat_value in actor_act_stats.items():
                            writer.add_scalar(f'Activations/{stat_name}', stat_value, total_steps)

                        # Warn if too many dead neurons
                        dead_ratios = [v for k, v in actor_act_stats.items() if 'dead_ratio' in k]
                        if dead_ratios and max(dead_ratios) > 0.5:
                            print(f"⚠️  Warning: {max(dead_ratios)*100:.1f}% dead neurons detected in actor!")

                        # Warn if tanh saturated
                        saturated_ratios = [v for k, v in actor_act_stats.items() if 'saturated_ratio' in k]
                        if saturated_ratios and max(saturated_ratios) > 0.8:
                            print(f"⚠️  Warning: {max(saturated_ratios)*100:.1f}% saturated tanh activations in actor!")

                # Add values to episode tracking lists
                episode_critic_losses.extend(batch_critic_losses)
                episode_actor_losses.extend(batch_actor_losses)
                episode_temporal_losses.extend(batch_temporal_losses)
                episode_spatial_losses.extend(batch_spatial_losses)
                episode_zero_losses.extend(batch_zero_losses)
                episode_q_values.extend(batch_q_values)

            # Check if episode is done
            if any(terminations.values()) or any(truncations.values()):
                episode += 1

                # Calculate average reward across all agents
                avg_reward = sum(episode_rewards.values()) / len(episode_rewards)

                # Log common episode metrics
                writer.add_scalar('Episode/reward', avg_reward, episode)
                writer.add_scalar('Episode/steps', episode_steps, episode)

                # Log environment physics metrics
                avg_dpdx = np.mean(episode_dpdx_values) if episode_dpdx_values else 0
                min_dpdx = min(episode_dpdx_values) if episode_dpdx_values else 0
                max_dpdx = max(episode_dpdx_values) if episode_dpdx_values else 0

                writer.add_scalar('Episode/dpdx_mean', avg_dpdx, episode)
                writer.add_scalar('Episode/dpdx_min', min_dpdx, episode)
                writer.add_scalar('Episode/dpdx_max', max_dpdx, episode)

                # Log training metrics averaged over the episode
                if episode_critic_losses:
                    avg_critic_loss = np.mean(episode_critic_losses)
                    writer.add_scalar('Episode/critic_loss', avg_critic_loss, episode)

                if episode_actor_losses:
                    avg_actor_loss = np.mean(episode_actor_losses)
                    writer.add_scalar('Episode/actor_loss', avg_actor_loss, episode)

                # Log smoothness loss components
                if episode_temporal_losses:
                    writer.add_scalar('Episode/temporal_loss', np.mean(episode_temporal_losses), episode)
                if episode_spatial_losses:
                    writer.add_scalar('Episode/spatial_loss', np.mean(episode_spatial_losses), episode)
                if episode_zero_losses:
                    writer.add_scalar('Episode/zero_loss', np.mean(episode_zero_losses), episode)

                if episode_q_values:
                    avg_q_val = np.mean(episode_q_values)
                    writer.add_scalar('Episode/q_value', avg_q_val, episode)

                # Action statistics
                if episode_action_mean_history:
                    writer.add_scalar('Episode/action_mean', np.mean(episode_action_mean_history), episode)
                    writer.add_scalar('Episode/action_std', np.mean(episode_action_std_history), episode)

                # Log current noise scale for exploration tracking
                current_noise_scale = compute_noise_scale(episode, config)
                writer.add_scalar('Episode/noise_scale', current_noise_scale, episode)

                # Generate detailed histograms (every 5 episodes)
                if episode % 5 == 0:
                    # Actions histogram
                    if episode_actions:
                        writer.add_histogram('Histograms/actions', np.array(episode_actions), episode)

                    # Q-value distribution
                    if episode_q_values:
                        writer.add_histogram('Histograms/q_values', np.array(episode_q_values), episode)

                # Log field visualizations (every 10 episodes)
                if episode % 10 == 0:
                    try:
                        # Get current fields from environment
                        u_field = env.u_obs_field * env.om_max  # Denormalize
                        w_field = env.w_obs_field * env.om_max  # Denormalize

                        # Reconstruct action field from last actions
                        action_matrix = np.zeros((64, 64), dtype=np.float32)
                        for agent_id, action in actions.items():
                            i, j = map(int, agent_id.split("_")[1:])
                            x_start, x_end = i * 8, (i + 1) * 8
                            y_start, y_end = j * 8, (j + 1) * 8
                            action_matrix[x_start:x_end, y_start:y_end] = action

                        # Create field visualization with episode info
                        field_image = create_field_visualization(u_field, w_field, action_matrix, episode)

                        # Log to TensorBoard (HWC format)
                        writer.add_image('Fields/velocity_and_actions', field_image, episode, dataformats='HWC')

                        # Also log action change visualization (boom-boom detector)
                        if episode > 0:  # Need at least 2 episodes for comparison
                            action_change_image = create_action_change_visualization(
                                action_matrix, prev_action_matrix, episode
                            )
                            writer.add_image('Fields/action_changes', action_change_image, episode, dataformats='HWC')

                        # Update previous action matrix for next visualization
                        prev_action_matrix = action_matrix.copy()

                        print(f"  └─ Field visualizations logged for episode {episode}")

                    except Exception as e:
                        print(f"⚠️  Warning: Could not create field visualization: {e}")

                # Check for best reward and save checkpoint
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    save_checkpoint(maddpg, replay_buffer, checkpoint_dir, total_steps, episode, best_reward, is_best=True)
                    writer.add_scalar('Episode/best_reward', best_reward, episode)

                # Reset environment and metrics
                observations, infos = env.reset()
                prev_actions = {agent: np.zeros((8, 8), dtype=np.float32) for agent in agents}
                episode_rewards = {agent: 0.0 for agent in agents}
                episode_steps = 0
                episode_dpdx_values = []
                episode_q_values = []
                episode_actor_losses = []
                episode_critic_losses = []
                episode_temporal_losses = []
                episode_spatial_losses = []
                episode_zero_losses = []
                episode_actions = []
                episode_reward_history = []
                episode_action_mean_history = []
                episode_action_std_history = []
                # Note: prev_action_matrix persists across episodes for visualization

                # Print progress
                print(f"Episode {episode} - Avg Reward: {avg_reward:.3f}, Best: {best_reward:.3f}, Steps: {total_steps}")

            # Periodic checkpoint saving
            if total_steps % save_freq == 0:
                save_checkpoint(maddpg, replay_buffer, checkpoint_dir, total_steps, episode, best_reward)

                # Calculate and log training speed
                elapsed_time = (datetime.now() - training_start_time).total_seconds()
                steps_per_second = total_steps / elapsed_time
                steps_per_day = steps_per_second * 86400
                estimated_days = (max_steps - total_steps) / (steps_per_day + 1e-6)
                estimated_completion = datetime.now() + timedelta(days=estimated_days)

                # Print progress with speed metrics
                print(f"Step: {total_steps}/{max_steps}, Episode: {episode}, Buffer Size: {replay_buffer.size}")
                print(f"Training speed: {steps_per_second:.2f} steps/sec ({steps_per_day:.0f} steps/day)")
                print(f"Estimated time remaining: {estimated_days:.2f} days")
                print(f"Estimated completion date: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")

        # Training complete
        print("\nTraining complete!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    finally:
        # Save final checkpoint
        save_checkpoint(maddpg, replay_buffer, checkpoint_dir, total_steps, episode, best_reward, is_final=True)

        # Log training summary
        elapsed_time = (datetime.now() - training_start_time).total_seconds()
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        # Calculate final noise scale for summary
        noise_scale = compute_noise_scale(episode, config)

        summary_text = f"""
        Training completed or interrupted:
        - Total steps: {total_steps}/{max_steps}
        - Total episodes: {episode}
        - Best reward: {best_reward:.3f}
        - Training time: {hours}h {minutes}m {seconds}s
        - Final buffer size: {replay_buffer.size}/{buffer_size}
        - Final noise scale: {noise_scale:.6f}
        """

        writer.add_text('Summary', summary_text)

        # Close environment and writer
        env.close()
        writer.close()

        print(f"Total training time: {hours}h {minutes}m {seconds}s")
        print(f"Best average reward: {best_reward:.3f}")

def save_checkpoint(maddpg, replay_buffer, checkpoint_dir, total_steps, episode, best_reward, is_best=False, is_final=False):
    """Save checkpoint with model and training state."""

    # Create checkpoint dictionary
    checkpoint = {
        'maddpg_state_dict': maddpg.state_dict(),
        'total_steps': total_steps,
        'episode': episode,
        'best_reward': best_reward,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save regular checkpoint
    if is_final:
        checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{total_steps}.pt")

    torch.save(checkpoint, checkpoint_path)

    # Save replay buffer (always to the same file)
    buffer_path = os.path.join(checkpoint_dir, "latest_buffer.npz")
    replay_buffer.save(buffer_path)

    # Save best model if applicable
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(checkpoint, best_path)

        # Optionally save a copy of the buffer for the best model
        best_buffer_path = os.path.join(checkpoint_dir, "best_buffer.npz")
        replay_buffer.save(best_buffer_path)

        print(f"New best model saved with reward: {best_reward:.4f}")

def find_latest_checkpoint(checkpoint_dir):
    """Find the latest valid checkpoint and buffer."""

    checkpoint_path = None
    buffer_path = None

    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir)
                      if f.startswith("checkpoint_step_") and f.endswith('.pt')]

        # Sort checkpoints by step number
        checkpoints.sort(key=lambda x: int(x.split('_step_')[1].replace('.pt', '')), reverse=True)

        # Try checkpoints from newest to oldest
        for checkpoint in checkpoints:
            try:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
                buffer_path = os.path.join(checkpoint_dir, "latest_buffer.npz")

                # Verify checkpoint file integrity
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
                if 'maddpg_state_dict' not in checkpoint_data:
                    continue

                # Verify buffer file exists
                if not os.path.exists(buffer_path):
                    continue

                step_num = checkpoint.split('_step_')[1].replace('.pt', '')
                print(f"Found valid checkpoint at step {step_num}")
                return checkpoint_path, buffer_path

            except Exception as e:
                print(f"Skipping corrupted checkpoint {checkpoint}: {str(e)}")
                continue

    return None, None

def main():
    parser = argparse.ArgumentParser(description='Train MADDPG with CNN for patch-based STW control')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from latest checkpoint')
    parser.add_argument('--checkpoint', type=str,
                      help='Resume from specific checkpoint')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to run on (cuda or cpu)')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup directories
    checkpoint_dir = "./checkpoints_pettingzoo_cnn"
    logs_dir = "./logs_pettingzoo_cnn"

    # Handle resuming training
    checkpoint_path = None
    buffer_path = None

    if args.checkpoint:
        # Resume from specific checkpoint
        checkpoint_path = args.checkpoint
        buffer_path = os.path.join(os.path.dirname(checkpoint_path), "latest_buffer.npz")

        if not os.path.exists(buffer_path):
            print(f"Warning: Could not find buffer file at {buffer_path}")
            print("Will attempt to train with an empty buffer")
            buffer_path = None
    elif args.resume:
        # Find latest valid checkpoint
        checkpoint_path, buffer_path = find_latest_checkpoint(checkpoint_dir)
        if not checkpoint_path:
            print("No valid checkpoint found. Starting fresh training.")

    # Print training setup summary
    print("\n" + "="*50)
    print("STW PettingZoo MADDPG Training with CNN Policies")
    print("="*50)
    print(f"Configuration file: {args.config}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print(f"Logs directory: {logs_dir}")
    print(f"Resume checkpoint: {checkpoint_path if checkpoint_path else 'None - Starting fresh'}")
    if checkpoint_path:
        print(f"Buffer file: {buffer_path if buffer_path else 'None - Using empty buffer'}")
    print(f"Device: {args.device if args.device else 'Auto-detect'}")
    print("="*50 + "\n")

    # Train MADDPG agents
    train_maddpg(
        config=config,
        checkpoint_dir=checkpoint_dir,
        logs_dir=logs_dir,
        device=args.device,
        resume_checkpoint=checkpoint_path,
        resume_buffer=buffer_path
    )

if __name__ == "__main__":
    main()
