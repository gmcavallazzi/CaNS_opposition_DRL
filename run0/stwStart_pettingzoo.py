import os
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, Optional, Dict, Any, List
import json
import argparse
import matplotlib.pyplot as plt

from stwEnv_pettingzoo import STWParallelEnv
from utils import load_config
from models_pettingzoo import (
    SharedPolicyMADDPG,
    BatchedReplayBuffer
)
import time

def train_maddpg(
    config: Dict[str, Any],
    checkpoint_dir: str = "./checkpoints_pettingzoo",
    logs_dir: str = "./logs_pettingzoo",
    device: Optional[str] = None,
    resume_checkpoint: Optional[str] = None,
    resume_buffer: Optional[str] = None,
):
    """
    Train MADDPG agents in the STW environment using PettingZoo with shared policy.
    Enhanced TensorBoard logging with consistent step vs episode tracking.
    
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
    
    # Get agent list and sample observation/action spaces
    agents = env.possible_agents
    obs_shape = env.observation_spaces[agents[0]].shape
    act_shape = env.action_spaces[agents[0]].shape
    num_agents = len(agents)
    
    print(f"Environment created with {num_agents} agents")
    print(f"Observation shape: {obs_shape}, Action shape: {act_shape}")
    
    # Extract training parameters from config
    max_steps = config['total_timesteps']
    batch_size = config['model']['batch_size']
    gradient_steps = config['model']['gradient_steps'] 
    train_freq = config['model']['train_freq']
    save_freq = config['training']['save_freq']
    
    # Log interval (default to 1 if not specified)
    log_interval = config['training'].get('log_interval', 1)
    
    # Create shared policy MADDPG trainer
    maddpg = SharedPolicyMADDPG(
        agents=agents,
        obs_shape=obs_shape,
        act_shape=act_shape,
        gamma=config['model']['gamma'],
        tau=config['model']['tau'],
        lr=config['model']['learning_rate'],
        device=device,
        pi_arch=config.get('net_arch', {}).get('pi', [64, 64]),
        qf_arch=config.get('net_arch', {}).get('qf', [64, 64, 64])
    )
    
    # Initialize replay buffer - always use the optimized batched buffer
    buffer_size = config['model']['buffer_size']
    replay_buffer = BatchedReplayBuffer(buffer_size, obs_shape, act_shape, num_agents, agents)
    
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
    
    # Setup tensorboard writer with more descriptive run name
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"maddpg_agents{num_agents}_arch{config.get('net_arch', {}).get('pi', [64, 64])[0]}_{current_time}"
    writer = SummaryWriter(f"{logs_dir}/{run_name}")
    
    # Save config with pretty formatting
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
    episode_actions = []
    
    # New metrics tracking
    episode_reward_history = []
    episode_action_mean_history = []
    episode_action_std_history = []
    
    # Reset environment
    observations, infos = env.reset()
    
    # Training metrics for ongoing tracking (not just at episode end)
    current_actor_loss = 0
    current_critic_loss = 0
    current_q_value = 0
    update_count = 0
    
    print("Starting training...")
    try:
        while total_steps < max_steps:
            # Select actions for all agents using batched processing
            actions = {}
            
            # Stack all observations into a single tensor
            all_obs = np.stack([observations[agent] for agent in agents]).astype(np.float32)
            
            # Convert to tensor
            all_obs_tensor = torch.FloatTensor(all_obs).to(device)
            
            # Get actions for all agents in a single forward pass
            with torch.no_grad():
                # Directly use the actor network
                all_actions = maddpg.actor(all_obs_tensor).cpu().numpy()
    
                # Add exploration noise
                noise_scale = config.get('model', {}).get('action_noise', {}).get('sigma', 0.1)
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
            
            # Store transitions for all agents
            # Optimize buffer storage with batched approach
            obs_batch = np.stack([observations[agent] for agent in agents])
            next_obs_batch = np.stack([next_observations[agent] for agent in agents])
            act_batch = np.stack([actions[agent] for agent in agents])
            rew_batch = np.array([rewards[agent] for agent in agents])
            done_batch = np.array([terminations[agent] for agent in agents])
            
            # Add batch to buffer directly
            replay_buffer.add_batch(obs_batch, act_batch, rew_batch, next_obs_batch, done_batch)
            
            # Update rewards and observations
            for agent in agents:
                episode_rewards[agent] += rewards[agent]
            
            # Mean reward for this step
            step_mean_reward = np.mean(list(rewards.values()))
            episode_reward_history.append(step_mean_reward)
            
            # Store dpdx for the episode
            first_agent = agents[0]
            dpdx_value = infos[first_agent].get('dpdx', 0)
            episode_dpdx_values.append(dpdx_value)
            
            # Always log dpdx to have continuous data
            writer.add_scalar('Physics/dpdx', dpdx_value, total_steps)
            
            # Log consistent step-level metrics 
            if total_steps % log_interval == 0:
                # Always log these basic metrics against steps
                writer.add_scalar('Step/mean_reward', step_mean_reward, total_steps)
                writer.add_scalar('Step/action_mean', action_mean, total_steps)
                writer.add_scalar('Step/action_std', action_std, total_steps)
                
                # Log loss information if we have it (after first update)
                if update_count > 0:
                    writer.add_scalar('Loss/critic', current_critic_loss, total_steps)
                    writer.add_scalar('Loss/actor', current_actor_loss, total_steps)
                    writer.add_scalar('Training/q_value', current_q_value, total_steps)
            
            observations = next_observations
            episode_steps += 1
            total_steps += 1
            
            # Train networks
            if replay_buffer.size > batch_size and total_steps % train_freq == 0:
                update_count += 1
                batch_critic_losses = []
                batch_actor_losses = []
                batch_q_values = []
                
                for _ in range(gradient_steps):
                    # Sample batch from buffer
                    batch = replay_buffer.sample(batch_size, device)
                    
                    # Batched update for shared policy
                    critic_loss, actor_loss = maddpg.update_batched(batch)
                    
                    # Record losses for episode tracking
                    batch_critic_losses.append(critic_loss)
                    batch_actor_losses.append(actor_loss)
                    
                    # Extract Q-values for tracking
                    with torch.no_grad():
                        critic_input_obs = batch['obs']
                        critic_input_act = batch['acts']
                        
                        # Get Q-values
                        q_values = maddpg.critic(critic_input_obs, critic_input_act)
                        batch_q_values.append(q_values.mean().item())
                
                # Store the latest values for logging against steps
                current_critic_loss = np.mean(batch_critic_losses)
                current_actor_loss = np.mean(batch_actor_losses)
                current_q_value = np.mean(batch_q_values)
                
                # Add values to episode tracking lists
                episode_critic_losses.extend(batch_critic_losses)
                episode_actor_losses.extend(batch_actor_losses)
                episode_q_values.extend(batch_q_values)
            
            # Check if episode is done
            if any(terminations.values()) or any(truncations.values()):
                episode += 1
                
                # Calculate average reward across all agents
                avg_reward = sum(episode_rewards.values()) / len(episode_rewards)
                
                # Log common episode metrics
                writer.add_scalar('Episode/reward', avg_reward, episode)
                writer.add_scalar('Episode/steps', episode_steps, episode)
                writer.add_scalar('Episode/total_steps', total_steps, episode)
                
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
                
                if episode_q_values:
                    avg_q_val = np.mean(episode_q_values)
                    writer.add_scalar('Episode/q_value', avg_q_val, episode)
                
                # Log buffer statistics
                writer.add_scalar('Episode/buffer_size', replay_buffer.size, episode)
                
                # Log aggregate reward statistics
                reward_values = list(episode_rewards.values())
                writer.add_scalar('Episode/reward_variance', np.var(reward_values), episode)
                
                # Action statistics
                if episode_action_mean_history:
                    writer.add_scalar('Episode/action_mean', np.mean(episode_action_mean_history), episode)
                    writer.add_scalar('Episode/action_std', np.mean(episode_action_std_history), episode)
                
                # Generate detailed histograms (every 5 episodes)
                if episode % 5 == 0:
                    # Actions histogram
                    if episode_actions:
                        writer.add_histogram('Histograms/actions', np.array(episode_actions), episode)
                    
                    # Observation histograms - all channels
                    sample_agent = agents[0]  # Just use the first agent for consistent tracking
                    if sample_agent in observations:
                        obs = observations[sample_agent]
                        # Log each channel separately
                        if obs.shape[2] >= 1:
                            writer.add_histogram('Histograms/obs_channel0', 
                                              obs[:,:,0].flatten(), episode)
                        if obs.shape[2] >= 2:
                            writer.add_histogram('Histograms/obs_channel1', 
                                              obs[:,:,1].flatten(), episode)
                    
                    # Q-value distribution
                    if episode_q_values:
                        writer.add_histogram('Histograms/q_values', np.array(episode_q_values), episode)
                
                # Check for best reward and save checkpoint
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    save_checkpoint(maddpg, replay_buffer, checkpoint_dir, total_steps, episode, best_reward, is_best=True)
                    writer.add_scalar('Episode/best_reward', best_reward, episode)
                
                # Reset environment and metrics
                observations, infos = env.reset()
                episode_rewards = {agent: 0.0 for agent in agents}
                episode_steps = 0
                episode_dpdx_values = []
                episode_q_values = []
                episode_actor_losses = []
                episode_critic_losses = []
                episode_actions = []
                episode_reward_history = []
                episode_action_mean_history = []
                episode_action_std_history = []
                
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
    parser = argparse.ArgumentParser(description='Train MADDPG for STW control using PettingZoo')
    parser.add_argument('--resume', action='store_true', 
                      help='Resume from latest checkpoint')
    parser.add_argument('--checkpoint', type=str, 
                      help='Resume from specific checkpoint')
    parser.add_argument('--halo', type=int, default=None,
                      help='Override halo value in config (observation window size)')
    parser.add_argument('--device', type=str, default=None,
                      help='Device to run on (cuda or cpu)')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override halo if specified
    if args.halo is not None:
        config['halo'] = args.halo
        print(f"Overriding halo value to {args.halo}")
    
    # Setup directories - simplified to only use grid with shared policy
    checkpoint_dir = f"./checkpoints_pettingzoo_grid_shared"
    logs_dir = f"./logs_pettingzoo_grid_shared"
    
    # Handle resuming training
    checkpoint_path = None
    buffer_path = None
    
    if args.checkpoint:
        # Resume from specific checkpoint
        checkpoint_path = args.checkpoint
        step_num = os.path.basename(checkpoint_path).split('_step_')[1].replace('.pt', '')
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
    print("STW PettingZoo MADDPG Training with Enhanced Logging")
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