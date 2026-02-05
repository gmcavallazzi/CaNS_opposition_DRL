"""
Point-Based Multi-Agent DRL Models.

This module implements:
- MLPActorPoint: Tiny MLP actor (3→16→1 mapping) for point-based control
- CNNCriticPoint: Hybrid critic that reconstructs 64×64 field and uses CNN
- BatchedReplayBufferPoint: Replay buffer for 4096 scalar observations
- SharedPolicyMADDPGPoint: MADDPG with shared MLP actor for all 4096 agents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
import os

# Import existing critic and smoothness losses from patch-based models
from models_consistency import CNNCriticCircular, global_zero_mean_loss


# ============================================================================
# MLP Actor for Point-Based Control
# ============================================================================

class MLPActorPoint(nn.Module):
    """
    Tiny MLP actor network for point-based control.

    Each agent observes 2 or 3 scalars:
    - 2 channels: [u, w] velocities (backwards compatible)
    - 3 channels: [u, w, prev_action] (with action memory)

    Outputs a single scalar action.

    Architecture (SIMPLIFIED):
        Input: (2 or 3,) scalars
          ↓
        Linear(2/3, 16) + LayerNorm + ReLU + Dropout
          ↓
        Linear(16, 1) + tanh
          ↓
        Output: (1,) scalar action in [-1, 1]

    Side branch for consistency loss (optional):
        Similarity Head: Linear(16, 16)
    """
    def __init__(self,
                 hidden_dim: int = 16,
                 dropout_rate: float = 0.05,
                 similarity_dim: int = 16,
                 input_channels: int = 2):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.similarity_dim = similarity_dim
        self.input_channels = input_channels

        # Single hidden layer
        self.fc1 = nn.Linear(input_channels, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # Action output head
        self.action_head = nn.Linear(hidden_dim, 1)

        # Similarity features (optional, for consistency loss)
        self.similarity_fc = nn.Linear(hidden_dim, similarity_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights for stable training."""
        # Xavier init for hidden layer
        nn.init.xavier_uniform_(self.fc1.weight, gain=1.0)
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0.0)

        # Small init for output layer (avoid large initial actions)
        nn.init.uniform_(self.action_head.weight, -0.003, 0.003)
        if self.action_head.bias is not None:
            nn.init.uniform_(self.action_head.bias, -0.003, 0.003)

        # Similarity head init
        nn.init.xavier_uniform_(self.similarity_fc.weight, gain=1.0)
        if self.similarity_fc.bias is not None:
            nn.init.constant_(self.similarity_fc.bias, 0.0)

        # LayerNorm init
        nn.init.constant_(self.ln1.weight, 1.0)
        nn.init.constant_(self.ln1.bias, 0.0)

    def forward(self, obs: torch.Tensor, return_similarity: bool = False):
        """
        Forward pass.

        Args:
            obs: [batch, input_channels] - scalar observations
            return_similarity: If True, also return similarity features

        Returns:
            action: [batch, 1] - scalar action in [-1, 1]
            similarity_features: [batch, similarity_dim] (optional)
        """
        # Single hidden layer
        x = self.fc1(obs)          # [batch, hidden_dim]
        x = self.ln1(x)
        x = self.relu(x)
        features = self.dropout(x)  # [batch, hidden_dim]

        # Action output
        action = torch.tanh(self.action_head(features))  # [batch, 1]

        if return_similarity:
            similarity = self.similarity_fc(features)  # [batch, similarity_dim]
            return action, similarity
        return action


# ============================================================================
# CNN Critic for Point-Based Control (Hybrid Approach)
# ============================================================================

class CNNCriticPoint(nn.Module):
    """
    Hybrid critic for point-based agents.
    Reconstructs 64×64 fields from point data, then uses CNN.

    Architecture:
        Input: Point observations [batch, 4096, obs_channels] + actions [batch, 4096, 1]
          ↓
        Reshape to 64×64 grids: [batch, obs_channels, 64, 64] + [batch, 1, 64, 64]
          ↓
        Use existing CNNCriticCircular architecture (circular convolutions + MLP)
          ↓
        Output: [batch, 1] - Q-value

    Benefits:
    - Captures spatial patterns and coordination
    - Reuses proven architecture from patch-based system
    - Circular padding for periodic boundaries
    """
    def __init__(self,
                 conv_channels: List[int] = None,
                 mlp_layers: List[int] = None,
                 dropout_rate: float = 0.05,
                 obs_channels: int = 2):
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 32]  # Same as patch-based
        if mlp_layers is None:
            mlp_layers = [256, 128]       # Same as patch-based

        self.obs_channels = obs_channels

        # Reuse existing CNN critic architecture
        self.cnn_critic = CNNCriticCircular(
            conv_channels=conv_channels,
            mlp_layers=mlp_layers,
            dropout_rate=dropout_rate,
            input_channels=obs_channels  # Will concatenate action → obs_channels+1
        )

    def forward(self, obs_points: torch.Tensor, act_points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs_points: [batch, 4096, obs_channels] - point observations
            act_points: [batch, 4096, 1] - point actions

        Returns:
            Q-values: [batch, 1]
        """
        batch_size = obs_points.size(0)

        # Reconstruct 64×64 fields from 4096 points
        # Each agent (i, j) in 0..63 maps to grid position [i, j]

        # Reshape observations: [batch, 4096, obs_channels] → [batch, obs_channels, 64, 64]
        obs_fields = obs_points.view(batch_size, 64, 64, self.obs_channels)
        obs_fields = obs_fields.permute(0, 3, 1, 2)  # [batch, obs_channels, 64, 64]

        # Reshape actions: [batch, 4096, 1] → [batch, 1, 64, 64]
        act_field = act_points.view(batch_size, 64, 64, 1)
        act_field = act_field.permute(0, 3, 1, 2)  # [batch, 1, 64, 64]

        # Use CNN critic (it will concatenate obs_fields and act_field internally)
        return self.cnn_critic(obs_fields, act_field)


# ============================================================================
# Simplified Consistency Loss for Point-Based System
# ============================================================================

def observation_consistency_loss_point(
    sim_features: torch.Tensor,
    actions: torch.Tensor,
    tau: float = 0.9,
    margin: float = 0.1,
    sample_size: int = 512
) -> torch.Tensor:
    """
    Simplified consistency loss for point-based agents.
    If two agents have similar observations, their actions should be similar.

    IMPORTANT: With 4096 agents, full pairwise comparison is expensive (16M comparisons).
    This implementation samples a subset of agent pairs to keep computation tractable.

    Args:
        sim_features: [batch, 4096, feat_dim] - similarity features from actor
        actions: [batch, 4096, 1] - scalar actions
        tau: Similarity threshold (cosine similarity)
        margin: Allowed action difference
        sample_size: Number of agents to sample for pairwise comparison

    Returns:
        Scalar loss
    """
    batch_size, n_agents, feat_dim = sim_features.shape
    device = sim_features.device

    # Sample subset of agents for efficiency
    if n_agents > sample_size:
        # Random sampling
        sampled_indices = torch.randperm(n_agents, device=device)[:sample_size]
        sim_features_sampled = sim_features[:, sampled_indices, :]
        actions_sampled = actions[:, sampled_indices, :]
        n_sampled = sample_size
    else:
        sim_features_sampled = sim_features
        actions_sampled = actions
        n_sampled = n_agents

    # Normalize features for cosine similarity
    features_norm = F.normalize(sim_features_sampled, dim=-1)

    # Pairwise similarity
    similarity = torch.bmm(features_norm, features_norm.transpose(1, 2))

    # Mask: pairs above threshold, excluding self-comparisons
    eye_mask = torch.eye(n_sampled, device=device).bool().unsqueeze(0)
    mask = (similarity > tau) & (~eye_mask)

    if not mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Action differences
    actions_scalar = actions_sampled.squeeze(-1)  # [batch, n_sampled]
    act_diff = torch.cdist(actions_scalar.unsqueeze(-1),
                          actions_scalar.unsqueeze(-1), p=2).squeeze(-1)

    # Penalize differences beyond margin
    act_diff_penalized = F.relu(act_diff - margin)
    masked_penalty = act_diff_penalized * mask.float()

    loss = masked_penalty.sum() / (mask.sum() + 1e-8)
    return loss


# ============================================================================
# Batched Replay Buffer for Point-Based System
# ============================================================================

class BatchedReplayBufferPoint:
    """
    Replay buffer for point-based multi-agent system.

    Stores transitions for 4096 agents with scalar observations and actions.
    """
    def __init__(self, capacity: int, n_agents: int = 4096, agent_ids: List[str] = None,
                 obs_channels: int = 2):
        self.capacity = capacity
        self.n_agents = n_agents
        self.obs_channels = obs_channels
        self.agent_ids = agent_ids

        # Buffers for scalar data
        self.obs_buf = np.zeros((capacity, n_agents, obs_channels), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, n_agents, obs_channels), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, n_agents, 1), dtype=np.float32)
        self.prev_acts_buf = np.zeros((capacity, n_agents, 1), dtype=np.float32)
        self.rews_buf = np.zeros((capacity, n_agents), dtype=np.float32)
        self.done_buf = np.zeros((capacity, n_agents), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add_batch(self, obs_batch: np.ndarray, acts_batch: np.ndarray, prev_acts_batch: np.ndarray,
                  rews_batch: np.ndarray, next_obs_batch: np.ndarray, dones_batch: np.ndarray):
        """
        Add batch of transitions for all 4096 agents.

        Args:
            obs_batch: [4096, obs_channels]
            acts_batch: [4096, 1]
            prev_acts_batch: [4096, 1]
            rews_batch: [4096]
            next_obs_batch: [4096, obs_channels]
            dones_batch: [4096]
        """
        self.obs_buf[self.ptr] = obs_batch
        self.next_obs_buf[self.ptr] = next_obs_batch
        self.acts_buf[self.ptr] = acts_batch
        self.prev_acts_buf[self.ptr] = prev_acts_batch
        self.rews_buf[self.ptr] = rews_batch
        self.done_buf[self.ptr] = dones_batch

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str) -> Dict[str, torch.Tensor]:
        """
        Sample random batch of transitions.

        Returns:
            batch: Dictionary with tensors on specified device
        """
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = {
            'obs': torch.FloatTensor(self.obs_buf[idxs]).to(device),
            'acts': torch.FloatTensor(self.acts_buf[idxs]).to(device),
            'prev_acts': torch.FloatTensor(self.prev_acts_buf[idxs]).to(device),
            'rews': torch.FloatTensor(self.rews_buf[idxs]).to(device),
            'next_obs': torch.FloatTensor(self.next_obs_buf[idxs]).to(device),
            'done': torch.FloatTensor(self.done_buf[idxs]).to(device)
        }
        return batch

    def save(self, filepath: str):
        """Save replay buffer to disk."""
        save_dict = {
            'capacity': self.capacity,
            'n_agents': self.n_agents,
            'obs_channels': self.obs_channels,
            'ptr': self.ptr,
            'size': self.size,
            'obs_buf': self.obs_buf[:self.size],
            'next_obs_buf': self.next_obs_buf[:self.size],
            'acts_buf': self.acts_buf[:self.size],
            'prev_acts_buf': self.prev_acts_buf[:self.size],
            'rews_buf': self.rews_buf[:self.size],
            'done_buf': self.done_buf[:self.size],
            'agent_ids': self.agent_ids
        }
        np.savez_compressed(filepath, **save_dict)
        print(f"Replay buffer saved to {filepath} (size={self.size})")

    def load(self, filepath: str):
        """Load replay buffer from disk."""
        if not os.path.exists(filepath):
            print(f"Warning: Replay buffer file not found: {filepath}")
            return False

        data = np.load(filepath, allow_pickle=True)

        # Load saved data
        saved_size = data['size']
        saved_obs_channels = data['obs_channels']

        # Handle channel mismatch (backwards compatibility)
        if saved_obs_channels != self.obs_channels:
            print(f"Warning: Buffer has {saved_obs_channels} channels, but model expects {self.obs_channels}")
            if saved_obs_channels < self.obs_channels:
                # Pad with zeros (e.g., loading 2-channel into 3-channel buffer)
                print(f"Padding with zeros for missing channels")
                self.obs_buf[:saved_size, :, :saved_obs_channels] = data['obs_buf']
                self.next_obs_buf[:saved_size, :, :saved_obs_channels] = data['next_obs_buf']
            else:
                # Drop extra channels (e.g., loading 3-channel into 2-channel buffer)
                print(f"Dropping extra channels")
                self.obs_buf[:saved_size] = data['obs_buf'][:, :, :self.obs_channels]
                self.next_obs_buf[:saved_size] = data['next_obs_buf'][:, :, :self.obs_channels]
        else:
            self.obs_buf[:saved_size] = data['obs_buf']
            self.next_obs_buf[:saved_size] = data['next_obs_buf']

        # Load other buffers (no channel dimension)
        self.acts_buf[:saved_size] = data['acts_buf']
        self.prev_acts_buf[:saved_size] = data['prev_acts_buf']
        self.rews_buf[:saved_size] = data['rews_buf']
        self.done_buf[:saved_size] = data['done_buf']

        self.ptr = int(data['ptr'])
        self.size = saved_size

        print(f"Replay buffer loaded from {filepath} (size={self.size})")
        return True


# ============================================================================
# Shared Policy MADDPG for Point-Based System
# ============================================================================

class SharedPolicyMADDPGPoint:
    """
    Multi-Agent DDPG with shared MLP policy for 4096 point-based agents.

    Key features:
    - Tiny MLP actor (3→16→1 mapping) shared across all 4096 agents
    - Hybrid CNN critic (reconstructs 64×64 field, then CNN)
    - Differentiable zero-mean constraint for mass conservation
    - Optional consistency loss (expensive with 4096 agents, use sampling)
    """
    def __init__(
        self,
        agents: List[str],
        gamma: float = 0.995,
        tau: float = 0.005,
        lr: float = 1e-3,
        critic_lr: float = 3e-4,
        dropout_rate: float = 0.05,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        actor_hidden_dim: int = 16,
        critic_conv_channels: List[int] = None,
        critic_mlp_layers: List[int] = None,
        gradient_clip: float = 0.5,
        lambda_temporal: float = 0.0,  # Disabled (action memory in obs)
        lambda_global_mean: float = 0.05,  # Global zero-mean constraint
        # Consistency loss parameters
        consistency_enable: bool = False,  # Start disabled (expensive)
        consistency_lambda: float = 0.1,
        consistency_tau: float = 0.9,
        consistency_margin: float = 0.1,
        consistency_sample_size: int = 512,
        consistency_warmup_steps: int = 5000,
        similarity_dim: int = 16,
        input_channels: int = 2  # 2 for [u,w], 3 for [u,w,prev_action]
    ):
        self.agents = agents
        self.n_agents = len(agents)  # 4096
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.weight_decay = weight_decay
        self.gradient_clip = gradient_clip

        # Learning rates
        self.actor_lr = lr
        self.critic_lr = critic_lr

        # Smoothness penalty weights
        self.lambda_temporal = lambda_temporal
        self.lambda_global_mean = lambda_global_mean

        # Consistency loss parameters
        self.consistency_enable = consistency_enable
        self.consistency_lambda = consistency_lambda
        self.consistency_tau = consistency_tau
        self.consistency_margin = consistency_margin
        self.consistency_sample_size = consistency_sample_size
        self.consistency_warmup_steps = consistency_warmup_steps
        self.similarity_dim = similarity_dim

        # Store input channels
        self.input_channels = input_channels

        # Training step counter for warmup
        self.training_step = 0

        # Default architectures
        if critic_conv_channels is None:
            critic_conv_channels = [32, 64, 32]
        if critic_mlp_layers is None:
            critic_mlp_layers = [256, 128]

        print(f"Initializing Point-Based MADDPG:")
        print(f"  Number of agents: {self.n_agents} (64×64 grid)")
        print(f"  Input channels: {input_channels} ({'[u,w,prev_action]' if input_channels == 3 else '[u,w]'})")
        print(f"  Actor hidden dim: {actor_hidden_dim} (tiny MLP!)")
        print(f"  Critic conv channels: {critic_conv_channels}")
        print(f"  Critic MLP layers: {critic_mlp_layers}")
        print(f"  Temporal smoothness: {lambda_temporal} (disabled, use action memory)")
        print(f"  Global zero-mean: {lambda_global_mean}")
        print(f"  Consistency loss - enabled: {consistency_enable}, lambda: {consistency_lambda}")
        print(f"  Consistency sample size: {consistency_sample_size} (out of {self.n_agents})")

        # Create networks
        self.actor = MLPActorPoint(actor_hidden_dim, dropout_rate, similarity_dim, input_channels).to(device)
        self.actor_target = MLPActorPoint(actor_hidden_dim, dropout_rate, similarity_dim, input_channels).to(device)

        # Critic: Hybrid CNN (reconstructs field from points)
        self.critic = CNNCriticPoint(critic_conv_channels, critic_mlp_layers, dropout_rate, input_channels).to(device)
        self.critic_target = CNNCriticPoint(critic_conv_channels, critic_mlp_layers, dropout_rate, input_channels).to(device)

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=weight_decay)

        print(f"  Actor parameters: {sum(p.numel() for p in self.actor.parameters())}")
        print(f"  Critic parameters: {sum(p.numel() for p in self.critic.parameters())}")

    def select_actions_batched(self, all_obs: torch.Tensor) -> np.ndarray:
        """
        Select actions for all 4096 agents in one forward pass.

        Args:
            all_obs: [4096, obs_channels] - observations for all agents

        Returns:
            all_actions: [4096, 1] - scalar actions, zero-mean corrected
        """
        with torch.no_grad():
            # Single forward pass for all agents (weight sharing!)
            all_actions_raw = self.actor(all_obs)  # [4096, 1]
            if isinstance(all_actions_raw, tuple):
                all_actions_raw = all_actions_raw[0]

            # Global zero-mean correction (CRITICAL: must be differentiable in training)
            global_mean = all_actions_raw.mean()
            all_actions = all_actions_raw - global_mean

            return all_actions.cpu().numpy()

    def update_batched(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float, Dict[str, float]]:
        """
        Update networks using batch of experiences.

        Batch contents:
            obs: [batch_size, 4096, obs_channels]
            acts: [batch_size, 4096, 1]
            prev_acts: [batch_size, 4096, 1]
            rews: [batch_size, 4096]
            next_obs: [batch_size, 4096, obs_channels]
            done: [batch_size, 4096]

        Returns:
            critic_loss, actor_loss, loss_breakdown
        """
        obs_batch = batch['obs']
        act_batch = batch['acts']
        prev_act_batch = batch['prev_acts']
        rew_batch = batch['rews']
        next_obs_batch = batch['next_obs']
        done_batch = batch['done']

        batch_size = obs_batch.size(0)

        # Increment training step counter
        self.training_step += 1

        # ========================================================================
        # Update Critic
        # ========================================================================

        with torch.no_grad():
            # Reshape for actor: [batch_size*4096, obs_channels]
            next_obs_flat = next_obs_batch.view(batch_size * self.n_agents, self.input_channels)

            # Get next actions (single forward pass for all 4096 agents)
            next_actions_flat = self.actor_target(next_obs_flat)  # [batch_size*4096, 1]
            if isinstance(next_actions_flat, tuple):
                next_actions_flat = next_actions_flat[0]
            next_actions_raw = next_actions_flat.view(batch_size, self.n_agents, 1)

            # Zero-mean correction (per batch element)
            next_global_means = next_actions_raw.view(batch_size, -1).mean(dim=1, keepdim=True)
            next_global_means = next_global_means.view(batch_size, 1, 1)
            next_actions = next_actions_raw - next_global_means

            # Target Q-values
            target_q = self.critic_target(next_obs_batch, next_actions)

            # TD target (use mean reward across agents)
            mean_reward = rew_batch.mean(dim=1, keepdim=True)
            done_flag = done_batch[:, 0].unsqueeze(-1)
            target_value = mean_reward + self.gamma * (1.0 - done_flag) * target_q

        # Current Q-values
        current_q = self.critic(obs_batch, act_batch)

        # Critic loss
        critic_loss = F.mse_loss(current_q, target_value)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()

        # ========================================================================
        # Update Actor
        # ========================================================================

        obs_flat = obs_batch.view(batch_size * self.n_agents, self.input_channels)

        # Get actions and similarity features
        actions_flat, sim_features_flat = self.actor(obs_flat, return_similarity=True)
        actions_raw = actions_flat.view(batch_size, self.n_agents, 1)
        sim_features = sim_features_flat.view(batch_size, self.n_agents, self.similarity_dim)

        # Zero-mean correction (DIFFERENTIABLE!)
        global_means = actions_raw.view(batch_size, -1).mean(dim=1, keepdim=True)
        global_means = global_means.view(batch_size, 1, 1)
        actions = actions_raw - global_means

        # Base Q-loss (maximize Q-value)
        q_loss = -self.critic(obs_batch, actions).mean()

        # Temporal smoothness (action memory already in obs, usually disabled)
        temporal_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_temporal > 0:
            temporal_loss = torch.mean((actions - prev_act_batch) ** 2)

        # Global zero-mean loss (encourage policy to output near-zero mean)
        global_mean_loss = (actions.mean()) ** 2

        # Consistency loss (optional, expensive for 4096 agents)
        consistency_loss = torch.tensor(0.0, device=self.device)
        if self.consistency_enable and self.training_step >= self.consistency_warmup_steps:
            consistency_loss = observation_consistency_loss_point(
                sim_features, actions,
                tau=self.consistency_tau,
                margin=self.consistency_margin,
                sample_size=self.consistency_sample_size
            )

        # Combined actor loss
        actor_loss = (q_loss +
                     self.lambda_temporal * temporal_loss +
                     self.lambda_global_mean * global_mean_loss +
                     self.consistency_lambda * consistency_loss)

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()

        # Soft update targets
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        # Prepare loss breakdown (include all losses for compatibility with training script)
        loss_breakdown = {
            'q_loss': q_loss.item(),
            'temporal_loss': temporal_loss.item(),
            'spatial_loss': 0.0,  # Not used in point-based (no patches)
            'zero_loss': 0.0,     # Deprecated (use global_mean_loss instead)
            'global_mean_loss': global_mean_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'actor_grad_norm': actor_grad_norm.item(),
            'critic_grad_norm': critic_grad_norm.item()
        }

        return critic_loss.item(), actor_loss.item(), loss_breakdown

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update of target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_step': self.training_step
        }
        torch.save(checkpoint, filepath)
        print(f"Model checkpoint saved to {filepath}")

    def load(self, filepath: str):
        """Load model checkpoint."""
        if not os.path.exists(filepath):
            print(f"Warning: Checkpoint file not found: {filepath}")
            return False

        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)

        print(f"Model checkpoint loaded from {filepath} (training_step={self.training_step})")
        return True
