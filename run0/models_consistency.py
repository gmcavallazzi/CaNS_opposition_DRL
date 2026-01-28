"""
Models with Observation-Similarity Consistency Loss for Smooth Multi-Agent Control.

This module implements:
- Actor with standard padding and similarity encoder branch
- Critic with circular padding for periodic domain
- Observation-similarity consistency loss
- Modified MADDPG class with consistency regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Optional, Any


# ============================================================================
# Circular Padding Convolution (for Critic on periodic domain)
# ============================================================================

class CircularConv2d(nn.Module):
    """
    Conv2d with circular (periodic) padding.
    Appropriate for the full 64x64 domain which is periodic in x and y.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, bias: bool = True):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply circular padding
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='circular')
        return self.conv(x)


# ============================================================================
# CNN Actor with Similarity Branch
# ============================================================================

class CNNActorWithSimilarity(nn.Module):
    """
    CNN-based actor network with similarity feature extraction branch.

    Each agent observes an 8x8 patch with 2 channels (u, w velocities)
    and outputs an 8x8 grid of actions.

    Uses STANDARD padding (not circular) since the 8x8 patch is a local
    window into the larger domain, not a periodic structure itself.

    The similarity branch extracts features used for the observation-
    similarity consistency loss during training.
    """
    def __init__(self, conv_channels: List[int] = None, dropout_rate: float = 0.05,
                 similarity_dim: int = 16):
        super().__init__()

        # Default encoder-decoder architecture
        if conv_channels is None:
            conv_channels = [16, 32]  # Encoder channels

        self.similarity_dim = similarity_dim

        # Encoder: Extract spatial features from input
        # Uses standard padding since patches are NOT periodic
        encoder_layers = []
        in_channels = 2  # u, w velocities

        for out_channels in conv_channels:
            encoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Standard padding
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout2d(dropout_rate)
            ])
            in_channels = out_channels

        self.encoder = nn.Sequential(*encoder_layers)

        # Similarity branch: Extract compact feature vector for consistency loss
        # Conv 1x1 to reduce channels, then global average pooling
        self.similarity_conv = nn.Conv2d(conv_channels[-1], similarity_dim, kernel_size=1)
        self.similarity_pool = nn.AdaptiveAvgPool2d(1)

        # Decoder: Generate action field from features
        decoder_layers = []
        in_channels = conv_channels[-1]
        for i in range(len(conv_channels) - 1, -1, -1):
            out_channels = conv_channels[i - 1] if i > 0 else 1
            decoder_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # Standard padding
                nn.BatchNorm2d(out_channels) if out_channels > 1 else nn.Identity(),
                nn.ReLU() if out_channels > 1 else nn.Identity(),
                nn.Dropout2d(dropout_rate) if out_channels > 1 else nn.Identity()
            ])
            in_channels = out_channels

        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            # Use smaller initialization for stability
            # Last layer in decoder will be initialized with small values
            nn.init.xavier_uniform_(module.weight, gain=0.01 if module.out_channels == 1 else 1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor, return_similarity: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            obs: [batch_size, 2, 8, 8] - velocity observations
            return_similarity: If True, also return similarity features

        Returns:
            actions: [batch_size, 8, 8] - action grid in [-1, 1]
            similarity_features: [batch_size, similarity_dim] (optional)
        """
        # Encode spatial features
        features = self.encoder(obs)  # [batch, 32, 8, 8]

        # Extract similarity features if needed
        similarity_features = None
        if return_similarity:
            sim_feat = self.similarity_conv(features)  # [batch, 16, 8, 8]
            sim_feat = self.similarity_pool(sim_feat)  # [batch, 16, 1, 1]
            similarity_features = sim_feat.view(sim_feat.size(0), -1)  # [batch, 16]

        # Decode to action field
        actions = self.decoder(features)

        # Apply tanh and squeeze channel dimension
        actions = torch.tanh(actions.squeeze(1))  # [batch_size, 8, 8]

        if return_similarity:
            return actions, similarity_features
        return actions


# ============================================================================
# CNN Critic with Circular Padding
# ============================================================================

class CNNCriticCircular(nn.Module):
    """
    Convolutional critic network with circular padding for periodic domain.

    Processes 64x64 grids with 3 channels (u, w observations + actions).
    Uses circular padding since the full domain is periodic in x and y.
    """
    def __init__(self, conv_channels: List[int] = None,
                 mlp_layers: List[int] = None, dropout_rate: float = 0.05):
        super().__init__()

        # Default architectures
        if conv_channels is None:
            conv_channels = [32, 64, 32]
        if mlp_layers is None:
            mlp_layers = [256, 128]

        # Convolutional layers with circular padding for spatial processing
        conv_layers = []
        in_channels = 3  # 2 for u,w + 1 for actions

        for out_channels in conv_channels:
            conv_layers.extend([
                CircularConv2d(in_channels, out_channels, kernel_size=3),
                nn.GroupNorm(4, out_channels),  # GroupNorm is more stable than BatchNorm for RL
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # Reduce spatial dimensions
                nn.Dropout2d(dropout_rate)
            ])
            in_channels = out_channels

        self.conv_net = nn.Sequential(*conv_layers)

        # Calculate flattened size after convolutions
        # 64x64 -> 32x32 -> 16x16 -> 8x8 (3 pooling operations)
        final_spatial_size = 64 // (2 ** len(conv_channels))
        conv_output_size = conv_channels[-1] * (final_spatial_size ** 2)

        # MLP layers for final Q-value prediction
        mlp_layers_full = []
        mlp_layers_full.append(nn.LayerNorm(conv_output_size))

        prev_dim = conv_output_size
        for h_dim in mlp_layers:
            mlp_layers_full.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = h_dim

        self.mlp_net = nn.Sequential(*mlp_layers_full)

        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if module == self.output_layer:
                nn.init.uniform_(module.weight, -0.003, 0.003)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.003, 0.003)
            else:
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs_fields: torch.Tensor, act_field: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            obs_fields: [batch_size, 2, 64, 64] - u,w velocity fields
            act_field: [batch_size, 1, 64, 64] - action field

        Returns:
            Q-values: [batch_size, 1]
        """
        # Combine observations and actions
        x = torch.cat([obs_fields, act_field], dim=1)  # [batch_size, 3, 64, 64]

        # Apply convolutional layers
        x = self.conv_net(x)

        # Flatten for MLP
        x = x.view(x.size(0), -1)

        # Apply MLP layers
        x = self.mlp_net(x)

        # Output Q-value
        return self.output_layer(x)


# ============================================================================
# Observation-Similarity Consistency Loss
# ============================================================================

def observation_consistency_loss(
    sim_features: torch.Tensor,
    actions: torch.Tensor,
    tau: float = 0.9,
    margin: float = 0.1,
    boundary_only: bool = True
) -> torch.Tensor:
    """
    Penalize action differences for agents with similar observations.
    Uses only observation features (no neighbor action info).

    This encourages the shared policy to be smooth in observation space,
    which translates to spatial smoothness when adjacent flow regions
    have similar features.

    Args:
        sim_features: [batch, n_agents, feat_dim] - similarity features from actor
        actions: [batch, n_agents, 8, 8] - actions for all agents
        tau: Similarity threshold (0.85-0.95 recommended)
        margin: Allowed action difference before penalty (0.05-0.2)
        boundary_only: If True, only compare boundary pixels of patches

    Returns:
        Scalar loss value
    """
    batch_size, n_agents, feat_dim = sim_features.shape
    device = sim_features.device

    # 1. Normalize features for cosine similarity
    features_norm = F.normalize(sim_features, dim=-1)  # [batch, n_agents, feat_dim]

    # 2. Pairwise cosine similarity: [batch, n_agents, n_agents]
    similarity = torch.bmm(features_norm, features_norm.transpose(1, 2))

    # 3. Mask: pairs above threshold, excluding self-comparisons
    eye_mask = torch.eye(n_agents, device=device).bool().unsqueeze(0)  # [1, n_agents, n_agents]
    mask = (similarity > tau) & (~eye_mask)

    if not mask.any():
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 4. Extract actions for comparison
    if boundary_only:
        # Only compare boundary pixels (28 per agent: 8+8+6+6)
        # Top row: 8 pixels
        top = actions[:, :, 0, :]         # [B, N, 8]
        # Bottom row: 8 pixels
        bottom = actions[:, :, -1, :]     # [B, N, 8]
        # Left column (excluding corners already in top/bottom): 6 pixels
        left = actions[:, :, 1:-1, 0]     # [B, N, 6]
        # Right column (excluding corners): 6 pixels
        right = actions[:, :, 1:-1, -1]   # [B, N, 6]
        actions_cmp = torch.cat([top, bottom, left, right], dim=-1)  # [B, N, 28]
    else:
        actions_cmp = actions.view(batch_size, n_agents, -1)  # [B, N, 64]

    # 5. Pairwise L2 distance between action vectors
    # Use cdist for efficient pairwise distance computation
    act_diff = torch.cdist(actions_cmp, actions_cmp, p=2)  # [B, N, N]

    # 6. Apply margin: only penalize differences beyond margin
    act_diff_penalized = F.relu(act_diff - margin)

    # 7. Apply mask and average
    masked_penalty = act_diff_penalized * mask.float()
    num_pairs = mask.sum() + 1e-8  # Avoid division by zero
    loss = masked_penalty.sum() / num_pairs

    return loss


# ============================================================================
# Smoothness Loss Functions (from original models)
# ============================================================================

def temporal_smoothness_loss(actions_t: torch.Tensor, actions_t_prev: torch.Tensor) -> torch.Tensor:
    """
    Compute temporal smoothness penalty: ||a(t) - a(t-1)||^2

    Args:
        actions_t: Current actions [batch_size, n_agents, 8, 8]
        actions_t_prev: Previous actions [batch_size, n_agents, 8, 8]

    Returns:
        Scalar loss value
    """
    return torch.mean((actions_t - actions_t_prev) ** 2)


def spatial_smoothness_loss(actions: torch.Tensor) -> torch.Tensor:
    """
    Compute spatial smoothness penalty using finite differences.
    Penalizes large gradients within each 8x8 patch.

    Args:
        actions: [batch_size, n_agents, 8, 8]

    Returns:
        Scalar loss value
    """
    # Compute gradients in x and y directions using finite differences
    grad_x = actions[:, :, :, 1:] - actions[:, :, :, :-1]  # [batch, n_agents, 8, 7]
    grad_y = actions[:, :, 1:, :] - actions[:, :, :-1, :]  # [batch, n_agents, 7, 8]

    # Sum of squared gradients
    loss_x = torch.mean(grad_x ** 2)
    loss_y = torch.mean(grad_y ** 2)

    return loss_x + loss_y


def zero_mean_loss(actions: torch.Tensor) -> torch.Tensor:
    """
    Compute zero-mean penalty: mean(a)^2 per agent.

    Args:
        actions: [batch_size, n_agents, 8, 8]

    Returns:
        Scalar loss value
    """
    # Compute mean per agent (average over the 8x8 grid)
    mean_per_agent = actions.mean(dim=(2, 3))  # [batch_size, n_agents]

    return torch.mean(mean_per_agent ** 2)


# ============================================================================
# Shared Policy MADDPG with Consistency Loss
# ============================================================================

class SharedPolicyMADDPGConsistency:
    """
    Multi-Agent DDPG with shared CNN policies, smoothness constraints,
    and observation-similarity consistency loss.

    Key improvements:
    - Actor uses standard padding (patches are not periodic)
    - Critic uses circular padding (full domain is periodic)
    - Consistency loss encourages smooth actions for similar observations
    - Temporal loss disabled by default (causes confusion with old policy actions)
    """
    def __init__(
        self,
        agents: List[str],
        gamma: float = 0.995,
        tau: float = 0.01,
        lr: float = 1e-3,
        critic_lr: float = 3e-4,
        dropout_rate: float = 0.05,
        weight_decay: float = 1e-5,
        device: str = "cpu",
        actor_channels: List[int] = None,
        critic_conv_channels: List[int] = None,
        critic_mlp_layers: List[int] = None,
        gradient_clip: float = 1.0,
        lambda_temporal: float = 0.0,  # Disabled by default
        lambda_spatial: float = 0.5,
        lambda_zero: float = 0.1,
        # Consistency loss parameters
        consistency_enable: bool = True,
        consistency_lambda: float = 0.1,
        consistency_tau: float = 0.9,
        consistency_margin: float = 0.1,
        consistency_boundary_only: bool = True,
        consistency_warmup_steps: int = 5000,
        similarity_dim: int = 16
    ):
        self.agents = agents
        self.n_agents = len(agents)
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
        self.lambda_spatial = lambda_spatial
        self.lambda_zero = lambda_zero

        # Consistency loss parameters
        self.consistency_enable = consistency_enable
        self.consistency_lambda = consistency_lambda
        self.consistency_tau = consistency_tau
        self.consistency_margin = consistency_margin
        self.consistency_boundary_only = consistency_boundary_only
        self.consistency_warmup_steps = consistency_warmup_steps
        self.similarity_dim = similarity_dim

        # Training step counter for warmup
        self.training_step = 0

        # Default architectures
        if actor_channels is None:
            actor_channels = [16, 32]
        if critic_conv_channels is None:
            critic_conv_channels = [32, 64, 32]
        if critic_mlp_layers is None:
            critic_mlp_layers = [256, 128]

        print(f"Initializing MADDPG with Consistency Loss:")
        print(f"  Number of agents: {self.n_agents}")
        print(f"  Actor channels: {actor_channels}")
        print(f"  Critic conv channels: {critic_conv_channels}")
        print(f"  Critic MLP layers: {critic_mlp_layers}")
        print(f"  Smoothness penalties - temporal: {lambda_temporal}, spatial: {lambda_spatial}, zero-mean: {lambda_zero}")
        print(f"  Consistency loss - enabled: {consistency_enable}, lambda: {consistency_lambda}")
        print(f"  Consistency params - tau: {consistency_tau}, margin: {consistency_margin}, boundary_only: {consistency_boundary_only}")

        # Create networks with new architecture
        self.actor = CNNActorWithSimilarity(actor_channels, dropout_rate, similarity_dim).to(device)
        self.actor_target = CNNActorWithSimilarity(actor_channels, dropout_rate, similarity_dim).to(device)

        # Critic with circular padding
        self.critic = CNNCriticCircular(critic_conv_channels, critic_mlp_layers, dropout_rate).to(device)
        self.critic_target = CNNCriticCircular(critic_conv_channels, critic_mlp_layers, dropout_rate).to(device)

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr, weight_decay=weight_decay)

    def select_actions_batched(self, all_obs: torch.Tensor) -> np.ndarray:
        """
        Select actions for all agents in a batched forward pass.

        Args:
            all_obs: [n_agents, 2, 8, 8] - observations for all agents

        Returns:
            all_actions: [n_agents, 8, 8] - actions for all agents
        """
        with torch.no_grad():
            # CNN expects [Batch*N_Agents, 2, 8, 8]
            # all_obs is [N_Agents, 2, 8, 8] which works as a batch of 64
            all_actions = self.actor(all_obs, return_similarity=False)
            if isinstance(all_actions, tuple):
                all_actions = all_actions[0]
            return all_actions.cpu().numpy()

    def update_batched(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float, Dict[str, float]]:
        """
        Update actor and critic networks using a batch of experiences.

        Returns:
            critic_loss, actor_loss, loss_breakdown (dict with smoothness/consistency components)
        """
        obs_batch = batch['obs']             # [batch_size, n_agents, 2, 8, 8]
        act_batch = batch['acts']            # [batch_size, n_agents, 8, 8]
        prev_act_batch = batch['prev_acts']  # [batch_size, n_agents, 8, 8]
        rew_batch = batch['rews']            # [batch_size, n_agents]
        next_obs_batch = batch['next_obs']   # [batch_size, n_agents, 2, 8, 8]
        done_batch = batch['done']           # [batch_size, n_agents]

        batch_size = obs_batch.size(0)

        # Increment training step counter
        self.training_step += 1

        # ========================================================================
        # Update Critic
        # ========================================================================

        with torch.no_grad():
            # Reshape for actor: [batch_size * n_agents, 2, 8, 8]
            next_obs_flat = next_obs_batch.view(batch_size * self.n_agents, 2, 8, 8)

            # Get next actions from target actor
            next_actions_flat = self.actor_target(next_obs_flat, return_similarity=False)
            if isinstance(next_actions_flat, tuple):
                next_actions_flat = next_actions_flat[0]
            next_actions = next_actions_flat.view(batch_size, self.n_agents, 8, 8)

            # Prepare spatial data for critic
            next_obs_fields, next_act_field = self._prepare_spatial_data(
                next_obs_batch, next_actions
            )

            # Compute target Q-values
            target_q = self.critic_target(next_obs_fields, next_act_field)

            # Calculate TD target
            mean_reward = rew_batch.mean(dim=1, keepdim=True)  # Shared reward
            done_flag = done_batch[:, 0].unsqueeze(-1)
            target_value = mean_reward + self.gamma * (1.0 - done_flag) * target_q

        # Prepare current spatial data for critic
        obs_fields, act_field = self._prepare_spatial_data(obs_batch, act_batch)

        # Compute current Q-values
        current_q = self.critic(obs_fields, act_field)

        # Critic loss
        critic_loss = F.mse_loss(current_q, target_value)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clip)
        self.critic_optimizer.step()

        # ========================================================================
        # Update Actor
        # ========================================================================

        # Reshape observations for actor
        obs_flat = obs_batch.view(batch_size * self.n_agents, 2, 8, 8)

        # Get actions and similarity features from current policy
        actions_flat, sim_features_flat = self.actor(obs_flat, return_similarity=True)
        actions = actions_flat.view(batch_size, self.n_agents, 8, 8)
        sim_features = sim_features_flat.view(batch_size, self.n_agents, self.similarity_dim)

        # Prepare spatial data for critic
        obs_fields_new, act_field_new = self._prepare_spatial_data(obs_batch, actions)

        # Base actor loss: maximize Q-value
        q_loss = -self.critic(obs_fields_new, act_field_new).mean()

        # Temporal smoothness penalty (disabled by default)
        temporal_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_temporal > 0:
            temporal_loss = temporal_smoothness_loss(actions, prev_act_batch)

        # Spatial smoothness penalty
        spatial_loss = spatial_smoothness_loss(actions)

        # Zero-mean penalty
        zero_loss = zero_mean_loss(actions)

        # Consistency loss (with warmup)
        consistency_loss = torch.tensor(0.0, device=self.device)
        if self.consistency_enable and self.training_step >= self.consistency_warmup_steps:
            consistency_loss = observation_consistency_loss(
                sim_features,
                actions,
                tau=self.consistency_tau,
                margin=self.consistency_margin,
                boundary_only=self.consistency_boundary_only
            )

        # Combined actor loss
        actor_loss = (q_loss +
                     self.lambda_temporal * temporal_loss +
                     self.lambda_spatial * spatial_loss +
                     self.lambda_zero * zero_loss +
                     self.consistency_lambda * consistency_loss)

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clip)
        self.actor_optimizer.step()

        # Update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        # Loss breakdown for logging
        loss_breakdown = {
            'q_loss': q_loss.item(),
            'temporal_loss': temporal_loss.item() if isinstance(temporal_loss, torch.Tensor) else temporal_loss,
            'spatial_loss': spatial_loss.item(),
            'zero_loss': zero_loss.item(),
            'consistency_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss
        }

        return critic_loss.item(), actor_loss.item(), loss_breakdown

    def _prepare_spatial_data(
        self,
        obs_batch: torch.Tensor,
        act_batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert patch-based data to full spatial fields for critic.

        Args:
            obs_batch: [batch_size, n_agents, 2, 8, 8] - observations
            act_batch: [batch_size, n_agents, 8, 8] - actions

        Returns:
            obs_fields: [batch_size, 2, 64, 64] - full velocity fields
            act_field: [batch_size, 1, 64, 64] - full action field
        """
        batch_size = obs_batch.size(0)

        # Reconstruct full 64x64 fields from 8x8 patches
        # Agents are arranged in 8x8 grid
        obs_fields_u = torch.zeros(batch_size, 64, 64, device=obs_batch.device)
        obs_fields_w = torch.zeros(batch_size, 64, 64, device=obs_batch.device)
        act_fields = torch.zeros(batch_size, 64, 64, device=act_batch.device)

        agent_idx = 0
        for i in range(8):  # 8 patches in x
            for j in range(8):  # 8 patches in y
                # Extract patch from agent
                obs_patch = obs_batch[:, agent_idx, :, :, :]  # [batch, 2, 8, 8]
                act_patch = act_batch[:, agent_idx, :, :]     # [batch, 8, 8]

                # Place in full field
                x_start, x_end = i * 8, (i + 1) * 8
                y_start, y_end = j * 8, (j + 1) * 8

                obs_fields_u[:, x_start:x_end, y_start:y_end] = obs_patch[:, 0, :, :]
                obs_fields_w[:, x_start:x_end, y_start:y_end] = obs_patch[:, 1, :, :]
                act_fields[:, x_start:x_end, y_start:y_end] = act_patch

                agent_idx += 1

        # Stack observation fields
        obs_fields = torch.stack([obs_fields_u, obs_fields_w], dim=1)  # [batch, 2, 64, 64]
        act_field = act_fields.unsqueeze(1)  # [batch, 1, 64, 64]

        return obs_fields, act_field

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update of target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * source_param.data
            )

    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for saving."""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary."""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.actor_target.load_state_dict(state_dict['actor_target'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])
        self.training_step = state_dict.get('training_step', 0)


# ============================================================================
# Batched Replay Buffer with Previous Actions (same as original)
# ============================================================================

class BatchedReplayBuffer:
    """
    Replay buffer for shared policy training with smoothness penalties.
    Stores previous actions for temporal smoothness computation.
    """
    def __init__(
        self,
        capacity: int,
        n_agents: int,
        agent_ids: List[str]
    ):
        self.capacity = capacity
        self.n_agents = n_agents
        self.agent_ids = agent_ids

        # Each agent has 2x8x8=128 obs features and 8x8=64 action values
        self.obs_dim = 128  # 2 * 8 * 8
        self.act_dim = 64   # 8 * 8

        # Initialize buffers
        # Store spatial data directly
        self.obs_buf = np.zeros((capacity, n_agents, 2, 8, 8), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, n_agents, 2, 8, 8), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, n_agents, 8, 8), dtype=np.float32)
        self.prev_acts_buf = np.zeros((capacity, n_agents, 8, 8), dtype=np.float32)
        self.rews_buf = np.zeros((capacity, n_agents), dtype=np.float32)
        self.done_buf = np.zeros((capacity, n_agents), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add_batch(
        self,
        obs_batch: np.ndarray,       # [n_agents, 2, 8, 8]
        acts_batch: np.ndarray,      # [n_agents, 8, 8]
        prev_acts_batch: np.ndarray, # [n_agents, 8, 8]
        rews_batch: np.ndarray,      # [n_agents]
        next_obs_batch: np.ndarray,  # [n_agents, 2, 8, 8]
        dones_batch: np.ndarray      # [n_agents]
    ):
        """Add a batch of transitions to the buffer."""
        self.obs_buf[self.ptr] = obs_batch
        self.next_obs_buf[self.ptr] = next_obs_batch
        self.acts_buf[self.ptr] = acts_batch
        self.prev_acts_buf[self.ptr] = prev_acts_batch
        self.rews_buf[self.ptr] = rews_batch
        self.done_buf[self.ptr] = dones_batch

        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = {
            "obs": torch.as_tensor(self.obs_buf[idxs], device=device),
            "next_obs": torch.as_tensor(self.next_obs_buf[idxs], device=device),
            "acts": torch.as_tensor(self.acts_buf[idxs], device=device),
            "prev_acts": torch.as_tensor(self.prev_acts_buf[idxs], device=device),
            "rews": torch.as_tensor(self.rews_buf[idxs], device=device),
            "done": torch.as_tensor(self.done_buf[idxs], device=device)
        }

        return batch

    def save(self, path: str):
        """Save buffer state to disk."""
        np.savez(
            path,
            obs=self.obs_buf[:self.size],
            next_obs=self.next_obs_buf[:self.size],
            acts=self.acts_buf[:self.size],
            prev_acts=self.prev_acts_buf[:self.size],
            rews=self.rews_buf[:self.size],
            done=self.done_buf[:self.size],
            ptr=self.ptr,
            size=self.size
        )

    def load(self, path: str):
        """Load buffer state from disk."""
        data = np.load(path)

        # Load data
        load_size = data['size'] if 'size' in data else len(data['obs'])
        self.obs_buf[:load_size] = data['obs']
        self.next_obs_buf[:load_size] = data['next_obs']
        self.acts_buf[:load_size] = data['acts']

        # Handle backward compatibility
        if 'prev_acts' in data:
            self.prev_acts_buf[:load_size] = data['prev_acts']
        else:
            print("Warning: Loading old buffer without prev_acts, initializing to zeros")
            self.prev_acts_buf[:load_size] = 0.0

        self.rews_buf[:load_size] = data['rews']
        self.done_buf[:load_size] = data['done']
        self.ptr = int(data['ptr']) if 'ptr' in data else load_size % self.capacity
        self.size = min(load_size, self.capacity)

        print(f"Loaded buffer with {self.size} transitions")

    @property
    def full(self) -> bool:
        """Check if buffer is full."""
        return self.size == self.capacity
