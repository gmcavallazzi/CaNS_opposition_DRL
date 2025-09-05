import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict

# MLP-based actor network
class MLPActor(nn.Module):
    """MLP-based actor network."""
    def __init__(self, obs_shape: Tuple[int, ...], act_shape: Tuple[int, ...], 
                 dropout_rate: float = 0.05, hidden_layers: List[int] = None):
        super().__init__()
        
        # Calculate flattened input size
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = int(np.prod(act_shape))
        
        # Use provided architecture or default
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]  # Enhanced architecture
        
        # Build the network layers dynamically
        layers = [nn.Flatten()]
        
        # Input layer
        prev_dim = self.obs_dim
        
        # Add layer normalization after input
        layers.append(nn.LayerNorm(prev_dim))
        
        # Hidden layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))  # Layer normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        
        # Output layer with tanh activation
        self.output_layer = nn.Linear(prev_dim, self.act_dim)
        
        # Create the network
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module == self.output_layer:  # Output layer
                nn.init.uniform_(module.weight, -0.003, 0.003)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.003, 0.003)
            else:  # Hidden layers
                nn.init.xavier_uniform_(module.weight, gain=1.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        x = self.output_layer(x)
        return torch.tanh(x)  # Explicit tanh activation for final output

# MLP-based critic network
class MLPCritic(nn.Module):
    """MLP-based critic network that takes observation and action of all agents."""
    def __init__(self, obs_shape: Tuple[int, ...], act_shape: Tuple[int, ...], 
                 n_agents: int, dropout_rate: float = 0.05, hidden_layers: List[int] = None):
        super().__init__()
        
        # Calculate flattened input sizes
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = int(np.prod(act_shape))
        
        # Total input size is observations and actions from all agents
        self.input_dim = self.obs_dim * n_agents + self.act_dim * n_agents
        
        # Use provided architecture or default
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]  # Enhanced architecture
        
        # Build the network layers dynamically
        layers = []
        
        # Add layer normalization after input
        layers.append(nn.LayerNorm(self.input_dim))
        
        # Input layer
        prev_dim = self.input_dim
        
        # Hidden layers
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))  # Layer normalization
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)  # Q-value output
        
        # Create the network
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module == self.output_layer:  # Output layer
                nn.init.uniform_(module.weight, -0.003, 0.003)
                if module.bias is not None:
                    nn.init.uniform_(module.bias, -0.003, 0.003)
            else:  # Hidden layers
                nn.init.xavier_uniform_(module.weight, gain=1.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs_all: torch.Tensor, act_all: torch.Tensor) -> torch.Tensor:
        # Combine all observations and actions
        x = torch.cat([obs_all, act_all], dim=1)
        x = self.net(x)
        return self.output_layer(x)

class SharedPolicyMADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient with shared policies.
    All agents share the same policy network weights.
    """
    def __init__(
        self,
        agents: List[str],
        obs_shape: Tuple[int, ...],
        act_shape: Tuple[int, ...],
        gamma: float = 0.99,
        tau: float = 0.01,
        lr: float = 1e-4,  # Reduced learning rate
        dropout_rate: float = 0.05,  # Reduced dropout
        weight_decay: float = 1e-5,  # Added L2 regularization
        device: str = "cpu",
        pi_arch: List[int] = None,
        qf_arch: List[int] = None
    ):
        self.agents = agents
        self.n_agents = len(agents)  # This line must come before using self.n_agents
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.weight_decay = weight_decay
        
        # Network architectures
        if pi_arch is None:
            pi_arch = [128, 64, 32]  # Updated architecture
        if qf_arch is None:
            qf_arch = [128, 64, 32]  # Updated architecture
            
        print(f"Using MLP networks for all agents")
        print(f"Actor architecture: {pi_arch}")
        print(f"Critic architecture: {qf_arch}")
        # Create networks
        self.actor = MLPActor(obs_shape, act_shape, dropout_rate, pi_arch).to(device)
        self.actor_target = MLPActor(obs_shape, act_shape, dropout_rate, pi_arch).to(device)
        self.critic = MLPCritic(obs_shape, act_shape, self.n_agents, dropout_rate, qf_arch).to(device)
        self.critic_target = MLPCritic(obs_shape, act_shape, self.n_agents, dropout_rate, qf_arch).to(device)
        
        # Initialize target networks with same weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Setup optimizers with weight decay
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr, weight_decay=weight_decay)
    
    
    def select_action(self, agent_id: str, obs: torch.Tensor) -> np.ndarray:
        """Select action for a specific agent given its observation."""
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
        return action
    
    def select_actions_batched(self, all_obs: torch.Tensor) -> np.ndarray:
        """Select actions for all agents in a single batched forward pass."""
        with torch.no_grad():
            # Process all observations in a single forward pass
            all_actions = self.actor(all_obs).cpu().numpy()
        return all_actions
    
    def update_batched(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, float]:
        """Update actor and critic networks using a batch of experiences (optimized for shared policy)."""
        obs_batch = batch['obs']           # (batch_size, n_agents * obs_dim)
        act_batch = batch['acts']          # (batch_size, n_agents * act_dim)
        rew_batch = batch['rews']          # (batch_size, n_agents)
        next_obs_batch = batch['next_obs'] # (batch_size, n_agents * obs_dim)
        done_batch = batch['done']         # (batch_size, n_agents)
        
        batch_size = obs_batch.size(0)
        
        # Compute target Q-values (batched)
        with torch.no_grad():
            # Reshape observations for target actor
            next_obs_reshaped = self._reshape_obs_for_actor(next_obs_batch)
            # Get next actions from target actor (batched)
            next_actions = self.actor_target(next_obs_reshaped)
            
            # Add noise to next actions for smoothing
            noise = torch.randn_like(next_actions) * 0.1
            next_actions = torch.clamp(next_actions + noise, -1, 1)
            
            # Reshape for critic input
            next_act_batch = self._reshape_actions_for_critic(next_actions)
            
            # Compute target Q-values
            target_q = self.critic_target(next_obs_batch, next_act_batch)
            
            # Calculate targets for all agents
            # We take the mean reward across all agents since they share the same policy
            mean_reward = rew_batch.mean(dim=1, keepdim=True)
            # Use any agent's done flag (all agents terminate together)
            done_flag = done_batch[:, 0].unsqueeze(-1)
            
            target_value = mean_reward + self.gamma * (1.0 - done_flag) * target_q
        
        # Update critic (batched)
        current_q = self.critic(obs_batch, act_batch)
        critic_loss = F.mse_loss(current_q, target_value)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Update actor (batched)
        # Reshape observations for actor
        obs_reshaped = self._reshape_obs_for_actor(obs_batch)
        
        # Get actions from current policy (batched)
        actions = self.actor(obs_reshaped)
        
        # Reshape for critic input
        act_batch_new = self._reshape_actions_for_critic(actions)
        
        # Compute actor loss
        actor_loss = -self.critic(obs_batch, act_batch_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
        
    def _reshape_obs_for_actor(self, obs_batch: torch.Tensor) -> torch.Tensor:
        """Reshape observation batch for actor network."""
        batch_size = obs_batch.size(0)
        obs_dim = int(np.prod(self.obs_shape))
        
        # Reshape to [batch_size, n_agents, obs_dim]
        reshaped = obs_batch.view(batch_size, self.n_agents, obs_dim)
        # Flatten batch and agents dimensions
        reshaped = reshaped.view(batch_size * self.n_agents, obs_dim)
        return reshaped
            
    def _reshape_actions_for_critic(self, actions: torch.Tensor) -> torch.Tensor:
        """Reshape actions from actor to format expected by critic."""
        batch_size = actions.size(0) // self.n_agents
        act_dim = actions.size(1)
        
        # Reshape to [batch_size, n_agents, act_dim]
        reshaped = actions.view(batch_size, self.n_agents, act_dim)
        # Flatten agent and action dimensions
        reshaped = reshaped.view(batch_size, self.n_agents * act_dim)
        return reshaped
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update of target network parameters."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.tau) * target_param.data + self.tau * source_param.data)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for saving."""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dictionary."""
        self.actor.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.actor_target.load_state_dict(state_dict['actor_target'])
        self.critic_target.load_state_dict(state_dict['critic_target'])
        self.actor_optimizer.load_state_dict(state_dict['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state_dict['critic_optimizer'])

class BatchedReplayBuffer:
    """
    Optimized replay buffer for shared policy training.
    Stores transitions more efficiently for large numbers of agents.
    """
    def __init__(
        self, 
        capacity: int, 
        obs_shape: Tuple[int, ...], 
        act_shape: Tuple[int, ...],
        n_agents: int,
        agent_ids: List[str]
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.n_agents = n_agents
        self.agent_ids = agent_ids
        
        # Calculate flattened dimensions
        self.obs_dim = int(np.prod(obs_shape))
        self.act_dim = int(np.prod(act_shape))
        
        # Initialize buffers - store all agents' data together
        self.obs_buf = np.zeros((capacity, n_agents, self.obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, n_agents, self.obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, n_agents, self.act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((capacity, n_agents), dtype=np.float32)
        self.done_buf = np.zeros((capacity, n_agents), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0
    
    def add_batch(
        self, 
        obs_batch: np.ndarray, 
        acts_batch: np.ndarray,
        rews_batch: np.ndarray,
        next_obs_batch: np.ndarray,
        dones_batch: np.ndarray
    ):
        """Add a batch of transitions to the buffer."""
        self.obs_buf[self.ptr] = obs_batch.reshape(self.n_agents, self.obs_dim)
        self.next_obs_buf[self.ptr] = next_obs_batch.reshape(self.n_agents, self.obs_dim)
        self.acts_buf[self.ptr] = acts_batch.reshape(self.n_agents, self.act_dim)
        self.rews_buf[self.ptr] = rews_batch
        self.done_buf[self.ptr] = dones_batch
        
        # Update pointer and size
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add(
        self, 
        obs: Dict[str, np.ndarray], 
        acts: Dict[str, np.ndarray],
        rews: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        dones: Dict[str, bool]
    ):
        """Add a transition to the buffer (compatible with original interface)."""
        # Convert dictionaries to arrays
        obs_batch = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        next_obs_batch = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        acts_batch = np.zeros((self.n_agents, self.act_dim), dtype=np.float32)
        rews_batch = np.zeros(self.n_agents, dtype=np.float32)
        dones_batch = np.zeros(self.n_agents, dtype=np.float32)
        
        for i, agent_id in enumerate(self.agent_ids):
            # Flatten observation and action if needed
            flat_obs = obs[agent_id].reshape(-1)
            flat_next_obs = next_obs[agent_id].reshape(-1)
            flat_act = acts[agent_id].reshape(-1)
            
            # Store in arrays
            obs_batch[i] = flat_obs
            next_obs_batch[i] = flat_next_obs
            acts_batch[i] = flat_act
            rews_batch[i] = rews[agent_id]
            dones_batch[i] = float(dones[agent_id])
        
        # Add batch to buffer
        self.add_batch(obs_batch, acts_batch, rews_batch, next_obs_batch, dones_batch)
    
    def sample(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        # Create batch
        batch = {
            # Reshape observations and actions for critic input
            # (batch_size, n_agents * obs_dim) and (batch_size, n_agents * act_dim)
            "obs": torch.as_tensor(self.obs_buf[idxs].reshape(batch_size, -1), device=device),
            "next_obs": torch.as_tensor(self.next_obs_buf[idxs].reshape(batch_size, -1), device=device),
            "acts": torch.as_tensor(self.acts_buf[idxs].reshape(batch_size, -1), device=device),
            # Keep rewards and dones as (batch_size, n_agents)
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
            rews=self.rews_buf[:self.size],
            done=self.done_buf[:self.size],
            ptr=self.ptr,
            size=self.size
        )
    
    def load(self, path: str):
        """Load buffer state from disk."""
        data = np.load(path)
        
        # Check if shapes match
        if data['obs'].shape[1:] != (self.n_agents, self.obs_dim):
            raise ValueError(
                f"Cannot load buffer with incompatible shapes. "
                f"Expected ({self.n_agents}, {self.obs_dim}), got {data['obs'].shape[1:]}"
            )
        
        # Load data
        load_size = data['size'] if 'size' in data else len(data['obs'])
        self.obs_buf[:load_size] = data['obs']
        self.next_obs_buf[:load_size] = data['next_obs']
        self.acts_buf[:load_size] = data['acts']
        self.rews_buf[:load_size] = data['rews']
        self.done_buf[:load_size] = data['done']
        self.ptr = int(data['ptr']) if 'ptr' in data else load_size % self.capacity
        self.size = min(load_size, self.capacity)
        
        print(f"Loaded buffer with {self.size} transitions")
    
    @property
    def full(self) -> bool:
        """Check if buffer is full."""
        return self.size == self.capacity