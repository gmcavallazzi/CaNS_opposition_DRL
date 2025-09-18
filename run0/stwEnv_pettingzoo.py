import numpy as np
from mpi4py import MPI
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from utils import load_config, compute_reward
import os
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

class STWParallelEnv(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "stw_v0"}
    
    def __init__(self, config, render_mode=None):
        """
        Multi-agent environment for the STW control problem using PettingZoo.

        Args:
            config: Configuration dictionary
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        self.config = config
        self.render_mode = render_mode
    
        # Grid dimensions
        self.grid_i = self.config['grid']['target']['i']
        self.grid_j = self.config['grid']['target']['j']
    
        # Each grid point is an agent
        self._num_agents = self.grid_i * self.grid_j
        self.possible_agents = [f"agent_{i}_{j}" for i in range(self.grid_i) for j in range(self.grid_j)]
    
        # Initialize agent list
        self.agents = self.possible_agents[:]
    
        # Get halo size from config (default to 1 if not specified)
        self.halo = self.config.get('halo', 1)

        # Cache frequently accessed config values
        self.om_max = self.config['action']['om_max']
        
    
        # Setup observation and action spaces
        self._setup_spaces()
    
        # Initialize environment state
        self.current_step = 0
        self.episode_length = self.config['training']['start_episode_length']
    
        # Class-level counter for tracking total steps
        self.total_steps = 0
    
        # Setup MPI communication
        self.setup_mpi()
        self.rank = self.common_comm.Get_rank()
    
        # Initialize simulation data attributes
        self.dpdx = 0.0
        self.last_action = {}
        self.last_rewards = {}
        self.last_observations = {}
    
        # Flag to track if we've captured the initial observation
        self.initial_obs_captured = False
        self.initial_u_obs_mat = None
        self.initial_w_obs_mat = None

        # Initialize field shift coordinates if enabled
        if self.config.get('field_shift', {}).get('enable', False):
            Lx = self.config['field_shift']['domain_size']['Lx']
            Ly = self.config['field_shift']['domain_size']['Ly']
            self.shift_x = np.linspace(0, Lx, self.grid_i, endpoint=False)
            self.shift_y = np.linspace(0, Ly, self.grid_j, endpoint=False)

    def _setup_spaces(self):
        """Set up the observation and action spaces for agents."""
        # Observation space
        # Grid-based observation space
        obs_height = 2 * self.halo + 1 if self.halo > 0 else 1
        obs_width = 2 * self.halo + 1 if self.halo > 0 else 1
        self.observation_spaces = {
            agent: spaces.Box(
                low=0, 
                high=255, 
                shape=(obs_height, obs_width, 2),  # 2 channels (u, w)
                dtype=np.uint8
            ) for agent in self.possible_agents
        }
        
        # Action space
        self.action_spaces = {
            agent: spaces.Box(
                low=-1,
                high=1,
                shape=(1,),  # Single continuous action
                dtype=np.float32
            ) for agent in self.possible_agents
        }
    
    def setup_mpi(self):
        """Initialize MPI communication with the simulation."""
        print("Python: Setting up MPI communication")
        self.sub_comm = MPI.COMM_SELF.Spawn(
            './cans', 
            args=[], 
            maxprocs=self.config['maxprocs'][0]
        )
        print("Python: Spawned cans process")
        
        self.common_comm = self.sub_comm.Merge(False)
        print("Python: Merged communicators")
        
        # Initial synchronization
        sync_flag = np.array([1], dtype=np.int32)
        print("Python: Sending initial sync signal")
        self.common_comm.Bcast([sync_flag, MPI.INT], root=0)
        print("Python: Initial sync complete")
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        print("Python: reset() called")
        self.current_step = 0
        
    
        # Reset agent list
        self.agents = self.possible_agents[:]
    
        # Initialize observations
        observations = {}
        infos = {}

        # Use initial observation if available, otherwise use zeros
        if self.initial_obs_captured:
            # Use the stored initial observation
            u_obs_mat = self.initial_u_obs_mat.copy()
            w_obs_mat = self.initial_w_obs_mat.copy()
        else:
            # Fallback to zeros if initial observation hasn't been captured yet
            # (This should only happen on the very first reset before any steps)
            u_obs_mat = np.zeros((self.grid_i, self.grid_j), dtype=np.uint8)
            w_obs_mat = np.zeros((self.grid_i, self.grid_j), dtype=np.uint8)
            
        for agent in self.agents:
            i, j = map(int, agent.split("_")[1:])
            observations[agent] = self.get_local_observation(u_obs_mat, w_obs_mat, i, j)
            infos[agent] = {"step": self.current_step, "total_steps": self.total_steps}
    
        self.last_observations = observations
        return observations, infos
    
    def step(self, actions):
        """
        Execute one time step within the environment.
        
        Args:
            actions: Dictionary of actions for each agent
        
        Returns:
            observations: Dictionary of observations for each agent
            rewards: Dictionary of rewards for each agent
            terminations: Dictionary indicating if episodes are done
            truncations: Dictionary indicating if episodes are truncated
            infos: Dictionary containing additional information
        """
        # Send control message based on step
        control_msg = b'START' if self.current_step == 0 else b'CONTN'
        self.common_comm.Bcast([control_msg, MPI.CHAR], root=0)
        
        # Process actions based on agent type            
        # Convert actions dictionary to action matrix
        action_matrix = np.zeros((self.grid_i, self.grid_j))
    
        # First collect all action values
        all_actions = np.array([action[0] for action in actions.values()])
    
        # Calculate the mean and subtract from all actions to enforce zero-net-mass-flux
        action_mean = np.mean(all_actions)
    
        # Apply zero-mean normalization when filling the action matrix
        for agent, action in actions.items():
            i, j = map(int, agent.split("_")[1:])
            action_matrix[i, j] = action[0] - action_mean
        
        self.last_action = actions
        
        # Send actions to simulation - this is what goes to CaNS
        amp_send = np.double(action_matrix * self.om_max)
        
        
        self.common_comm.Send([amp_send, MPI.DOUBLE], dest=1, tag=1)
        
        # Initialize arrays for receiving data
        self.u_obs_all = np.zeros((self.grid_i, self.grid_j), dtype=np.float64)
        self.w_obs_all = np.zeros((self.grid_i, self.grid_j), dtype=np.float64)
        self.dpdx = np.array(0.0, dtype=np.float64)
        
        # Receive observation data
        self.common_comm.Recv([self.u_obs_all, MPI.DOUBLE], source=1, tag=5)
        self.common_comm.Recv([self.w_obs_all, MPI.DOUBLE], source=1, tag=9)
        self.common_comm.Recv([self.dpdx, MPI.DOUBLE], source=1, tag=4)
        
        
        # Process observations: subtract mean and divide by om_max
        # For u velocity component
        u_mean = np.mean(self.u_obs_all)
        self.u_obs_mat = (self.u_obs_all - u_mean) / self.om_max
        
        # For w velocity component
        w_mean = np.mean(self.w_obs_all)
        self.w_obs_mat = (self.w_obs_all - w_mean) / self.om_max
        
        # Store the initial observation if this is the first step of the first episode
        if not self.initial_obs_captured and self.total_steps == 0:
            self.initial_u_obs_mat = self.u_obs_mat.copy()
            self.initial_w_obs_mat = self.w_obs_mat.copy()
            self.initial_obs_captured = True
            print("Initial observation captured for future resets")

        # Apply field shifting if enabled
        if self.config.get('field_shift', {}).get('enable', False):
            u_avg = np.mean(self.u_obs_mat)
            dx_shift = u_avg * self.config['field_shift']['dt']
            self.u_obs_mat = self.shift_field_subgrid(self.shift_x, self.shift_y, self.u_obs_mat, dx_shift)
            self.w_obs_mat = self.shift_field_subgrid(self.shift_x, self.shift_y, self.w_obs_mat, dx_shift)

        # Compute global reward component
        global_reward = float(compute_reward(self.dpdx, self.config))
        
        # Create observations, rewards, terminations, truncations, infos dictionaries
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            i, j = map(int, agent.split("_")[1:])
                
            # Get local observation
            observations[agent] = self.get_local_observation(self.u_obs_mat, self.w_obs_mat, i, j)
                
            # Combine global and local rewards
            rewards[agent] = self.config['reward']['global_weight'] * global_reward 
                
            terminations[agent] = self.current_step >= self.episode_length
            truncations[agent] = False
            infos[agent] = {
                'dpdx': float(self.dpdx),
                'step': self.current_step,
                'total_steps': self.total_steps,
                'global_reward': global_reward
            }
        
        # Update step counters
        self.current_step += 1
        self.total_steps += 1
        
        # Handle episode completion
        self._check_simulation_end()
        
        self.last_observations = observations
        self.last_rewards = rewards
        
        return observations, rewards, terminations, truncations, infos



    def get_local_observation(self, u_obs, w_obs, i, j):
        """
        Extract local observation for a grid-based agent with periodic boundary conditions.
        Size depends on the halo parameter in config.
        Directly processes raw observations by multiplying with om_max.
        """
        om_max = self.om_max
    
        if self.halo == 0:
            # Single point observation (1x1x2)
            local_obs = np.zeros((1, 1, 2), dtype=np.float32)
            local_obs[0, 0, 0] = u_obs[i, j] * om_max  # u velocity at agent's position
            local_obs[0, 0, 1] = w_obs[i, j] * om_max  # w velocity at agent's position
        else:
            # Square observation with size (2*halo+1) x (2*halo+1) x 2
            size = 2 * self.halo + 1
            local_obs = np.zeros((size, size, 2), dtype=np.float32)
        
            for di in range(-self.halo, self.halo + 1):
                for dj in range(-self.halo, self.halo + 1):
                    # Apply periodic boundary conditions
                    obs_i = (i + di) % self.grid_i
                    obs_j = (j + dj) % self.grid_j
                
                    # Map to observation array indices
                    local_i = di + self.halo
                    local_j = dj + self.halo
                
                    # First channel: u velocity, directly multiplied by om_max
                    local_obs[local_i, local_j, 0] = u_obs[obs_i, obs_j] * om_max
                    # Second channel: w velocity, directly multiplied by om_max
                    local_obs[local_i, local_j, 1] = w_obs[obs_i, obs_j] * om_max

        return local_obs

    def shift_field_subgrid(self, x, y, field, dx_shift):
        """Shift field by dx_shift using subgrid interpolation"""
        nx, ny = field.shape
        Lx = x[-1] - x[0] + (x[1] - x[0])  # Add one grid spacing for correct domain size

        # Create interpolator
        interp = RegularGridInterpolator((x, y), field,
                                       bounds_error=False,
                                       fill_value=None,  # Use nearest for out-of-bounds
                                       method='linear')

        # Create shifted coordinates (with periodic boundary conditions)
        X, Y = np.meshgrid(x, y, indexing='ij')
        X_shifted = X - dx_shift

        # Handle periodic boundaries
        X_shifted = X_shifted % Lx

        # Create points for interpolation
        points = np.column_stack([X_shifted.ravel(), Y.ravel()])

        # Interpolate
        field_shifted = interp(points).reshape(nx, ny)

        return field_shifted

    def _check_simulation_end(self):
        """Check simulation state and handle MPI communication."""
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        # Check if overall simulation is complete
        if self.total_steps >= self.config['total_timesteps']:
            # Overall simulation is complete
            print("Python: Sending ENDED - simulation complete")
            self.common_comm.Bcast([b'ENDED', MPI.CHAR], root=0)
            self.common_comm.Free()
            self.sub_comm.Disconnect()
        elif self.current_step >= self.episode_length:
            # Episode just completed
            print("Python: Sending CONTR - episode complete")
            file_num = np.random.randint(1, 10)
            src_file = os.path.join(data_dir, f"fld_{file_num:04d}.bin")
            dst_file = os.path.join(data_dir, "fld.bin")
            
            if self.rank == 0:
                print(f"Copying file {file_num:04d} to fld.bin")
                os.system(f"cp {src_file} {dst_file}")
            
            self.common_comm.Bcast([b'CONTR', MPI.CHAR], root=0)
        else:
            print("Python: Sending CONTN - continuing episode")
            self.common_comm.Bcast([b'CONTN', MPI.CHAR], root=0)
   
    
    def close(self):
        """Clean up resources."""
        
        try:
            self.common_comm.Bcast([b'ENDED', MPI.CHAR], root=0)
            self.common_comm.Free()
            self.sub_comm.Disconnect()
        except Exception as e:
            print(f"Error during environment cleanup: {e}")
    
        
    def observation_space(self, agent):
        """Return the observation space for a specific agent."""
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        """Return the action space for a specific agent."""
        return self.action_spaces[agent]
    
    def set_episode_length(self, episode_length):
        """Set the episode length for curriculum learning."""
        self.episode_length = episode_length
