import numpy as np
from mpi4py import MPI
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from utils import load_config, compute_reward, compute_diversity_penalty
import os
import matplotlib.pyplot as plt

class STWParallelEnvPoint(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "stw_point_v0"}

    def __init__(self, config, render_mode=None):
        """
        Point-based multi-agent environment for STW control.
        4096 agents (64x64 grid), each controlling a single point in the action field.

        Args:
            config: Configuration dictionary
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        self.config = config
        self.render_mode = render_mode

        # Full domain dimensions (64x64)
        self.full_grid_i = 64
        self.full_grid_j = 64

        # 4096 agents total (64x64 grid of points, one agent per point)
        self._num_agents = self.full_grid_i * self.full_grid_j

        # Create agent IDs: agent_0_0, agent_0_1, ..., agent_63_63
        self.possible_agents = [
            f"agent_{i}_{j}"
            for i in range(self.full_grid_i)
            for j in range(self.full_grid_j)
        ]

        # Initialize agent list
        self.agents = self.possible_agents[:]

        # Cache frequently accessed config values
        self.om_max = self.config['action']['om_max']
        self.action_scaling_factor = self.config['action']['scaling_factor']

        # Check if prev_action should be included in observations
        self.include_prev_action_in_obs = config.get('observation', {}).get('include_prev_action', False)
        self.prev_action_scale = config.get('observation', {}).get('prev_action_scale', 1.0)

        # Initialize prev_action field (64x64, always track)
        self.prev_action_field = np.zeros((64, 64), dtype=np.float32)

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
        self.initial_u_obs_field = None
        self.initial_w_obs_field = None

        # Initialize field shift coordinates if enabled
        if self.config.get('field_shift', {}).get('enable', False):
            Lx = self.config['field_shift']['domain_size']['Lx']
            Ly = self.config['field_shift']['domain_size']['Ly']
            self.shift_x = np.linspace(0, Lx, self.full_grid_i, endpoint=False)
            self.shift_y = np.linspace(0, Ly, self.full_grid_j, endpoint=False)

    def _setup_spaces(self):
        """Set up the observation and action spaces for point-based agents."""
        # Observation space: each agent observes scalar values at single point
        # If include_prev_action_in_obs: (3,) - [u, w, prev_action]
        # Otherwise: (2,) - [u, w] (backwards compatible)
        n_channels = 3 if self.include_prev_action_in_obs else 2
        self.observation_spaces = {
            agent: spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_channels,),  # Scalar observation per channel
                dtype=np.float32
            ) for agent in self.possible_agents
        }

        # Action space: each agent outputs single scalar action
        # Shape: (1,) - scalar action value
        self.action_spaces = {
            agent: spaces.Box(
                low=-1,
                high=1,
                shape=(1,),  # Single scalar action per agent
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

        # Reset prev_action_field to zero
        self.prev_action_field = np.zeros((64, 64), dtype=np.float32)

        # Reset agent list
        self.agents = self.possible_agents[:]

        # Initialize observations
        observations = {}
        infos = {}

        # Use initial observation if available, otherwise use zeros
        if self.initial_obs_captured:
            # Use the stored initial observation
            u_obs_field = self.initial_u_obs_field.copy()
            w_obs_field = self.initial_w_obs_field.copy()
        else:
            # Fallback to zeros if initial observation hasn't been captured yet
            u_obs_field = np.zeros((self.full_grid_i, self.full_grid_j), dtype=np.float32)
            w_obs_field = np.zeros((self.full_grid_i, self.full_grid_j), dtype=np.float32)

        for agent in self.agents:
            i, j = map(int, agent.split("_")[1:])
            observations[agent] = self.get_point_observation(u_obs_field, w_obs_field, i, j)
            infos[agent] = {"step": self.current_step, "total_steps": self.total_steps}

        self.last_observations = observations
        return observations, infos

    def step(self, actions):
        """
        Execute one time step within the environment.

        Args:
            actions: Dictionary mapping agent_id -> scalar action (1,)

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

        # Reconstruct full 64x64 action matrix from point-based actions
        action_matrix = self._reconstruct_action_field(actions)

        # NOTE: Zero-mean constraint is now applied in the training script BEFORE
        # actions are sent here, to ensure gradient flow and exact zero-mean guarantee.
        # The actions received here should already be exactly zero-mean.

        # Update prev_action_field for next step (store exactly what gets executed)
        self.prev_action_field = action_matrix.copy()

        self.last_action = actions

        # Send actions to simulation
        amp_send = np.double(action_matrix * self.om_max * self.action_scaling_factor)
        self.common_comm.Send([amp_send, MPI.DOUBLE], dest=1, tag=1)

        # Initialize arrays for receiving data
        u_obs_all = np.zeros((self.full_grid_i, self.full_grid_j), dtype=np.float64)
        w_obs_all = np.zeros((self.full_grid_i, self.full_grid_j), dtype=np.float64)
        self.dpdx = np.array(0.0, dtype=np.float64)

        # Receive observation data
        self.common_comm.Recv([u_obs_all, MPI.DOUBLE], source=1, tag=5)
        self.common_comm.Recv([w_obs_all, MPI.DOUBLE], source=1, tag=9)
        self.common_comm.Recv([self.dpdx, MPI.DOUBLE], source=1, tag=4)

        # Process observations: subtract mean and normalize
        u_mean = np.mean(u_obs_all)
        self.u_obs_field = (u_obs_all - u_mean) / self.om_max

        w_mean = np.mean(w_obs_all)
        self.w_obs_field = (w_obs_all - w_mean) / self.om_max

        # Store the initial observation if this is the first step
        if not self.initial_obs_captured and self.total_steps == 0:
            self.initial_u_obs_field = self.u_obs_field.copy()
            self.initial_w_obs_field = self.w_obs_field.copy()
            self.initial_obs_captured = True
            print("Initial observation captured for future resets")

        # Apply field shifting if enabled
        if self.config.get('field_shift', {}).get('enable', False):
            u_avg = np.mean(self.u_obs_field)
            dx_shift = u_avg * self.config['field_shift']['dt']
            self.u_obs_field = self.shift_field_subgrid(
                self.shift_x, self.shift_y, self.u_obs_field, dx_shift
            )
            self.w_obs_field = self.shift_field_subgrid(
                self.shift_x, self.shift_y, self.w_obs_field, dx_shift
            )

        # Compute global reward component
        global_reward = float(compute_reward(self.dpdx, self.config))

        # Compute diversity penalty (note: may need adaptation for point-based)
        diversity_penalty = compute_diversity_penalty(actions, self.config)

        # Create observations, rewards, terminations, truncations, infos dictionaries
        observations = {}
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
            i, j = map(int, agent.split("_")[1:])

            # Get point observation
            observations[agent] = self.get_point_observation(
                self.u_obs_field, self.w_obs_field, i, j
            )

            # All agents receive the same global reward
            rewards[agent] = (
                self.config['reward']['global_weight'] * global_reward +
                diversity_penalty
            )

            terminations[agent] = self.current_step >= self.episode_length
            truncations[agent] = False
            infos[agent] = {
                'dpdx': float(self.dpdx),
                'step': self.current_step,
                'total_steps': self.total_steps,
                'global_reward': global_reward,
                'diversity_penalty': diversity_penalty
            }

        # Update step counters
        self.current_step += 1
        self.total_steps += 1

        # Handle episode completion
        self._check_simulation_end()

        self.last_observations = observations
        self.last_rewards = rewards

        return observations, rewards, terminations, truncations, infos

    def _reconstruct_action_field(self, actions):
        """
        Reconstruct full 64x64 action matrix from 4096 scalar actions.

        Args:
            actions: Dictionary mapping agent_id -> scalar_action (1,)
                     e.g., {'agent_0_0': array([0.5]), 'agent_0_1': array([-0.3]), ...}

        Returns:
            action_matrix: (64, 64) numpy array
        """
        action_matrix = np.zeros((self.full_grid_i, self.full_grid_j), dtype=np.float32)

        for agent, action_scalar in actions.items():
            i, j = map(int, agent.split("_")[1:])
            # Extract scalar from (1,) array
            action_matrix[i, j] = action_scalar[0]

        return action_matrix

    def get_point_observation(self, u_obs_field, w_obs_field, i, j):
        """
        Extract observation at single point (i, j).

        Args:
            u_obs_field: Full 64x64 u-velocity field (normalized)
            w_obs_field: Full 64x64 w-velocity field (normalized)
            i, j: Point indices in 64x64 grid

        Returns:
            obs: (2,) or (3,) array = [u[i,j], w[i,j]] or [u[i,j], w[i,j], prev_action[i,j]]
        """
        # Extract scalar values at point (i, j)
        u_point = u_obs_field[i, j] * self.om_max
        w_point = w_obs_field[i, j] * self.om_max

        # Conditional inclusion of prev_action
        if self.include_prev_action_in_obs:
            # Extract previous action at this point
            prev_action_point = self.prev_action_field[i, j]
            # Stack as 3-channel [u, w, prev_action]
            obs = np.array([
                u_point,
                w_point,
                self.prev_action_scale * prev_action_point
            ], dtype=np.float32)
        else:
            # Original 2-channel [u, w] - backwards compatible
            obs = np.array([u_point, w_point], dtype=np.float32)

        return obs

    def shift_field_subgrid(self, x, y, field, dx_shift):
        """Shift field by dx_shift using numpy-based interpolation with periodic boundaries"""
        nx, ny = field.shape
        dx = x[1] - x[0]  # Grid spacing in x direction
        Lx = x[-1] - x[0] + dx  # Domain length

        # Calculate shift in grid units
        shift_grid_units = dx_shift / dx

        # Create result array
        field_shifted = np.zeros_like(field)

        # For each grid point, interpolate from shifted position
        for i in range(nx):
            for j in range(ny):
                # Calculate source position (shifted backwards)
                source_x_continuous = i - shift_grid_units

                # Handle periodic boundaries
                source_x_continuous = source_x_continuous % nx

                # Bilinear interpolation indices
                i1 = int(np.floor(source_x_continuous)) % nx
                i2 = (i1 + 1) % nx

                # Interpolation weight
                wx = source_x_continuous - np.floor(source_x_continuous)

                # Linear interpolation in x direction (y stays the same)
                field_shifted[i, j] = (1 - wx) * field[i1, j] + wx * field[i2, j]

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
