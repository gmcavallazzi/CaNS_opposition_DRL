import numpy as np
from mpi4py import MPI
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
from utils import load_config, compute_reward, compute_local_reward, img_rescale, rescale_amp
import os
import matplotlib.pyplot as plt
import sys
import multiprocessing as mp


def agent_processing_worker():
    """Worker function for parallel agent observation processing"""
    # Get parent communicator
    parentcomm = MPI.Comm.Get_parent()
    
    # Get worker info
    myid = MPI.COMM_WORLD.Get_rank()
    mysize = MPI.COMM_WORLD.Get_size()
    
    # Create worker communicator
    workercomm = MPI.COMM_WORLD.Dup()
    
    # Merge with parent
    intracomm = parentcomm.Merge(high=True)
    ourid = intracomm.Get_rank()
    
    while True:
        # Receive work signal
        work_signal = np.empty(1, dtype=np.int32)
        if ourid == 1:  # First worker receives from main process
            intracomm.Recv(work_signal, source=0, tag=300)
        
        # Broadcast to all workers
        workercomm.Bcast(work_signal, root=0)
        
        if work_signal[0] == -1:  # Shutdown signal
            break
            
        # Receive data from main process
        if ourid == 1:
            print(f"WORKER {myid}: Receiving data from main process")
            # Receive grid dimensions
            grid_dims = np.empty(2, dtype=np.int32)
            intracomm.Recv(grid_dims, source=0, tag=301)
            print(f"WORKER {myid}: Received grid dimensions {grid_dims}")
            
            # Receive observation data (keep as float64)
            grid_size = grid_dims[0] * grid_dims[1]
            u_obs_flat = np.empty(grid_size, dtype=np.float64)
            w_obs_flat = np.empty(grid_size, dtype=np.float64)
            print(f"WORKER {myid}: Receiving {grid_size} observation values")
            intracomm.Recv(u_obs_flat, source=0, tag=302)
            intracomm.Recv(w_obs_flat, source=0, tag=303)
            print(f"WORKER {myid}: Received observation data")
            
            # Receive om_max only (halo assumed to be 0)
            om_max_arr = np.empty(1, dtype=np.float64)
            intracomm.Recv(om_max_arr, source=0, tag=304)
            print(f"WORKER {myid}: Received om_max = {om_max_arr[0]}")
        else:
            grid_dims = np.empty(2, dtype=np.int32)
            grid_size = None
            u_obs_flat = None
            w_obs_flat = None
            om_max_arr = np.empty(1, dtype=np.float64)
        
        # Broadcast to all workers
        print(f"WORKER {myid}: Broadcasting data to all workers")
        workercomm.Bcast(grid_dims, root=0)
        
        if grid_size is None:
            grid_size = grid_dims[0] * grid_dims[1]
            u_obs_flat = np.empty(grid_size, dtype=np.float64)
            w_obs_flat = np.empty(grid_size, dtype=np.float64)
            
        workercomm.Bcast(u_obs_flat, root=0)
        workercomm.Bcast(w_obs_flat, root=0)
        workercomm.Bcast(om_max_arr, root=0)
        print(f"WORKER {myid}: Broadcast complete")
        
        # Extract parameters
        grid_i, grid_j = grid_dims
        om_max = om_max_arr[0]
        
        # Reshape to matrices (keep as float64)
        u_obs_mat = u_obs_flat.reshape(grid_i, grid_j)
        w_obs_mat = w_obs_flat.reshape(grid_i, grid_j)
        
        # Determine work chunk for this worker
        total_agents = grid_i * grid_j
        agents_per_worker = total_agents // mysize
        start_idx = myid * agents_per_worker
        end_idx = (myid + 1) * agents_per_worker if myid < mysize - 1 else total_agents
        
        # Process assigned agents (halo = 0 only)
        num_agents_for_worker = end_idx - start_idx
        print(f"WORKER {myid}: Processing {num_agents_for_worker} agents (indices {start_idx}-{end_idx})")
        
        local_observations = []
        local_agent_ids = []
        
        for agent_idx in range(start_idx, end_idx):
            i = agent_idx // grid_j
            j = agent_idx % grid_j
            agent_id = f"agent_{i}_{j}"
            local_agent_ids.append(agent_id)
            
            # Create 1x1x2 observation (halo=0)
            local_obs = np.zeros((1, 1, 2), dtype=np.float32)
            local_obs[0, 0, 0] = u_obs_mat[i, j] * om_max
            local_obs[0, 0, 1] = w_obs_mat[i, j] * om_max
            
            local_observations.append(local_obs.flatten())
        
        print(f"WORKER {myid}: Finished processing, gathering results")
        # Gather all observations
        all_observations = workercomm.gather(local_observations, root=0)
        all_agent_ids = workercomm.gather(local_agent_ids, root=0)
        print(f"WORKER {myid}: Gather complete")
        
        # Send results back to main process
        if myid == 0:
            print(f"WORKER 0: Flattening gathered data from {mysize} workers")
            # Flatten gathered data
            flat_observations = []
            flat_agent_ids = []
            for worker_obs, worker_ids in zip(all_observations, all_agent_ids):
                flat_observations.extend(worker_obs)
                flat_agent_ids.extend(worker_ids)
            
            # Send number of observations
            num_obs = len(flat_observations)
            print(f"WORKER 0: Sending {num_obs} observations to main process")
            result_count = np.array([num_obs], dtype=np.int32)
            intracomm.Send(result_count, dest=0, tag=350)
            
            # Send observations as one large array
            if num_obs > 0:
                print(f"WORKER 0: Sending observation data")
                all_obs_array = np.array(flat_observations, dtype=np.float32).flatten()
                intracomm.Send(all_obs_array, dest=0, tag=351)
                
                # Send agent IDs
                print(f"WORKER 0: Sending agent IDs")
                agent_id_str = ",".join(flat_agent_ids)
                agent_id_bytes = agent_id_str.encode('utf-8')
                id_length = np.array([len(agent_id_bytes)], dtype=np.int32)
                intracomm.Send(id_length, dest=0, tag=352)
                intracomm.Send([agent_id_bytes, MPI.CHAR], dest=0, tag=353)
                print(f"WORKER 0: All data sent to main process")
    
    # Cleanup
    workercomm.Free()
    intracomm.Free()
    parentcomm.Disconnect()


class STWParallelEnvCustom(ParallelEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "name": "stw_custom_v0"}
    
    def __init__(self, config, render_mode=None, save_images=False, image_save_dir="./evaluation_images", num_workers=None):
        """
        Custom multi-agent environment with parallel agent processing.
        
        Args:
            config: Configuration dictionary
            render_mode: Rendering mode ("human" or "rgb_array")
            save_images: Whether to save images during evaluation
            image_save_dir: Directory to save images
            num_workers: Number of parallel workers (None for auto-detect based on grid size)
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
        
        # Get halo size from config
        self.halo = self.config.get('halo', 1)
        
        # Image saving setup
        self.save_images = save_images
        self.image_save_dir = image_save_dir
        self.step_counter = 0
        self.episode_counter = 0
        
        if self.save_images:
            os.makedirs(self.image_save_dir, exist_ok=True)
            print(f"Saving images to: {self.image_save_dir}")
        
        # Setup observation and action spaces
        self._setup_spaces()
        
        # Initialize environment state
        self.current_step = 0
        self.episode_length = self.config['training']['start_episode_length']
        
        # Class-level counter for tracking total steps
        self.total_steps = 0
        
        # Determine number of workers
        if num_workers is None:
            if self._num_agents > 100000:  # For very large grids
                self.num_workers = min(mp.cpu_count(), 8)
            elif self._num_agents > 10000:  # For medium grids
                self.num_workers = min(mp.cpu_count(), 4)
            else:
                self.num_workers = 0  # No parallelization for small grids
        else:
            self.num_workers = num_workers
            
        print(f"Using {self.num_workers} workers for {self._num_agents} agents")
        
        # Setup MPI communication
        self.setup_mpi()
        self.rank = self.common_comm.Get_rank()
        
        # Setup worker processes if needed
        if self.num_workers > 0:
            self.setup_workers()
        else:
            self.worker_comm = None
            self.intracomm_workers = None
        
        # Initialize simulation data attributes
        self.dpdx = 0.0
        self.last_action = {}
        self.last_rewards = {}
        self.last_observations = {}
        
        # Flag to track if we've captured the initial observation
        self.initial_obs_captured = False
        self.initial_u_obs_mat = None
        self.initial_w_obs_mat = None

    def _setup_spaces(self):
        """Set up the observation and action spaces for agents."""
        # Observation space
        obs_height = 2 * self.halo + 1 if self.halo > 0 else 1
        obs_width = 2 * self.halo + 1 if self.halo > 0 else 1
        self.observation_spaces = {
            agent: spaces.Box(
                low=0, 
                high=255, 
                shape=(obs_height, obs_width, 2),
                dtype=np.uint8
            ) for agent in self.possible_agents
        }
        
        # Action space
        self.action_spaces = {
            agent: spaces.Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float32
            ) for agent in self.possible_agents
        }
    
    def setup_mpi(self):
        """Initialize MPI communication with CaNS simulation."""
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
    
    def setup_workers(self):
        """Setup parallel worker processes for agent processing."""
        print(f"Python: Spawning {self.num_workers} worker processes")
        self.worker_comm = MPI.COMM_SELF.Spawn(
            'python', 
            args=[__file__, 'worker'], 
            maxprocs=self.num_workers
        )
        print("Python: Worker processes spawned")
        
        # Create merged communicator
        self.intracomm_workers = self.worker_comm.Merge(high=False)
        print(f"Python: Worker communicator created - size={self.intracomm_workers.Get_size()}")
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        print("Python: reset() called")
        self.current_step = 0
        
        # Reset image counters
        if self.save_images:
            self.step_counter = 0
            self.episode_counter += 1
            print(f"Starting episode {self.episode_counter}")
    
        # Reset agent list
        self.agents = self.possible_agents[:]
    
        # Initialize observations
        observations = {}
        infos = {}

        # Use initial observation if available, otherwise use zeros
        if self.initial_obs_captured:
            u_obs_mat = self.initial_u_obs_mat.copy()
            w_obs_mat = self.initial_w_obs_mat.copy()
        else:
            u_obs_mat = np.zeros((self.grid_i, self.grid_j), dtype=np.uint8)
            w_obs_mat = np.zeros((self.grid_i, self.grid_j), dtype=np.uint8)
            
        # Get observations using parallel processing if available
        if self.num_workers > 0:
            observations = self._process_agents_parallel(u_obs_mat, w_obs_mat)
        else:
            observations = self._process_agents_sequential(u_obs_mat, w_obs_mat)
            
        # Add infos
        for agent in self.agents:
            infos[agent] = {"step": self.current_step, "total_steps": self.total_steps}
    
        self.last_observations = observations
        return observations, infos
    
    def step(self, actions):
        """Execute one time step with parallel agent processing."""
        # Send control message based on step
        control_msg = b'START' if self.current_step == 0 else b'CONTN'
        self.common_comm.Bcast([control_msg, MPI.CHAR], root=0)
        
        # Process actions (same as original)
        action_matrix = np.zeros((self.grid_i, self.grid_j))
        all_actions = np.array([action[0] for action in actions.values()])
        action_mean = np.mean(all_actions)
        
        for agent, action in actions.items():
            i, j = map(int, agent.split("_")[1:])
            action_matrix[i, j] = action[0] - action_mean
        
        self.last_action = actions
        
        # Send actions to CaNS (unchanged)
        amp_send = np.double(action_matrix * self.config['action']['om_max'])
        
        if self.save_images:
            self._save_actions_to_cans_image(amp_send)
        
        self.common_comm.Send([amp_send, MPI.DOUBLE], dest=1, tag=1)
        
        # Receive observation data from CaNS (unchanged)
        print(f"PYTHON: About to receive observation data from CaNS for {self.grid_i}x{self.grid_j} grid")
        self.u_obs_all = np.zeros((self.grid_i, self.grid_j), dtype=np.float64)
        self.w_obs_all = np.zeros((self.grid_i, self.grid_j), dtype=np.float64)
        self.dpdx = np.array(0.0, dtype=np.float64)
        
        print("PYTHON: Receiving u_obs_all from CaNS...")
        self.common_comm.Recv([self.u_obs_all, MPI.DOUBLE], source=1, tag=5)
        print(f"PYTHON: Received u_obs_all, shape={self.u_obs_all.shape}, range=[{np.min(self.u_obs_all):.6f}, {np.max(self.u_obs_all):.6f}]")
        
        print("PYTHON: Receiving w_obs_all from CaNS...")
        self.common_comm.Recv([self.w_obs_all, MPI.DOUBLE], source=1, tag=9)
        print(f"PYTHON: Received w_obs_all, shape={self.w_obs_all.shape}, range=[{np.min(self.w_obs_all):.6f}, {np.max(self.w_obs_all):.6f}]")
        
        print("PYTHON: Receiving dpdx from CaNS...")
        self.common_comm.Recv([self.dpdx, MPI.DOUBLE], source=1, tag=4)
        print(f"PYTHON: Received dpdx = {self.dpdx}")
        print("PYTHON: All CaNS data received successfully!")
        
        if self.save_images:
            self._save_raw_observations_images(self.u_obs_all, self.w_obs_all)
        
        # Process observations (unchanged)
        print("PYTHON: Processing observations...")
        u_mean = np.mean(self.u_obs_all)
        print(f"PYTHON: u_mean = {u_mean}")
        self.u_obs_mat = (self.u_obs_all - u_mean) / self.config['action']['om_max']
        print(f"PYTHON: u_obs_mat processed, range=[{np.min(self.u_obs_mat):.6f}, {np.max(self.u_obs_mat):.6f}]")
        
        w_mean = np.mean(self.w_obs_all)
        print(f"PYTHON: w_mean = {w_mean}")
        self.w_obs_mat = (self.w_obs_all - w_mean) / self.config['action']['om_max']
        print(f"PYTHON: w_obs_mat processed, range=[{np.min(self.w_obs_mat):.6f}, {np.max(self.w_obs_mat):.6f}]")
        print(f"PYTHON: om_max = {self.config['action']['om_max']}")
        
        # Store initial observation if needed
        if not self.initial_obs_captured and self.total_steps == 0:
            self.initial_u_obs_mat = self.u_obs_mat.copy()
            self.initial_w_obs_mat = self.w_obs_mat.copy()
            self.initial_obs_captured = True
            print("PYTHON: Initial observation captured for future resets")
        
        # Compute global reward
        print("PYTHON: Computing global reward...")
        global_reward = float(compute_reward(self.dpdx, self.config))
        print(f"PYTHON: Global reward = {global_reward}")
        
        # Process agents in parallel for observations
        print(f"PYTHON: Processing agents - using {self.num_workers} workers")
        if self.num_workers > 0:
            print("PYTHON: Starting parallel agent processing...")
            observations = self._process_agents_parallel(self.u_obs_mat, self.w_obs_mat)
            print("PYTHON: Parallel agent processing complete!")
        else:
            print("PYTHON: Starting sequential agent processing...")
            observations = self._process_agents_sequential(self.u_obs_mat, self.w_obs_mat)
            print("PYTHON: Sequential agent processing complete!")
        
        # Create other dictionaries
        rewards = {}
        terminations = {}
        truncations = {}
        infos = {}

        for agent in self.agents:
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
        self.step_counter += 1
        
        # Handle episode completion
        self._check_simulation_end()
        
        self.last_observations = observations
        self.last_rewards = rewards
        
        return observations, rewards, terminations, truncations, infos

    def _process_agents_parallel(self, u_obs_mat, w_obs_mat):
        """Process agent observations using parallel workers."""
        print(f"MAIN: Starting parallel processing for {self._num_agents} agents")
        
        # Send work signal to workers
        work_signal = np.array([1], dtype=np.int32)
        print("MAIN: Sending work signal to workers")
        self.intracomm_workers.Send(work_signal, dest=1, tag=300)
        
        # Send grid dimensions
        grid_dims = np.array([self.grid_i, self.grid_j], dtype=np.int32)
        print(f"MAIN: Sending grid dimensions {self.grid_i}x{self.grid_j}")
        self.intracomm_workers.Send(grid_dims, dest=1, tag=301)
        
        # Send observation data (keep as float64)
        u_obs_flat = u_obs_mat.flatten().astype(np.float64)
        w_obs_flat = w_obs_mat.flatten().astype(np.float64)
        print(f"MAIN: Sending observation data (u: {u_obs_flat.shape}, w: {w_obs_flat.shape})")
        self.intracomm_workers.Send(u_obs_flat, dest=1, tag=302)
        self.intracomm_workers.Send(w_obs_flat, dest=1, tag=303)
        
        # Send only om_max (halo assumed 0)
        om_max_arr = np.array([self.config['action']['om_max']], dtype=np.float64)
        print(f"MAIN: Sending om_max = {om_max_arr[0]}")
        self.intracomm_workers.Send(om_max_arr, dest=1, tag=304)
        
        print("MAIN: Waiting for results from workers...")
        # Receive results
        result_count = np.empty(1, dtype=np.int32)
        self.intracomm_workers.Recv(result_count, source=1, tag=350)
        print(f"MAIN: Received result count: {result_count[0]}")
        
        observations = {}
        if result_count[0] > 0:
            # For halo=0: observation size is 1x1x2 = 2
            obs_size = 2
            
            # Receive all observations
            print(f"MAIN: Receiving {result_count[0] * obs_size} observation values")
            all_obs_flat = np.empty(result_count[0] * obs_size, dtype=np.float32)
            self.intracomm_workers.Recv(all_obs_flat, source=1, tag=351)
            
            # Receive agent IDs
            print("MAIN: Receiving agent IDs")
            id_length = np.empty(1, dtype=np.int32)
            self.intracomm_workers.Recv(id_length, source=1, tag=352)
            agent_id_bytes = np.empty(id_length[0], dtype=np.uint8)
            self.intracomm_workers.Recv([agent_id_bytes, MPI.CHAR], source=1, tag=353)
            
            # Reconstruct observations
            print("MAIN: Reconstructing observations dictionary")
            agent_ids = agent_id_bytes.tobytes().decode('utf-8').split(',')
            all_obs = all_obs_flat.reshape(result_count[0], 1, 1, 2)  # halo=0: 1x1x2
            
            for i, agent_id in enumerate(agent_ids):
                observations[agent_id] = all_obs[i]
        
        print(f"MAIN: Parallel processing complete, got {len(observations)} observations")
        return observations

    def _process_agents_sequential(self, u_obs_mat, w_obs_mat):
        """Process agent observations sequentially (fallback method)."""
        observations = {}
        for agent in self.agents:
            i, j = map(int, agent.split("_")[1:])
            observations[agent] = self.get_local_observation(u_obs_mat, w_obs_mat, i, j)
        return observations

    def get_local_observation(self, u_obs, w_obs, i, j):
        """Extract local observation (same as original)."""
        om_max = self.config['action']['om_max']
    
        if self.halo == 0:
            local_obs = np.zeros((1, 1, 2), dtype=np.float32)
            local_obs[0, 0, 0] = u_obs[i, j] * om_max
            local_obs[0, 0, 1] = w_obs[i, j] * om_max
        else:
            size = 2 * self.halo + 1
            local_obs = np.zeros((size, size, 2), dtype=np.float32)
        
            for di in range(-self.halo, self.halo + 1):
                for dj in range(-self.halo, self.halo + 1):
                    obs_i = (i + di) % self.grid_i
                    obs_j = (j + dj) % self.grid_j
                    local_i = di + self.halo
                    local_j = dj + self.halo
                    local_obs[local_i, local_j, 0] = u_obs[obs_i, obs_j] * om_max
                    local_obs[local_i, local_j, 1] = w_obs[obs_i, obs_j] * om_max
                
        return local_obs

    def _save_actions_to_cans_image(self, amp_send):
        """Save the scaled actions that are sent to CaNS (same as original)."""
        try:
            plt.figure(figsize=(10, 8))
            im = plt.imshow(amp_send, cmap='RdBu_r', origin='lower')
            plt.colorbar(im, label='Action Value (scaled by om_max)')
            plt.title(f'Actions Sent to CaNS - Episode {self.episode_counter}, Step {self.step_counter}')
            plt.xlabel('j (grid points)')
            plt.ylabel('i (grid points)')
            
            mean_val = np.mean(amp_send)
            std_val = np.std(amp_send)
            min_val = np.min(amp_send)
            max_val = np.max(amp_send)
            plt.suptitle(f'Mean: {mean_val:.6f}, Std: {std_val:.6f}, Range: [{min_val:.6f}, {max_val:.6f}]', 
                        fontsize=10)
            
            filename = f"actions_to_cans_ep{self.episode_counter:03d}_step{self.step_counter:04d}.png"
            filepath = os.path.join(self.image_save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            if self.step_counter <= 3:
                print(f"Saved actions to CaNS: {filename}")
            
        except Exception as e:
            print(f"Error saving actions to CaNS image: {e}")

    def _save_raw_observations_images(self, u_obs_raw, w_obs_raw):
        """Save the raw u and w observations (same as original)."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            im1 = ax1.imshow(u_obs_raw, cmap='viridis', origin='lower')
            ax1.set_title(f'U Velocity Observations - Episode {self.episode_counter}, Step {self.step_counter}')
            ax1.set_xlabel('j (grid points)')
            ax1.set_ylabel('i (grid points)')
            cbar1 = plt.colorbar(im1, ax=ax1, label='U Velocity')
            
            u_mean = np.mean(u_obs_raw)
            u_std = np.std(u_obs_raw)
            u_min = np.min(u_obs_raw)
            u_max = np.max(u_obs_raw)
            ax1.text(0.02, 0.98, f'Mean: {u_mean:.6f}\nStd: {u_std:.6f}\nMin: {u_min:.6f}\nMax: {u_max:.6f}', 
                    transform=ax1.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            im2 = ax2.imshow(w_obs_raw, cmap='plasma', origin='lower')
            ax2.set_title(f'W Velocity Observations - Episode {self.episode_counter}, Step {self.step_counter}')
            ax2.set_xlabel('j (grid points)')
            ax2.set_ylabel('i (grid points)')
            cbar2 = plt.colorbar(im2, ax=ax2, label='W Velocity')
            
            w_mean = np.mean(w_obs_raw)
            w_std = np.std(w_obs_raw)
            w_min = np.min(w_obs_raw)
            w_max = np.max(w_obs_raw)
            ax2.text(0.02, 0.98, f'Mean: {w_mean:.6f}\nStd: {w_std:.6f}\nMin: {w_min:.6f}\nMax: {w_max:.6f}', 
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            
            filename = f"observations_from_cans_ep{self.episode_counter:03d}_step{self.step_counter:04d}.png"
            filepath = os.path.join(self.image_save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            if self.step_counter <= 3:
                print(f"Saved observations from CaNS: {filename}")
            
        except Exception as e:
            print(f"Error saving raw observations images: {e}")

    def _check_simulation_end(self):
        """Check simulation state (same as original)."""
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        
        if self.total_steps >= self.config['total_timesteps']:
            print("Python: Sending ENDED - simulation complete")
            self.common_comm.Bcast([b'ENDED', MPI.CHAR], root=0)
            self.common_comm.Free()
            self.sub_comm.Disconnect()
        elif self.current_step >= self.episode_length:
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
   
    def render(self):
        """Render the environment (same as original)."""
        if self.render_mode == "human":
            print(f"Step: {self.current_step}, dpdx: {self.dpdx:.6f}")
            return None
        elif self.render_mode == "rgb_array":
            u_normalized = (self.u_obs_mat / 255.0)
            w_normalized = (self.w_obs_mat / 255.0)
            
            r_channel = u_normalized
            g_channel = w_normalized
            b_channel = (u_normalized + w_normalized) / 2
            
            rgb_array = np.stack([r_channel, g_channel, b_channel], axis=2)
            return rgb_array
        
        return None
    
    def close(self):
        """Clean up resources including workers."""
        try:
            # Shutdown workers if they exist
            if self.num_workers > 0 and self.intracomm_workers is not None:
                shutdown_signal = np.array([-1], dtype=np.int32)
                self.intracomm_workers.Send(shutdown_signal, dest=1, tag=300)
                self.worker_comm.Disconnect()
                self.intracomm_workers.Free()
            
            # Shutdown CaNS
            self.common_comm.Bcast([b'ENDED', MPI.CHAR], root=0)
            self.common_comm.Free()
            self.sub_comm.Disconnect()
        except Exception as e:
            print(f"Error during environment cleanup: {e}")
    
    def seed(self, seed=None):
        """Set random seed for reproducibility."""
        if seed is not None:
            np.random.seed(seed)
        
    def observation_space(self, agent):
        """Return the observation space for a specific agent."""
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        """Return the action space for a specific agent."""
        return self.action_spaces[agent]
    
    def set_episode_length(self, episode_length):
        """Set the episode length for curriculum learning."""
        self.episode_length = episode_length


# Main entry point for worker processes
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        agent_processing_worker()