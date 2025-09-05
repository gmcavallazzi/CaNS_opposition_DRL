#!/bin/bash
#SBATCH -D /users/addh496/sharedscratch/CaNS_DRL2.4/pz_guastoni0_highdim/run0
#SBATCH -J pz_run0
#SBATCH --nodes=4
#!SBATCH --ntasks=64
#SBATCH --ntasks-per-node=33
#SBATCH --time=72:00:00
#SBATCH --exclusive
#SBATCH --output=R-%x.%j.out
#SBATCH --partition=nodes

# Load environment setup
flight env activate gridware
module load compilers/gcc/11.2.0
module load mpi/openmpi/4.1.1
module load fftw/3.3.10
#
# Conda setup
__conda_setup="$('/users/addh496/sharedscratch/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/users/addh496/sharedscratch/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/users/addh496/sharedscratch/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/users/addh496/sharedscratch/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda activate stw_pettingzoo
export PYTHONPATH=$(python -c "import sys; print(':'.join(sys.path))")

# OpenMP settings
export OMP_NUM_THREADS=1

# UCX-specific settings (optimized for mlx5_0)
export UCX_TLS=rc_mlx5,dc_mlx5,ud_mlx5,sm,self
export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_TRAFFIC_CLASS=105
export UCX_IB_GID_INDEX=3
export UCX_IB_SL=3
export UCX_RNDV_THRESH=16384
export UCX_MAX_RNDV_RAILS=1
export UCX_MEMTYPE_CACHE=n

# OpenMPI settings for spawning
export OMPI_MCA_rmaps_base_mapping_policy=node:PE=1
export OMPI_MCA_rmaps_base_oversubscribe=1
export OMPI_MCA_orte_allowed_exit_without_sync=1
export OMPI_MCA_mpi_show_handle_leaks=1

# Create hostfile
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.tmp
while read node; do
    echo "$node slots=48"
done < hostfile.tmp > hostfile
rm hostfile.tmp

echo "Generated hostfile contains:"
cat hostfile

# Print debug information
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_JOB_NODELIST"

#python mpi_cleanup.py

mpirun \
  --verbose \
  --mca plm_rsh_agent srun \
  --mca pml ucx \
  --mca btl ^vader,tcp,openib,uct \
  --hostfile hostfile \
  --map-by node \
  --bind-to core \
  -n 1 \
  -x LD_LIBRARY_PATH \
  -x UCX_TLS \
  -x UCX_NET_DEVICES \
  -x UCX_IB_TRAFFIC_CLASS \
  -x UCX_IB_GID_INDEX \
  -x UCX_IB_SL \
  -x UCX_RNDV_THRESH \
  -x UCX_MAX_RNDV_RAILS \
  -x UCX_MEMTYPE_CACHE \
  -x PYTHONPATH \
  python evaluate_custom_grid.py ./checkpoints_pettingzoo_grid_shared/best_model.pt --grid_i 192 --grid_j 192 --episodes 5 --no_save
  #python stw_utils_pettingzoo.py evaluate --config config.yaml --num_episodes 5 --policy_path ./checkpoints_pettingzoo_grid_shared/best_model.pt
  #python stwStart_pettingzoo.py 
  #python stw_utils_pettingzoo.py evaluate --config config.yaml --num_episodes 1 --policy_path ./checkpoints_pettingzoo_grid_shared/checkpoint_step_43200.pt
  #python debug.py
# Clean up
rm -f hostfile