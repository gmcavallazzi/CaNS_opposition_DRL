#!/bin/bash
#SBATCH -D /users/addh496/sharedscratch/CaNS_DRL2.4/pz_guastoni0_highdim/run0_550_opt
#SBATCH -J pz550btl
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=48
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
  --mca btl tcp,vader,self \
  --mca btl_tcp_if_include ib0 \
  --mca oob_tcp_if_include ib0 \
  --mca pml ob1 \
  --mca btl_base_warn_component_unused 0 \
  --mca coll_tuned_use_dynamic_rules 1 \
  --mca coll_tuned_barrier_algorithm 1 \
  --mca coll_tuned_bcast_algorithm 1 \
  --mca coll_tuned_reduce_algorithm 1 \
  --hostfile hostfile \
  --map-by node \
  --bind-to core \
  -n 1 \
  -x LD_LIBRARY_PATH \
  -x PYTHONPATH \
  python evaluate_custom_grid.py ./checkpoints_pettingzoo_grid_shared/best_model.pt --grid_i 576 --grid_j 576 --episodes 5 --no_save --num_workers 32
  #python stw_utils_pettingzoo.py evaluate --config config.yaml --num_episodes 5 --policy_path ./checkpoints_pettingzoo_grid_shared/best_model.pt
  #python stwStart_pettingzoo.py 
  #python stw_utils_pettingzoo.py evaluate --config config.yaml --num_episodes 1 --policy_path ./checkpoints_pettingzoo_grid_shared/checkpoint_step_43200.pt
  #python debug.py
# Clean up
rm -f hostfile