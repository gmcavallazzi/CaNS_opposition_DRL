#!/bin/bash
#SBATCH -D /users/addh496/sharedscratch/CaNS_DRL2.4/pz_guastoni0_highdim/ib_spawn_test
#SBATCH -J simple_comparison
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=10:00
#SBATCH --exclusive
#SBATCH --output=simple_%j.out
#SBATCH --partition=nodes

module load compilers/gcc/11.2.0
module load mpi/openmpi/4.1.1

echo "=== SIMPLE SPAWN vs PERSISTENT COMPARISON ==="
echo "Controller tells 8 workers to do intensive collective computation"
echo ""

cd $SLURM_SUBMIT_DIR
make clean && make

# Create hostfile
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.tmp
while read node; do
    echo "$node slots=48"
done < hostfile.tmp > hostfile
rm hostfile.tmp

echo "--- TEST 1: SPAWN METHOD ---"
mpirun \
  --mca btl tcp,self \
  --mca btl_tcp_if_include ib0 \
  --mca btl_base_warn_component_unused 0 \
  --hostfile hostfile \
  --bind-to core \
  -n 1 \
  ./simple_spawn

echo ""
echo "--- TEST 2: PERSISTENT METHOD ---"
mpirun \
  --mca btl tcp,self \
  --mca btl_tcp_if_include ib0 \
  --mca btl_base_warn_component_unused 0 \
  --hostfile hostfile \
  --bind-to core \
  -n 9 \
  ./simple_persistent

# Clean up
rm -f hostfile

echo ""
echo "=== COMPARISON COMPLETE ==="