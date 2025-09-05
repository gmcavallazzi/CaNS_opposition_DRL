#!/bin/bash
#SBATCH -D /users/addh496/sharedscratch/CaNS_DRL2.4/pz_guastoni0_highdim/ib_spawn_test
#SBATCH -J simple_spawn_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=10:00
#SBATCH --exclusive
#SBATCH --output=simple_%j.out
#SBATCH --partition=nodes

module load compilers/gcc/11.2.0
module load mpi/openmpi/4.1.1

echo "=== SIMPLE SPAWN vs PERSISTENT TEST ==="
echo "Using 1 node, 9 processes (1 controller + 8 workers)"
echo ""

cd $SLURM_SUBMIT_DIR
make clean && make

# Create simple hostfile
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.tmp
while read node; do
    echo "$node slots=48"
done < hostfile.tmp > hostfile
rm hostfile.tmp

echo "--- TEST 1: PERSISTENT WORKERS ---"
timeout 300s mpirun \
  --mca btl tcp,self \
  --mca btl_tcp_if_include ib0 \
  --mca btl_base_warn_component_unused 0 \
  --bind-to core \
  -n 9 \
  ./persistent_workers

echo ""
echo "--- TEST 2: SPAWN-BASED (if persistent works) ---"
echo "Skipping spawn test for now - focus on persistent worker performance"

# Clean up
rm -f hostfile

echo ""
echo "=== TEST COMPLETED ==="