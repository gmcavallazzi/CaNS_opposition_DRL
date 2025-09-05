#!/bin/bash
#SBATCH -D /users/addh496/sharedscratch/CaNS_DRL2.4/pz_guastoni0_highdim/ib_spawn_test
#SBATCH -J ib_test_openib
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --time=10:00
#SBATCH --exclusive
#SBATCH --output=test_openib.%j.out
#SBATCH --partition=nodes

# Load environment
module load compilers/gcc/11.2.0
module load mpi/openmpi/4.1.1

echo "=== Testing Native OpenIB BTL with Spawning ==="
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: 2"

cd $SLURM_SUBMIT_DIR

# Build the test
make clean && make

if [ ! -f parent ] || [ ! -f child ]; then
    echo "Build failed!"
    exit 1
fi

# Test 1: Native OpenIB
echo "=== Test 1: Native OpenIB BTL ==="
mpirun \
  --mca btl openib,vader,self \
  --mca btl_openib_use_eager_rdma 1 \
  --mca btl_openib_eager_limit 32768 \
  --mca pml ob1 \
  --bind-to core \
  -n 4 \
  ./parent 2>&1 | tee openib_results.log

echo ""
echo "=== Test 1 Results (check for errors above) ==="
echo ""

# Test 2: TCP over IB (fallback)
echo "=== Test 2: TCP over IB (fallback) ==="
mpirun \
  --mca btl tcp,vader,self \
  --mca btl_tcp_if_include ib0 \
  --mca pml ob1 \
  --bind-to core \
  -n 4 \
  ./parent 2>&1 | tee tcp_ib_results.log

echo ""
echo "=== Performance Comparison ==="
echo "OpenIB results:"
grep "Bandwidth:" openib_results.log | head -4
echo ""
echo "TCP/IB results:"  
grep "Bandwidth:" tcp_ib_results.log | head -4

echo ""
echo "=== Test completed ==="