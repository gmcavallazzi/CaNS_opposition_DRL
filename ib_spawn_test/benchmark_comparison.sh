#!/bin/bash
#SBATCH -D /users/addh496/sharedscratch/CaNS_DRL2.4/pz_guastoni0_highdim/ib_spawn_test
#SBATCH -J benchmark_spawn_vs_persistent
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=16
#SBATCH --time=30:00
#SBATCH --exclusive
#SBATCH --output=benchmark_%j.out
#SBATCH --partition=nodes

# Load environment
module load compilers/gcc/11.2.0
module load mpi/openmpi/4.1.1

echo "=== SPAWN vs PERSISTENT WORKERS BENCHMARK ==="
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Total processes available: $SLURM_NTASKS"
echo "Testing with different core counts and matrix sizes"
echo ""

cd $SLURM_SUBMIT_DIR

# Build all tests
echo "Building tests..."
make clean && make
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "         PERFORMANCE COMPARISON"
echo "=========================================="

# Test configurations: [total_procs, matrix_size, episodes]
declare -a configs=(
    "8 512 3"
    "16 1024 3"  
    "32 1024 3"
)

for config in "${configs[@]}"; do
    read -r total_procs matrix_size episodes <<< "$config"
    child_procs=$((total_procs - 1))  # 1 parent + N children for spawn test
    
    echo ""
    echo "===========================================" 
    echo "Configuration: ${total_procs} cores, ${matrix_size}x${matrix_size} matrix, ${episodes} episodes"
    echo "==========================================="
    
    # Test 1: Spawn-based approach (using your working settings)
    echo ""
    echo "--- TEST 1: SPAWN-BASED (current approach) ---"
    echo "Parent procs: 1, Child procs: ${child_procs}"
    
    # Create hostfile for spawning (same as your working setup)
    scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.tmp
    while read node; do
        echo "$node slots=16"
    done < hostfile.tmp > hostfile
    rm hostfile.tmp
    
    timeout 600s mpirun \
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
      ./intensive_parent 2>&1 | tee spawn_${total_procs}_${matrix_size}.log
    
    if [ ${PIPESTATUS[0]} -eq 124 ]; then
        echo "SPAWN TEST TIMED OUT (>10 minutes)"
    fi
    
    # Test 2: Persistent worker approach (also with same settings for fairness)
    echo ""
    echo "--- TEST 2: PERSISTENT WORKERS (proposed approach) ---" 
    echo "Controller procs: 1, Worker procs: ${child_procs}"
    
    timeout 600s mpirun \
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
      -n ${total_procs} \
      -x LD_LIBRARY_PATH \
      ./persistent_workers 2>&1 | tee persistent_${total_procs}_${matrix_size}.log
    
    if [ ${PIPESTATUS[0]} -eq 124 ]; then
        echo "PERSISTENT TEST TIMED OUT (>10 minutes)"
    fi
    
    # Extract and compare results
    echo ""
    echo "--- RESULTS COMPARISON ---"
    spawn_time=$(grep "Average per episode" spawn_${total_procs}_${matrix_size}.log | awk '{print $4}' || echo "FAILED")
    spawn_throughput=$(grep "Throughput:" spawn_${total_procs}_${matrix_size}.log | awk '{print $2}' || echo "FAILED")
    
    persistent_time=$(grep "Average per episode" persistent_${total_procs}_${matrix_size}.log | awk '{print $4}' || echo "FAILED")
    persistent_throughput=$(grep "Throughput:" persistent_${total_procs}_${matrix_size}.log | awk '{print $2}' || echo "FAILED")
    
    echo "SPAWN:      ${spawn_time} ms/episode,      ${spawn_throughput} episodes/sec"
    echo "PERSISTENT: ${persistent_time} ms/episode, ${persistent_throughput} episodes/sec"
    
    if [[ "$spawn_time" != "FAILED" && "$persistent_time" != "FAILED" ]]; then
        speedup=$(echo "scale=2; $spawn_time / $persistent_time" | bc -l 2>/dev/null || echo "N/A")
        echo "SPEEDUP: ${speedup}x faster with persistent workers"
    fi
    
    echo "==========================================="
done

echo ""
echo "=== BENCHMARK COMPLETED ==="
echo ""
# Clean up hostfile
rm -f hostfile

echo "Summary logs created:"
ls -la spawn_*.log persistent_*.log 2>/dev/null || echo "No detailed logs found"

echo ""
echo "Key findings:"
echo "1. Check if persistent workers show consistent speedup"
echo "2. Look for any initialization/spawn overhead differences"  
echo "3. Verify computational results are equivalent"
echo ""
echo "If persistent workers are faster, this approach should work"
echo "for your CaNS+DRL integration without MPI spawning issues."