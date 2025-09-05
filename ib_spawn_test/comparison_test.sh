#!/bin/bash
#SBATCH -D /users/addh496/sharedscratch/CaNS_DRL2.4/pz_guastoni0_highdim/ib_spawn_test
#SBATCH -J python_fortran_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
#SBATCH --time=10:00
#SBATCH --exclusive
#SBATCH --output=comparison_%j.out
#SBATCH --partition=nodes

module load compilers/gcc/11.2.0
module load mpi/openmpi/4.1.1

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

echo "=== PYTHON CONTROLLER + FORTRAN WORKERS COMPARISON ==="
echo ""

cd $SLURM_SUBMIT_DIR
make clean && make

# Create hostfile
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile.tmp
while read node; do
    echo "$node slots=48"
done < hostfile.tmp > hostfile
rm hostfile.tmp

echo "--- TEST 1: SPAWN METHOD (Python spawns Fortran workers) ---"
mpirun \
  --mca btl tcp,self \
  --mca btl_tcp_if_include ib0 \
  --mca btl_base_warn_component_unused 0 \
  --mca btl_base_verbose 0 \
  --hostfile hostfile \
  --bind-to core \
  -n 1 \
  python python_spawn_test.py 2>&1 | grep -v -E "(mca:|btl:|select:|bml:)"

echo ""
echo "--- TEST 2: PERSISTENT METHOD (Python + Python workers) ---"
mpirun \
  --mca btl tcp,self \
  --mca btl_tcp_if_include ib0 \
  --mca btl_base_warn_component_unused 0 \
  --mca btl_base_verbose 0 \
  --hostfile hostfile \
  --bind-to core \
  -n 9 \
  python python_persistent_test.py 2>&1 | grep -v -E "(mca:|btl:|select:|bml:)"

# Clean up
rm -f hostfile

echo ""
echo "=== COMPARISON COMPLETE ==="