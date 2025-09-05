#!/usr/bin/env python3
import numpy as np
from mpi4py import MPI
import sys

def python_worker_main():
    """Main function for spawned Python workers"""
    print("PY_WORKER: Starting")
    
    # Get parent communicator (connection to Python1)
    parentcomm = MPI.Comm.Get_parent()
    print("PY_WORKER: Got parent communicator")
    
    # Get rank and size in MPI_COMM_WORLD (Python workers only)
    myid = MPI.COMM_WORLD.Get_rank()
    mysize = MPI.COMM_WORLD.Get_size()
    print(f"PY_WORKER: rank {myid} of {mysize}")
    
    # Create pythoncomm (Python workers only, like canscomm)
    pythoncomm = MPI.COMM_WORLD.Dup()
    print(f"PY_WORKER {myid}: pythoncomm created")
    
    # Merge with parent to create intracomm_python (Python1 + Python workers)
    intracomm_python = parentcomm.Merge(high=True)  # Workers get higher ranks
    ourid = intracomm_python.Get_rank()
    oursize = intracomm_python.Get_size()
    print(f"PY_WORKER {myid}: intracomm_python rank {ourid}, size {oursize}")
    
    # Wait for data from Python1 via intracomm_python (only rank 1 receives)
    data = np.empty(2, dtype=np.float64)
    
    # Wait for Python1 to send data (no barriers needed)
    
    if ourid == 1:  # First Python worker in intracomm_python
        print(f"PY_WORKER {myid}: receiving from Python1")
        intracomm_python.Recv(data, source=0, tag=200)
        print(f"PY_WORKER {myid}: received {data}")
    
    # Broadcast data to all Python workers via pythoncomm
    pythoncomm.Bcast(data, root=0)
    print(f"PY_WORKER {myid}: broadcasted data = {data}")
    
    # Perform collective operation in pythoncomm
    local_sum = np.sum(data) * (myid + 1)  # Each worker contributes differently
    total_result = pythoncomm.allreduce(local_sum, op=MPI.SUM)
    if myid == 0:
        print(f"PY_WORKER: collective result in pythoncomm = {total_result}")
    
    # Send result back to Python1 (only rank 0 in pythoncomm sends)
    if myid == 0:
        result_array = np.array([total_result])
        print(f"PY_WORKER {myid}: sending result {total_result} to Python1")
        intracomm_python.Send(result_array, dest=0, tag=250)
        print(f"PY_WORKER {myid}: result sent")
    
    
    # Cleanup
    pythoncomm.Free()
    intracomm_python.Free()
    parentcomm.Disconnect()
    print(f"PY_WORKER {myid}: finished")

def main():
    # Check if this is a worker process by looking at command line args
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        python_worker_main()
        return
    
    print("PYTHON1: Starting orchestrator")
    
    # Python1 is rank 0 in MPI_COMM_WORLD
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    print(f"PYTHON1: rank={rank}, size={size}")
    
    # ===== FORTRAN SETUP =====
    print("PYTHON1: Spawning 2 Fortran workers")
    fortran_comm = MPI.COMM_SELF.Spawn('./fortran_worker', maxprocs=2)
    print("PYTHON1: Fortran workers spawned")
    
    # Create intracomm_fortran (Python1 + Fortran workers)
    intracomm_fortran = fortran_comm.Merge(high=False)  # Python1 gets rank 0
    print(f"PYTHON1: intracomm_fortran created - size={intracomm_fortran.Get_size()}")
    
    # ===== PYTHON WORKERS SETUP =====
    print("PYTHON1: Spawning 2 Python workers")
    python_comm = MPI.COMM_SELF.Spawn('python', args=['main.py', 'worker'], maxprocs=2)
    print("PYTHON1: Python workers spawned")
    
    # Create intracomm_python (Python1 + Python workers)
    intracomm_python = python_comm.Merge(high=False)  # Python1 gets rank 0
    print(f"PYTHON1: intracomm_python created - size={intracomm_python.Get_size()}")
    
    
    # ===== FORTRAN WORKFLOW =====
    print("PYTHON1: Starting Fortran workflow")
    
    # Send data to Fortran via intracomm_fortran
    fortran_data = np.array([10.0, 20.0])
    print(f"PYTHON1: Sending {fortran_data} to Fortran")
    intracomm_fortran.Send(fortran_data, dest=1, tag=100)  # Send to first Fortran worker
    
    # Receive result from Fortran
    fortran_result = np.empty(1, dtype=np.float64)
    print("PYTHON1: Waiting for Fortran result")
    intracomm_fortran.Recv(fortran_result, source=1, tag=150)
    print(f"PYTHON1: Received Fortran result: {fortran_result[0]}")
    
    # ===== PYTHON WORKERS WORKFLOW =====
    print("PYTHON1: Starting Python workers workflow")
    
    # Send Fortran result to Python workers via intracomm_python
    python_data = np.array([fortran_result[0], fortran_result[0]*2])
    print(f"PYTHON1: Sending {python_data} to Python workers")
    intracomm_python.Send(python_data, dest=1, tag=200)  # Send to first Python worker
    
    # Receive result from Python workers
    python_result = np.empty(1, dtype=np.float64)
    print("PYTHON1: Waiting for Python workers result")
    intracomm_python.Recv(python_result, source=1, tag=250)
    print(f"PYTHON1: Received Python workers result: {python_result[0]}")
    
    # ===== CLEANUP =====
    print("PYTHON1: Final result processing complete")
    print(f"PYTHON1: Fortran collective result: {fortran_result[0]}")
    print(f"PYTHON1: Python collective result: {python_result[0]}")
    
    print("PYTHON1: Cleaning up")
    fortran_comm.Disconnect()
    intracomm_fortran.Free()
    python_comm.Disconnect()
    intracomm_python.Free()
    print("PYTHON1: Finished")

if __name__ == "__main__":
    main()