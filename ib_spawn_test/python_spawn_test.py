#!/usr/bin/env python3
import numpy as np
import time
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    print("=== SPAWN TEST: Python controller spawns Fortran workers ===")
    print("Python: Starting spawn test...")
    
    # Data to send to workers (larger for multi-node test)
    data = np.full(500000, 3.14, dtype=np.float64)
    result = np.zeros_like(data)
    
    start_time = time.time()
    
    print("Python: About to spawn 64 Fortran workers...")
    # Spawn 64 Fortran workers (should distribute across 2 nodes)
    worker_comm = MPI.COMM_WORLD.Spawn('./fortran_worker', args=[], maxprocs=64)
    print("Python: Workers spawned successfully")
    
    print("Python: Broadcasting data to workers...")
    # Send data to workers
    worker_comm.Bcast(data, root=MPI.ROOT)
    print("Python: Data broadcast complete")
    
    print("Python: Waiting for result from worker rank 0...")
    # Receive result from rank 0 of the workers
    worker_comm.Recv(result, source=0, tag=99)
    print("Python: Result received")
    
    end_time = time.time()
    
    print("Python: Skipping disconnect - let MPI auto-cleanup")
    # Skip explicit disconnect - let MPI_Finalize handle it
    
    print(f"Result sum: {np.sum(result):.2f}")
    print(f"Spawn total time: {(end_time - start_time)*1000:.3f} ms")
    
    # Small delay after printing results to let workers exit cleanly
    time.sleep(0.1)

if __name__ == "__main__":
    main()