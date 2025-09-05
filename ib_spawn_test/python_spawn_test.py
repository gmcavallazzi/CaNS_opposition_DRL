#!/usr/bin/env python3
import numpy as np
import time
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    print("=== SPAWN TEST: Python controller spawns Fortran workers ===")
    
    # Data to send to workers
    data = np.full(50000, 3.14, dtype=np.float64)
    result = np.zeros_like(data)
    
    start_time = time.time()
    
    # Spawn 8 Fortran workers
    worker_comm = MPI.COMM_WORLD.Spawn('./fortran_worker', args=[], maxprocs=8)
    
    # Send data to workers
    worker_comm.Bcast(data, root=MPI.ROOT)
    
    # Receive result from workers
    worker_comm.Reduce(None, result, op=MPI.SUM, root=MPI.ROOT)
    
    # Clean up
    worker_comm.Disconnect()
    
    end_time = time.time()
    
    print(f"Result sum: {np.sum(result):.2f}")
    print(f"Spawn total time: {(end_time - start_time)*1000:.3f} ms")

if __name__ == "__main__":
    main()