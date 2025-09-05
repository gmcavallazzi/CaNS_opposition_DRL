#!/usr/bin/env python3
import numpy as np
import time
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print("=== PERSISTENT TEST: Python controller with pre-allocated workers ===")
        
        # Data to send to workers (larger for multi-node test)
        data = np.full(500000, 3.14, dtype=np.float64)
        result = np.zeros_like(data)
        
        start_time = time.time()
        
        # Send data to all processes (workers will receive it)
        comm.Bcast(data, root=0)
        
        # Tell workers to start computation (send a "go" signal)
        for worker_rank in range(1, size):
            comm.send("compute", dest=worker_rank, tag=1)
        
        # Receive results from workers
        for worker_rank in range(1, size):
            worker_result = comm.recv(source=worker_rank, tag=2)
            result += worker_result
        
        end_time = time.time()
        
        print(f"Result sum: {np.sum(result):.2f}")
        print(f"Persistent total time: {(end_time - start_time)*1000:.3f} ms")
    
    else:
        # Worker process (Python ranks 1-64)
        # Receive data from controller
        data = np.empty(500000, dtype=np.float64)
        comm.Bcast(data, root=0)
        
        # Wait for compute signal
        signal = comm.recv(source=0, tag=1)
        
        # Do computation (simulating what Fortran workers would do)
        local_result = data * np.sin(data) * np.cos(data)
        
        # Send result back to controller  
        comm.send(local_result, dest=0, tag=2)

if __name__ == "__main__":
    main()