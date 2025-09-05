program simple_spawn
    use mpi_f08
    implicit none
    
    type(MPI_Comm) :: intercomm
    integer :: rank, size, ierr
    integer, parameter :: array_size = 100000
    real(8), allocatable :: data(:), result(:)
    real(8) :: start_time, end_time
    character(len=20) :: child_exe = './simple_child'
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    
    allocate(data(array_size), result(array_size))
    
    if (rank == 0) then
        write(*,'(A)') '=== SPAWN TEST: Controller + workers do collective operation ==='
    endif
    
    ! Initialize data
    data = 3.14_8
    result = 0.0_8
    
    start_time = MPI_Wtime()
    
    ! Spawn 8 workers
    call MPI_Comm_spawn(child_exe, MPI_ARGV_NULL, 8, &
                       MPI_INFO_NULL, 0, MPI_COMM_WORLD, &
                       intercomm, MPI_ERRCODES_IGNORE, ierr)
    
    ! Send work to workers
    call MPI_Bcast(data, array_size, MPI_DOUBLE_PRECISION, MPI_ROOT, intercomm, ierr)
    
    ! Workers do collective computation and send result back
    call MPI_Reduce(result, result, array_size, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_ROOT, intercomm, ierr)
    
    call MPI_Comm_disconnect(intercomm, ierr)
    
    end_time = MPI_Wtime()
    
    if (rank == 0) then
        write(*,'(A,F10.2)') 'Result sum: ', sum(result)
        write(*,'(A,F8.3,A)') 'Spawn total time: ', (end_time-start_time)*1000, ' ms'
    endif
    
    deallocate(data, result)
    call MPI_Finalize(ierr)
end program