program simple_persistent
    use mpi_f08
    implicit none
    
    integer :: rank, size, ierr
    integer, parameter :: array_size = 100000
    real(8), allocatable :: data(:), result(:), local_result(:)
    real(8) :: start_time, end_time
    logical :: is_controller
    integer :: i
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    
    is_controller = (rank == 0)
    allocate(data(array_size), result(array_size), local_result(array_size))
    
    if (is_controller) then
        write(*,'(A)') '=== PERSISTENT TEST: Controller + workers do collective operation ==='
        
        ! Initialize data
        data = 3.14_8
        result = 0.0_8
        
        start_time = MPI_Wtime()
    endif
    
    ! Broadcast data to all workers.
    call MPI_Bcast(data, array_size, MPI_DOUBLE_PRECISION, 0, MPI_COMM_WORLD, ierr)
    
    if (.not. is_controller) then
        ! Workers do intensive computation
        do i = 1, array_size
            local_result(i) = data(i) * sin(data(i)) * cos(data(i)) * exp(data(i) * 0.001_8)
        enddo
    else
        local_result = 0.0_8
    endif
    
    ! Collective operation among all processes
    call MPI_Allreduce(local_result, result, array_size, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    
    if (is_controller) then
        end_time = MPI_Wtime()
        write(*,'(A,F10.2)') 'Result sum: ', sum(result)
        write(*,'(A,F8.3,A)') 'Persistent total time: ', (end_time-start_time)*1000, ' ms'
    endif
    
    deallocate(data, result, local_result)
    call MPI_Finalize(ierr)
end program
