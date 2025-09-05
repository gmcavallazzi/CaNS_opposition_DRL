program simple_child
    use mpi_f08
    implicit none
    
    type(MPI_Comm) :: parent_comm
    integer :: rank, size, ierr
    integer, parameter :: array_size = 100000
    real(8), allocatable :: data(:), local_result(:)
    integer :: i
    
    call MPI_Init(ierr)
    call MPI_Comm_get_parent(parent_comm, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    
    allocate(data(array_size), local_result(array_size))
    
    ! Receive data from parent
    call MPI_Bcast(data, array_size, MPI_DOUBLE_PRECISION, 0, parent_comm, ierr)
    
    ! Do intensive computation
    do i = 1, array_size
        local_result(i) = data(i) * sin(data(i)) * cos(data(i)) * exp(data(i) * 0.001_8)
    enddo
    
    ! Collective operation among workers first
    call MPI_Allreduce(MPI_IN_PLACE, local_result, array_size, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    
    ! Send result back to parent
    call MPI_Reduce(local_result, local_result, array_size, MPI_DOUBLE_PRECISION, MPI_SUM, 0, parent_comm, ierr)
    
    deallocate(data, local_result)
    call MPI_Finalize(ierr)
end program