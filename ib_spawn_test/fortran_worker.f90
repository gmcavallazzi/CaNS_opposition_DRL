program fortran_worker
    use mpi_f08
    implicit none
    
    type(MPI_Comm) :: parent_comm
    integer :: rank, size, ierr
    integer, parameter :: array_size = 50000
    real(8), allocatable :: data(:), local_result(:)
    integer :: i
    
    call MPI_Init(ierr)
    call MPI_Comm_get_parent(parent_comm, ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    
    allocate(data(array_size), local_result(array_size))
    
    ! Receive data from Python parent
    call MPI_Bcast(data, array_size, MPI_DOUBLE_PRECISION, 0, parent_comm, ierr)
    
    ! Do intensive computation
    do i = 1, array_size
        local_result(i) = data(i) * sin(data(i)) * cos(data(i))
    enddo
    
    ! Collective among Fortran workers only
    call MPI_Allreduce(MPI_IN_PLACE, local_result, array_size, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    
    ! Only rank 0 sends result back to Python parent
    if (rank == 0) then
        call MPI_Send(local_result, array_size, MPI_DOUBLE_PRECISION, 0, 99, parent_comm, ierr)
    endif
    
    deallocate(data, local_result)
    call MPI_Finalize(ierr)
end program