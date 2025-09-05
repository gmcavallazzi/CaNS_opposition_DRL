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
    
    write(*,'(A,I0,A,I0)') 'Fortran worker ', rank, ' of ', size, ' started'
    
    allocate(data(array_size), local_result(array_size))
    
    write(*,'(A,I0,A)') 'Fortran worker ', rank, ': Waiting for data from Python...'
    ! Receive data from Python parent
    call MPI_Bcast(data, array_size, MPI_DOUBLE_PRECISION, 0, parent_comm, ierr)
    write(*,'(A,I0,A)') 'Fortran worker ', rank, ': Data received, starting computation...'
    
    ! Do intensive computation
    do i = 1, array_size
        local_result(i) = data(i) * sin(data(i)) * cos(data(i))
    enddo
    
    write(*,'(A,I0,A)') 'Fortran worker ', rank, ': Computation done, doing collective...'
    ! Collective among Fortran workers only
    call MPI_Allreduce(MPI_IN_PLACE, local_result, array_size, MPI_DOUBLE_PRECISION, MPI_SUM, MPI_COMM_WORLD, ierr)
    write(*,'(A,I0,A)') 'Fortran worker ', rank, ': Collective done'
    
    ! Only rank 0 sends result back to Python parent
    if (rank == 0) then
        write(*,'(A)') 'Fortran worker 0: Sending result back to Python...'
        call MPI_Send(local_result, array_size, MPI_DOUBLE_PRECISION, 0, 99, parent_comm, ierr)
        write(*,'(A)') 'Fortran worker 0: Result sent'
    endif
    
    write(*,'(A,I0,A)') 'Fortran worker ', rank, ': Exiting'
    
    deallocate(data, local_result)
    call MPI_Finalize(ierr)
end program