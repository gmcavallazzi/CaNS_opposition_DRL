program test_spawn_child
    use mpi_f08
    implicit none
    
    type(MPI_Comm) :: parent_comm
    integer :: rank, size, ierr, msg_size
    real(8), allocatable :: recv_data(:), send_data(:)
    integer :: parent_rank, parent_size
    
    call MPI_Init(ierr)
    call MPI_Comm_get_parent(parent_comm, ierr)
    
    if (parent_comm == MPI_COMM_NULL) then
        write(*,*) 'Child: No parent communicator found'
        stop
    endif
    
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    call MPI_Comm_remote_size(parent_comm, parent_size, ierr)
    
    write(*,'(A,I0,A,I0,A,I0)') 'Child: rank ', rank, ' of ', size, ', parent_size=', parent_size
    
    ! Test different message sizes (same as parent)
    integer, parameter :: num_tests = 4
    integer, parameter :: test_sizes(4) = [1024, 8192, 32768, 131072]
    integer :: i
    
    do i = 1, num_tests
        msg_size = test_sizes(i)
        allocate(recv_data(msg_size), send_data(msg_size))
        
        ! Receive data from parent
        call MPI_Recv(recv_data, msg_size, MPI_DOUBLE_PRECISION, rank, 100, parent_comm, MPI_STATUS_IGNORE, ierr)
        
        ! Process data (multiply by 2)
        send_data = recv_data * 2.0_8
        
        ! Send response back
        call MPI_Send(send_data, msg_size, MPI_DOUBLE_PRECISION, rank, 200, parent_comm, ierr)
        
        ! Barrier to sync timing
        call MPI_Barrier(parent_comm, ierr)
        
        deallocate(recv_data, send_data)
    enddo
    
    call MPI_Finalize(ierr)
end program