program test_spawn_parent
    use mpi_f08
    implicit none
    
    type(MPI_Comm) :: intercomm, intracomm
    integer :: rank, size, ierr, i, msg_size
    integer :: remote_size, remote_rank
    real(8), allocatable :: send_data(:), recv_data(:)
    real(8) :: start_time, end_time, bandwidth
    character(len=20) :: child_exe = './child'
    character(len=1024) :: info_msg
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    
    write(*,'(A,I0,A,I0)') 'Parent: rank ', rank, ' of ', size
    
    ! Test different message sizes (in doubles)
    integer, parameter :: num_tests = 4
    integer, parameter :: test_sizes(num_tests) = [1024, 8192, 32768, 131072]
    
    do i = 1, num_tests
        msg_size = test_sizes(i)
        allocate(send_data(msg_size), recv_data(msg_size))
        
        ! Initialize data
        send_data = real(rank + 1, 8) * 1.5_8
        recv_data = 0.0_8
        
        write(*,'(A,I0,A)') 'Testing with ', msg_size*8, ' bytes...'
        
        ! Spawn child processes
        call MPI_Comm_spawn(child_exe, MPI_ARGV_NULL, size, &
                           MPI_INFO_NULL, 0, MPI_COMM_WORLD, &
                           intercomm, MPI_ERRCODES_IGNORE, ierr)
        
        if (ierr /= MPI_SUCCESS) then
            write(*,*) 'Failed to spawn children'
            stop
        endif
        
        ! Test collective communication performance
        start_time = MPI_Wtime()
        
        ! Send data to children
        call MPI_Send(send_data, msg_size, MPI_DOUBLE_PRECISION, rank, 100, intercomm, ierr)
        
        ! Receive response from children  
        call MPI_Recv(recv_data, msg_size, MPI_DOUBLE_PRECISION, rank, 200, intercomm, MPI_STATUS_IGNORE, ierr)
        
        ! Barrier to sync timing
        call MPI_Barrier(intercomm, ierr)
        
        end_time = MPI_Wtime()
        
        ! Calculate bandwidth (MB/s)
        bandwidth = (msg_size * 8.0 * 2) / ((end_time - start_time) * 1024.0 * 1024.0)
        
        if (rank == 0) then
            write(*,'(A,F8.2,A,F10.2,A)') 'Time: ', (end_time-start_time)*1000, ' ms, Bandwidth: ', bandwidth, ' MB/s'
        endif
        
        ! Verify data integrity
        if (abs(recv_data(1) - (rank + 1) * 3.0_8) > 1e-10) then
            write(*,*) 'Data corruption detected!'
        endif
        
        ! Clean up
        call MPI_Comm_disconnect(intercomm, ierr)
        deallocate(send_data, recv_data)
        
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
    enddo
    
    if (rank == 0) write(*,*) 'All tests completed successfully'
    
    call MPI_Finalize(ierr)
end program