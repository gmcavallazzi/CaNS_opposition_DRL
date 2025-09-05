program intensive_spawn_parent
    use mpi_f08
    implicit none
    
    type(MPI_Comm) :: intercomm
    integer :: rank, size, ierr, episode, num_episodes
    integer :: matrix_size, child_procs
    real(8), allocatable :: matrix_a(:,:), matrix_b(:,:), result_matrix(:,:)
    real(8) :: episode_start, episode_end, total_start, total_end
    real(8) :: spawn_time, compute_time, comm_time, cleanup_time
    character(len=20) :: child_exe = './intensive_child'
    
    ! Problem parameters
    integer, parameter :: default_matrix_size = 1024
    integer, parameter :: default_episodes = 5
    integer, parameter :: default_child_procs = 8
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    
    ! Get parameters from command line or use defaults
    matrix_size = default_matrix_size
    num_episodes = default_episodes  
    child_procs = default_child_procs
    
    if (rank == 0) then
        write(*,'(A)') '=== SPAWN-BASED INTENSIVE TEST ==='
        write(*,'(A,I0)') 'Parent processes: ', size
        write(*,'(A,I0)') 'Child processes per episode: ', child_procs
        write(*,'(A,I0)') 'Matrix size: ', matrix_size
        write(*,'(A,I0)') 'Episodes: ', num_episodes
        write(*,'(A)') '================================='
    endif
    
    ! Allocate matrices
    allocate(matrix_a(matrix_size, matrix_size))
    allocate(matrix_b(matrix_size, matrix_size))  
    allocate(result_matrix(matrix_size, matrix_size))
    
    ! Initialize matrices with some data
    call initialize_matrices(matrix_a, matrix_b, matrix_size, rank)
    
    total_start = MPI_Wtime()
    
    do episode = 1, num_episodes
        episode_start = MPI_Wtime()
        
        if (rank == 0) write(*,'(A,I0,A)') 'Episode ', episode, ': Spawning children...'
        
        ! Time spawning
        spawn_time = MPI_Wtime()
        call MPI_Comm_spawn(child_exe, MPI_ARGV_NULL, child_procs, &
                           MPI_INFO_NULL, 0, MPI_COMM_WORLD, &
                           intercomm, MPI_ERRCODES_IGNORE, ierr)
        spawn_time = MPI_Wtime() - spawn_time
        
        if (ierr /= MPI_SUCCESS) then
            write(*,*) 'Failed to spawn children in episode ', episode
            cycle
        endif
        
        ! Time communication + computation
        comm_time = MPI_Wtime()
        
        ! Send work to children (matrix data + parameters)
        call MPI_Bcast(matrix_size, 1, MPI_INTEGER, MPI_ROOT, intercomm, ierr)
        call MPI_Bcast(matrix_a, matrix_size*matrix_size, MPI_DOUBLE_PRECISION, MPI_ROOT, intercomm, ierr)
        call MPI_Bcast(matrix_b, matrix_size*matrix_size, MPI_DOUBLE_PRECISION, MPI_ROOT, intercomm, ierr)
        
        ! Children do computation, then send results back
        call MPI_Reduce(MPI_IN_PLACE, result_matrix, matrix_size*matrix_size, &
                       MPI_DOUBLE_PRECISION, MPI_SUM, MPI_ROOT, intercomm, ierr)
        
        comm_time = MPI_Wtime() - comm_time
        
        ! Time cleanup
        cleanup_time = MPI_Wtime()
        call MPI_Comm_disconnect(intercomm, ierr)
        cleanup_time = MPI_Wtime() - cleanup_time
        
        episode_end = MPI_Wtime()
        
        if (rank == 0) then
            write(*,'(A,I0,A,F8.3,A)') 'Episode ', episode, ' completed in ', &
                  (episode_end-episode_start)*1000, ' ms'
            write(*,'(A,F6.1,A,F6.1,A,F6.1,A)') '  Spawn: ', spawn_time*1000, &
                  ' ms, Comm: ', comm_time*1000, ' ms, Cleanup: ', cleanup_time*1000, ' ms'
        endif
        
        ! Verify result (simple checksum)
        if (rank == 0) then
            if (abs(result_matrix(1,1)) < 1e-10) then
                write(*,*) 'Warning: Result seems incorrect'
            endif
        endif
        
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
    enddo
    
    total_end = MPI_Wtime()
    
    if (rank == 0) then
        write(*,'(A)') '================================='
        write(*,'(A,F10.3,A)') 'Total time: ', (total_end-total_start)*1000, ' ms'
        write(*,'(A,F10.3,A)') 'Average per episode: ', &
              (total_end-total_start)*1000/num_episodes, ' ms'
        write(*,'(A,F10.2,A)') 'Throughput: ', &
              num_episodes/(total_end-total_start), ' episodes/sec'
    endif
    
    deallocate(matrix_a, matrix_b, result_matrix)
    call MPI_Finalize(ierr)
    
contains
    
    subroutine initialize_matrices(a, b, n, rank_id)
        real(8), intent(out) :: a(:,:), b(:,:)
        integer, intent(in) :: n, rank_id
        integer :: i, j
        
        do j = 1, n
            do i = 1, n
                a(i,j) = sin(real(i+j+rank_id, 8) * 0.01_8)
                b(i,j) = cos(real(i*j+rank_id, 8) * 0.01_8)  
            enddo
        enddo
    end subroutine initialize_matrices
    
end program