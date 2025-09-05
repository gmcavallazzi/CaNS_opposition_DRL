program persistent_workers
    use mpi_f08
    implicit none
    
    integer :: rank, size, ierr, episode, num_episodes
    integer :: matrix_size, worker_procs, controller_rank
    real(8), allocatable :: matrix_a(:,:), matrix_b(:,:), result_matrix(:,:), local_result(:,:)
    real(8) :: episode_start, episode_end, total_start, total_end
    real(8) :: compute_time, comm_time
    logical :: is_controller
    type(MPI_Comm) :: worker_comm
    integer :: worker_rank, worker_size
    
    ! Problem parameters
    integer, parameter :: default_matrix_size = 1024
    integer, parameter :: default_episodes = 5
    integer, parameter :: CONTROLLER_TAG = 100
    integer, parameter :: WORK_TAG = 200
    integer, parameter :: RESULT_TAG = 300
    integer, parameter :: STOP_TAG = 999
    
    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    
    ! Parameters - in real use these could come from command line
    matrix_size = default_matrix_size
    num_episodes = default_episodes
    controller_rank = 0  ! Rank 0 is the controller
    worker_procs = size - 1  ! All other ranks are workers
    is_controller = (rank == controller_rank)
    
    if (worker_procs < 1) then
        if (rank == 0) write(*,*) 'Error: Need at least 2 processes (1 controller + 1 worker)'
        call MPI_Finalize(ierr)
        stop
    endif
    
    ! Create worker communicator (excludes controller)
    if (is_controller) then
        worker_comm = MPI_COMM_NULL
        worker_rank = -1
        worker_size = -1
    else
        call MPI_Comm_split(MPI_COMM_WORLD, 1, rank, worker_comm, ierr)
        call MPI_Comm_rank(worker_comm, worker_rank, ierr)
        call MPI_Comm_size(worker_comm, worker_size, ierr)
    endif
    
    if (is_controller) then
        ! CONTROLLER CODE
        write(*,'(A)') '=== PERSISTENT WORKER TEST ==='
        write(*,'(A,I0)') 'Total processes: ', size
        write(*,'(A,I0)') 'Worker processes: ', worker_procs
        write(*,'(A,I0)') 'Matrix size: ', matrix_size
        write(*,'(A,I0)') 'Episodes: ', num_episodes
        write(*,'(A)') '==============================='
        
        ! Allocate matrices
        allocate(matrix_a(matrix_size, matrix_size))
        allocate(matrix_b(matrix_size, matrix_size))  
        allocate(result_matrix(matrix_size, matrix_size))
        
        total_start = MPI_Wtime()
        
        do episode = 1, num_episodes
            episode_start = MPI_Wtime()
            
            write(*,'(A,I0,A)') 'Episode ', episode, ': Sending work to persistent workers...'
            
            ! Initialize matrices with episode-specific data
            call initialize_matrices(matrix_a, matrix_b, matrix_size, episode)
            
            comm_time = MPI_Wtime()
            
            ! Send work parameters to all workers
            call send_work_to_workers(matrix_a, matrix_b, matrix_size, worker_procs, &
                                    MPI_COMM_WORLD, controller_rank)
            
            ! Collect results from workers
            call collect_results_from_workers(result_matrix, matrix_size, worker_procs, &
                                            MPI_COMM_WORLD, controller_rank)
            
            comm_time = MPI_Wtime() - comm_time
            
            episode_end = MPI_Wtime()
            
            write(*,'(A,I0,A,F8.3,A)') 'Episode ', episode, ' completed in ', &
                  (episode_end-episode_start)*1000, ' ms'
            write(*,'(A,F6.1,A)') '  Communication time: ', comm_time*1000, ' ms'
            
            ! Verify result (simple checksum)
            if (abs(result_matrix(1,1)) < 1e-10) then
                write(*,*) 'Warning: Result seems incorrect'
            endif
        enddo
        
        ! Send stop signal to workers
        call send_stop_signal(worker_procs, MPI_COMM_WORLD, controller_rank)
        
        total_end = MPI_Wtime()
        
        write(*,'(A)') '==============================='
        write(*,'(A,F10.3,A)') 'Total time: ', (total_end-total_start)*1000, ' ms'
        write(*,'(A,F10.3,A)') 'Average per episode: ', &
              (total_end-total_start)*1000/num_episodes, ' ms'
        write(*,'(A,F10.2,A)') 'Throughput: ', &
              num_episodes/(total_end-total_start), ' episodes/sec'
        
        deallocate(matrix_a, matrix_b, result_matrix)
        
    else
        ! WORKER CODE
        allocate(matrix_a(matrix_size, matrix_size))
        allocate(matrix_b(matrix_size, matrix_size))
        allocate(local_result(matrix_size, matrix_size))
        
        write(*,'(A,I0,A)') 'Worker ', rank, ' ready for work'
        
        ! Worker main loop - wait for work until stop signal
        do
            if (.not. receive_work_or_stop(matrix_a, matrix_b, matrix_size, &
                                         MPI_COMM_WORLD, controller_rank)) exit
            
            ! Do the computation
            compute_time = MPI_Wtime()
            call compute_intensive_work(matrix_a, matrix_b, local_result, matrix_size, &
                                      worker_rank, worker_size, worker_comm)
            compute_time = MPI_Wtime() - compute_time
            
            write(*,'(A,I0,A,F8.3,A)') 'Worker ', rank, ' computation time: ', &
                  compute_time*1000, ' ms'
            
            ! Send results back
            call send_results_to_controller(local_result, matrix_size, &
                                          MPI_COMM_WORLD, controller_rank, rank)
        enddo
        
        write(*,'(A,I0,A)') 'Worker ', rank, ' shutting down'
        deallocate(matrix_a, matrix_b, local_result)
    endif
    
    if (.not. is_controller .and. worker_comm /= MPI_COMM_NULL) then
        call MPI_Comm_free(worker_comm, ierr)
    endif
    
    call MPI_Finalize(ierr)
    
contains
    
    subroutine initialize_matrices(a, b, n, episode_id)
        real(8), intent(out) :: a(:,:), b(:,:)
        integer, intent(in) :: n, episode_id
        integer :: i, j
        
        do j = 1, n
            do i = 1, n
                a(i,j) = sin(real(i+j+episode_id, 8) * 0.01_8)
                b(i,j) = cos(real(i*j+episode_id, 8) * 0.01_8)  
            enddo
        enddo
    end subroutine initialize_matrices
    
    subroutine send_work_to_workers(a, b, n, num_workers, comm, controller)
        real(8), intent(in) :: a(:,:), b(:,:)
        integer, intent(in) :: n, num_workers, controller
        type(MPI_Comm), intent(in) :: comm
        integer :: worker, ierr
        
        do worker = 1, num_workers
            call MPI_Send(a, n*n, MPI_DOUBLE_PRECISION, worker, WORK_TAG, comm, ierr)
            call MPI_Send(b, n*n, MPI_DOUBLE_PRECISION, worker, WORK_TAG, comm, ierr)
        enddo
    end subroutine send_work_to_workers
    
    subroutine collect_results_from_workers(result, n, num_workers, comm, controller)
        real(8), intent(out) :: result(:,:)
        integer, intent(in) :: n, num_workers, controller
        type(MPI_Comm), intent(in) :: comm
        real(8), allocatable :: worker_result(:,:)
        integer :: worker, ierr
        
        allocate(worker_result(n, n))
        result = 0.0_8
        
        do worker = 1, num_workers
            call MPI_Recv(worker_result, n*n, MPI_DOUBLE_PRECISION, worker, &
                         RESULT_TAG, comm, MPI_STATUS_IGNORE, ierr)
            result = result + worker_result
        enddo
        
        deallocate(worker_result)
    end subroutine collect_results_from_workers
    
    subroutine send_stop_signal(num_workers, comm, controller)
        integer, intent(in) :: num_workers, controller
        type(MPI_Comm), intent(in) :: comm
        integer :: worker, dummy, ierr
        
        dummy = 0
        do worker = 1, num_workers
            call MPI_Send(dummy, 1, MPI_INTEGER, worker, STOP_TAG, comm, ierr)
        enddo
    end subroutine send_stop_signal
    
    function receive_work_or_stop(a, b, n, comm, controller) result(continue_work)
        real(8), intent(out) :: a(:,:), b(:,:)
        integer, intent(in) :: n, controller
        type(MPI_Comm), intent(in) :: comm
        logical :: continue_work
        integer :: ierr, tag
        type(MPI_Status) :: status
        integer :: dummy
        
        ! Check what message is coming
        call MPI_Probe(controller, MPI_ANY_TAG, comm, status, ierr)
        tag = status%MPI_TAG
        
        if (tag == STOP_TAG) then
            call MPI_Recv(dummy, 1, MPI_INTEGER, controller, STOP_TAG, comm, status, ierr)
            continue_work = .false.
        else
            call MPI_Recv(a, n*n, MPI_DOUBLE_PRECISION, controller, WORK_TAG, comm, status, ierr)
            call MPI_Recv(b, n*n, MPI_DOUBLE_PRECISION, controller, WORK_TAG, comm, status, ierr)
            continue_work = .true.
        endif
    end function receive_work_or_stop
    
    subroutine compute_intensive_work(a, b, result, n, w_rank, w_size, w_comm)
        real(8), intent(in) :: a(:,:), b(:,:)
        real(8), intent(out) :: result(:,:)
        integer, intent(in) :: n, w_rank, w_size
        type(MPI_Comm), intent(in) :: w_comm
        integer :: i, j, k, start_row, end_row, rows_per_proc
        
        ! Determine work distribution among workers
        rows_per_proc = n / w_size
        start_row = w_rank * rows_per_proc + 1
        end_row = start_row + rows_per_proc - 1
        if (w_rank == w_size - 1) end_row = n  ! Last worker gets remaining rows
        
        ! Initialize result
        result = 0.0_8
        
        ! Same intensive computation as spawn version
        do j = 1, n
            do i = start_row, end_row
                do k = 1, n
                    ! Matrix multiplication plus extra operations
                    result(i,j) = result(i,j) + a(i,k) * b(k,j)
                    result(i,j) = result(i,j) + sin(a(i,k)) * cos(b(k,j))
                    result(i,j) = result(i,j) * exp(-1e-6_8 * abs(result(i,j)))
                enddo
            enddo
        enddo
    end subroutine compute_intensive_work
    
    subroutine send_results_to_controller(result, n, comm, controller, sender)
        real(8), intent(in) :: result(:,:)
        integer, intent(in) :: n, controller, sender
        type(MPI_Comm), intent(in) :: comm
        integer :: ierr
        
        call MPI_Send(result, n*n, MPI_DOUBLE_PRECISION, controller, RESULT_TAG, comm, ierr)
    end subroutine send_results_to_controller
    
end program