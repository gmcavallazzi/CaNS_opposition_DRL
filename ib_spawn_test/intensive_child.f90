program intensive_spawn_child
    use mpi_f08
    implicit none
    
    type(MPI_Comm) :: parent_comm
    integer :: rank, size, ierr, matrix_size
    real(8), allocatable :: matrix_a(:,:), matrix_b(:,:), local_result(:,:)
    real(8) :: compute_start, compute_end
    integer :: i, j, k, start_row, end_row, rows_per_proc
    
    call MPI_Init(ierr)
    call MPI_Comm_get_parent(parent_comm, ierr)
    
    if (parent_comm == MPI_COMM_NULL) then
        write(*,*) 'Child: No parent communicator found'
        call MPI_Finalize(ierr)
        stop
    endif
    
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)
    
    ! Receive problem size
    call MPI_Bcast(matrix_size, 1, MPI_INTEGER, 0, parent_comm, ierr)
    
    ! Allocate matrices
    allocate(matrix_a(matrix_size, matrix_size))
    allocate(matrix_b(matrix_size, matrix_size))
    allocate(local_result(matrix_size, matrix_size))
    
    ! Receive matrix data
    call MPI_Bcast(matrix_a, matrix_size*matrix_size, MPI_DOUBLE_PRECISION, 0, parent_comm, ierr)
    call MPI_Bcast(matrix_b, matrix_size*matrix_size, MPI_DOUBLE_PRECISION, 0, parent_comm, ierr)
    
    ! Determine work distribution
    rows_per_proc = matrix_size / size
    start_row = rank * rows_per_proc + 1
    end_row = start_row + rows_per_proc - 1
    if (rank == size - 1) end_row = matrix_size  ! Last proc gets remaining rows
    
    ! write(*,'(A,I0,A,I0,A,I0,A,I0)') 'Child ', rank, ' computing rows ', start_row, ' to ', end_row
    
    ! Initialize local result
    local_result = 0.0_8
    
    ! Intensive computation: Matrix multiplication with extra operations
    compute_start = MPI_Wtime()
    
    do j = 1, matrix_size
        do i = start_row, end_row
            do k = 1, matrix_size
                ! Matrix multiplication plus some extra floating point operations
                local_result(i,j) = local_result(i,j) + matrix_a(i,k) * matrix_b(k,j)
                
                ! Add some extra computation to make it more intensive
                local_result(i,j) = local_result(i,j) + sin(matrix_a(i,k)) * cos(matrix_b(k,j))
                local_result(i,j) = local_result(i,j) * exp(-1e-6_8 * abs(local_result(i,j)))
            enddo
        enddo
    enddo
    
    compute_end = MPI_Wtime()
    
    ! write(*,'(A,I0,A,F8.3,A)') 'Child ', rank, ' computation time: ', &
    !       (compute_end-compute_start)*1000, ' ms'
    
    ! Send results back to parent
    call MPI_Reduce(local_result, MPI_IN_PLACE, matrix_size*matrix_size, &
                   MPI_DOUBLE_PRECISION, MPI_SUM, 0, parent_comm, ierr)
    
    ! Cleanup
    deallocate(matrix_a, matrix_b, local_result)
    call MPI_Finalize(ierr)
end program