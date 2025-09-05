program fortran_worker
  use mpi
  implicit none
  
  integer :: ierr, myid, mysize, ourid, oursize
  integer :: parentcomm, intracomm, group, cansgroup, canscomm
  integer, allocatable :: canscomm_ranks(:)
  double precision :: data(2), collective_result, collective_result_to_send
  integer :: i
  
  print*, 'F: Starting'
  
  ! Initialize MPI for spawned process
  call MPI_INIT(ierr)
  print*, 'F: MPI_INIT done, ierr=', ierr
  
  ! Get parent and world info
  call MPI_COMM_GET_PARENT(parentcomm, ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, myid, ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, mysize, ierr)
  print*, 'F: rank', myid, 'of', mysize
  
  ! Create canscomm (Fortran-only)
  call MPI_COMM_GROUP(MPI_COMM_WORLD, group, ierr)
  allocate(canscomm_ranks(mysize))
  do i = 1, mysize
    canscomm_ranks(i) = i - 1
  end do
  call MPI_GROUP_INCL(group, mysize, canscomm_ranks, cansgroup, ierr)
  call MPI_COMM_CREATE(MPI_COMM_WORLD, cansgroup, canscomm, ierr)
  print*, 'F', myid, ': canscomm created'
  
  ! Merge with parent to create intracomm - children use high=true
  call MPI_INTERCOMM_MERGE(parentcomm, .true., intracomm, ierr)
  if (ierr /= MPI_SUCCESS) then
    print*, 'F', myid, ': Error in MPI_INTERCOMM_MERGE, ierr=', ierr
  end if
  call MPI_COMM_RANK(intracomm, ourid, ierr)
  call MPI_COMM_SIZE(intracomm, oursize, ierr)
  print*, 'F', myid, ': intracomm rank', ourid, 'size', oursize
  
  ! Collective operation in canscomm
  collective_result = real(myid + 1, kind=8)
  call MPI_ALLREDUCE(MPI_IN_PLACE, collective_result, 1, MPI_DOUBLE_PRECISION, MPI_SUM, canscomm, ierr)
  if (myid == 0) print*, 'F: collective result=', collective_result
  
  
  ! Receive from Python (only rank 1 in intracomm)
  if (ourid == 1) then
    print*, 'F', myid, ': receiving from Python'
    call MPI_RECV(data, 2, MPI_DOUBLE_PRECISION, 0, 100, intracomm, MPI_STATUS_IGNORE, ierr)
    print*, 'F', myid, ': received', data
  end if
  
  ! Broadcast to all Fortran processes
  call MPI_BCAST(data, 2, MPI_DOUBLE_PRECISION, 0, canscomm, ierr)
  print*, 'F', myid, ': final data=', data
  
  ! Send processed data back to Python (only rank 0 in canscomm sends)
  if (myid == 0) then
    ! Process the data with collective result and send single value
    collective_result_to_send = collective_result
    print*, 'F', myid, ': sending collective result to Python', collective_result_to_send
    call MPI_SEND(collective_result_to_send, 1, MPI_DOUBLE_PRECISION, 0, 150, intracomm, ierr)
    print*, 'F', myid, ': collective result sent to Python'
  end if
  
  
  ! Cleanup
  call MPI_COMM_FREE(canscomm, ierr)
  call MPI_COMM_FREE(intracomm, ierr)
  call MPI_COMM_DISCONNECT(parentcomm, ierr)
  call MPI_FINALIZE(ierr)
  print*, 'F', myid, ': done'
  
end program fortran_worker