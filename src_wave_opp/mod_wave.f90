! -
!
! SPDX-FileCopyrightText: Copyright (c) 2017-2022 Pedro Costa and the CaNS contributors. All rights reserved.
! SPDX-License-Identifier: MIT
!
! -
module mod_wave
use mod_types
use mpi
use decomp_2d
implicit none
public
!
! Two-wave superposition parameters
real(rp), protected :: wave_amplitude_1, wave_k_1, wave_omega_1
real(rp), protected :: wave_amplitude_2, wave_k_2, wave_omega_2
!
! 2D slice storage for boundary conditions
real(rp), allocatable, dimension(:,:) :: bc_bottom, bc_top

contains

subroutine read_wave(myid)
  implicit none
  integer, intent(in) :: myid
  integer :: iunit, ierr
  namelist /wave/ wave_amplitude_1, wave_k_1, wave_omega_1, &
                  wave_amplitude_2, wave_k_2, wave_omega_2
  !
  wave_amplitude_1 = -999.0_rp
  wave_k_1 = -999.0_rp
  wave_omega_1 = -999.0_rp
  wave_amplitude_2 = -999.0_rp
  wave_k_2 = -999.0_rp
  wave_omega_2 = -999.0_rp
  !
  open(newunit=iunit, file='wave.nml', status='old', action='read', iostat=ierr)
  if (ierr /= 0) then
    if (myid == 0) then
      print *, 'Error opening wave.nml file, iostat = ', ierr
      print *, 'Aborting...'
    end if
    call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
  end if
  !
  read(iunit, nml=wave, iostat=ierr)
  if (ierr /= 0) then
    if (myid == 0) then
      print *, 'Error reading namelist from wave.nml, iostat = ', ierr
      print *, 'Aborting...'
    end if
    call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
  end if
  !
  close(iunit)
  !
  if (wave_amplitude_1 == -999.0_rp .or. wave_k_1 == -999.0_rp .or. wave_omega_1 == -999.0_rp .or. &
      wave_amplitude_2 == -999.0_rp .or. wave_k_2 == -999.0_rp .or. wave_omega_2 == -999.0_rp) then
    if (myid == 0) then
      print *, 'Error: Some variables were not properly read from namelist'
      print *, 'Aborting...'
    end if
    call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
  end if
  !
  if (myid == 0) then
    print *, 'Successfully read from wave.nml (two-wave superposition):'
    print *, '  Wave 1: A = ', wave_amplitude_1, ', k = ', wave_k_1, ', omega = ', wave_omega_1
    print *, '  Wave 2: A = ', wave_amplitude_2, ', k = ', wave_k_2, ', omega = ', wave_omega_2
  end if
end subroutine read_wave

subroutine read_2d_slice(filename, ipencil, comm, ng, n, lo, plane_data)
  !
  ! Read a 2D binary slice (XY plane) with MPI-IO
  !
  implicit none
  character(len=*), intent(in) :: filename
  integer, intent(in) :: ipencil, comm
  integer, dimension(3), intent(in) :: ng, n, lo
  real(rp), dimension(:,:), intent(out) :: plane_data
  integer :: fh, ierr
  integer(kind=MPI_OFFSET_KIND) :: disp
  integer, dimension(2) :: sizes, subsizes, starts
  integer :: type_plane, n1, n2
  !
  ! Determine local array sizes based on pencil decomposition
  select case(ipencil)
  case(1) ! X-pencil
    n1 = xsize(1)
    n2 = xsize(2)
    sizes = [nx_global, ny_global]
    subsizes = [xsize(1), xsize(2)]
    starts = [xstart(1)-1, xstart(2)-1]
  case(2) ! Y-pencil
    n1 = ysize(1)
    n2 = ysize(2)
    sizes = [nx_global, ny_global]
    subsizes = [ysize(1), ysize(2)]
    starts = [ystart(1)-1, ystart(2)-1]
  case(3) ! Z-pencil
    n1 = zsize(1)
    n2 = zsize(2)
    sizes = [nx_global, ny_global]
    subsizes = [zsize(1), zsize(2)]
    starts = [zstart(1)-1, zstart(2)-1]
  end select
  !
  ! Open file and read
  call MPI_FILE_OPEN(comm, trim(filename), MPI_MODE_RDONLY, MPI_INFO_NULL, fh, ierr)
  if (ierr /= MPI_SUCCESS) then
    print*, 'Error opening file: ', trim(filename), ' ierr=', ierr
    call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
  end if
  !
  call MPI_TYPE_CREATE_SUBARRAY(2, sizes, subsizes, starts, &
       MPI_ORDER_FORTRAN, MPI_REAL_RP, type_plane, ierr)
  call MPI_TYPE_COMMIT(type_plane, ierr)
  disp = 0_MPI_OFFSET_KIND
  call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL_RP, type_plane, 'native', MPI_INFO_NULL, ierr)
  call MPI_FILE_READ_ALL(fh, plane_data, product(subsizes), MPI_REAL_RP, MPI_STATUS_IGNORE, ierr)
  call MPI_TYPE_FREE(type_plane, ierr)
  call MPI_FILE_CLOSE(fh, ierr)
  !
end subroutine read_2d_slice

subroutine apply_wave_bc(time, lo, dl, n, is_bound, w, istep, ipencil, comm, ng, datadir)
  implicit none
  real(rp), intent(in) :: time
  integer,  intent(in), dimension(3) :: lo, n, ng
  real(rp), intent(in), dimension(3) :: dl
  logical,  intent(in), dimension(0:1,3) :: is_bound
  real(rp), intent(inout), dimension(0:,0:,0:) :: w
  integer,  intent(in) :: istep, ipencil, comm
  character(len=*), intent(in) :: datadir
  character(len=256) :: filename_bot, filename_top
  character(len=7) :: istep_str
  integer :: i, j, istep_cyclic
  !
  ! Cycle through available files (0 to 800)
  istep_cyclic = mod(istep, 801)
  !
  ! Construct filenames based on cyclic iteration
  write(istep_str, '(i7.7)') istep_cyclic
  filename_bot = trim(datadir)//'slices/vez_slice_bot_fld_'//trim(istep_str)//'.bin'
  filename_top = trim(datadir)//'slices/vez_slice_top_fld_'//trim(istep_str)//'.bin'
  !
  ! Read bottom BC slice
  if (is_bound(0,3)) then
    if (.not. allocated(bc_bottom)) allocate(bc_bottom(n(1), n(2)))
    call read_2d_slice(filename_bot, ipencil, comm, ng, n, lo, bc_bottom)
    ! Apply to bottom boundary (k=0)
    do j = 1, n(2)
      do i = 1, n(1)
        w(i,j,0) = bc_bottom(i,j)
      end do
    end do
  end if
  !
  ! Read top BC slice
  if (is_bound(1,3)) then
    if (.not. allocated(bc_top)) allocate(bc_top(n(1), n(2)))
    call read_2d_slice(filename_top, ipencil, comm, ng, n, lo, bc_top)
    ! Apply to top boundary (k=n(3))
    do j = 1, n(2)
      do i = 1, n(1)
        w(i,j,n(3)) = bc_top(i,j)
      end do
    end do
  end if
end subroutine apply_wave_bc

end module
