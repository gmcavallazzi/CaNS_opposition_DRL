! -
!
! SPDX-FileCopyrightText: Copyright (c) 2017-2022 Pedro Costa and the CaNS contributors. All rights reserved.
! SPDX-License-Identifier: MIT
!
! -
module mod_profile
  use mpi
  use decomp_2d
  use mod_common_mpi, only: myid, ierr
  use mod_types
  implicit none
  private
  public :: init_profiles, write_yavg_profile, write_xavg_profile, finalize_profiles
  !
  ! file units for profiles
  !
  integer, save :: iunit_u_yavg, iunit_w_yavg  ! y-averaged (profiles in x)
  integer, save :: iunit_u_xavg, iunit_w_xavg  ! x-averaged (profiles in y)
  logical, save :: is_initialized = .false.
  !
  contains
  !
  subroutine init_profiles(datadir)
    !
    ! Initialize the profile output files for both x and y averaging
    ! This should be called once at the beginning
    !
    implicit none
    character(len=*), intent(in) :: datadir
    !
    if (myid == 0) then
      ! Y-averaged profiles (profiles in x-direction)
      open(newunit=iunit_u_yavg, file=trim(datadir)//'profile_u_k1_yavg.bin', &
           form='unformatted', access='stream', status='replace', action='write')
      open(newunit=iunit_w_yavg, file=trim(datadir)//'profile_w_k1_yavg.bin', &
           form='unformatted', access='stream', status='replace', action='write')
      ! X-averaged profiles (profiles in y-direction)
      open(newunit=iunit_u_xavg, file=trim(datadir)//'profile_u_k1_xavg.bin', &
           form='unformatted', access='stream', status='replace', action='write')
      open(newunit=iunit_w_xavg, file=trim(datadir)//'profile_w_k1_xavg.bin', &
           form='unformatted', access='stream', status='replace', action='write')
    end if
    is_initialized = .true.
    !
  end subroutine init_profiles
  !
  subroutine write_yavg_profile(ng,lo,hi,u,w)
    !
    ! Computes spanwise (y-direction) average of u and w at k=1
    ! Produces profiles in x-direction
    !
    ! ng    -> global domain sizes
    ! lo,hi -> lower and upper extents of local arrays
    ! u,w   -> 3D velocity fields
    !
    implicit none
    integer , intent(in), dimension(3) :: ng, lo, hi
    real(rp), intent(in), dimension(lo(1)-1:,lo(2)-1:,lo(3)-1:) :: u, w
    !
    real(rp), allocatable, dimension(:) :: u_avg, w_avg
    integer :: i, j, k
    real(rp) :: ny_inv
    !
    ! k=1 is the first point in z-direction
    k = 1
    !
    ! Normalization factor for y-averaging
    ny_inv = 1.0_rp / real(ng(2), rp)
    !
    ! Allocate arrays (size ng(1) for x-direction)
    allocate(u_avg(ng(1)), w_avg(ng(1)))
    u_avg(:) = 0.0_rp
    w_avg(:) = 0.0_rp
    !
    ! Compute local contribution: sum over y-direction for each x
    do j = lo(2), hi(2)
      do i = lo(1), hi(1)
        u_avg(i) = u_avg(i) + u(i, j, k)
        w_avg(i) = w_avg(i) + w(i, j, k)
      end do
    end do
    !
    ! Sum contributions from all ranks
    call MPI_ALLREDUCE(MPI_IN_PLACE, u_avg(1), ng(1), MPI_REAL_RP, MPI_SUM, MPI_COMM_WORLD, ierr)
    call MPI_ALLREDUCE(MPI_IN_PLACE, w_avg(1), ng(1), MPI_REAL_RP, MPI_SUM, MPI_COMM_WORLD, ierr)
    !
    ! Normalize and write (only rank 0)
    if (myid == 0) then
      u_avg(:) = u_avg(:) * ny_inv
      w_avg(:) = w_avg(:) * ny_inv
      write(iunit_u_yavg) u_avg
      write(iunit_w_yavg) w_avg
      flush(iunit_u_yavg)
      flush(iunit_w_yavg)
    end if
    !
    deallocate(u_avg, w_avg)
    !
  end subroutine write_yavg_profile
  !
  subroutine write_xavg_profile(ng,lo,hi,u,w)
    !
    ! Computes streamwise (x-direction) average of u and w at k=1
    ! Produces profiles in y-direction
    !
    ! ng    -> global domain sizes
    ! lo,hi -> lower and upper extents of local arrays
    ! u,w   -> 3D velocity fields
    !
    implicit none
    integer , intent(in), dimension(3) :: ng, lo, hi
    real(rp), intent(in), dimension(lo(1)-1:,lo(2)-1:,lo(3)-1:) :: u, w
    !
    real(rp), allocatable, dimension(:) :: u_avg, w_avg
    integer :: i, j, k
    real(rp) :: nx_inv
    !
    ! k=1 is the first point in z-direction
    k = 1
    !
    ! Normalization factor for x-averaging
    nx_inv = 1.0_rp / real(ng(1), rp)
    !
    ! Allocate arrays (size ng(2) for y-direction)
    allocate(u_avg(ng(2)), w_avg(ng(2)))
    u_avg(:) = 0.0_rp
    w_avg(:) = 0.0_rp
    !
    ! Compute local contribution: sum over x-direction for each y
    do j = lo(2), hi(2)
      do i = lo(1), hi(1)
        u_avg(j) = u_avg(j) + u(i, j, k)
        w_avg(j) = w_avg(j) + w(i, j, k)
      end do
    end do
    !
    ! Sum contributions from all ranks
    call MPI_ALLREDUCE(MPI_IN_PLACE, u_avg(1), ng(2), MPI_REAL_RP, MPI_SUM, MPI_COMM_WORLD, ierr)
    call MPI_ALLREDUCE(MPI_IN_PLACE, w_avg(1), ng(2), MPI_REAL_RP, MPI_SUM, MPI_COMM_WORLD, ierr)
    !
    ! Normalize and write (only rank 0)
    if (myid == 0) then
      u_avg(:) = u_avg(:) * nx_inv
      w_avg(:) = w_avg(:) * nx_inv
      write(iunit_u_xavg) u_avg
      write(iunit_w_xavg) w_avg
      flush(iunit_u_xavg)
      flush(iunit_w_xavg)
    end if
    !
    deallocate(u_avg, w_avg)
    !
  end subroutine write_xavg_profile
  !
  subroutine finalize_profiles()
    !
    ! Close all profile output files
    !
    implicit none
    !
    if (myid == 0 .and. is_initialized) then
      close(iunit_u_yavg)
      close(iunit_w_yavg)
      close(iunit_u_xavg)
      close(iunit_w_xavg)
    end if
    !
  end subroutine finalize_profiles
  !
end module mod_profile
