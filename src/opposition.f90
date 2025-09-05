! -
!
! SPDX-FileCopyrightText: Copyright (c) 2017-2022 Pedro Costa and the CaNS contributors. All rights reserved.
! SPDX-License-Identifier: MIT
!
! -
module mod_opposition
use mod_types
use mpi
!@acc use cudecomp
implicit none
public
!
!
! opposition control read input and routines
!
!
!integer , protected :: z_sense, mysize
!real(rp), protected :: alpha_opp
integer , protected :: z_sense, n_walls
real(rp), protected :: alpha_opp
! Needed for routines
real(rp), allocatable, dimension(:,:,:) :: var_opp

contains
subroutine read_opposition(myid)
    implicit none
    integer, intent(in) :: myid
    integer :: iunit, ierr
    namelist /opp/ z_sense, n_walls, alpha_opp
    
    ! Initialize with invalid values to check if they're properly read
    z_sense = -999
    n_walls = -999
    alpha_opp = -999.0_rp
    
    open(newunit=iunit, file='opp.nml', status='old', action='read', iostat=ierr)
    if (ierr /= 0) then
        if (myid == 0) then
            print *, 'Error opening opp.nml file, iostat = ', ierr
            print *, 'Aborting...'
        end if
        call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
    end if
    
    read(iunit, nml=opp, iostat=ierr)
    if (ierr /= 0) then
        if (myid == 0) then
            print *, 'Error reading namelist from opp.nml, iostat = ', ierr
            print *, 'Aborting...'
        end if
        call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
    end if
    
    close(iunit)
    
    ! Verify values were properly read
    if (z_sense == -999 .or. alpha_opp == -999.0_rp .or. n_walls == -999) then
        if (myid == 0) then
            print *, 'Error: Some variables were not properly read from namelist'
            print *, 'Aborting...'
        end if
        call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
    end if
    
    if (myid == 0) then
        print *, 'Successfully read from opp.nml:'
        print *, '  alpha_opp = ', alpha_opp
        print *, '  z_sense = ', z_sense
        print *, '  n_walls = ', n_walls
    end if
end subroutine read_opposition

subroutine opposition(n,alpha_opp,z,var_opp,w)
  implicit none
  integer , intent(in),    dimension(3)     :: n
  real(rp), intent(in)                      :: alpha_opp
  integer , intent(in)                      :: z
  real(rp), intent(in),    dimension(:,:,0:)   :: var_opp
  real(rp), intent(inout), dimension(0:,0:,0:) :: w
  integer  :: i,j,ierr

  do j=1,n(2)
    do i=1,n(1)
      w(i,j,z*n(3)) = alpha_opp * var_opp(i,j,z*n(3))
    end do
  end do

end subroutine opposition

subroutine update_opp(n, n_walls, w, var_opp, myid)
  use mod_drl, only: var_opp_marl
  implicit none
  integer, intent(in), dimension(3) :: n
  integer, intent(in) :: n_walls
  real(rp), intent(in), dimension(0:,0:,0:) :: w
  real(rp), intent(inout), dimension(:,:,0:) :: var_opp
  integer, intent(in) :: myid
  integer :: i,j

  var_opp(:,:,0) = var_opp_marl(:,:)
  if (n_walls.eq.2) var_opp(:,:,1) = var_opp_marl(:,:)
end subroutine update_opp

end module
