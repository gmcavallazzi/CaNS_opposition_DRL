program pod_fortran
  ! ---------------------------------------------------------------
  ! POD of two 2D velocity slices via SVD (LAPACK dgesdd).
  !
  ! Reads the same binary files as the Python script, stacks them
  ! as columns of a snapshot matrix, performs the economy SVD,
  ! truncates at 90 % energy, and writes:
  !   pod_fortran_result.dat  — plain text summary
  !   pod_mode1.bin           — reconstruction from mode 1 (most energetic)
  !   pod_modeN.bin           — reconstruction from mode N (least energetic retained)
  !
  ! Compile (requires LAPACK/BLAS):
  !   gfortran -O2 -o pod_fortran pod_fortran.f90 -llapack -lblas
  !
  ! Run:
  !   ./pod_fortran
  ! ---------------------------------------------------------------
  implicit none

  ! --- parameters ---
  integer, parameter :: dp      = selected_real_kind(15, 307)
  integer, parameter :: nx      = 192
  integer, parameter :: ny      = 192
  integer, parameter :: npts    = nx * ny        ! spatial points per field
  integer, parameter :: nsnap   = 2              ! number of snapshots (vex, vez)
  real(dp), parameter :: energy_threshold = 0.9_dp

  ! --- file names ---
  character(len=256), parameter :: dir_data = '../data_slices/'
  character(len=256), parameter :: file_vex = trim(dir_data) // 'vex_slice_19_fld_0449500.bin'
  character(len=256), parameter :: file_vez = trim(dir_data) // 'vez_slice_19_fld_0449500.bin'
  character(len=256), parameter :: file_out = 'pod_fortran_result.dat'
  character(len=256), parameter :: file_mode1 = 'pod_mode1.bin'
  character(len=256), parameter :: file_modeN = 'pod_modeN.bin'

  ! --- working arrays ---
  ! Snapshot matrix  X(npts, nsnap)   — columns are snapshots
  real(dp), allocatable :: X(:,:)
  ! SVD: X = U * diag(S) * Vt
  real(dp), allocatable :: U(:,:), S(:), Vt(:,:)
  ! Reconstructed fields
  real(dp), allocatable :: recon_mode1(:,:), recon_modeN(:,:)

  ! LAPACK workspace
  real(dp), allocatable :: work(:)
  integer, allocatable  :: iwork(:)
  integer :: lwork, info

  ! loop / misc
  integer :: i, j, n_keep, unit_io
  real(dp) :: cumulative, total_energy

  ! --- allocate ---
  allocate( X(npts, nsnap) )
  allocate( U(npts, nsnap), S(nsnap), Vt(nsnap, nsnap) )
  allocate( recon_mode1(npts, nsnap), recon_modeN(npts, nsnap) )

  ! --- read binary slices ---
  call read_bin(file_vex, X(:,1), npts)
  call read_bin(file_vez, X(:,2), npts)

  ! --- SVD via LAPACK dgesdd (divide-and-conquer, economy) ---
  ! Query optimal workspace
  lwork = -1
  allocate(work(1), iwork(8*nsnap))
  call dgesdd('S', npts, nsnap, X, npts, S, U, npts, Vt, nsnap, &
              work, lwork, iwork, info)
  lwork = int(work(1))
  deallocate(work)
  allocate(work(lwork))

  ! Actual factorisation (overwrites X)
  call dgesdd('S', npts, nsnap, X, npts, S, U, npts, Vt, nsnap, &
              work, lwork, iwork, info)
  if (info /= 0) then
     write(*,*) 'FATAL: dgesdd returned info =', info
     stop 1
  end if

  ! --- energy truncation ---
  total_energy = sum(S**2)
  cumulative   = 0.0_dp
  n_keep       = 0
  do i = 1, nsnap
     cumulative = cumulative + S(i)**2
     n_keep     = i
     if (cumulative / total_energy >= energy_threshold) exit
  end do

  ! --- reconstruct from single modes ---
  ! mode k reconstruction:  X_k = U(:,k) * S(k) * Vt(k,:)
  call reconstruct_mode(U, S, Vt, 1, npts, nsnap, recon_mode1)
  call reconstruct_mode(U, S, Vt, n_keep, npts, nsnap, recon_modeN)

  ! --- write binary reconstruction files (Fortran column-major, same as input) ---
  call write_bin(file_mode1, recon_mode1(:,1), npts)
  call write_bin(file_mode1, recon_mode1(:,2), npts)  ! appends vez after vex

  call write_bin(file_modeN, recon_modeN(:,1), npts)
  call write_bin(file_modeN, recon_modeN(:,2), npts)

  ! --- write text summary ---
  open(newunit=unit_io, file=file_out, status='replace', action='write')
  write(unit_io,'(A)')   '=== POD Fortran Summary ==='
  write(unit_io,'(A,I0)') 'Grid points (nx*ny): ', npts
  write(unit_io,'(A,I0)') 'Snapshots           : ', nsnap
  write(unit_io,'(A,I0)') 'Modes retained (90%): ', n_keep
  write(unit_io,'(A)') ' '
  write(unit_io,'(A)') 'Singular values:'
  do i = 1, nsnap
     write(unit_io,'(A,I2,A,ES23.16)') '  S(', i, ') = ', S(i)
  end do
  write(unit_io,'(A)') ' '
  write(unit_io,'(A)') 'Cumulative energy fraction:'
  cumulative = 0.0_dp
  do i = 1, nsnap
     cumulative = cumulative + S(i)**2
     write(unit_io,'(A,I2,A,F10.6)') '  mode ', i, ' : ', cumulative/total_energy
  end do
  write(unit_io,'(A)') ' '
  write(unit_io,'(A,A)') 'Most energetic mode reconstruction  : ', trim(file_mode1)
  write(unit_io,'(A,A)') 'Least energetic retained recon.     : ', trim(file_modeN)
  close(unit_io)

  write(*,'(A)') 'POD complete.  See pod_fortran_result.dat for details.'
  write(*,'(A,I0,A)') 'Modes retained: ', n_keep, ' / ' // char(ichar('0')+nsnap)

  ! --- cleanup ---
  deallocate(X, U, S, Vt, work, iwork)
  deallocate(recon_mode1, recon_modeN)

contains

  subroutine read_bin(fname, buf, n)
    ! Read n doubles from an unformatted binary file.
    character(len=*), intent(in)  :: fname
    integer,          intent(in)  :: n
    real(dp),         intent(out) :: buf(n)
    integer :: funit

    open(newunit=funit, file=trim(fname), status='old', access='stream', &
         form='unformatted', iostat=info)
    if (info /= 0) then
       write(*,*) 'ERROR: cannot open ', trim(fname)
       stop 1
    end if
    read(funit) buf
    close(funit)
    write(*,'(A,A,A,I0,A)') 'Read ', trim(fname), '  (', n, ' doubles)'
  end subroutine

  subroutine write_bin(fname, buf, n)
    ! Append n doubles to an unformatted binary file.
    ! First call with a given fname must be preceded by a delete or the file
    ! will be overwritten on the first open (position='append' creates if needed).
    character(len=*), intent(in) :: fname
    integer,          intent(in) :: n
    real(dp),         intent(in) :: buf(n)
    integer :: funit
    logical :: exists

    inquire(file=trim(fname), exist=exists)
    if (.not. exists) then
       open(newunit=funit, file=trim(fname), status='new', access='stream', &
            form='unformatted')
    else
       open(newunit=funit, file=trim(fname), status='old', access='stream', &
            form='unformatted', position='append')
    end if
    write(funit) buf
    close(funit)
  end subroutine

  subroutine reconstruct_mode(U, S, Vt, k, m, n, recon)
    ! recon(:,j) = U(:,k) * S(k) * Vt(k,j)
    real(dp), intent(in)  :: U(m,n), S(n), Vt(n,n)
    integer,  intent(in)  :: k, m, n
    real(dp), intent(out) :: recon(m,n)
    integer :: j
    do j = 1, n
       recon(:,j) = U(:,k) * S(k) * Vt(k,j)
    end do
  end subroutine

end program pod_fortran
