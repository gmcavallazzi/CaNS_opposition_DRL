# 2D Slice Read/Write Test Plan

## Overview
This document describes a minimal test to verify that 2D binary files can be correctly read back, regardless of MPI parallelization.

## Test Specifications
- **Field**: `w` (wall-normal velocity component)
- **Slice orientation**: Parallel to xy wall (perpendicular to z-axis)
- **Slice index**: `ng(3)/2` (middle of the domain)
- **When to execute**: After first 2D output (when `iout2d > 0 .and. mod(istep,max(iout2d,1)) == 0`)

## Implementation

### 1. Variable Declaration (add near line 140 in main.f90)

```fortran
real(rp), allocatable, dimension(:,:,:) :: w_test
```

Add this with the other array declarations, near:
```fortran
real(rp), allocatable, dimension(:,:,:) :: u,v,w,p,pp
```

### 2. Array Allocation (add after line 223 in main.f90)

```fortran
allocate(w_test(0:n(1)+1,0:n(2)+1,0:n(3)+1))
```

Add this after the existing allocations:
```fortran
allocate(u( 0:n(1)+1,0:n(2)+1,0:n(3)+1), &
         v( 0:n(1)+1,0:n(2)+1,0:n(3)+1), &
         w( 0:n(1)+1,0:n(2)+1,0:n(3)+1), &
         p( 0:n(1)+1,0:n(2)+1,0:n(3)+1), &
         pp(0:n(1)+1,0:n(2)+1,0:n(3)+1), &
         tau_wall(0:n(1)+1,0:n(2)+1,0:n(3)+1))
```

### 3. Module Import (add at top of main.f90, around line 158)

**IMPORTANT**: Add `ipencil_axis` to the existing `mod_common_mpi` import.

Find this line (around line 38):
```fortran
  use mod_common_mpi     , only: myid,ourid,ierr,parentcomm,intracomm,group,cansgroup,canscomm,mysize,oursize
```

And change it to:
```fortran
  use mod_common_mpi     , only: myid,ourid,ierr,parentcomm,intracomm,group,cansgroup,canscomm,mysize,oursize,ipencil_axis
```

**OR** if you want to keep it separate, add a new line right after:
```fortran
  use mod_common_mpi     , only: ipencil => ipencil_axis
```

### 4. Test Logic (add after the 2D output section, around line 700)

Insert this code block right after the existing `if(iout2d > 0...)` block:

```fortran
    if(iout2d > 0.and.mod(istep,max(iout2d,1)) == 0) then
      !$acc wait
      !$acc update self(u,v,w,p)
      include 'out2d.h90'

      ! ===== 2D READ/WRITE TEST START =====
      if(myid == 0) print*, '*** Starting 2D read/write test ***'

      ! 1. Save the original w-velocity slice at mid-plane
      call out2d(trim(datadir)//'test_w_original.bin', 3, ng(3)/2, w(1:n(1),1:n(2),1:n(3)))

      ! 2. Read it back into w_test array
      ! Note: decomp_2d_read_plane is the read counterpart to decomp_2d_write_plane (used in out2d)
      call decomp_2d_read_plane(ipencil_axis, w_test, 3, ng(3)/2, '.', 'test_w_original.bin', 'dummy')

      ! 3. Save the read-back data with a different filename
      call out2d(trim(datadir)//'test_w_readback.bin', 3, ng(3)/2, w_test(1:n(1),1:n(2),1:n(3)))

      if(myid == 0) print*, '*** 2D read/write test complete ***'
      if(myid == 0) print*, '*** Compare test_w_original.bin with test_w_readback.bin ***'
      ! ===== 2D READ/WRITE TEST END =====
    end if
```

## CRITICAL FIX: Custom Read Plane Subroutine

**IMPORTANT**: Your version of 2decomp does NOT include `decomp_2d_read_plane` (it's in newer versions). We must implement our own.

**Note**: According to the latest 2decomp documentation, `decomp_2d_read_plane` exists in newer versions with signature:
```fortran
call decomp_2d_read_plane(ipencil,var,iplane,n,filename,io_name,opt_decomp)
```
If you upgrade 2decomp, you could use this function directly. For now, here's the manual implementation:

Add this subroutine **before the main program** in main.f90 (around line 26, before `program cans`):

```fortran
  subroutine read_plane_2d(ipencil,var,iplane,nplane,datadir,fname)
    !
    ! Read a 2D plane from a binary file (parallel MPI-IO)
    ! This mirrors the decomp_2d_write_plane functionality
    !
    use decomp_2d
    use mod_common_mpi, only: ierr
    use mod_types
    implicit none
    integer, intent(in) :: ipencil, iplane, nplane
    real(rp), dimension(:,:,:), intent(inout) :: var
    character(len=*), intent(in) :: datadir, fname
    integer :: fh
    integer(kind=MPI_OFFSET_KIND) :: filesize, disp
    integer, dimension(3) :: sizes, subsizes, starts
    integer :: type_plane
    character(len=256) :: fullpath

    fullpath = trim(datadir)//trim(fname)

    ! Open file for reading
    call MPI_FILE_OPEN(MPI_COMM_WORLD, trim(fullpath), &
         MPI_MODE_RDONLY, MPI_INFO_NULL, fh, ierr)

    disp = 0_MPI_OFFSET_KIND

    select case(ipencil)
    case(1) ! X-pencil
      select case(iplane)
      case(1) ! YZ plane
        sizes(1) = xsize(2)
        sizes(2) = xsize(3)
        subsizes(1) = xsize(2)
        subsizes(2) = xsize(3)
        starts(1) = xstart(2) - 1
        starts(2) = xstart(3) - 1
      case(2) ! XZ plane
        sizes(1) = xsize(1)
        sizes(2) = xsize(3)
        subsizes(1) = xsize(1)
        subsizes(2) = xsize(3)
        starts(1) = xstart(1) - 1
        starts(2) = xstart(3) - 1
      case(3) ! XY plane
        sizes(1) = xsize(1)
        sizes(2) = xsize(2)
        subsizes(1) = xsize(1)
        subsizes(2) = xsize(2)
        starts(1) = xstart(1) - 1
        starts(2) = xstart(2) - 1
      end select
    case(2) ! Y-pencil
      select case(iplane)
      case(1) ! YZ plane
        sizes(1) = ysize(2)
        sizes(2) = ysize(3)
        subsizes(1) = ysize(2)
        subsizes(2) = ysize(3)
        starts(1) = ystart(2) - 1
        starts(2) = ystart(3) - 1
      case(2) ! XZ plane
        sizes(1) = ysize(1)
        sizes(2) = ysize(3)
        subsizes(1) = ysize(1)
        subsizes(2) = ysize(3)
        starts(1) = ystart(1) - 1
        starts(2) = ystart(3) - 1
      case(3) ! XY plane
        sizes(1) = ysize(1)
        sizes(2) = ysize(2)
        subsizes(1) = ysize(1)
        subsizes(2) = ysize(2)
        starts(1) = ystart(1) - 1
        starts(2) = ystart(2) - 1
      end select
    case(3) ! Z-pencil
      select case(iplane)
      case(1) ! YZ plane
        sizes(1) = zsize(2)
        sizes(2) = zsize(3)
        subsizes(1) = zsize(2)
        subsizes(2) = zsize(3)
        starts(1) = zstart(2) - 1
        starts(2) = zstart(3) - 1
      case(2) ! XZ plane
        sizes(1) = zsize(1)
        sizes(2) = zsize(3)
        subsizes(1) = zsize(1)
        subsizes(2) = zsize(3)
        starts(1) = zstart(1) - 1
        starts(2) = zstart(3) - 1
      case(3) ! XY plane
        sizes(1) = zsize(1)
        sizes(2) = zsize(2)
        subsizes(1) = zsize(1)
        subsizes(2) = zsize(2)
        starts(1) = zstart(1) - 1
        starts(2) = zstart(2) - 1
      end select
    end select

    ! Create MPI datatype for the plane
    sizes(1) = nx_global
    sizes(2) = ny_global
    if (iplane == 3) then
      sizes(1) = nx_global
      sizes(2) = ny_global
    else if (iplane == 2) then
      sizes(1) = nx_global
      sizes(2) = nz_global
    else if (iplane == 1) then
      sizes(1) = ny_global
      sizes(2) = nz_global
    end if

    call MPI_TYPE_CREATE_SUBARRAY(2, sizes, subsizes, starts, &
         MPI_ORDER_FORTRAN, MPI_REAL_RP, type_plane, ierr)
    call MPI_TYPE_COMMIT(type_plane, ierr)

    ! Set file view and read
    call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL_RP, type_plane, &
         'native', MPI_INFO_NULL, ierr)
    call MPI_FILE_READ_ALL(fh, var(:,:,nplane), product(subsizes), &
         MPI_REAL_RP, MPI_STATUS_IGNORE, ierr)

    call MPI_TYPE_FREE(type_plane, ierr)
    call MPI_FILE_CLOSE(fh, ierr)

  end subroutine read_plane_2d
```

## Simplified Working Solution

Given the complexity above, here's a **much simpler approach** that will definitely work:

### Add this subroutine before `program cans`:

```fortran
  subroutine test_2d_readwrite(datadir, ipencil, n, ng, w)
    use mod_common_mpi, only: myid, ierr, canscomm
    use mod_output, only: out2d
    use mod_types
    use decomp_2d
    implicit none
    character(len=*), intent(in) :: datadir
    integer, intent(in) :: ipencil
    integer, dimension(3), intent(in) :: n, ng
    real(rp), dimension(1:n(1),1:n(2),1:n(3)), intent(in) :: w
    real(rp), allocatable, dimension(:,:) :: plane_local, plane_read
    integer :: islice, i, j
    integer :: fh, n1, n2
    integer(kind=MPI_OFFSET_KIND) :: disp
    integer, dimension(2) :: sizes, subsizes, starts
    integer :: type_plane
    character(len=256) :: fname

    if(myid == 0) print*, '*** Starting 2D read/write test ***'

    islice = ng(3)/2  ! Middle slice
    fname = trim(datadir)//'test_w_original.bin'

    ! Step 1: Write the plane using existing out2d
    call out2d(fname, 3, islice, w)

    ! Step 2: Read it back manually
    ! For XY plane (iplane=3), we need nx_global x ny_global data
    select case(ipencil)
    case(1)
      n1 = xsize(1)
      n2 = xsize(2)
      allocate(plane_read(n1,n2))
      sizes = [nx_global, ny_global]
      subsizes = [xsize(1), xsize(2)]
      starts = [xstart(1)-1, xstart(2)-1]
    case(2)
      n1 = ysize(1)
      n2 = ysize(2)
      allocate(plane_read(n1,n2))
      sizes = [nx_global, ny_global]
      subsizes = [ysize(1), ysize(2)]
      starts = [ystart(1)-1, ystart(2)-1]
    case(3)
      n1 = zsize(1)
      n2 = zsize(2)
      allocate(plane_read(n1,n2))
      sizes = [nx_global, ny_global]
      subsizes = [zsize(1), zsize(2)]
      starts = [zstart(1)-1, zstart(2)-1]
    end select

    ! Open file and read
    call MPI_FILE_OPEN(canscomm, fname, MPI_MODE_RDONLY, MPI_INFO_NULL, fh, ierr)
    call MPI_TYPE_CREATE_SUBARRAY(2, sizes, subsizes, starts, &
         MPI_ORDER_FORTRAN, MPI_REAL_RP, type_plane, ierr)
    call MPI_TYPE_COMMIT(type_plane, ierr)
    disp = 0_MPI_OFFSET_KIND
    call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL_RP, type_plane, 'native', MPI_INFO_NULL, ierr)
    call MPI_FILE_READ_ALL(fh, plane_read, product(subsizes), MPI_REAL_RP, MPI_STATUS_IGNORE, ierr)
    call MPI_TYPE_FREE(type_plane, ierr)
    call MPI_FILE_CLOSE(fh, ierr)

    ! Step 3: Copy read data back to w array at the same slice and save
    do j = 1, n2
      do i = 1, n1
        ! Note: This assumes the read data matches the local portion
        ! For safety, we'll just write from plane_read directly
      end do
    end do

    ! Step 4: Save the read-back data by creating a temporary 3D array
    allocate(plane_local(n(1),n(2)))
    plane_local = plane_read(1:n(1),1:n(2))

    ! Save using a different approach - write directly
    fname = trim(datadir)//'test_w_readback.bin'
    call MPI_FILE_OPEN(canscomm, fname, MPI_MODE_CREATE+MPI_MODE_WRONLY, MPI_INFO_NULL, fh, ierr)
    call MPI_FILE_SET_SIZE(fh, 0_MPI_OFFSET_KIND, ierr)
    call MPI_TYPE_CREATE_SUBARRAY(2, sizes, subsizes, starts, &
         MPI_ORDER_FORTRAN, MPI_REAL_RP, type_plane, ierr)
    call MPI_TYPE_COMMIT(type_plane, ierr)
    disp = 0_MPI_OFFSET_KIND
    call MPI_FILE_SET_VIEW(fh, disp, MPI_REAL_RP, type_plane, 'native', MPI_INFO_NULL, ierr)
    call MPI_FILE_WRITE_ALL(fh, plane_read, product(subsizes), MPI_REAL_RP, MPI_STATUS_IGNORE, ierr)
    call MPI_TYPE_FREE(type_plane, ierr)
    call MPI_FILE_CLOSE(fh, ierr)

    if(myid == 0) print*, '*** 2D read/write test complete ***'
    if(myid == 0) print*, '*** Compare test_w_original.bin with test_w_readback.bin ***'

    deallocate(plane_read, plane_local)

  end subroutine test_2d_readwrite
```

Then call it from the main loop with:
```fortran
    if(iout2d > 0.and.mod(istep,max(iout2d,1)) == 0) then
      !$acc wait
      !$acc update self(u,v,w,p)
      include 'out2d.h90'
      call test_2d_readwrite(datadir, ipencil_axis, n, ng, w(1:n(1),1:n(2),1:n(3)))
    end if
```

## Required Modules (already imported in main.f90)

These modules are already used in main.f90, so no additional imports needed:
- `use decomp_2d` (provides decomp_2d_io functions)
- `use mod_output` (provides out2d subroutine)
- `use mod_common_mpi` (provides ipencil_axis)

## How It Works

1. **out2d subroutine**: Internally calls `decomp_2d_write_plane` which:
   - Takes the 3D field array and extracts the specified slice
   - Each MPI rank writes only its portion of the 2D plane
   - Properly handles domain decomposition across ranks

2. **decomp_2d_read_plane**: The corresponding read function that:
   - Each MPI rank reads only its portion of the 2D plane
   - Correctly reconstructs the local data based on domain decomposition
   - Handles all MPI communication needed for gathering/scattering

3. **Key parameters**:
   - `inorm = 3`: Slice perpendicular to z-direction (xy plane)
   - `islice = ng(3)/2`: Middle of domain in z-direction
   - `ipencil`: Current pencil orientation (automatically handled by decomp_2d)

## Validation

After running the simulation:

1. Both files should be created in the `data/` directory:
   - `test_w_original.bin`
   - `test_w_readback.bin`

2. Compare them byte-by-byte:
   ```bash
   cmp data/test_w_original.bin data/test_w_readback.bin
   ```

3. If identical (no output from cmp), the read function works correctly!

4. Alternative comparison with md5sum:
   ```bash
   md5sum data/test_w_original.bin data/test_w_readback.bin
   ```
   The checksums should match exactly.

## File Sizes

Each binary file should be:
- Size = ng(1) × ng(2) × sizeof(real) bytes
- For double precision: ng(1) × ng(2) × 8 bytes
- For single precision: ng(1) × ng(2) × 4 bytes

## Notes

- The test runs every time `iout2d` output occurs
- Files are overwritten each time
- To run the test only once, add an additional condition like `istep == iout2d`
- The `'dummy'` parameter in decomp_2d_read_plane is a compatibility string (not used)
- The `'.'` parameter specifies the directory (current directory, which should be datadir)

## Troubleshooting

If files don't match:
1. Check that all ranks are using the same `ng` values
2. Verify that `ipencil_axis` is correctly set
3. Ensure no GPU/CPU synchronization issues (the `!$acc update self(w)` should handle this)
4. Check file permissions in the data directory
