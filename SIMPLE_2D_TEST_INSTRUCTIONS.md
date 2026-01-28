# Simple 2D Slice Test - Quick Instructions

## What This Does
Tests that you can correctly read 2D binary slices regardless of MPI parallelization.
Saves a W-velocity slice, reads it back, and saves it again for byte-by-byte comparison.

## Installation (2 steps)

### Step 1: Add import to main.f90

Add this line with the other `use` statements at the top of main.f90 (around line 49):

```fortran
  use mod_test_2d_io
```

Should look like:
```fortran
  use mod_output         , only: out0d,gen_alias,out1d,out1d_chan,out2d,out3d,write_log_output,write_visu_2d,write_visu_3d
  use mod_test_2d_io     ! <-- ADD THIS LINE
```

### Step 2: Add test call in main.f90

Inside the time loop, add this ONE line after the existing 2D output section (around line 700):

```fortran
    if(iout2d > 0.and.mod(istep,max(iout2d,1)) == 0) then
      !$acc wait
      !$acc update self(u,v,w,p)
      include 'out2d.h90'
      call test_2d_slice_io(datadir, ipencil_axis, n, ng, w(1:n(1),1:n(2),1:n(3)))  ! <-- ADD THIS LINE
    end if
```

**IMPORTANT**: You need `ipencil_axis` imported! Add it to line 38:
```fortran
  use mod_common_mpi     , only: myid,ourid,ierr,parentcomm,intracomm,group,cansgroup,canscomm,mysize,oursize,ipencil_axis
```

## Compile and Run

The new `src/test_2d_io.f90` file will be automatically compiled by your Makefile.

```bash
make
```

Then run normally. When 2D output occurs, you'll see:
```
*** [TEST] Starting 2D slice read/write test ***
*** [TEST] Original slice written to: data/test_w_original.bin
*** [TEST] Slice read back successfully
*** [TEST] Read-back slice written to: data/test_w_readback.bin
*** [TEST] Complete! Compare files to verify:
    cmp data/test_w_original.bin data/test_w_readback.bin
```

## Verify It Works

After running, compare the two files:
```bash
cmp data/test_w_original.bin data/test_w_readback.bin
```

If there's **no output**, the files are identical = SUCCESS! ✓

If they differ, you'll see an error message.

## That's It!

- **1 new file**: `src/test_2d_io.f90` (already created)
- **2 lines added** to `main.f90`:
  - 1 `use` statement
  - 1 function call

Done!
