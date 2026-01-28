# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**CaNS (Canonical Navier-Stokes Solver)** - A Direct Numerical Simulation (DNS) solver for 3D incompressible Navier-Stokes equations with traveling wave boundary conditions for wall-bounded turbulence.

This is a modified version of CaNS with streamwise-traveling wave boundary conditions applied to the wall-normal velocity component.

## Code Structure

### Core Time-Stepping Loop (main.f90)

The main simulation loop uses 3rd-order low-storage Runge-Kutta time integration with three substeps per timestep:
1. **Momentum advance** (`mod_rk`, `mod_mom`) - advection + diffusion terms
2. **Pressure Poisson solve** (`mod_fillps`, `mod_solver`) - ensures incompressibility
3. **Velocity correction** (`mod_correc`) - projects velocity field to divergence-free
4. **Wave boundary condition** (`mod_wave`) - applies traveling wave BC at walls

### Key Physics Modules

- **mod_rk**: Runge-Kutta time integration kernel
- **mod_mom**: Momentum equation kernels with heavy OpenACC/OpenMP parallelization
  - Can operate in full 3D mode (`mom_xyz_ad`) or split 1D mode for implicit diffusion
- **mod_correc**: Velocity correction to enforce continuity
- **mod_fillps**: Constructs RHS of Poisson equation for pressure
- **mod_wave**: Traveling wave boundary condition implementation
  - Applies sinusoidal wave pattern to wall-normal velocity at both walls
  - Wave form: w = A*sin(k*x - ω*t) at z=0 and z=nz

### Poisson Solver Architecture

The pressure Poisson solver uses FFT-based spectral methods with 2D pencil decomposition:

1. **Spatial transforms** via `mod_solver` or `mod_solver_gpu`:
   - x-pencil → FFT in x direction
   - Transpose to y-pencil → FFT in y direction
   - Transpose to z-pencil
   - Gaussian elimination in z (tridiagonal solve on eigenvalues)
   - Inverse FFTs back to physical space

2. **Parallelization**:
   - CPU: Uses `decomp_2d` library with FFTW3
   - GPU: Uses `cuDecomp` with cuFFT for GPU-aware communication

3. **Implicit diffusion** (optional, `_IMPDIFF` flag):
   - Reuses same solver infrastructure for implicit viscous terms
   - Can be full 3D or 1D (z-direction only, `_IMPDIFF_1D`)

### Boundary Conditions

Controlled via character codes in `input.nml`:
- `'P'` = Periodic
- `'D'` = Dirichlet
- `'N'` = Neumann

Format: `cbcvel(lower:upper, x:y:z, u:v:w)` for velocity, `cbcpre(lower:upper, x:y:z)` for pressure

### Domain Decomposition

Uses 2D pencil decomposition where:
- MPI distributes work across x-y planes (controlled by `dims` in `input.nml`)
- z-direction remains contiguous for efficient tridiagonal solves
- Halo exchanges occur during finite difference stencil operations
- FFT transpositions require global communication patterns

## Configuration Files

### input.nml

Main physics and simulation parameters (namelist `&dns`):

**Grid & Domain**:
- `ng(1:3)` - Global grid resolution [nx, ny, nz]
- `l(1:3)` - Physical domain size
- `gtype`, `gr` - Grid stretching type and ratio

**Time Stepping**:
- `cfl` - CFL number for stability
- `dtmax` - Maximum allowed timestep
- `dt_f` - Fixed timestep (negative = adaptive)
- `visci` - Inverse viscosity (= Reynolds number for channel flow)

**Simulation Control**:
- `nstep` - Maximum timesteps
- `time_max` - Maximum simulation time
- `stop_type(1:3)` - Enable stopping criteria [nstep, time_max, wall_clock]
- `icheck` - Frequency for stability/divergence checks
- `isave` - Checkpoint save frequency

**Initial Conditions**:
- `inivel` - Initial velocity field type (e.g., `'poi'` for Poiseuille)
- `is_wallturb` - Initialize with wall turbulence perturbations
- `restart` - Load from checkpoint if `.true.`

**Forcing**:
- `is_forced(1:3)` - Enable constant bulk velocity forcing [x,y,z]
- `velf(1:3)` - Target bulk velocities
- `bforce(1:3)` - Constant body force (alternative to bulk forcing)

**Parallelization**:
- `dims(1:2)` - MPI processor grid [x-pencils, y-pencils] (0 = auto)

**GPU Options** (namelist `&cudecomp`):
- `cudecomp_t_comm_backend` - Transpose communication backend
- `cudecomp_is_t_enable_nccl` - Enable NCCL for GPU-direct communication
- `cudecomp_is_t_enable_nvshmem` - Enable NVSHMEM

### wave.nml

Traveling wave boundary condition parameters (namelist `&wave`):
- `wave_amplitude` - Wave amplitude A (in physical units)
- `wave_k` - Wavenumber k = 2π/λ (in rad/length units)
- `wave_omega` - Angular frequency ω (in rad/time units)

## Build Configuration

Compile-time features controlled via preprocessor flags:

- `_OPENACC` - Enable GPU acceleration via OpenACC
- `_IMPDIFF` - Enable implicit diffusion solver
- `_IMPDIFF_1D` - Restrict implicit diffusion to z-direction only
- `_TIMING` - Enable detailed performance timing
- `_DEBUG` - Enable debug output
- `_DEBUG_SOLVER` - Test solver correctness at initialization

## Key Computational Patterns

### OpenACC GPU Directives

- **Data regions**: `!$acc enter data copyin(...)` for persistent device arrays
- **Async execution**: `async(1)` for stream-based overlap of computation/communication
- **Update clauses**: `!$acc update self(...)` to sync host before I/O
- **Kernels**: `!$acc kernels default(present)` for automatic parallelization

### MPI Communication Patterns

- **Halo exchanges**: Via `decomp_2d` update_halo routines during spatial derivatives
- **Global transpositions**: During FFT-based Poisson solve (pencil reordering)
- **Reductions**: For global diagnostics (bulk velocity, divergence norms, etc.)

### Wave Boundary Condition Implementation

Located in `mod_wave` (mod_wave.f90:72-99):

The traveling wave BC is applied at both walls (z=0 and z=nz) during the `bounduvw` call when the `pfix` flag is present.

**Wave formula**: w(x,t) = A*sin(k*x - ω*t)

**MPI-safe implementation**:
- Computes global x-coordinate from local indices: `x = (lo(1) + i - 1.5) * dl(1)`
- `lo(1)` provides the global starting index for this MPI rank's subdomain
- Each rank independently computes its portion of the wave pattern
- Only ranks with physical wall boundaries (`is_bound(0,3)` or `is_bound(1,3)`) apply the BC

## Common Development Workflow

### Modifying Physics

When adding new forcing terms or modifying momentum equations:
1. Edit `mod_mom` for new physics kernels
2. Update `mod_rk` if coupling to other fields required
3. Add parameters to `mod_param` and `input.nml` if configuration needed
4. Ensure OpenACC directives added for GPU compatibility (`!$acc kernels` or `!$acc parallel loop`)

### Modifying Wave Boundary Conditions

The wave BC logic is in `mod_wave`:
- `read_wave` (mod_wave.f90:17-55) - Reads wave parameters from wave.nml
- `apply_wave_bc` (mod_wave.f90:57-97) - Applies traveling wave BC at walls

Called from `bounduvw` in bound.f90:85 when `pfix` flag is present.

To modify wave parameters, edit wave.nml. To change the wave formula, modify the `apply_wave_bc` subroutine.

### Adding Boundary Conditions

Boundary condition logic is in `mod_bound`:
- `bounduvw` - Applies velocity BCs
- `boundp` - Applies pressure BCs
- `updt_rhs_b` - Updates RHS for implicit solvers with BC contributions

### File I/O Conventions

- `data/` directory (controlled by `datadir` in input.nml) for all output
- Binary checkpoint format: `fld.bin` or `fld_XXXXXXX.bin`
- Checkpoint contains: u,v,w,p fields + time + istep
- ASCII diagnostics: `time.out`, `forcing.out`, `tau.out`
- Post-processing: 1D profiles (out1d.h90), 2D slices (out2d.h90), 3D fields (out3d.h90)

## Important Notes for Code Modification

1. **Maintain OpenACC compatibility**: Any new arrays must have corresponding `!$acc enter data` directives and be declared `present` in compute regions

2. **Preserve low-storage RK pattern**: Avoid creating temporary field copies; reuse existing arrays where possible

3. **Boundary condition order matters**: Always call `bounduvw` after velocity updates and `boundp` after pressure updates

4. **Grid indexing**: Arrays use 0-based indexing with ghost cells: `dimension(0:n(1)+1, 0:n(2)+1, 0:n(3)+1)`

5. **Staggered grid**: Velocity components live at cell faces, pressure at cell centers. Indexing in `mod_mom` accounts for this.

6. **FFT solver assumptions**: Requires at least one periodic direction for spectral accuracy. Non-periodic directions use finite differences in z.

7. **Wave BC timing**: The wave BC is applied after pressure correction (when `pfix` flag is present) to ensure it's the final boundary value applied before the next RK substep.
