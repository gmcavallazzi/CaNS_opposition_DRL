import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import rc
import time
from mpi4py import MPI

# Enable LaTeX formatting
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('axes', labelsize=12)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)
rc('legend', fontsize=10)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

def read_single_field_binary(filenamei, iskip=None):
    """
    Read single field binary file from CaNS simulation
    
    Parameters:
    - filenamei: filename of the binary field
    - iskip: data sampling interval (default: [1,1,1])
    
    Returns:
    - data: 3D numpy array of the field
    - grid coordinates
    """
    # Default parameters
    iprecision = 8            # precision of the real-valued data
    r0 = np.array([0.,0.,0.]) # domain origin
    non_uniform_grid = True
    
    # Set precision
    precision = 'float64' if iprecision == 8 else 'float32'
    
    # Read geometry file
    geofile = "geometry.out"
    geo = np.loadtxt(geofile, comments="!", max_rows=2)
    ng = geo[0,:].astype('int')
    l = geo[1,:]
    dl = l / (1. * ng)
    
    # Create grid coordinates
    xp = np.arange(r0[0]+dl[0]/2., r0[0]+l[0], dl[0])  # centered x grid
    yp = np.arange(r0[1]+dl[1]/2., r0[1]+l[1], dl[1])  # centered y grid
    zp = np.arange(r0[2]+dl[2]/2., r0[2]+l[2], dl[2])  # centered z grid
    
    # Staggered grid coordinates
    xu = xp + dl[0]/2.
    yv = yp + dl[1]/2.
    zw = zp + dl[2]/2.
    
    # Non-uniform grid handling
    if non_uniform_grid:
        with open('grid.bin', 'rb') as f:
            grid_z = np.fromfile(f, dtype=precision)
        grid_z = np.reshape(grid_z, (ng[2], 4), order='F')
        zp = r0[2] + np.transpose(grid_z[:,2])  # centered z grid
        zw = r0[2] + np.transpose(grid_z[:,3])  # staggered z grid
    
    # Set default iskip if not provided
    if iskip is None:
        iskip = np.ones(3, dtype=int)
    
    # Compute output grid dimensions
    n = (ng[:] / iskip[:]).astype(int)
    
    # Read binary file
    with open(filenamei, 'rb') as f:
        fld = np.fromfile(f, dtype=precision)
    
    # Reshape data (Fortran order)
    data = np.reshape(fld, (n[0], n[1], n[2]), order='F')
    
    # Subsample grid coordinates
    xp = xp[0:ng[0]:iskip[0]]
    yp = yp[0:ng[1]:iskip[1]]
    zp = zp[0:ng[2]:iskip[2]]
    xu = xu[0:ng[0]:iskip[0]]
    yv = yv[0:ng[1]:iskip[1]]
    zw = zw[0:ng[2]:iskip[2]]
    
    return data, xp, yp, zp, xu, yv, zw

def compute_2d_spectrum(velocity_data, z_idx, dimensions, re_tau):
    """Compute 2D pre-multiplied energy spectrum"""
    # Extract the z-plane
    plane = velocity_data[:, :, z_idx]
    
    # Compute FFT for each snapshot first
    fft_data = np.fft.rfft2(plane)
    
    # Compute power spectrum with proper normalization
    power = np.abs(fft_data)**2 / (plane.shape[0] * plane.shape[1])**2
    
    # Calculate wavenumbers in plus units
    Lx_plus = dimensions[0] * re_tau
    Ly_plus = dimensions[1] * re_tau
    
    # Create wavenumber arrays
    kx = np.fft.fftfreq(plane.shape[0], d=1.0/plane.shape[0]) * (2*np.pi/Lx_plus)
    ky = np.fft.rfftfreq(plane.shape[1], d=1.0/plane.shape[1]) * (2*np.pi/Ly_plus)
    
    # Create meshgrid for multiplying with power spectrum
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    # Pre-multiplied spectrum
    premult_spectrum = KX * KY * power
    
    return kx, ky, premult_spectrum

def process_file(filename, target_w_plus, dimensions, re_tau):
    """Process a single velocity file and compute its 2D spectrum"""
    try:
        # Read velocity data
        velocity_data, _, _, zp, _, _, _ = read_single_field_binary(filename)
        
        # Compute z+ values
        z_plus = zp * re_tau
        
        # Find z-index closest to target
        z_idx = np.argmin(np.abs(z_plus - target_w_plus))
        
        # Compute 2D spectrum
        kx, ky, spectrum = compute_2d_spectrum(
            velocity_data, z_idx, dimensions, re_tau
        )
        
        return {
            'kx': kx,
            'ky': ky,
            'spectrum': spectrum,
            'z_plus': z_plus[z_idx]
        }
        
    except Exception as e:
        print(f"  Error processing {filename}: {str(e)}")
        return None

def plot_2d_spectrum(kx, ky, Lx_plus, Ly_plus, spectrum, component, z_plus, save_path=None):
    """Plot 2D pre-multiplied energy spectrum using wavelengths with enhanced formatting"""
    # Create figure with higher DPI for better quality
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Convert wavenumbers to wavelengths
    lambda_x = 2*np.pi / np.maximum(np.abs(kx), 1e-10)
    lambda_y = 2*np.pi / np.maximum(np.abs(ky), 1e-10)
    
    # Create meshgrid for plotting
    LX, LY = np.meshgrid(lambda_x, lambda_y, indexing='ij')
    
    # Set axes to log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Find appropriate contour levels
    if np.max(spectrum) > 0:
        max_val = np.max(spectrum)
        min_val = max(np.min(spectrum[spectrum > 0]), max_val * 1e-3)
        
        # Create more contour levels for smoother appearance
        levels = np.logspace(np.log10(min_val), np.log10(max_val), 15)
        
        # Define a better colormap
        cmap = plt.cm.viridis
        
        # Draw contours with smoother transitions
        cs = ax.contourf(LX, LY, spectrum, levels=levels, 
                       cmap=cmap, locator=plt.LogLocator(), alpha=0.95)
        
        # Add contour lines for better visualization
        contour_lines = ax.contour(LX, LY, spectrum, levels=levels[::3], 
                                 colors='k', alpha=0.2, linewidths=0.5)
        
        # Find the global maximum in the spectrum
        max_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
        peak_x_idx, peak_y_idx = max_idx
        
        peak_lambda_x = LX[peak_x_idx, peak_y_idx]
        peak_lambda_y = LY[peak_x_idx, peak_y_idx]
        
        # Add a triangle marker at the peak location
        ax.scatter(peak_lambda_x, peak_lambda_y, s=200, marker='^', color='red', 
                  edgecolor='black', zorder=10)
        
        # Set improved labels with LaTeX formatting
        ax.set_xlabel(r'$\lambda_x^+$', fontsize=14)
        ax.set_ylabel(r'$\lambda_y^+$', fontsize=14)
        
        # Create component label for title
        if component == 'uu':
            comp_label = 'u^+u^+'
        elif component == 'vv':
            comp_label = 'v^+v^+'
        elif component == 'ww':
            comp_label = 'w^+w^+'
        else:
            comp_label = component
            
        # Set title with LaTeX formatting
        ax.set_title(r'$\kappa_x \kappa_y \phi_{' + comp_label + '}$ at $z^+ = ' + f'{z_plus:.1f}$', 
                    fontsize=16)
        
        # Set axis limits to match data
        ax.set_xlim(lambda_x.min(), Lx_plus)
        ax.set_ylim(lambda_y.min(), Ly_plus)
        
        # Enhance grid
        ax.grid(True, alpha=0.3, linestyle='--', which='both', color='gray')
        
        # Improve tick formatting
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Add background color to improve contrast
        ax.set_facecolor('#f8f8f8')
        
        # Add box around the plot
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    else:
        print(f"Warning: No positive values in spectrum")
        return None, None

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Start timer (only on rank 0)
    if rank == 0:
        start_time = time.time()
        print(f"Running with {size} MPI processes")
    
    # Define the geometry parameters (read on all ranks)
    try:
        data = np.loadtxt("geometry.out", comments="!", max_rows=2)
        ng = data[0, :].astype('int')
        dimensions = data[1, :]
    except:
        print(f"Rank {rank}: Using default dimensions")
        ng = np.array([384, 192, 100])  # Grid size (nx, ny, nz)
        dimensions = np.array([25.13, 6.28, 1.0])  # Box dimensions
    
    if rank == 0:
        print(f"Grid dimensions: {ng}")
        print(f"Domain dimensions: {dimensions}")
    
    # Parameters
    re_tau = 1083.0  # Reynolds number
    target_w_plus = 100.0  # Target wall-normal location
    
    # Calculate domain sizes in plus units
    Lx_plus = dimensions[0] * re_tau
    Ly_plus = dimensions[1] * re_tau
    
    # Rank 0 collects file list and distributes work
    if rank == 0:
        # Find velocity files
        u_files = sorted(glob.glob("vex_fld_*.bin"))
        v_files = sorted(glob.glob("vey_fld_*.bin"))
        w_files = sorted(glob.glob("vez_fld_*.bin"))
        
        if not u_files:
            print("No velocity files found! Make sure you're in the correct directory.")
            # Send empty lists to all ranks to signal termination
            for r in range(1, size):
                comm.send([], dest=r, tag=10)  # u_files
                comm.send([], dest=r, tag=11)  # v_files
                comm.send([], dest=r, tag=12)  # w_files
            return
        
        print(f"Found {len(u_files)} x-velocity files")
        print(f"Found {len(v_files)} y-velocity files")
        print(f"Found {len(w_files)} z-velocity files")
        
        # Distribute files evenly among ranks
        u_chunks = [[] for _ in range(size)]
        v_chunks = [[] for _ in range(size)]
        w_chunks = [[] for _ in range(size)]
        
        # Distribute files
        for i, f in enumerate(u_files):
            u_chunks[i % size].append(f)
        for i, f in enumerate(v_files):
            v_chunks[i % size].append(f)
        for i, f in enumerate(w_files):
            w_chunks[i % size].append(f)
        
        # Keep rank 0's chunks
        my_u_files = u_chunks[0]
        my_v_files = v_chunks[0]
        my_w_files = w_chunks[0]
        
        # Send chunks to other ranks
        for r in range(1, size):
            comm.send(u_chunks[r], dest=r, tag=10)
            comm.send(v_chunks[r], dest=r, tag=11)
            comm.send(w_chunks[r], dest=r, tag=12)
    else:
        # Receive file chunks from rank 0
        my_u_files = comm.recv(source=0, tag=10)
        my_v_files = comm.recv(source=0, tag=11)
        my_w_files = comm.recv(source=0, tag=12)
    
    # Process files assigned to this rank
    local_results = {'uu': [], 'vv': [], 'ww': []}
    
    # Process components
    for component, files in [('uu', my_u_files), ('vv', my_v_files), ('ww', my_w_files)]:
        if rank == 0:
            print(f"\nRank {rank}: Processing {len(files)} {component}-files...")
        
        # Process files
        for i, filename in enumerate(files):
            if rank == 0 and i % 5 == 0:  # Print progress less frequently
                progress = (i + 1) / len(files) * 100
                print(f"Rank {rank}: Processing {os.path.basename(filename)} ({progress:.1f}%)")
            
            # Process file
            result = process_file(filename, target_w_plus, dimensions, re_tau)
            if result is not None:
                local_results[component].append(result)
    
    # Gather all results to rank 0
    if rank == 0:
        # Initialize with rank 0's results
        all_results = {'uu': local_results['uu'], 'vv': local_results['vv'], 'ww': local_results['ww']}
        
        # Receive from other ranks
        for r in range(1, size):
            remote_results = comm.recv(source=r, tag=20)
            all_results['uu'].extend(remote_results['uu'])
            all_results['vv'].extend(remote_results['vv'])
            all_results['ww'].extend(remote_results['ww'])
    else:
        # Send results to rank 0
        comm.send(local_results, dest=0, tag=20)
    
    # Rank 0 averages results and generates plots
    if rank == 0:
        # Process each component
        for component in ['uu', 'vv', 'ww']:
            results = all_results[component]
            
            if not results:
                print(f"No valid results for {component}!")
                continue
            
            print(f"\nAveraging {len(results)} {component} spectra...")
            
            # Extract data from first result to initialize
            kx = results[0]['kx']
            ky = results[0]['ky']
            shape = results[0]['spectrum'].shape
            
            # Initialize arrays for averaging
            avg_spectrum = np.zeros(shape)
            z_plus_values = []
            
            # Add all results
            for res in results:
                if res['spectrum'].shape == shape:
                    avg_spectrum += res['spectrum']
                    z_plus_values.append(res['z_plus'])
            
            # Average the spectrum
            avg_spectrum /= len(results)
            avg_z_plus = np.mean(z_plus_values)
            
            print(f"Average z+ = {avg_z_plus:.2f}")
            
            # Plot and save
            fig, ax = plot_2d_spectrum(
                kx, ky, Lx_plus, Ly_plus, avg_spectrum, component, avg_z_plus,
                save_path=f"2d_spectrum_{component}_averaged.png"
            )
            
            if fig is not None:
                plt.close(fig)
    
        # Print execution time
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
