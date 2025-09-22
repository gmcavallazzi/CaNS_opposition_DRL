import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import rc
import time
import pickle
from mpi4py import MPI

# Enable LaTeX formatting
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('axes', labelsize=12)
rc('xtick', labelsize=10)
rc('ytick', labelsize=10)
rc('legend', fontsize=10)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Configuration
iprecision = 8  # Precision of real-valued data
my_dtype = 'float64' if iprecision == 8 else 'float32'

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

def calculate_w_plus(w, re_tau=1083.0):
    """Calculate w+ values (wall units for wall-normal coordinate)"""
    return w * re_tau

def compute_xz_spectrum(velocity_data, direction='x', dimensions=None, re_tau=1083.0):
    """
    Compute spectrum in x or y direction for all z-planes with correct physical scaling
    
    Parameters:
    -----------
    velocity_data : ndarray
        Velocity field data (nx, ny, nz)
    direction : str
        Direction for FFT computation ('x' or 'y')
    dimensions : ndarray
        Physical dimensions of the domain (Lx, Ly, Lz)
    re_tau : float
        Reynolds number based on friction velocity
    
    Returns:
    --------
    wavenumbers : ndarray
        Wavenumber array along the specified direction (in plus units)
    spectra : ndarray
        Spectrum for each z-plane
    """
    # Get data dimensions - now data is (nx, ny, nz)
    nx, ny, nz = velocity_data.shape
    
    # Set default dimensions if not provided
    if dimensions is None:
        dimensions = np.array([2*np.pi, np.pi, 2])
    
    # Choose axis and domain size based on direction
    if direction == 'x':
        axis = 0
        n = nx
        L = dimensions[0]  # Domain length in x
    elif direction == 'y':
        axis = 1
        n = ny
        L = dimensions[1]  # Domain length in y
    else:
        raise ValueError("Direction must be 'x' or 'y'")
    
    # Convert domain length to plus units
    L_plus = L * re_tau
    
    # Calculate fundamental wavenumber (in plus units)
    k0_plus = 2 * np.pi / L_plus
    
    # Calculate wavenumbers with proper scaling (in plus units)
    # The FFT gives modes 0, 1, 2, ..., n/2
    wavenumbers = np.fft.rfftfreq(n) * n * k0_plus
    
    # Initialize array to store spectra for each z-plane
    n_fft = n // 2 + 1  # Number of FFT points in rfft
    spectra = np.zeros((n_fft, nz))
    
    # Compute spectrum for each z-plane
    for iz in range(nz):
        if direction == 'x':
            # Take data at this z-plane
            # For each y, compute FFT along x
            fft_data = np.zeros((n_fft, ny), dtype=complex)
            for iy in range(ny):
                # Extract 1D array for this y and z
                line_data = velocity_data[:, iy, iz]
                # Compute FFT along x
                fft_data[:, iy] = np.fft.rfft(line_data)
            
            # Compute power spectrum, average over y direction
            power = np.mean(np.abs(fft_data)**2, axis=1)
            
            # Pre-multiply by wavenumber and store for this z-plane
            spectra[:, iz] = wavenumbers * power
            
        elif direction == 'y':
            # Take data at this z-plane
            # For each x, compute FFT along y
            fft_data = np.zeros((nx, n_fft), dtype=complex)
            for ix in range(nx):
                # Extract 1D array for this x and z
                line_data = velocity_data[ix, :, iz]
                # Compute FFT along y
                fft_data[ix, :] = np.fft.rfft(line_data)
            
            # Compute power spectrum, average over x direction
            power = np.mean(np.abs(fft_data)**2, axis=0)
            
            # Pre-multiply by wavenumber and store for this z-plane
            spectra[:, iz] = wavenumbers * power
    
    return wavenumbers, spectra

def process_file(filename, component, dimensions, re_tau=1083.0):
    """
    Process a single velocity file and compute its spectra
    
    Parameters:
    -----------
    filename : str
        Velocity file to process
    component : str
        Velocity component (u, v, w)
    dimensions : ndarray
        Physical dimensions of the domain
    re_tau : float
        Reynolds number based on friction velocity
    
    Returns:
    --------
    dict
        Dictionary containing the spectrum results
    """
    try:
        # Read velocity data
        velocity_data, _, _, z, _, _, _ = read_single_field_binary(filename)
        
        if velocity_data is None:
            return None
        
        # Calculate w+ values
        z_plus = calculate_w_plus(z, re_tau)
        
        # Compute spectra for all z-planes with correct physical scaling
        kx, spectra_x = compute_xz_spectrum(velocity_data, direction='x', dimensions=dimensions, re_tau=re_tau)
        ky, spectra_y = compute_xz_spectrum(velocity_data, direction='y', dimensions=dimensions, re_tau=re_tau)
        
        # Return result
        return {
            'kx': kx,
            'ky': ky,
            'spectra_x': spectra_x,
            'spectra_y': spectra_y,
            'z_plus': z_plus
        }
        
    except Exception as e:
        print(f"  Error processing {filename}: {str(e)}")
        return None

def combine_spectra_safely(results):
    """Safely combine spectrum results from multiple files"""
    if not results:
        return None
    
    # Initialize accumulators
    combined = {
        'kx': results[0]['kx'],
        'spectra_x': np.zeros_like(results[0]['spectra_x']),
        'ky': results[0]['ky'],
        'spectra_y': np.zeros_like(results[0]['spectra_y']),
        'z_plus': results[0]['z_plus']
    }
    
    # Add all valid results (with matching dimensions)
    count_x = 0
    count_y = 0
    
    for r in results:
        if r is None:
            continue
            
        # Handle x-direction spectrum
        if r['spectra_x'].shape == combined['spectra_x'].shape:
            combined['spectra_x'] += r['spectra_x']
            count_x += 1
        
        # Handle y-direction spectrum
        if r['spectra_y'].shape == combined['spectra_y'].shape:
            combined['spectra_y'] += r['spectra_y']
            count_y += 1
    
    # Avoid division by zero
    if count_x > 0:
        combined['spectra_x'] /= count_x
    if count_y > 0:
        combined['spectra_y'] /= count_y
    
    print(f"Combined {count_x} x-spectra and {count_y} y-spectra")
    
    return combined

def plot_z_lambda_spectrum(results, component, direction='x', re_tau=1083.0, save_path=None):
    """
    Plot spectrum with wavelength on x-axis and z+ on y-axis using a semilogx plot
    with enhanced visual formatting
    
    Parameters:
    -----------
    results : dict
        Dictionary containing the spectrum data
    component : str
        Velocity component label (u, v, w)
    direction : str
        Direction for spectrum ('x' or 'y')
    re_tau : float
        Reynolds number based on friction velocity
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig, ax : tuple
        Figure and axes objects
    """
    # Create figure with higher DPI for better quality
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    
    # Extract data
    if direction == 'x':
        k = results['kx']
        spectra = results['spectra_x']
        wavelength_symbol = r'_x'
        wavenumber_symbol = r'_x'
    else:  # y direction
        k = results['ky']
        spectra = results['spectra_y']
        wavelength_symbol = r'_y'
        wavenumber_symbol = r'_y'
    
    z_plus = results['z_plus']
    
    # Filter out zero or very small wavenumbers for log scaling
    min_k = 1e-10
    valid = k > min_k
    
    # Extract valid data
    k_valid = k[valid]
    spectra_valid = spectra[valid, :]
    
    # Convert to wavelength (lambda = 2π/k) - wavenumbers are already in plus units
    lambda_k = 2 * np.pi / k_valid
    
    # Create meshgrid for contour plotting
    X, Y = np.meshgrid(lambda_k, z_plus)
    
    # Transpose spectra for correct orientation
    Z = spectra_valid.T
    
    # Find appropriate contour levels
    if np.max(Z) > 0:
        max_val = np.max(Z)
        min_val = np.max([np.min(Z[Z > 0]), max_val * 1e-3])
        
        # Create more contour levels for smoother appearance
        levels = np.logspace(np.log10(min_val), np.log10(max_val), 15)
        
        # Define a better colormap
        cmap = plt.cm.viridis
        
        # Draw filled contours with smoother transitions
        cs = ax.contourf(X, Y, Z, levels=levels, 
                       cmap=cmap, locator=plt.LogLocator(), alpha=0.95)
        
        # Add contour lines for better visualization
        contour_lines = ax.contour(X, Y, Z, levels=levels[::3], 
                                 colors='k', alpha=0.2, linewidths=0.5)
        
        # For y direction and high Re_tau, find both inner and outer peaks only for u-component
        if direction == 'y' and re_tau > 750 and component == 'u':
            # Define masks for small and large wavelengths
            small_lambda_mask = X < 1000  # Region where lambda_y+ < 1000
            large_lambda_mask = X >= 1000  # Region where lambda_y+ >= 1000
            
            # Create masked arrays
            Z_small = Z.copy()
            Z_large = Z.copy()
            
            # Apply masks for finding peaks in different regions
            Z_small[~small_lambda_mask] = 0
            Z_large[~large_lambda_mask] = 0
            
            # Find peaks in each region
            if np.max(Z_small) > 0:
                small_max_idx = np.unravel_index(np.argmax(Z_small), Z.shape)
                small_peak_z_idx, small_peak_lambda_idx = small_max_idx
                
                small_peak_z = Y[small_peak_z_idx, small_peak_lambda_idx]
                small_peak_lambda = X[small_peak_z_idx, small_peak_lambda_idx]
                
                # Add a triangle marker at the inner peak location
                ax.scatter(small_peak_lambda, small_peak_z, s=200, marker='^', color='red', 
                          edgecolor='black', zorder=10)
            
            if np.max(Z_large) > 0:
                large_max_idx = np.unravel_index(np.argmax(Z_large), Z.shape)
                large_peak_z_idx, large_peak_lambda_idx = large_max_idx
                
                large_peak_z = Y[large_peak_z_idx, large_peak_lambda_idx]
                large_peak_lambda = X[large_peak_z_idx, large_peak_lambda_idx]
                
                # Add a square marker at the outer peak location
                ax.scatter(large_peak_lambda, large_peak_z, s=200, marker='s', color='green', 
                          edgecolor='black', zorder=10)
                
        else:
            # For all other cases, just find the global maximum
            max_idx = np.unravel_index(np.argmax(Z), Z.shape)
            peak_z_idx, peak_lambda_idx = max_idx
            
            peak_z = Y[peak_z_idx, peak_lambda_idx]
            peak_lambda = X[peak_z_idx, peak_lambda_idx]
            
            # Add a triangle marker at the peak location
            ax.scatter(peak_lambda, peak_z, s=200, marker='^', color='red', 
                      edgecolor='black', zorder=10)
        
        # Semilog x
        ax.set_xscale('log')
        
        # Set labels with improved LaTeX formatting
        ax.set_xlabel(r'$\lambda^+' + wavelength_symbol + '$', fontsize=14)
        ax.set_ylabel(r'$z^+$', fontsize=14)
        
        # Create component label for title
        if component == 'u':
            comp_label = 'u^+u^+'
        elif component == 'v':
            comp_label = 'v^+v^+'
        elif component == 'w':
            comp_label = 'w^+w^+'
        else:
            comp_label = component
            
        # Set title with LaTeX formatting including Re_tau
        ax.set_title(r'$\kappa' + wavenumber_symbol + '\phi_{' + comp_label + '}$' + r', $Re_\tau = ' + f'{re_tau:.0f}$', 
                    fontsize=16)
        
        # Enhance grid
        ax.grid(True, alpha=0.3, linestyle='--', which='both', color='gray')
        
        # Improve tick formatting
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # Adjust layout
        plt.tight_layout()
        
        # Add background color to improve contrast
        ax.set_facecolor('#f8f8f8')
        
        # Add box around the plot
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('black')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    else:
        print(f"Warning: No positive values in spectrum for {component}-component in {direction} direction")
        return None, None
    
def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Process and plot lambda-z spectra')
    parser.add_argument('--re_tau', type=float, default=1083.0, help='Reynolds number based on friction velocity')
    parser.add_argument('--output_dir', type=str, default='lambda_z_spectra', help='Directory for saving plots')
    args = parser.parse_args()
    
    # Set Reynolds number from argument
    re_tau = args.re_tau
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Create output directories if they don't exist
    if rank == 0:
        os.makedirs('spectra_data', exist_ok=True)
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Start timer (only on rank 0)
    if rank == 0:
        start_time = time.time()
        print(f"Running with {size} MPI processes")
        print(f"Using Reynolds number Re_tau = {re_tau}")
    
    # Define the geometry parameters (read on all ranks)
    geofile = "geometry.out"
    try:
        data = np.loadtxt(geofile, comments="!", max_rows=2)
        ng = data[0, :].astype('int')
        dimensions = data[1, :]
    except:
        print(f"Rank {rank}: Warning: Could not read geometry.out, using default dimensions")
        ng = np.array([192, 192, 192])  # Default grid size
        dimensions = np.array([2*np.pi, np.pi, 2])  # Default box dimensions
    
    if rank == 0:
        print(f"Domain dimensions: {dimensions}")
    
    # Rank 0 collects file list and distributes work
    if rank == 0:
        # Find all velocity files
        vex_files = sorted(glob.glob("vex_fld_*.bin"))
        vey_files = sorted(glob.glob("vey_fld_*.bin"))
        vez_files = sorted(glob.glob("vez_fld_*.bin"))
        
        if not vex_files:
            print("No velocity files found! Make sure you're in the correct directory.")
            # Send empty lists to all ranks to signal termination
            for r in range(1, size):
                comm.send([], dest=r, tag=10)  # vex
                comm.send([], dest=r, tag=11)  # vey
                comm.send([], dest=r, tag=12)  # vez
            return
        
        print(f"Found {len(vex_files)} x-velocity files")
        print(f"Found {len(vey_files)} y-velocity files")
        print(f"Found {len(vez_files)} z-velocity files")
        
        # Distribute files evenly among ranks
        vex_chunks = [[] for _ in range(size)]
        vey_chunks = [[] for _ in range(size)]
        vez_chunks = [[] for _ in range(size)]
        
        # Distribute x-velocity files
        for i, f in enumerate(vex_files):
            vex_chunks[i % size].append(f)
        
        # Distribute y-velocity files
        for i, f in enumerate(vey_files):
            vey_chunks[i % size].append(f)
        
        # Distribute z-velocity files
        for i, f in enumerate(vez_files):
            vez_chunks[i % size].append(f)
        
        # Keep rank 0's chunks
        my_vex_files = vex_chunks[0]
        my_vey_files = vey_chunks[0]
        my_vez_files = vez_chunks[0]
        
        # Send chunks to other ranks
        for r in range(1, size):
            comm.send(vex_chunks[r], dest=r, tag=10)
            comm.send(vey_chunks[r], dest=r, tag=11)
            comm.send(vez_chunks[r], dest=r, tag=12)
    else:
        # Receive file chunks from rank 0
        my_vex_files = comm.recv(source=0, tag=10)
        my_vey_files = comm.recv(source=0, tag=11)
        my_vez_files = comm.recv(source=0, tag=12)
    
    # Process files assigned to this rank
    local_results = {'u': [], 'v': [], 'w': []}
    
    # Process components
    for component, files in [('u', my_vex_files), ('v', my_vey_files), ('w', my_vez_files)]:
        if rank == 0:
            print(f"\nRank {rank}: Processing {len(files)} {component}-velocity files...")
        
        # Process files
        for i, filename in enumerate(files):
            # Print progress for every file
            progress = (i + 1) / len(files) * 100
            print(f"Rank {rank}: Processing {os.path.basename(filename)} ({progress:.1f}%)")
            
            # Process file
            result = process_file(filename, component, dimensions, re_tau)
            if result is not None:
                local_results[component].append(result)
    
    # Combine local results for each component
    local_combined = {}
    for component in ['u', 'v', 'w']:
        if local_results[component]:
            local_combined[component] = combine_spectra_safely(local_results[component])
    
    # Gather all results to rank 0
    if rank == 0:
        # Initialize with rank 0's results
        all_results = {'u': [], 'v': [], 'w': []}
        
        # Add rank 0's combined results
        for component in ['u', 'v', 'w']:
            if component in local_combined:
                all_results[component].append(local_combined[component])
        
        # Receive from other ranks
        for r in range(1, size):
            remote_combined = comm.recv(source=r, tag=20)
            for component in ['u', 'v', 'w']:
                if component in remote_combined:
                    all_results[component].append(remote_combined[component])
    else:
        # Send results to rank 0
        comm.send(local_combined, dest=0, tag=20)
    
    # Rank 0 combines all results, saves data, and generates plots
    if rank == 0:
        final_results = {}
        
        # Combine all results for each component
        for component in ['u', 'v', 'w']:
            if all_results[component]:
                # Flatten the list of results
                flat_results = []
                for res in all_results[component]:
                    if res is not None:
                        flat_results.append(res)
                
                if flat_results:
                    final_results[component] = combine_spectra_safely(flat_results)
        
        # Save the processed data
        if final_results:
            # Add metadata
            metadata = {
                'dimensions': dimensions,
                're_tau': re_tau,
                'date_processed': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Create data to save
            save_data = {
                'metadata': metadata,
                'spectra': final_results
            }
            
            # Save to file
            output_file = os.path.join('spectra_data', f'lambda_z_spectra_re{re_tau:.0f}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(save_data, f)
            print(f"\nSaved spectra data to {output_file}")
        
        # Plot contour results for each component
        for component, results in final_results.items():
            # Skip if no results
            if results is None:
                continue
            
            # Plot λ-z+ spectrum for x-direction
            fig_x, _ = plot_z_lambda_spectrum(
                results, 
                component, 
                direction='x',
                re_tau=re_tau,
                save_path=os.path.join(args.output_dir, f"lambda_z_spectrum_{component}_x_re{re_tau:.0f}.png")
            )
            if fig_x:
                plt.close(fig_x)
            
            # Plot λ-z+ spectrum for y-direction (with dual peaks for high Re_tau)
            fig_y, _ = plot_z_lambda_spectrum(
                results, 
                component, 
                direction='y',
                re_tau=re_tau,
                save_path=os.path.join(args.output_dir, f"lambda_z_spectrum_{component}_y_re{re_tau:.0f}.png")
            )
            if fig_y:
                plt.close(fig_y)
        
        # Print execution time
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()