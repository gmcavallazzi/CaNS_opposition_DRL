import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import rc
import time
from mpi4py import MPI
import pickle
import argparse

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

def process_file(filename, z_plus_values, dimensions, re_tau):
    """Process a single velocity file and compute its 2D spectrum at multiple z+ locations"""
    try:
        # Read velocity data
        velocity_data, _, _, zp, _, _, _ = read_single_field_binary(filename)
        
        # Compute z+ values
        z_plus_actual = zp * re_tau
        
        # Initialize results for this file
        file_results = {}
        
        # Process for each requested z+ value
        for target_z_plus in z_plus_values:
            # Find z-index closest to target
            z_idx = np.argmin(np.abs(z_plus_actual - target_z_plus))
            actual_z_plus = z_plus_actual[z_idx]
            
            # Compute 2D spectrum
            kx, ky, spectrum = compute_2d_spectrum(
                velocity_data, z_idx, dimensions, re_tau
            )
            
            # Store in results
            file_results[actual_z_plus] = {
                'kx': kx,
                'ky': ky,
                'spectrum': spectrum
            }
        
        return file_results
        
    except Exception as e:
        print(f"  Error processing {filename}: {str(e)}")
        return None

def plot_2d_spectrum(kx, ky, Lx_plus, Ly_plus, spectrum, component, z_plus, re_tau, save_path=None):
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
        
        # Find maxima based on component and Re_tau
        # For uu component with high Re_tau, find two peaks
        if component == 'uu' and re_tau > 750:
            # Create masks for the two regions (small and large wavelengths)
            small_lambda_mask = LX < 1000  # Region where lambda_x+ < 1000
            large_lambda_mask = LX >= 1000  # Region where lambda_x+ >= 1000
            
            # Apply masks to spectrum
            small_spectrum = spectrum.copy()
            large_spectrum = spectrum.copy()
            
            # Zero out regions we don't want to consider for each maximum
            small_spectrum[~small_lambda_mask] = 0
            large_spectrum[~large_lambda_mask] = 0
            
            # Find maximum in small wavelength region
            if np.max(small_spectrum) > 0:
                small_max_idx = np.unravel_index(np.argmax(small_spectrum), spectrum.shape)
                small_peak_x_idx, small_peak_y_idx = small_max_idx
                small_peak_lambda_x = LX[small_peak_x_idx, small_peak_y_idx]
                small_peak_lambda_y = LY[small_peak_x_idx, small_peak_y_idx]
                
                # Add a triangle marker at the small peak location (no label)
                ax.scatter(small_peak_lambda_x, small_peak_lambda_y, s=200, marker='^', color='red', 
                          edgecolor='black', zorder=10)
            
            # Find maximum in large wavelength region
            if np.max(large_spectrum) > 0:
                large_max_idx = np.unravel_index(np.argmax(large_spectrum), spectrum.shape)
                large_peak_x_idx, large_peak_y_idx = large_max_idx
                large_peak_lambda_x = LX[large_peak_x_idx, large_peak_y_idx]
                large_peak_lambda_y = LY[large_peak_x_idx, large_peak_y_idx]
                
                # Add a square marker at the large peak location (no label)
                ax.scatter(large_peak_lambda_x, large_peak_lambda_y, s=200, marker='s', color='green', 
                          edgecolor='black', zorder=10)
        else:
            # For all other components or low Re_tau, just find the global maximum
            max_idx = np.unravel_index(np.argmax(spectrum), spectrum.shape)
            peak_x_idx, peak_y_idx = max_idx
            
            peak_lambda_x = LX[peak_x_idx, peak_y_idx]
            peak_lambda_y = LY[peak_x_idx, peak_y_idx]
            
            # Add a triangle marker at the peak location (no label)
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
            
        # Set title with LaTeX formatting including Re_tau
        ax.set_title(r'$\kappa_x \kappa_y \phi_{' + comp_label + '}$ at $z^+ = ' + f'{z_plus:.1f}' + r'$, $Re_\tau = ' + f'{re_tau:.0f}$', 
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
        
        # No colorbar as requested
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    else:
        print(f"Warning: No positive values in spectrum")
        return None, None

def process_data(z_plus_values, re_tau=1083.0):
    """Process velocity files and compute spectra at multiple z+ values"""
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Start timer (only on rank 0)
    if rank == 0:
        start_time = time.time()
        print(f"Running with {size} MPI processes")
        print(f"Computing spectra at z+ values: {z_plus_values}")
        print(f"Using Reynolds number Re_tau = {re_tau}")
    
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
            return None
        
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
        
        # Check if we received empty lists (for early termination)
        if not my_u_files:
            return None
    
    # Process files assigned to this rank
    local_results = {'uu': {}, 'vv': {}, 'ww': {}}
    
    # Initialize with empty lists for each z+ value
    for z_plus in z_plus_values:
        for component in local_results:
            local_results[component][z_plus] = []
    
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
            result = process_file(filename, z_plus_values, dimensions, re_tau)
            if result is not None:
                # Organize results by z+ value
                for z_plus, data in result.items():
                    # Find the closest target z+ value
                    closest_z_plus = min(z_plus_values, key=lambda x: abs(x - z_plus))
                    if abs(closest_z_plus - z_plus) < 5.0:  # Only include if reasonably close
                        if closest_z_plus not in local_results[component]:
                            local_results[component][closest_z_plus] = []
                        local_results[component][closest_z_plus].append(data)
    
    # Gather all results to rank 0
    if rank == 0:
        # Initialize with rank 0's results
        all_results = {'uu': {}, 'vv': {}, 'ww': {}}
        
        # Copy rank 0's results to all_results
        for component in ['uu', 'vv', 'ww']:
            all_results[component] = local_results[component].copy()
        
        # Receive from other ranks
        for r in range(1, size):
            remote_results = comm.recv(source=r, tag=20)
            
            # Merge remote results into all_results
            for component in ['uu', 'vv', 'ww']:
                for z_plus, data_list in remote_results[component].items():
                    if z_plus not in all_results[component]:
                        all_results[component][z_plus] = []
                    all_results[component][z_plus].extend(data_list)
    else:
        # Send results to rank 0
        comm.send(local_results, dest=0, tag=20)
        return None
    
    # Only rank 0 continues from here
    if rank == 0:
        # Average results
        averaged_results = {'uu': {}, 'vv': {}, 'ww': {}}
        
        # Process each component
        for component in ['uu', 'vv', 'ww']:
            print(f"\nProcessing {component} spectra...")
            
            for z_plus, results in all_results[component].items():
                if not results:
                    print(f"No valid results for {component} at z+ ≈ {z_plus:.1f}!")
                    continue
                
                print(f"Averaging {len(results)} {component} spectra at z+ ≈ {z_plus:.1f}...")
                
                # Extract data from first result to initialize
                kx = results[0]['kx']
                ky = results[0]['ky']
                shape = results[0]['spectrum'].shape
                
                # Initialize array for averaging
                avg_spectrum = np.zeros(shape)
                
                # Add all results
                valid_results = 0
                for res in results:
                    if res['spectrum'].shape == shape:
                        avg_spectrum += res['spectrum']
                        valid_results += 1
                
                if valid_results > 0:
                    # Average the spectrum
                    avg_spectrum /= valid_results
                    
                    # Store the averaged result
                    averaged_results[component][z_plus] = {
                        'kx': kx,
                        'ky': ky,
                        'spectrum': avg_spectrum
                    }
        
        # Create output directory if it doesn't exist
        os.makedirs('spectra_data', exist_ok=True)
        
        # Save the processed data
        output_file = os.path.join('spectra_data', 'averaged_spectra.pkl')
        
        # Add metadata
        metadata = {
            'dimensions': dimensions,
            're_tau': re_tau,
            'Lx_plus': Lx_plus,
            'Ly_plus': Ly_plus,
            'date_processed': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Combined data to save
        save_data = {
            'metadata': metadata,
            'spectra': averaged_results
        }
        
        # Save to pickle file
        with open(output_file, 'wb') as f:
            pickle.dump(save_data, f)
        
        print(f"\nSaved averaged spectra data to {output_file}")
        
        # Print execution time
        elapsed_time = time.time() - start_time
        print(f"\nTotal execution time: {elapsed_time:.2f} seconds")
        
        return save_data

def plot_spectra(data_file=None, z_plus_values=None, output_dir='spectra_plots'):
    """Plot spectra from saved data at specified z+ values"""
    # Load data if a file is provided
    if data_file is not None:
        print(f"Loading spectra data from {data_file}")
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
    else:
        print("No data file provided!")
        return
    
    # Extract metadata
    metadata = data['metadata']
    spectra = data['spectra']
    re_tau = metadata['re_tau']
    
    print(f"Using Reynolds number Re_tau = {re_tau}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get components
    components = list(spectra.keys())
    
    # If no z+ values are specified, use all available
    if z_plus_values is None:
        # Get all available z+ values from the uu component
        if 'uu' in spectra:
            z_plus_values = sorted(list(spectra['uu'].keys()))
        else:
            z_plus_values = sorted(list(spectra[components[0]].keys()))
        
        print(f"Available z+ values: {[f'{z:.1f}' for z in z_plus_values]}")
    
    # Plot each component at each z+ value
    for component in components:
        print(f"\nPlotting {component} spectra...")
        
        for z_plus in z_plus_values:
            # Find the closest z+ value that exists
            closest_z_plus = min(spectra[component].keys(), 
                                key=lambda x: abs(x - z_plus))
            
            # Skip if the closest value is too far
            if abs(closest_z_plus - z_plus) > 5.0:
                print(f"  Skipping z+ = {z_plus:.1f} (not available, closest is {closest_z_plus:.1f})")
                continue
            
            print(f"  Plotting z+ ≈ {closest_z_plus:.1f} (requested: {z_plus:.1f})")
            
            # Get the data
            data = spectra[component][closest_z_plus]
            
            # File path for saving
            save_path = os.path.join(output_dir, 
                                    f"2d_spectrum_{component}_z{closest_z_plus:.1f}_re{re_tau:.0f}.png")
            
            # Plot with re_tau parameter
            fig, ax = plot_2d_spectrum(
                data['kx'], data['ky'], 
                metadata['Lx_plus'], metadata['Ly_plus'], 
                data['spectrum'], component, closest_z_plus, re_tau,
                save_path=save_path
            )
            
            if fig is not None:
                plt.close(fig)
    
    print(f"\nPlots saved to {output_dir}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process and plot 2D energy spectra from DNS data')
    parser.add_argument('--mode', type=str, choices=['process', 'plot'], required=True,
                      help='Operation mode: process data or plot existing data')
    parser.add_argument('--z_plus', type=float, nargs='+', default=[15.0, 30.0, 50.0, 100.0, 200.0],
                      help='List of z+ values to process/plot')
    parser.add_argument('--re_tau', type=float, default=1083.0,
                      help='Reynolds number based on friction velocity')
    parser.add_argument('--data_file', type=str, default=None,
                      help='Path to saved spectra data file (for plot mode)')
    parser.add_argument('--output_dir', type=str, default='spectra_plots',
                      help='Directory for saving plots')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get MPI rank for conditionals
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    # Process mode
    if args.mode == 'process':
        # Pass re_tau to process_data
        process_data(args.z_plus, args.re_tau)
    
    # Plot mode (only on rank 0)
    elif rank == 0 and args.mode == 'plot':
        # If no data_file is provided, try to use default
        if args.data_file is None:
            default_file = os.path.join('spectra_data', 'averaged_spectra.pkl')
            if os.path.exists(default_file):
                args.data_file = default_file
                print(f"Using default data file: {default_file}")
            else:
                print("Error: No data file provided and default file not found!")
                print("Please specify a data file with --data_file or run in 'process' mode first.")
                return
        
        # Plot the data
        plot_spectra(data_file=args.data_file, z_plus_values=args.z_plus, output_dir=args.output_dir)

if __name__ == "__main__":
    main()