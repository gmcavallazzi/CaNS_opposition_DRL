import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import rc
import time
import matplotlib.cm as cm

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

def calculate_w_plus(w, re_tau=180.0):
    """Calculate w+ values (wall units for wall-normal coordinate)"""
    return w * re_tau

def compute_xz_spectrum(velocity_data, direction='x', dimensions=None, re_tau=180.0):
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

def plot_z_lambda_spectrum(results, component, direction='x', save_path=None):
    """
    Plot spectrum with wavelength on x-axis and z+ on y-axis using a semilogx plot
    
    Parameters:
    -----------
    results : dict
        Dictionary containing the spectrum data
    component : str
        Velocity component label (u, v, w)
    direction : str
        Direction for spectrum ('x' or 'y')
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig, ax : tuple
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    if direction == 'x':
        k = results['kx']
        spectra = results['spectra_x']
        direction_label = 'streamwise'
    else:  # y direction
        k = results['ky']
        spectra = results['spectra_y']
        direction_label = 'spanwise'
    
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
        
        # Create logarithmically spaced contour levels
        levels = np.logspace(np.log10(min_val), np.log10(max_val), 10)
        
        # Define a colormap
        cmap = plt.cm.viridis
        
        # Draw filled contours
        cs = ax.contourf(X, Y, Z, levels=levels, 
                       cmap=cmap, locator=plt.LogLocator())
        
        # Add colorbar
        plt.colorbar(cs, ax=ax, label=f'Pre-multiplied Spectrum Intensity')
        
        # Semilog x
        ax.set_xscale('log')
        
        # Set labels
        ax.set_xlabel(r'$\lambda_{' + direction + '}^+$')
        ax.set_ylabel(r'$z^+$')
        
        # Set title
        ax.set_title(f'{component}-{direction_label} Pre-multiplied Spectrum')
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        
        # Adjust layout
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    else:
        print(f"Warning: No positive values in spectrum for {component}-component in {direction} direction")
        return None, None

def plot_cross_sections(results, component, direction='x', z_indices=None, save_path=None):
    """
    Plot cross-sections of the spectrum at specific z+ values
    
    Parameters:
    -----------
    results : dict
        Dictionary containing the spectrum data
    component : str
        Velocity component label (u, v, w)
    direction : str
        Direction for spectrum ('x' or 'y')
    z_indices : list of int, optional
        Indices for z+ cross-sections. If None, will choose automatically
    save_path : str, optional
        Path to save the figure
    
    Returns:
    --------
    fig, ax : tuple
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    if direction == 'x':
        k = results['kx']
        spectra = results['spectra_x']
        direction_label = 'streamwise'
    else:  # y direction
        k = results['ky']
        spectra = results['spectra_y']
        direction_label = 'spanwise'
    
    z_plus = results['z_plus']
    
    # Filter out zero or very small wavenumbers for log scaling
    min_k = 1e-10
    valid = k > min_k
    
    # Extract valid data
    k_valid = k[valid]
    spectra_valid = spectra[valid, :]
    
    # Convert to wavelength (lambda = 2π/k)
    lambda_k = 2 * np.pi / k_valid
    
    # If z_indices not provided, choose some sensible defaults
    if z_indices is None:
        nz = len(z_plus)
        if nz > 20:
            # Choose 5 points logarithmically spaced in z+
            z_min_idx = 5  # Skip the very first points near the wall
            z_max_idx = nz - 1
            z_indices = np.unique(np.logspace(
                np.log10(z_min_idx), np.log10(z_max_idx), 5).astype(int))
        else:
            # Few points, use them all
            z_indices = np.arange(0, nz, max(1, nz // 5))
    
    # Plot cross-sections
    colors = plt.cm.viridis(np.linspace(0, 1, len(z_indices)))
    
    for i, z_idx in enumerate(z_indices):
        if z_idx < len(z_plus):
            z_val = z_plus[z_idx]
            ax.loglog(lambda_k, spectra_valid[:, z_idx], 
                     color=colors[i], label=f'$z^+ = {z_val:.1f}$')
    
    # Set labels
    ax.set_xlabel(r'$\lambda_{' + direction + '}^+$')
    ax.set_ylabel(r'$k_{' + direction + '} E_{' + component + '}$')
    
    # Set title
    ax.set_title(f'{component}-{direction_label} Pre-multiplied Spectrum Cross-sections')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    
    # Add legend
    ax.legend(loc='best')
    
    # Adjust layout
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def main():
    # Start timer
    start_time = time.time()
    
    # Define the geometry parameters
    geofile = "geometry.out"
    try:
        data = np.loadtxt(geofile, comments="!", max_rows=2)
        ng = data[0, :].astype('int')
        dimensions = data[1, :]
    except:
        print("Warning: Could not read geometry.out, using default dimensions")
        ng = np.array([192, 192, 192])  # Default grid size
        dimensions = np.array([2*np.pi, np.pi, 2])  # Default box dimensions
    
    # Parameters
    re_tau = 180.0  # Reynolds number based on friction velocity
    
    # Find all velocity files
    vex_files = sorted(glob.glob("vex_fld_*.bin"))
    vey_files = sorted(glob.glob("vey_fld_*.bin"))
    vez_files = sorted(glob.glob("vez_fld_*.bin"))
    
    if not vex_files:
        print("No velocity files found! Make sure you're in the correct directory.")
        return
    
    print(f"Found {len(vex_files)} x-velocity files")
    print(f"Found {len(vey_files)} y-velocity files")
    print(f"Found {len(vez_files)} z-velocity files")
    
    # Process all available files
    print("Processing all available files...")
    
    # Initialize dictionary to store accumulated spectra results
    spectra_results = {}
    
    # Process velocity components sequentially 
    for component, files in [('u', vex_files), ('v', vey_files), ('w', vez_files)]:
        print(f"\nProcessing {component}-velocity component...")
        
        # Process files sequentially
        results = []
        for i, filename in enumerate(files):
            try:
                progress = (i + 1) / len(files) * 100
                print(f"Processing {os.path.basename(filename)} ({progress:.1f}%)")
                
                # Read velocity data - provides data in (nx, ny, nz) format
                velocity_data, _, _, z, _, _, _ = read_single_field_binary(filename)
                
                if velocity_data is None:
                    continue
                
                # Calculate z+ values
                z_plus = calculate_w_plus(z, re_tau)
                
                # Compute spectra for all z-planes with correct physical scaling
                kx, spectra_x = compute_xz_spectrum(velocity_data, direction='x', dimensions=dimensions, re_tau=re_tau)
                ky, spectra_y = compute_xz_spectrum(velocity_data, direction='y', dimensions=dimensions, re_tau=re_tau)
                
                # Store result
                results.append({
                    'kx': kx,
                    'ky': ky,
                    'spectra_x': spectra_x,
                    'spectra_y': spectra_y,
                    'z_plus': z_plus
                })
                
            except Exception as e:
                print(f"  Error processing {filename}: {str(e)}")
        
        if not results:
            print(f"No valid results for {component}-velocity component!")
            continue
        
        # Combine results safely
        spectra_results[component] = combine_spectra_safely(results)
    
    # Plot and save results
    for component, results in spectra_results.items():
        # Skip if no results
        if results is None:
            continue
        
        # Plot λ-z+ spectrum for x-direction
        fig_x, _ = plot_z_lambda_spectrum(
            results, 
            component, 
            direction='x',
            save_path=f"lambda_z_spectrum_{component}_x.png"
        )
        if fig_x:
            plt.close(fig_x)
        
        # Plot λ-z+ spectrum for y-direction
        fig_y, _ = plot_z_lambda_spectrum(
            results, 
            component, 
            direction='y',
            save_path=f"lambda_z_spectrum_{component}_y.png"
        )
        if fig_y:
            plt.close(fig_y)
        
        # Plot cross-sections at specific z+ values for x-direction
        fig_x_cs, _ = plot_cross_sections(
            results, 
            component, 
            direction='x',
            save_path=f"lambda_z_cross_sections_{component}_x.png"
        )
        if fig_x_cs:
            plt.close(fig_x_cs)
        
        # Plot cross-sections at specific z+ values for y-direction
        fig_y_cs, _ = plot_cross_sections(
            results, 
            component, 
            direction='y',
            save_path=f"lambda_z_cross_sections_{component}_y.png"
        )
        if fig_y_cs:
            plt.close(fig_y_cs)
    
    # Print execution time
    elapsed_time = time.time() - start_time
    print(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()