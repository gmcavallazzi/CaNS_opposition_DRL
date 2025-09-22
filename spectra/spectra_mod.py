import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from matplotlib import rc
import time

# Enable LaTeX formatting
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
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

def plot_2d_spectrum(kx, ky, Lx_plus, Ly_plus, spectrum, component, z_plus, save_path=None):
    """Plot 2D pre-multiplied energy spectrum using wavelengths"""
    # Convert wavenumbers to wavelengths
    lambda_x = 2*np.pi / np.maximum(np.abs(kx), 1e-10)
    lambda_y = 2*np.pi / np.maximum(np.abs(ky), 1e-10)
    
    # Create meshgrid for plotting
    LX, LY = np.meshgrid(lambda_x, lambda_y, indexing='ij')
    
    # Create figure with log-log scale
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Find appropriate contour levels
    if np.max(spectrum) > 0:
        max_val = np.max(spectrum)
        min_val = max(np.min(spectrum[spectrum > 0]), max_val * 1e-3)
        levels = np.logspace(np.log10(min_val), np.log10(max_val), 10)
        
        # Draw contours
        cs = ax.contourf(LX, LY, spectrum, levels=levels, 
                       cmap=plt.cm.viridis, locator=plt.LogLocator())
        
        # Set axis labels and title
        ax.set_xlabel(r'$\lambda_x^+$')
        ax.set_ylabel(r'$\lambda_y^+$')
        ax.set_title(r'2D Pre-multiplied ' + component + r' Spectrum at $z^+ = %.2f$' % z_plus)
        
        # Set axis limits to match data
        ax.set_xlim(lambda_x.min(), Lx_plus)
        ax.set_ylim(lambda_y.min(), Ly_plus)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', which='both')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig, ax
    else:
        print(f"Warning: No positive values in spectrum")
        return None, None

def main():
    # Start timer
    start_time = time.time()
    
    # Define the geometry parameters
    try:
        data = np.loadtxt("geometry.out", comments="!", max_rows=2)
        ng = data[0, :].astype('int')
        dimensions = data[1, :]
    except:
        print("Using default dimensions")
        ng = np.array([384, 192, 100])  # Grid size (nx, ny, nz)
        dimensions = np.array([25.13, 6.28, 1.0])  # Box dimensions
    
    print(f"Grid dimensions: {ng}")
    print(f"Domain dimensions: {dimensions}")
    
    # Parameters
    re_tau = 180.0  # Reynolds number
    target_w_plus = 15.0  # Target wall-normal location
    
    # Find velocity files
    u_files = sorted(glob.glob("vex_fld_*.bin"))
    v_files = sorted(glob.glob("vey_fld_*.bin"))
    w_files = sorted(glob.glob("vez_fld_*.bin"))
    
    if not u_files:
        print("No velocity files found!")
        return
    
    print(f"Found {len(u_files)} x-velocity files")
    print(f"Found {len(v_files)} y-velocity files")
    print(f"Found {len(w_files)} z-velocity files")
    
    # Create global variables for domain
    global Lx_plus, Ly_plus
    Lx_plus = dimensions[0] * re_tau
    Ly_plus = dimensions[1] * re_tau
    
    # Process each velocity component
    for component, files in [('uu', u_files), ('vv', v_files), ('ww', w_files)]:
        print(f"\nProcessing {component}-spectrum...")
        
        # Initialize list to store spectra
        all_spectra = []
        z_plus_values = []
        
        # Compute spectra for each snapshot
        for filename in files:
            try:
                print(f"Reading {os.path.basename(filename)}")
                
                # Read velocity data
                velocity_data, _, _, zp, _, _, _ = read_single_field_binary(filename)
                
                # Compute z+ values
                z_plus = zp * re_tau
                
                # Find z-index closest to target
                z_idx = np.argmin(np.abs(z_plus - target_w_plus))
                print(f"  Using z+ = {z_plus[z_idx]:.2f} (index {z_idx})")
                
                # Compute 2D spectrum
                kx, ky, spectrum = compute_2d_spectrum(
                    velocity_data, z_idx, dimensions, re_tau
                )
                
                all_spectra.append(spectrum)
                z_plus_values.append(z_plus[z_idx])
                
            except Exception as e:
                print(f"  Error processing {filename}: {str(e)}")
        
        if not all_spectra:
            print(f"No valid results for {component}!")
            continue
        
        # Average spectra
        avg_spectrum = np.mean(np.array(all_spectra), axis=0)
        avg_z_plus = np.mean(z_plus_values)
        
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