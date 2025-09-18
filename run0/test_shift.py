import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# Set matplotlib to show plots
plt.ion()

def generate_test_field(nx=64, ny=64, Lx=2.67, Ly=0.8):
    """Generate a test field with waves - no noise"""
    # Create coordinate arrays
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Generate field with structures that vary primarily in x (no noise)
    field = (0.08 * np.sin(2*np.pi*X/Lx) +                                    # Main x-variation
             0.05 * np.cos(4*np.pi*X/Lx) * np.sin(np.pi*Y/Ly) +              # x-structure with y-modulation  
             0.03 * np.sin(6*np.pi*X/Lx) +                                    # Higher freq x-variation
             0.02 * np.sin(3*np.pi*X/Lx) * (1 + 0.5*np.cos(2*np.pi*Y/Ly)))   # x-wave with y-envelope
    
    # Ensure zero average
    field = field - np.mean(field)
    
    # Scale to desired range (±0.12)
    field_range = np.max(field) - np.min(field)
    field = field * (0.24 / field_range)  # 0.24 = 2 * 0.12
    
    # Add back a constant (simulating what was subtracted: 0.2-0.6)
    base_value = np.random.uniform(0.2, 0.6)
    field_with_base = field + base_value
    
    return x, y, field, field_with_base, base_value

def shift_field_subgrid(x, y, field, dx_shift):
    """Shift field by dx_shift using subgrid interpolation"""
    nx, ny = field.shape
    Lx = x[-1] - x[0]
    
    # Create interpolator
    interp = RegularGridInterpolator((x, y), field, 
                                   bounds_error=False, 
                                   fill_value=None,  # Use nearest for out-of-bounds
                                   method='linear')
    
    # Create shifted coordinates (with periodic boundary conditions)
    X, Y = np.meshgrid(x, y, indexing='ij')
    X_shifted = X - dx_shift
    
    # Handle periodic boundaries
    X_shifted = X_shifted % Lx
    
    # Create points for interpolation
    points = np.column_stack([X_shifted.ravel(), Y.ravel()])
    
    # Interpolate
    field_shifted = interp(points).reshape(nx, ny)
    
    return field_shifted

# Parameters
nx, ny = 64, 64
Lx, Ly = 2.67, 0.8
dt = 0.01  # time step

# Generate test field
x, y, field_zero_avg, field_with_base, base_value = generate_test_field(nx, ny, Lx, Ly)

print(f"Generated field:")
print(f"  Base value (subtracted): {base_value:.3f}")
print(f"  Zero-average field range: [{np.min(field_zero_avg):.4f}, {np.max(field_zero_avg):.4f}]")
print(f"  Field with base range: [{np.min(field_with_base):.4f}, {np.max(field_with_base):.4f}]")
print(f"  Mean of zero-average field: {np.mean(field_zero_avg):.6f}")

# Simulate a velocity field (this could be your 'u')
# For testing, let's use the field itself as a velocity estimate
u_avg = np.mean(field_zero_avg)  # This should be ~0
print(f"  Average 'velocity' u: {u_avg:.6f}")

# Test different shift amounts - focus on smaller, more realistic values
dx_shifts = [0.01, 0.03, 0.08]  # These are dx = u*dt for different scenarios

# Calculate and print grid spacing
dx_grid = Lx / nx
dy_grid = Ly / ny
print(f"Grid spacing:")
print(f"  dx_grid = {dx_grid:.6f}")
print(f"  dy_grid = {dy_grid:.6f}")

# Create figure with vertical layout - proper aspect ratio
fig_width = 16
fig_height = 12
fig, axes = plt.subplots(len(dx_shifts)+1, 1, figsize=(fig_width, fig_height))

# Plot original field at the top
im0 = axes[0].imshow(field_zero_avg.T, extent=[0, Lx, 0, Ly], 
                     origin='lower', cmap='RdBu_r', aspect='equal',
                     interpolation='bilinear')  # Smooth interpolation
axes[0].set_title('Original Field', fontsize=16, pad=5)
axes[0].set_ylabel('y', fontsize=14)
axes[0].set_xlabel('x', fontsize=14)
axes[0].set_xlim(0, Lx)
axes[0].set_ylim(0, Ly)

# Plot shifted fields below
for i, dx_shift in enumerate(dx_shifts):
    # Shift the field
    field_shifted = shift_field_subgrid(x, y, field_zero_avg, dx_shift)
    
    # Plot shifted field with smooth interpolation
    im = axes[i+1].imshow(field_shifted.T, extent=[0, Lx, 0, Ly], 
                         origin='lower', cmap='RdBu_r', aspect='equal',
                         vmin=np.min(field_zero_avg), vmax=np.max(field_zero_avg),
                         interpolation='bilinear')  # Smooth interpolation
    axes[i+1].set_title(f'Shifted by dx = {dx_shift:.3f} (= {dx_shift/dx_grid:.2f} × dx_grid)', 
                       fontsize=16, pad=5)
    axes[i+1].set_ylabel('y', fontsize=14)
    axes[i+1].set_xlabel('x', fontsize=14)
    axes[i+1].set_xlim(0, Lx)
    axes[i+1].set_ylim(0, Ly)
    
    # Calculate and print shift statistics
    diff = field_shifted - field_zero_avg
    print(f"Shift dx={dx_shift:.3f}: RMS difference = {np.sqrt(np.mean(diff**2)):.6f}")

# Minimize whitespace
plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08, hspace=0.3)

# Save the figure as PNG
plt.savefig('subgrid_interpolation_test.png', dpi=300, bbox_inches='tight', pad_inches=0.1)
print("Plot saved as 'subgrid_interpolation_test.png'")

plt.show()

# Force the plot to display
plt.draw()
plt.pause(0.1)

# Test grid resolution effects
dx_grid = Lx / nx
dy_grid = Ly / ny
print(f"\nGrid resolution:")
print(f"  dx_grid = {dx_grid:.4f}")
print(f"  dy_grid = {dy_grid:.4f}")
print(f"  Shifts as fraction of grid spacing:")
for dx_shift in dx_shifts:
    print(f"    dx={dx_shift:.3f} is {dx_shift/dx_grid:.2f} * dx_grid")

# Test interpolation accuracy with a known analytical shift
print(f"\nTesting interpolation accuracy:")
# Create a simple sinusoidal field
x_test = np.linspace(0, Lx, nx)
y_test = np.linspace(0, Ly, ny)
X_test, Y_test = np.meshgrid(x_test, y_test, indexing='ij')
analytical_field = 0.1 * np.sin(2*np.pi*X_test/Lx)

# Shift by exactly dx_grid/2 (should be well-resolved by linear interpolation)
dx_test = dx_grid / 2
shifted_numerical = shift_field_subgrid(x_test, y_test, analytical_field, dx_test)

# Analytical shifted field
X_analytical_shifted = X_test - dx_test
analytical_shifted = 0.1 * np.sin(2*np.pi*X_analytical_shifted/Lx)

interp_error = np.sqrt(np.mean((shifted_numerical - analytical_shifted)**2))
print(f"  Interpolation error for sinusoidal field: {interp_error:.8f}")