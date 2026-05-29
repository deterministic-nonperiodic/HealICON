import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

# Suppress timedelta warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset and sort coordinates to strictly increasing for matplotlib contourf
ds = xr.open_dataset('output_tides_zm.nc')
ds = ds.sortby(['lat', 'height'])

# Map periods
p_12 = np.timedelta64(12, 'h')
p_24 = np.timedelta64(24, 'h')

# Dictionary to map modes to their coordinates
modes = {
    'DW1': {'period': p_24, 'm': -1, 'type': 'Symmetric'},
    'SW2': {'period': p_12, 'm': -2, 'type': 'Symmetric'},
    'DE3': {'period': p_24, 'm': 3, 'type': 'Symmetric'},
    'DE3_asy': {'period': p_24, 'm': 3, 'type': 'Antisymmetric'},
}

# The user asked to visualize the main tides as function of height and latitude for different modes.
# We will just plot the dominant symmetric modes for DW1, SW2, DE3, and the antisymmetric for DE3 to show the separation.

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, (name, meta) in enumerate(modes.items()):
    ax = axes[i]
    
    # Select symmetric or antisymmetric
    var_name = 'temp_amp_sym' if meta['type'] == 'Symmetric' else 'temp_amp_asy'
    
    try:
        # We use method='nearest' to be safe against slight float precision issues in xarray coords
        data = ds[var_name].sel(period=meta['period'], m=meta['m'], method='nearest')
    except Exception as e:
        print(f"Error selecting {name}: {e}")
        continue
        
    y = data.height / 1000.0  # Convert to km
    x = data.lat
    
    # Contour plot
    vmax = float(data.max())
    levels = np.linspace(0, vmax, 20)
    
    cf = ax.contourf(x, y, data, levels=levels, cmap='inferno', extend='max')
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('Amplitude [K]', fontsize=10)
    
    # Formatting
    ax.set_title(f"{name} Tide Amplitude ({meta['type']})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Latitude [deg]", fontsize=12)
    ax.set_ylabel("Height [km]", fontsize=12)
    
    # Tides are typically plotted symmetric around the equator
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

# Save to the artifacts directory
# Save
out_dir = "."
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "tides_plot.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {out_path}")
