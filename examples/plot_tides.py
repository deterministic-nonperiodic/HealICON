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
    'SE2': {'period': p_12, 'm': 2, 'type': 'Symmetric'},
    'DW1_asy': {'period': p_24, 'm': -1, 'type': 'Antisymmetric'},
    'DE3_asy': {'period': p_24, 'm': 3, 'type': 'Antisymmetric'},
}

fig, axes = plt.subplots(3, 2, figsize=(14, 15), sharex=True, sharey=True)
axes = axes.flatten()

lat_ticks = [-60, -30, 0, 30, 60]
lat_labels = ['60°S', '30°S', '0°', '30°N', '60°N']
vmax = 6.0
levels = np.linspace(0, vmax, 13)

for i, (name, meta) in enumerate(modes.items()):
    ax = axes[i]
    var_name = 'temp_amp_sym' if meta['type'] == 'Symmetric' else 'temp_amp_asy'
    
    try:
        data = ds[var_name].sel(period=meta['period'], m=meta['m'], method='nearest')
    except Exception as e:
        print(f"Error selecting {name}: {e}")
        continue
        
    y = data.height / 1000.0  # Convert to km
    x = data.lat
    
    cf = ax.contourf(x, y, data, levels=levels, cmap='inferno', extend='max')
    
    ax.set_title(f"{name.split('_')[0]} ({meta['type']})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Latitude", fontsize=12)
    ax.set_ylabel("Height / km", fontsize=12)
    
    ax.set_xlim(-60, 60)
    ax.set_ylim(60, 110)
    ax.set_xticks(lat_ticks)
    ax.set_xticklabels(lat_labels)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Hide labels on inner axes
    ax.label_outer()

fig.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.1)
cbar_ax = fig.add_axes([0.26, 0.08, 0.52, 0.015])
cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
cbar.set_label('Amplitude / K', fontsize=14)

out_dir = "."
os.makedirs(out_dir, exist_ok=True)
amp_out_path = os.path.join(out_dir, "tides_plot.png")
plt.savefig(amp_out_path, dpi=300, bbox_inches='tight')
print(f"Saved amplitude plot to {amp_out_path}")

# Phase Plot
fig_pha, axes_pha = plt.subplots(3, 2, figsize=(14, 15), sharex=True, sharey=True)
axes_pha = axes_pha.flatten()

levels_pha = np.linspace(-np.pi, np.pi, 20)

for i, (name, meta) in enumerate(modes.items()):
    ax = axes_pha[i]
    var_name = 'temp_pha_sym' if meta['type'] == 'Symmetric' else 'temp_pha_asy'
    
    try:
        data = ds[var_name].sel(period=meta['period'], m=meta['m'], method='nearest')
    except Exception as e:
        print(f"Error selecting {name} phase: {e}")
        continue
        
    y = data.height / 1000.0  # Convert to km
    x = data.lat
    
    cf_pha = ax.contourf(x, y, data, levels=levels_pha, cmap='twilight_shifted', extend='both')
    
    ax.set_title(f"{name.split('_')[0]} Tide Phase ({meta['type']})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Latitude", fontsize=12)
    ax.set_ylabel("Height / km", fontsize=12)
    
    ax.set_xlim(-60, 60)
    ax.set_ylim(60, 110)
    ax.set_xticks(lat_ticks)
    ax.set_xticklabels(lat_labels)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Hide labels on inner axes
    ax.label_outer()

fig_pha.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.1)
cbar_ax_pha = fig_pha.add_axes([0.26, 0.08, 0.52, 0.015])
cbar_pha = fig_pha.colorbar(cf_pha, cax=cbar_ax_pha, orientation='horizontal', ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar_pha.ax.set_xticklabels(['$-\pi$', '$-\pi$/2', '0', '$\pi$/2', '$\pi$'])
cbar_pha.set_label('Phase / rad', fontsize=14)

pha_out_path = os.path.join(out_dir, "tides_phase_plot.png")
plt.savefig(pha_out_path, dpi=300, bbox_inches='tight')
print(f"Saved phase plot to {pha_out_path}")
