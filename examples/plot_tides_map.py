import xarray as xr
import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

ds = xr.open_dataset('output_tides.nc')

p_12 = np.timedelta64(12, 'h')
p_24 = np.timedelta64(24, 'h')

modes = {
    'DW1': {'period': p_24, 'm': -1, 'type': 'Symmetric'},
    'SW2': {'period': p_12, 'm': -2, 'type': 'Symmetric'},
    'DE3': {'period': p_24, 'm': 3, 'type': 'Symmetric'},
    'SE2': {'period': p_12, 'm': 2, 'type': 'Symmetric'},
}

target_height = 100000 # 100 km

fig = plt.figure(figsize=(16, 12))

for i, (name, meta) in enumerate(modes.items()):
    var_name = 'temp_amp_sym' if meta['type'] == 'Symmetric' else 'temp_amp_asy'
    try:
        data = ds[var_name].sel(period=meta['period'], m=meta['m'], height=target_height, method='nearest')
    except Exception as e:
        print(f"Skipping {name}: {e}")
        continue
        
    map_data = data.values
    # Reorder to RING for healpy plotting
    map_ring = hp.reorder(map_data, n2r=True)
    
    hp.mollview(map_ring, title=f"{name} Tide Amplitude (~100km)", 
                cmap='inferno', sub=(2, 2, i+1), max=float(map_data.max()), min=0, 
                unit='Amplitude [K]')
    hp.graticule()

out_dir = "."
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "tides_map.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to {out_path}")
