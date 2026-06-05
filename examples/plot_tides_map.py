import xarray as xr
import numpy as np
import os
import warnings
from healicon.visualize import plot_map

warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    input_file = 'output_tides.nc'
    print(f"Loading {input_file}...")
    try:
        ds = xr.open_dataset(input_file)
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Please ensure you have generated it first.")
        exit(1)

    p_12 = np.timedelta64(12, 'h')
    p_24 = np.timedelta64(24, 'h')

    modes = {
        'DW1': {'period': p_24, 'm': -1, 'type': 'Symmetric'},
        'SW2': {'period': p_12, 'm': -2, 'type': 'Symmetric'},
        'DE3': {'period': p_24, 'm': 3, 'type': 'Symmetric'},
        'SE2': {'period': p_12, 'm': 2, 'type': 'Symmetric'},
    }

    target_height = 100000 # 100 km

    out_dir = "."
    os.makedirs(out_dir, exist_ok=True)

    for name, meta in modes.items():
        var_name = 'temp_amp_sym' if meta['type'] == 'Symmetric' else 'temp_amp_asy'
        try:
            # Select the specific mode and height
            data_slice = ds.sel(period=meta['period'], m=meta['m'], height=target_height, method='nearest')
            prefix = f"tides_map_{name}"
            print(f"Plotting {name}...")
            plot_map(data_slice, var_name=var_name, out_dir=out_dir, prefix=prefix)
        except Exception as e:
            print(f"Skipping {name}: {e}")

    print("Done.")
