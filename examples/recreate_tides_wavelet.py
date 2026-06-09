import xarray as xr
import numpy as np
import logging
from healicon.wavelet import spherical_harmonic_wavelet_spectrum
import os

logging.basicConfig(level=logging.INFO)

def run():
    # 1. Load the original dataset to get dimensions
    input_file = 'UA-ICON_NWP_temp_DOM01_HL_60-100km_202501.nc'
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file}")

    print(f"Loading {input_file}...")
    ds_in = xr.open_dataset(input_file)
    
    # We will process temp variable
    # Dimensions are usually (time, height, cell) or (time, height, ncells)
    time_dim = 'time'
    cell_dim = [d for d in ds_in.dims if d in ['cell', 'ncells', 'cells', 'x']][0]
    height_dim = [d for d in ds_in.dims if d in ['height', 'z_mc', 'altitude']][0]

    height_vals = ds_in[height_dim].values
    cell_vals = ds_in[cell_dim].values
    
    # Original tidal modes
    m_vals = [-2, -1, 2, 3]
    period_vals = [np.timedelta64(24, 'h'), np.timedelta64(12, 'h')]
    
    # Initialize arrays
    n_m = len(m_vals)
    n_h = len(height_vals)
    n_p = len(period_vals)
    n_x = len(cell_vals)

    amp_sym = np.zeros((n_m, n_h, n_p, n_x))
    amp_asy = np.zeros((n_m, n_h, n_p, n_x))
    pha_sym = np.zeros((n_m, n_h, n_p, n_x))
    pha_asy = np.zeros((n_m, n_h, n_p, n_x))

    # Helper mapping
    mode_mapping = {
        -1: {'m_wavelet': 1, 'period': 24.0, 'p_idx': 0, 'dir': 'westward'},  # DW1
         3: {'m_wavelet': 3, 'period': 24.0, 'p_idx': 0, 'dir': 'eastward'},  # DE3
        -2: {'m_wavelet': 2, 'period': 12.0, 'p_idx': 1, 'dir': 'westward'},  # SW2
         2: {'m_wavelet': 2, 'period': 12.0, 'p_idx': 1, 'dir': 'eastward'}   # SE2
    }
    
    # Required positive wavenumbers for wavelet
    m_wavelets = [1, 2, 3]
    # Required periods for wavelet (in hours)
    wavelet_periods = [12.0, 24.0]

    for h_idx, h in enumerate(height_vals):
        print(f"\nProcessing height level {h_idx+1}/{n_h}: {h}")
        da_h = ds_in['temp'].isel(**{height_dim: h_idx})
        
        # We need to run wavelet for each required m_wavelet
        wavelet_outputs = {}
        for mw in m_wavelets:
            print(f"  Running wavelet for m={mw}")
            # The wavelet uses hours by default if dt=1.0 and time steps are 1 hour
            ds_wav = spherical_harmonic_wavelet_spectrum(
                da_h, zwn=mw, dt=1.0, 
                periods_to_reconstruct=wavelet_periods
            )
            wavelet_outputs[mw] = ds_wav

        # Now extract the temporal mean and place it into the output arrays
        for m_idx, m in enumerate(m_vals):
            meta = mode_mapping[m]
            mw = meta['m_wavelet']
            p_val = meta['period']
            p_idx = meta['p_idx']
            direction = meta['dir']
            
            ds_w = wavelet_outputs[mw]
            
            # Find the actual period string used in the dataset
            # by matching the prefix
            prefix_sym_w = f'amp_sym_{direction}_'
            matching_vars = [v for v in ds_w.data_vars if v.startswith(prefix_sym_w)]
            
            # Sort by how close the float value is to target p_val
            def get_dist(v):
                p_str = v.replace(prefix_sym_w, '')
                return abs(float(p_str) - p_val)
                
            closest_var = min(matching_vars, key=get_dist)
            actual_p_str = closest_var.replace(prefix_sym_w, '')
            
            # Extract variables (taking temporal mean to match stationary tide)
            amp_sym[m_idx, h_idx, p_idx, :] = ds_w[f'amp_sym_{direction}_{actual_p_str}'].mean(dim=time_dim).values
            amp_asy[m_idx, h_idx, p_idx, :] = ds_w[f'amp_asy_{direction}_{actual_p_str}'].mean(dim=time_dim).values
            
            pha_raw_sym = ds_w[f'pha_sym_{direction}_{actual_p_str}'].values
            mean_cos_sym = np.nanmean(np.cos(pha_raw_sym), axis=0)
            mean_sin_sym = np.nanmean(np.sin(pha_raw_sym), axis=0)
            pha_sym[m_idx, h_idx, p_idx, :] = np.arctan2(mean_sin_sym, mean_cos_sym)
            
            pha_raw_asy = ds_w[f'pha_asy_{direction}_{actual_p_str}'].values
            mean_cos_asy = np.nanmean(np.cos(pha_raw_asy), axis=0)
            mean_sin_asy = np.nanmean(np.sin(pha_raw_asy), axis=0)
            pha_asy[m_idx, h_idx, p_idx, :] = np.arctan2(mean_sin_asy, mean_cos_asy)

    # 5. Create final Dataset
    ds_out = xr.Dataset(
        data_vars={
            'temp_amp_sym': (['m', 'height', 'period', cell_dim], amp_sym),
            'temp_amp_asy': (['m', 'height', 'period', cell_dim], amp_asy),
            'temp_pha_sym': (['m', 'height', 'period', cell_dim], pha_sym),
            'temp_pha_asy': (['m', 'height', 'period', cell_dim], pha_asy),
        },
        coords={
            'm': m_vals,
            'height': height_vals,
            'period': period_vals,
            cell_dim: cell_vals
        }
    )
    
    out_file = 'wavelet_tides.nc'
    ds_out.to_netcdf(out_file)
    print(f"\nSaved wavelet tides to {out_file}")

    # Compute Zonal Mean using precise HEALPix rings
    print("\nComputing Zonal Mean...")
    from healicon.analysis import zonal_mean
    
    # We must be careful to compute circular mean for phases
    # So we compute zonal mean of cos(phase) and sin(phase) separately
    ds_cos_sym = np.cos(ds_out['temp_pha_sym'])
    ds_sin_sym = np.sin(ds_out['temp_pha_sym'])
    ds_cos_asy = np.cos(ds_out['temp_pha_asy'])
    ds_sin_asy = np.sin(ds_out['temp_pha_asy'])
    
    ds_components = xr.Dataset({
        'amp_sym': ds_out['temp_amp_sym'],
        'amp_asy': ds_out['temp_amp_asy'],
        'cos_sym': ds_cos_sym,
        'sin_sym': ds_sin_sym,
        'cos_asy': ds_cos_asy,
        'sin_asy': ds_sin_asy,
    })
    
    ds_zm_comp = zonal_mean(ds_components)
    
    ds_zm = xr.Dataset(coords=ds_zm_comp.coords)
    ds_zm['temp_amp_sym'] = ds_zm_comp['amp_sym']
    ds_zm['temp_amp_asy'] = ds_zm_comp['amp_asy']
    ds_zm['temp_pha_sym'] = np.arctan2(ds_zm_comp['sin_sym'], ds_zm_comp['cos_sym'])
    ds_zm['temp_pha_asy'] = np.arctan2(ds_zm_comp['sin_asy'], ds_zm_comp['cos_asy'])
    
    out_zm = 'wavelet_tides_zm.nc'
    ds_zm.to_netcdf(out_zm)
    print(f"Saved zonal mean to {out_zm}")

    # Plotting
    print("\nGenerating Plots...")
    from healicon.visualize import plot_tides, plot_map
    
    plot_tides(ds_zm, out_dir=".", prefix="wavelet_tides_plot")
    print("Generated tides plot")

    # Also generate the tides map for 100km
    target_height = 100000
    modes_to_plot = {
        'DW1': {'period': np.timedelta64(24, 'h'), 'm': -1, 'type': 'Symmetric'},
        'SW2': {'period': np.timedelta64(12, 'h'), 'm': -2, 'type': 'Symmetric'},
        'DE3': {'period': np.timedelta64(24, 'h'), 'm': 3, 'type': 'Symmetric'},
        'SE2': {'period': np.timedelta64(12, 'h'), 'm': 2, 'type': 'Symmetric'},
    }
    
    for name, meta in modes_to_plot.items():
        var_name = 'temp_amp_sym' if meta['type'] == 'Symmetric' else 'temp_amp_asy'
        data_slice = ds_out.sel(period=meta['period'], m=meta['m'], height=target_height, method='nearest')
        prefix = f"wavelet_tides_map_{name}"
        plot_map(data_slice, var_name=var_name, out_dir=".", prefix=prefix)
    print("Generated tides maps")

if __name__ == '__main__':
    run()
