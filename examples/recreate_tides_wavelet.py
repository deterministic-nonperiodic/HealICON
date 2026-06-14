"""
Reproduce tidal analysis using the SH-wavelet method.

Uses ``compute_wavelet_tidal_analysis`` to process all height levels and
tidal modes in a single call, producing output in the same format as
``compute_tidal_analysis`` for direct comparison.
"""
import xarray as xr
import numpy as np
import logging
import os

from healicon.wavelet import compute_wavelet_tidal_analysis
from healicon.analysis import regrade_resolution

logging.basicConfig(level=logging.INFO)


def run():
    # 1. Load and regrade
    input_file = 'UA-ICON_NWP_temp_DOM01_HL_60-110km_202501.nc'
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Missing {input_file}")

    print(f"Loading {input_file}...")
    ds_in = xr.open_dataset(input_file)
    ds_in = regrade_resolution(ds_in, new_nside=32)

    # 2. Run wavelet tidal analysis — all modes, all heights, one call
    ds_out = compute_wavelet_tidal_analysis(
        ds_in, 'temp',
        periods_hours=[24.0, 12.0],
        m_filters=[1, 2, -2, -3],  # DW1, SW2, SE2, DE3
        temporal_mean=True,         # time-averaged (LS-comparable)
    )

    out_file = 'wavelet_tides.nc'
    ds_out.to_netcdf(out_file)
    print(f"\nSaved wavelet tides to {out_file}")

    # 3. Compute Zonal Mean using precise HEALPix rings
    print("\nComputing Zonal Mean...")
    from healicon.extract import zonal_mean

    # Circular mean for phases: average cos and sin separately
    ds_components = xr.Dataset({
        'amp_sym': ds_out['temp_amp_sym'],
        'amp_asy': ds_out['temp_amp_asy'],
        'cos_sym': np.cos(ds_out['temp_pha_sym']),
        'sin_sym': np.sin(ds_out['temp_pha_sym']),
        'cos_asy': np.cos(ds_out['temp_pha_asy']),
        'sin_asy': np.sin(ds_out['temp_pha_asy']),
    })
    ds_components.attrs = ds_out.attrs

    ds_zm_comp = zonal_mean(ds_components)

    ds_zm = xr.Dataset(coords=ds_zm_comp.coords)
    ds_zm['temp_amp_sym'] = ds_zm_comp['amp_sym']
    ds_zm['temp_amp_asy'] = ds_zm_comp['amp_asy']
    ds_zm['temp_pha_sym'] = np.arctan2(ds_zm_comp['sin_sym'],
                                        ds_zm_comp['cos_sym'])
    ds_zm['temp_pha_asy'] = np.arctan2(ds_zm_comp['sin_asy'],
                                        ds_zm_comp['cos_asy'])

    out_zm = 'wavelet_tides_zm.nc'
    ds_zm.to_netcdf(out_zm)
    print(f"Saved zonal mean to {out_zm}")


    # 4. Plotting
    print("\nGenerating Plots...")
    from healicon.visualize import plot_tides, plot_map

    plot_tides(ds_zm, out_dir=".", prefix="wavelet_tides_plot_optimal", max_amplitude=6.0)
    print("Generated tides plot")

    # Generate tides maps at ~100 km
    target_height = 100000
    modes_to_plot = {
        'DW1': {'period': np.timedelta64(24, 'h'), 'm': 1,
                'type': 'Symmetric'},
        'SW2': {'period': np.timedelta64(12, 'h'), 'm': 2,
                'type': 'Symmetric'},
        'DE3': {'period': np.timedelta64(24, 'h'), 'm': -3,
                'type': 'Symmetric'},
        'SE2': {'period': np.timedelta64(12, 'h'), 'm': -2,
                'type': 'Symmetric'},
    }

    for name, meta in modes_to_plot.items():
        var_name = ('temp_amp_sym' if meta['type'] == 'Symmetric'
                    else 'temp_amp_asy')
        data_slice = ds_out.sel(period=meta['period'], m=meta['m'],
                                height=target_height, method='nearest')
        prefix = f"wavelet_tides_map_{name}_optimal"
        plot_map(data_slice, var_name=var_name, out_dir=".", prefix=prefix)
    print("Generated tides maps")


if __name__ == '__main__':
    run()
