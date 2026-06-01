import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

def plot_tides_comparison():
    saber_file = './saber_tides_zm.nc'
    icon_file = './icon_tides_zm.nc'
    saber_raw = './saber_lst.nc'
    
    if not os.path.exists(saber_file) or not os.path.exists(icon_file):
        print("Required files missing.")
        return

    ds_saber = xr.open_dataset(saber_file).sortby('lat')
    ds_icon = xr.open_dataset(icon_file).sortby(['lat', 'height'])
    
    # Get SABER altitudes in km
    ds_raw = xr.open_dataset(saber_raw)
    saber_alt = np.nanmean(ds_raw['tpaltitude'].values, axis=(1, 2))
    
    # Filter SABER to valid altitudes
    valid_alt = ~np.isnan(saber_alt)
    saber_alt = saber_alt[valid_alt]
    ds_saber = ds_saber.isel(altitude=valid_alt)

    modes = {
        'DW1': {'period': np.timedelta64(24, 'h'), 'm': -1, 'type': 'Symmetric'},
        'SW2': {'period': np.timedelta64(12, 'h'), 'm': -2, 'type': 'Symmetric'},
        'DE3': {'period': np.timedelta64(24, 'h'), 'm': 3, 'type': 'Symmetric'},
        'SE2': {'period': np.timedelta64(12, 'h'), 'm': 2, 'type': 'Symmetric'},
    }
    
    fig, axes = plt.subplots(4, 2, figsize=(14, 18), sharex=True, sharey=True)
    
    lat_ticks = [-60, -30, 0, 30, 60]
    lat_labels = ['60°S', '30°S', '0°', '30°N', '60°N']
    vmax = 15.0 # Max temp amplitude for color scale
    levels = np.linspace(0, vmax, 16)
    
    for i, (name, meta) in enumerate(modes.items()):
        # SABER Panel
        ax_saber = axes[i, 0]
        var_saber = 'ktemp_amp_sym' if meta['type'] == 'Symmetric' else 'ktemp_amp_asy'
        
        try:
            data_saber = ds_saber[var_saber].sel(period=meta['period'], m=meta['m'])
            cf = ax_saber.contourf(ds_saber['lat'], saber_alt, data_saber, levels=levels, cmap='inferno', extend='max')
        except KeyError:
            pass
            
        ax_saber.set_title(f"SABER {name} Amplitude")
        ax_saber.set_ylabel("Altitude (km)")
        ax_saber.set_ylim(60, 110)
        ax_saber.set_xticks(lat_ticks)
        ax_saber.set_xticklabels(lat_labels)
        
        # ICON Panel
        ax_icon = axes[i, 1]
        var_icon = 'temp_amp_sym' if meta['type'] == 'Symmetric' else 'temp_amp_asy'
        
        try:
            data_icon = ds_icon[var_icon].sel(period=meta['period'], m=meta['m'])
            icon_alt_km = ds_icon['height'] / 1000.0
            cf2 = ax_icon.contourf(ds_icon['lat'], icon_alt_km, data_icon, levels=levels, cmap='inferno', extend='max')
        except KeyError:
            pass
            
        ax_icon.set_title(f"UA-ICON {name} Amplitude")
        ax_icon.set_ylim(60, 110)
        ax_icon.set_xticks(lat_ticks)
        ax_icon.set_xticklabels(lat_labels)
        
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', label='Temperature Amplitude (K)')
    
    plt.subplots_adjust(hspace=0.3, bottom=0.1)
    
    out_path = 'tides_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    plot_tides_comparison()
