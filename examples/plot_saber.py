import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import warnings

# Suppress healpy warnings
warnings.filterwarnings("ignore")

def plot_saber_comparison(orig_file, binned_file, alt_index=200):
    """
    Plots the original SABER orbit tracks vs the binned HEALPix grid.
    """
    print(f"Loading original data from {orig_file}...")
    ds_orig = xr.open_dataset(orig_file)
    
    print(f"Loading binned data from {binned_file}...")
    ds_binned = xr.open_dataset(binned_file)

    # We will visualize Kinetic Temperature (ktemp)
    var_name = 'ktemp'
    
    # 1. Original Data (Altitude Slice)
    orig_temp = ds_orig[var_name].isel(altitude=alt_index).values
    orig_lat = ds_orig['tplatitude'].isel(altitude=alt_index).values
    orig_lon = ds_orig['tplongitude'].isel(altitude=alt_index).values
    
    # Filter out missing values (-999.0 or NaN)
    missing_val = ds_orig[var_name].attrs.get('missing_value', -999.0)
    valid = (orig_temp != missing_val) & (~np.isnan(orig_temp)) & (orig_lat != missing_val) & (orig_lon != missing_val)
    
    orig_temp = orig_temp[valid]
    orig_lat = orig_lat[valid]
    orig_lon = orig_lon[valid]
    
    # Convert longitudes to -180 to 180 for standard map plotting
    orig_lon = (orig_lon + 180) % 360 - 180

    # 2. Binned HEALPix Data (Altitude Slice)
    binned_temp = ds_binned[var_name].isel(altitude=alt_index).values
    nside = ds_binned.attrs.get('healpix_nside', 16)
    npix = hp.nside2npix(nside)
    
    # Replace NaNs with healpy's UNSEEN value for proper plotting
    hp_map = np.copy(binned_temp)
    hp_map[np.isnan(hp_map)] = hp.UNSEEN

    # Get altitude value for title
    alt_val = ds_orig['tpaltitude'].isel(altitude=alt_index).values
    valid_alt = alt_val[(alt_val != missing_val) & (~np.isnan(alt_val))]
    mean_alt = np.mean(valid_alt) if len(valid_alt) > 0 else 0.0

    # Setup figure
    fig = plt.figure(figsize=(14, 10))
    vmin = np.nanmin(orig_temp) if len(orig_temp) > 0 else 150
    vmax = np.nanmax(orig_temp) if len(orig_temp) > 0 else 300

    # Top Panel: Original Scatter Plot
    ax1 = fig.add_subplot(2, 1, 1)
    
    sc = ax1.scatter(orig_lon, orig_lat, c=orig_temp, cmap='inferno', s=5, vmin=vmin, vmax=vmax, alpha=0.7)
    
    ax1.set_title(f'Original SABER Orbit Paths (Altitude ~{mean_alt:.1f} km)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-90, 90)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Add colorbar for scatter
    cbar1 = fig.colorbar(sc, ax=ax1, orientation='vertical', pad=0.02)
    cbar1.set_label('Kinetic Temperature / K', fontsize=12)

    # Bottom Panel: HEALPix Mollweide Projection
    # healpy.mollview creates its own axis, so we just specify the sub subplot number
    hp.mollview(
        hp_map, 
        fig=fig.number, 
        sub=(2, 1, 2), 
        title=f'Binned HEALPix Grid (nside={nside})', 
        cmap='inferno', 
        min=vmin, 
        max=vmax,
        unit='Kinetic Temperature / K',
        nest=False,
        xsize=1600
    )

    plt.tight_layout()
    
    out_path = 'saber_comparison_plot.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    orig_file = '/media/deterministic-nonperiodic/DATA/ORIGIN/SABER_Temp_O3_H2O_January2025_v2.0.nc'
    binned_file = '/tmp/saber_out_v2.nc'
    
    if os.path.exists(orig_file) and os.path.exists(binned_file):
        plot_saber_comparison(orig_file, binned_file, alt_index=200)
    else:
        print("Required files not found. Make sure to run the converter first:")
        print(f"python -m healicon.cli convert {orig_file} {binned_file} -n 16")
