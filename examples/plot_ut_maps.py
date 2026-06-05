import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import warnings

# Suppress healpy warnings
warnings.filterwarnings("ignore")

def plot_ut_maps(binned_file, alt_index=200):
    """
    Plots the UT-binned HEALPix data for different universal times.
    """
    print(f"Loading binned data from {binned_file}...")
    ds = xr.open_dataset(binned_file)

    var_name = 'ktemp'
    
    if 'ut' not in ds.dims:
        raise ValueError("Dataset does not have an 'ut' dimension.")

    # We will plot 8 UT bins (every 3 hours)
    ut_indices = [0, 3, 6, 9, 12, 15, 18, 21]
    
    nside = ds.attrs.get('healpix_nside', 16)
    
    # Get altitude value for title
    ds_raw = xr.open_dataset('../saber_healpix.nc')
    alt_val = ds_raw['tpaltitude'].isel(altitude=alt_index).values
    valid_alt = alt_val[~np.isnan(alt_val)]
    mean_alt = np.mean(valid_alt) if len(valid_alt) > 0 else 0.0

    # Get colorbar limits across all selected LSTs to have a consistent scale
    temp_data = ds[var_name].isel(altitude=alt_index).values
    vmin = np.nanmin(temp_data) if not np.all(np.isnan(temp_data)) else 150
    vmax = np.nanmax(temp_data) if not np.all(np.isnan(temp_data)) else 300
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"SABER Kinetic Temperature at ~{mean_alt:.1f} km by Universal Time (UT)", fontsize=18, fontweight='bold', y=1.02)

    for idx, ut_idx in enumerate(ut_indices):
        ut_val = ds['ut'].isel(ut=ut_idx).values
        if np.issubdtype(type(ut_val), np.timedelta64):
            ut_val = float(ut_val / np.timedelta64(1, 'h'))
        else:
            ut_val = float(ut_val)
            
        data_slice = ds[var_name].isel(altitude=alt_index, ut=ut_idx).values
        
        # Replace NaNs with UNSEEN
        hp_map = np.copy(data_slice)
        hp_map[np.isnan(hp_map)] = hp.UNSEEN
        
        # Create subplot (2 rows, 4 columns)
        # healpy sub requires (nrows, ncols, index) where index is 1-based
        hp.mollview(
            hp_map, 
            fig=fig.number, 
            sub=(2, 4, idx + 1), 
            title=f'UT: {ut_val:.1f} hr', 
            cmap='inferno', 
            min=vmin, 
            max=vmax,
            cbar=True,
            unit='Temperature / K' if idx % 4 == 0 else '',
            nest=False
        )

    plt.tight_layout()
    out_path = 'saber_ut_maps.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    binned_file = '../saber_healpix.nc'
    
    if os.path.exists(binned_file):
        plot_ut_maps(binned_file, alt_index=200)
    else:
        print("Required file not found. Make sure to run the converter first.")
