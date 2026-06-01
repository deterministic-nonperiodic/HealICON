import os
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import warnings

# Suppress healpy warnings
warnings.filterwarnings("ignore")

def plot_lst_maps(binned_file, alt_index=200):
    """
    Plots the LST-binned HEALPix data for different local solar times.
    """
    print(f"Loading binned data from {binned_file}...")
    ds = xr.open_dataset(binned_file)

    var_name = 'ktemp'
    
    if 'lst' not in ds.dims:
        raise ValueError("Dataset does not have an 'lst' dimension.")

    # We will plot 8 LST bins (every 3 hours)
    lst_indices = [0, 3, 6, 9, 12, 15, 18, 21]
    
    nside = ds.attrs.get('healpix_nside', 16)
    
    # Get altitude value for title
    alt_val = ds['tpaltitude'].isel(altitude=alt_index).values
    valid_alt = alt_val[~np.isnan(alt_val)]
    mean_alt = np.mean(valid_alt) if len(valid_alt) > 0 else 0.0

    # Get colorbar limits across all selected LSTs to have a consistent scale
    temp_data = ds[var_name].isel(altitude=alt_index).values
    vmin = np.nanmin(temp_data) if not np.all(np.isnan(temp_data)) else 150
    vmax = np.nanmax(temp_data) if not np.all(np.isnan(temp_data)) else 300
    
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"SABER Kinetic Temperature at ~{mean_alt:.1f} km by Local Solar Time (LST)", fontsize=18, fontweight='bold', y=1.02)

    for idx, lst_idx in enumerate(lst_indices):
        lst_val = ds['lst'].isel(lst=lst_idx).values
        if np.issubdtype(type(lst_val), np.timedelta64):
            lst_val = float(lst_val / np.timedelta64(1, 'h'))
        else:
            lst_val = float(lst_val)
            
        data_slice = ds[var_name].isel(altitude=alt_index, lst=lst_idx).values
        
        # Replace NaNs with UNSEEN
        hp_map = np.copy(data_slice)
        hp_map[np.isnan(hp_map)] = hp.UNSEEN
        
        # Create subplot (2 rows, 4 columns)
        # healpy sub requires (nrows, ncols, index) where index is 1-based
        hp.mollview(
            hp_map, 
            fig=fig.number, 
            sub=(2, 4, idx + 1), 
            title=f'LST: {lst_val:.1f} hr', 
            cmap='inferno', 
            min=vmin, 
            max=vmax,
            cbar=True,
            unit='Temperature / K' if idx % 4 == 0 else '',
            nest=False
        )

    plt.tight_layout()
    out_path = 'saber_lst_maps.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    binned_file = './saber_lst.nc'
    
    if os.path.exists(binned_file):
        plot_lst_maps(binned_file, alt_index=200)
    else:
        print("Required file not found. Make sure to run the converter first:")
        print("python -m healicon.cli convert <input> ./saber_lst.nc -n 16 --lst-bins 24")
