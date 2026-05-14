import xarray as xr
from healicon.interpolate import interpolate_to_healpix
from healicon.analysis import filter_spatial
from plot import plot_quicklook
import os

def main():
    input_file = "UA-ICON_NWP_u_DOM01_ML_20250123T000000Z.nc"
    
    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please ensure the path is correct.")
        return
        
    print(f"Opening original dataset: {input_file}")
    ds_orig = xr.open_dataset(input_file)
    
    print("Selecting zonal wind ('u') at the first height level...")
    ds_subset = ds_orig[['u']].isel(time=0, height=0) 
    
    nside = 128
    print(f"Interpolating to HEALPix grid (nside={nside})...")
    ds_interp = interpolate_to_healpix(ds_subset, nside=nside)
    
    print("Computing interpolation result...")
    ds_interp = ds_interp.compute()
    
    lmax = 15
    print(f"Applying hard spectral low-pass filter (lmax = {lmax})...")
    ds_filtered = filter_spatial(ds_interp, lmax=lmax)
    
    print("Computing filter result...")
    ds_filtered = ds_filtered.compute()
    
    # Plot the comparison
    save_path = "u_wind_filtered_comparison_lmax.png"
    print(f"Generating quicklook plot and saving to {save_path}...")
    
    plot_quicklook(
        ds_interp, 
        ds_filtered, 
        var_name='u', 
        height_idx=0, 
        time_idx=0, 
        save_path=save_path,
        plot_nodes=False,
        orig_title=f'Unfiltered HEALPix (nside={nside})',
        interp_title=f'Filtered HEALPix (lmax={lmax})'
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
