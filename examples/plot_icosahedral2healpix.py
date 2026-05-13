import xarray as xr
from healicon.interpolate import interpolate_to_healpix
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
    
    # We will interpolate just the variable 'u' at the first height level and first time step
    print("Selecting zonal wind ('u') at the first height level...")
    # 'u' has dims (time, height, cell) or similar
    ds_subset = ds_orig[['u']].isel(time=0, height=0) 
    
    nside = 128
    print(f"Interpolating to HEALPix grid (nside={nside})...")
    
    ds_interp = interpolate_to_healpix(ds_subset, nside=nside)
    
    # Compute the result
    print("Computing interpolation result...")
    ds_interp = ds_interp.compute()
    ds_orig_computed = ds_subset.compute()
    
    # Plot the comparison
    save_path = "u_wind_new_comparison.png"
    print(f"Generating quicklook plot and saving to {save_path}...")
    
    plot_quicklook(
        ds_orig_computed, 
        ds_interp, 
        var_name='u', 
        height_idx=0, 
        time_idx=0, 
        save_path=save_path,
        plot_nodes=False,
        node_subsample=100,
        orig_title='Original Icosahedral Grid (R2B07)'
    )
    
    print("Done!")

if __name__ == "__main__":
    main()
