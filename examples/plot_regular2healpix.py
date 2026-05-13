import xarray as xr
from healicon.interpolate import interpolate_to_healpix
from plot import plot_quicklook
import os

def main():
    input_path = "/media/deterministic-nonperiodic/DATA/vortex/data/FALCON"
    input_file = f"{input_path}/UA-ICON_NWP_atm_DOM01_ML_20250219T030000Z.nc"
    
    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please ensure the path is correct.")
        return
        
    print(f"Opening original dataset: {input_file}")
    ds_orig = xr.open_dataset(input_file)
    
    # We will interpolate just the variable 'u' (zonal wind) at the first height level and first time step
    # to save memory and time for this example.
    print("Selecting zonal wind ('u') at the first height level...")
    # 'u' has dims (time, height, lat, lon)
    # We keep only 'u', and for speed we can subset the original dataset
    ds_subset = ds_orig[['u']].isel(time=0, height=20) # 10th level just to see some structure
    # Adding coords back because isel dropped scalar coords, but xarray keeps them if we just keep the ds
    
    nside = 256
    print(f"Interpolating to HEALPix grid (nside={nside})...")
    
    # We need to pass the ds with 'u' but keeping spatial dims
    # The interpolation function handles Dask chunks automatically
    ds_interp = interpolate_to_healpix(ds_subset, nside=nside)
    
    # Since we are interpolating lazily with dask, compute the result now
    print("Computing interpolation result...")
    ds_interp = ds_interp.compute()
    ds_orig_computed = ds_subset.compute()
    
    # Plot the comparison
    save_path = "u_wind_comparison.png"
    print(f"Generating quicklook plot and saving to {save_path}...")
    
    # The plot_quicklook handles 1D time/height correctly if passed a full dataset, 
    # but since we already subsetted height and time, they are scalars.
    plot_quicklook(ds_orig_computed, ds_interp, var_name='u', height_idx=0, time_idx=0, save_path=save_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
