import xarray as xr
import warnings
from healicon.visualize import plot_tides

# Suppress timedelta warnings
warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == '__main__':
    # Load dataset 
    input_file = 'jawara_tides_202303_zm.nc' # 'output_tides_zm.nc'
    print(f"Loading {input_file}...")
    
    try:
        ds = xr.open_dataset(input_file)
        
        # Use the healicon visualization module to generate publication-ready plots
        plot_tides(ds, out_dir=".", prefix="jawara_tides_2023")
        
        print("Done.")
    except FileNotFoundError:
        print(f"Error: Could not find {input_file}. Please ensure you have generated it first.")
