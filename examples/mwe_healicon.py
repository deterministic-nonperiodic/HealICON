import xarray as xr

from healicon.interpolate import interpolate_to_healpix
from plot import plot_quicklook

ds = xr.open_dataset("icon_grid_mwe.nc").isel(height=20, time=-1)

ds_hp = interpolate_to_healpix(ds, nside=64)

save_path = "u_wind_new_comparison.png"

plot_quicklook(
        ds[["u"]], 
        ds_hp, 
        var_name='u', 
        height_idx=0, 
        time_idx=0, 
        save_path=save_path,
        plot_nodes=False,
        node_subsample=100,
        orig_title='Original Icosahedral Grid (R2B04)'
    )
