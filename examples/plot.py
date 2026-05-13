import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np
import logging

logger = logging.getLogger(__name__)

def plot_quicklook(ds_orig: xr.Dataset, ds_interp: xr.Dataset, var_name: str, height_idx: int = 0, time_idx: int = 0, save_path: str = None, plot_nodes: bool = False, node_subsample: int = 10, orig_title: str = 'Original Grid'):
    """
    Plots a side-by-side comparison of the original dataset and the interpolated HEALPix dataset.
    
    Args:
        ds_orig: Original xarray Dataset
        ds_interp: Interpolated HEALPix xarray Dataset
        var_name: Name of the variable to plot
        height_idx: Index of the height/level dimension to plot
        time_idx: Index of the time dimension to plot
        save_path: Optional path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Extract data for original
    da_orig = ds_orig[var_name]
    if 'time' in da_orig.dims:
        da_orig = da_orig.isel(time=time_idx)
    # Handle possible height dimensions
    for dim in ['height', 'height_2', 'level']:
        if dim in da_orig.dims:
            da_orig = da_orig.isel({dim: height_idx})
            break
            
    # Extract data for interpolated
    da_interp = ds_interp[var_name]
    if 'time' in da_interp.dims:
        da_interp = da_interp.isel(time=time_idx)
    for dim in ['height', 'height_2', 'level']:
        if dim in da_interp.dims:
            da_interp = da_interp.isel({dim: height_idx})
            break
            
    # Determine common color limits and colormap
    if var_name in ['u', 'v', 'w', 'omega']:
        vmax = max(abs(np.nanmin(da_orig.values)), abs(np.nanmax(da_orig.values)))
        vmin = -vmax
        cmap = 'RdYlBu_r'
    else:
        vmin = min(np.nanmin(da_orig.values), np.nanmin(da_interp.values))
        vmax = max(np.nanmax(da_orig.values), np.nanmax(da_interp.values))
        cmap = 'viridis'
    
    # Common features for axes
    for ax in [ax1, ax2]:
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', alpha=0.7)
        gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
    # Plot original
    ax1.set_title(orig_title, fontsize=14, pad=10)
    
    # Detect grid type for original plotting
    lon_name, lat_name = None, None
    for name in ["lon", "longitude", "clon"]:
        if name in ds_orig.coords or name in ds_orig.variables:
            lon_name = name
            break
    for name in ["lat", "latitude", "clat"]:
        if name in ds_orig.coords or name in ds_orig.variables:
            lat_name = name
            break
            
    if lon_name and lat_name:
        lon_orig = ds_orig[lon_name].values
        lat_orig = ds_orig[lat_name].values
        
        lon_units = str(ds_orig[lon_name].attrs.get('units', '')).lower()
        lat_units = str(ds_orig[lat_name].attrs.get('units', '')).lower()
        
        is_rad = False
        if 'rad' in lon_units or 'rad' in lat_units:
            is_rad = True
        elif 'deg' not in lon_units and 'deg' not in lat_units:
            if np.nanmax(np.abs(lon_orig)) <= 2*np.pi + 0.1 and np.nanmax(np.abs(lat_orig)) <= np.pi/2 + 0.1:
                is_rad = True
                
        if is_rad:
            lon_orig = np.rad2deg(lon_orig)
            lat_orig = np.rad2deg(lat_orig)
        
        lon_min = np.nanmin(lon_orig)
        lon_max = np.nanmax(lon_orig)
        lat_min = np.nanmin(lat_orig)
        lat_max = np.nanmax(lat_orig)
        
        # Add a small buffer to the extent
        lon_buffer = (lon_max - lon_min) * 0.05 if lon_max > lon_min else 1.0
        lat_buffer = (lat_max - lat_min) * 0.05 if lat_max > lat_min else 1.0
        
        extent = [lon_min - lon_buffer, lon_max + lon_buffer, lat_min - lat_buffer, lat_max + lat_buffer]
        ax1.set_extent(extent, crs=ccrs.PlateCarree())
        ax2.set_extent(extent, crs=ccrs.PlateCarree())
        
        if len(ds_orig[lon_name].dims) == 1 and ds_orig[lon_name].dims[0] == lon_name:
            # Regular Grid
            mesh1 = ax1.pcolormesh(lon_orig, lat_orig, da_orig.values, transform=ccrs.PlateCarree(),
                                   vmin=vmin, vmax=vmax, cmap=cmap, shading='auto')
            if plot_nodes:
                X, Y = np.meshgrid(lon_orig[::node_subsample], lat_orig[::node_subsample])
                ax1.scatter(X, Y, c='k', s=0.5, alpha=0.5, transform=ccrs.PlateCarree())
        else:
            # Unstructured Grid
            mesh1 = ax1.scatter(lon_orig, lat_orig, c=da_orig.values, transform=ccrs.PlateCarree(),
                                vmin=vmin, vmax=vmax, cmap=cmap, s=5, marker='s', edgecolors='none')
            if plot_nodes:
                ax1.scatter(lon_orig[::node_subsample], lat_orig[::node_subsample], c='k', s=0.5, alpha=0.5, transform=ccrs.PlateCarree())
    
    # Calculate nside from npix
    npix = len(ds_interp.coords.get('cell', ds_interp['lon']))
    nside = int(np.sqrt(npix / 12))
    
    # Plot Interpolated (HEALPix is unstructured)
    ax2.set_title(f'HEALPix Grid (nside={nside})', fontsize=14, pad=10)
    
    lon_interp = ds_interp['lon'].values
    lat_interp = ds_interp['lat'].values
    
    mesh2 = ax2.scatter(lon_interp, lat_interp, c=da_interp.values, transform=ccrs.PlateCarree(),
                        vmin=vmin, vmax=vmax, cmap=cmap, s=5, marker='s', edgecolors='none')
                        
    # Shared Colorbar
    cbar_ax = fig.add_axes([0.26, 0.12, 0.5, 0.03]) # [left, bottom, width, height]
    unit_label = da_orig.attrs.get('units', '')
    long_name = da_orig.attrs.get('long_name', var_name)
    fig.colorbar(mesh1, cax=cbar_ax, orientation='horizontal', label=f"{long_name} [{unit_label}]")
    
    plt.subplots_adjust(bottom=0.22, wspace=0.15)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to {save_path}")
    else:
        plt.show()
        
    return fig, (ax1, ax2)
