import numpy as np
import xarray as xr
import healpy as hp
import logging

logger = logging.getLogger(__name__)

def _interp_healpix(data, theta, phi):
    """
    Helper function to interpolate HEALPix data along an axis.
    data: shape (..., npix)
    """
    orig_shape = data.shape
    npix = orig_shape[-1]
    data_2d = data.reshape(-1, npix)
    
    # get_interp_val defaults to nest=False (RING) which matches get_healpix_coords
    interp_vals = hp.get_interp_val(data_2d, theta, phi, nest=False)
    
    out_shape = orig_shape[:-1] + (len(theta),)
    return interp_vals.reshape(out_shape)

def extract_along_latitude(ds: xr.Dataset, lat: float, num_lons: int = None) -> xr.Dataset:
    """
    Extracts data along all longitudes for a specific latitude from a HEALPix dataset.
    
    Args:
        ds: xarray Dataset on a HEALPix grid (must have a 'cell' dimension).
        lat: Target latitude in degrees [-90, 90].
        num_lons: Number of longitude points to sample (default: number of HEALPix grid points).
        
    Returns:
        xr.Dataset with a 'lon' dimension containing the interpolated values.
    """
    if 'cell' not in ds.dims:
        raise ValueError("Dataset must have a 'cell' dimension representing the HEALPix grid.")
    
    npix = ds.sizes['cell']
    try:
        nside = hp.npix2nside(npix)
        if not hp.isnsideok(nside):
            raise ValueError()
    except Exception:
        raise ValueError(f"Number of cells ({npix}) is not a valid HEALPix npix.")
        
    if num_lons is None:
        num_lons = npix
    
    logger.info(f"Extracting data along latitude {lat} with {num_lons} longitude points.")
    
    # Generate target points
    lons = np.linspace(0, 360, num_lons, endpoint=False)
    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lons)
    
    thetas = np.full_like(phi, theta)
    
    out_ds = xr.Dataset(
        coords={
            'lon': lons,
            'lat': lat
        }
    )
    out_ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    out_ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}
    
    for var in ds.data_vars:
        if 'cell' in ds[var].dims:
            da = xr.apply_ufunc(
                _interp_healpix,
                ds[var],
                kwargs={'theta': thetas, 'phi': phi},
                input_core_dims=[['cell']],
                output_core_dims=[['lon']],
                dask="parallelized",
                output_dtypes=[ds[var].dtype],
                dask_gufunc_kwargs={'output_sizes': {'lon': len(lons)}}
            )
            out_ds[var] = da
            
            # Carry over attributes
            out_ds[var].attrs = ds[var].attrs
            
            # Carry over non-spatial coords
            for coord in ds[var].coords:
                if coord not in ['cell', 'lon', 'lat'] and coord in ds.coords:
                    out_ds.coords[coord] = ds.coords[coord]
        else:
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs
            
    # Copy global attributes
    out_ds.attrs = ds.attrs
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs['history'] = f"{history}{sep}Extracted along latitude {lat} with {num_lons} longitude points."
            
    return out_ds
