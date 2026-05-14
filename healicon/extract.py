import logging

import healpy as hp
import numpy as np
import xarray as xr

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
                dask_gufunc_kwargs={'output_sizes': {'lon': len(lons)}, 'allow_rechunk': True}
            )
            out_ds[var] = da

            # Carry over attributes
            out_ds[var].attrs = ds[var].attrs

            # Carry over non-spatial coords
            for coord in ds[var].coords:
                if coord not in ['cell', 'lon', 'lat'] and coord in ds.coords:
                    out_ds.coords[coord] = ds.coords[coord]
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs

    # Copy global attributes
    out_ds.attrs = ds.attrs
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs[
        'history'] = f"{history}{sep}Extracted along latitude {lat} with {num_lons} longitude points."

    return out_ds


def extract_along_longitude(ds: xr.Dataset, lon: float, num_lats: int = None) -> xr.Dataset:
    """
    Extracts data along all latitudes for a specific longitude from a HEALPix dataset.
    """
    if 'cell' not in ds.dims:
        raise ValueError("Dataset must have a 'cell' dimension representing the HEALPix grid.")

    npix = ds.sizes['cell']
    nside = hp.npix2nside(npix)

    if num_lats is None:
        num_lats = 4 * nside

    logger.info(f"Extracting data along longitude {lon} with {num_lats} latitude points.")

    # Generate target points
    lats = np.linspace(-90, 90, num_lats)
    theta = np.deg2rad(90.0 - lats)
    phis = np.full_like(theta, np.deg2rad(lon))

    out_ds = xr.Dataset(
        coords={
            'lat': lats,
            'lon': lon
        }
    )
    out_ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    out_ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}

    for var in ds.data_vars:
        if 'cell' in ds[var].dims:
            da = xr.apply_ufunc(
                _interp_healpix,
                ds[var],
                kwargs={'theta': theta, 'phi': phis},
                input_core_dims=[['cell']],
                output_core_dims=[['lat']],
                dask="parallelized",
                output_dtypes=[ds[var].dtype],
                dask_gufunc_kwargs={'output_sizes': {'lat': len(lats)}, 'allow_rechunk': True}
            )
            out_ds[var] = da
            out_ds[var].attrs = ds[var].attrs

            for coord in ds[var].coords:
                if coord not in ['cell', 'lon', 'lat'] and coord in ds.coords:
                    out_ds.coords[coord] = ds.coords[coord]
        else:
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs

    out_ds.attrs = ds.attrs
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs[
        'history'] = f"{history}{sep}Extracted along longitude {lon} with {num_lats} latitude points."

    return out_ds


def _zonal_mean_block(data_block, ring_indices, n_rings):
    """
    Compute the mean over HEALPix rings for a chunk of data.
    data_block: shape (..., npix)
    """
    orig_shape = data_block.shape
    npix = orig_shape[-1]
    data_2d = data_block.reshape(-1, npix)

    out_data = np.zeros((data_2d.shape[0], n_rings), dtype=data_2d.dtype)
    counts = np.bincount(ring_indices, minlength=n_rings)

    for i in range(data_2d.shape[0]):
        sums = np.bincount(ring_indices, weights=data_2d[i], minlength=n_rings)
        out_data[i] = sums / counts

    out_shape = orig_shape[:-1] + (n_rings,)
    return out_data.reshape(out_shape)


def zonal_mean(ds: xr.Dataset) -> xr.Dataset:
    """
    Computes the zonal mean (average along longitudes for each latitude ring).
    """
    if 'cell' not in ds.dims:
        raise ValueError("Dataset must have a 'cell' dimension.")

    npix = ds.sizes['cell']
    nside = hp.npix2nside(npix)
    n_rings = 4 * nside - 1

    logger.info(f"Computing zonal mean over {n_rings} latitude rings.")

    # Precompute ring indices and latitudes
    ring_indices = hp.pix2ring(nside, np.arange(npix)) - 1
    theta, _ = hp.pix2ang(nside, np.arange(npix))

    lats = np.zeros(n_rings)
    for i in range(n_rings):
        lats[i] = 90.0 - np.rad2deg(theta[ring_indices == i][0])

    out_ds = xr.Dataset(coords={'lat': lats})
    out_ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}

    for var in ds.data_vars:
        if 'cell' in ds[var].dims:
            da = xr.apply_ufunc(
                _zonal_mean_block,
                ds[var],
                kwargs={'ring_indices': ring_indices, 'n_rings': n_rings},
                input_core_dims=[['cell']],
                output_core_dims=[['lat']],
                dask="parallelized",
                output_dtypes=[ds[var].dtype],
                dask_gufunc_kwargs={'output_sizes': {'lat': n_rings}, 'allow_rechunk': True}
            )
            out_ds[var] = da.assign_coords(lat=lats)
            out_ds[var].attrs = ds[var].attrs

            for coord in ds[var].coords:
                if coord not in ['cell', 'lat', 'lon'] and coord in ds.coords:
                    out_ds.coords[coord] = ds.coords[coord]
        else:
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs

    out_ds.attrs = ds.attrs
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs['history'] = f"{history}{sep}Computed zonal mean over HEALPix rings."
    return out_ds


def extract_point(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    """
    Extracts the full time/height profile for the HEALPix pixel closest to the given lat/lon.
    """
    if 'cell' not in ds.dims:
        raise ValueError("Dataset must have a 'cell' dimension.")

    npix = ds.sizes['cell']
    nside = hp.npix2nside(npix)

    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)

    pix = hp.ang2pix(nside, theta, phi, nest=False)

    logger.info(f"Extracting point data at lat={lat}, lon={lon} (Mapped to HEALPix cell {pix}).")

    out_ds = ds.isel(cell=pix)

    out_ds = out_ds.assign_coords(lat=lat, lon=lon)
    out_ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    out_ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}

    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs[
        'history'] = f"{history}{sep}Extracted point data for lat={lat}, lon={lon} (pixel {pix})."

    return out_ds
