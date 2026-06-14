import logging

import healpy as hp
import numpy as np
import xarray as xr

from .grid import get_healpix_order, get_cells_dim, append_history

logger = logging.getLogger(__name__)


def _interp_healpix(data, theta, phi, is_nested=False):
    """
    Helper function to interpolate HEALPix data along an axis.
    data: shape (..., npix)
    """
    orig_shape = data.shape
    npix = orig_shape[-1]
    data_2d = data.reshape(-1, npix)

    interp_vals = hp.get_interp_val(data_2d, theta, phi, nest=is_nested)

    out_shape = orig_shape[:-1] + (len(theta),)
    return interp_vals.reshape(out_shape)


def extract_along_latitude(ds: xr.Dataset, lat: float, num_lons: int | None = None) -> xr.Dataset:
    """
    Extracts data along a specific latitude from a HEALPix dataset.

    Args:
        ds: Input xarray.Dataset on a HEALPix grid over time.
        lat: Latitude to extract (degrees).
        num_lons: Number of longitude points to interpolate.
    """
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'

    npix = ds.sizes[cell_dim]
    try:
        nside = hp.npix2nside(npix)
        if not hp.isnsideok(nside):
            raise ValueError()
    except Exception:
        raise ValueError(f"Number of cells ({npix}) is not a valid HEALPix npix.")

    if num_lons is None:
        num_lons = npix

    logger.info(f"Extracting data along latitude {lat} with {num_lons} longitude points.")

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
        if cell_dim in ds[var].dims:
            da = xr.apply_ufunc(
                _interp_healpix,
                ds[var],
                kwargs={'theta': thetas, 'phi': phi, 'is_nested': is_nested},
                input_core_dims=[[cell_dim]],
                output_core_dims=[['lon']],
                dask="parallelized",
                output_dtypes=[ds[var].dtype],
                dask_gufunc_kwargs={'output_sizes': {'lon': len(lons)}, 'allow_rechunk': True}
            )
            out_ds[var] = da
            out_ds[var].attrs = ds[var].attrs

            for coord in ds[var].coords:
                if coord not in [cell_dim, 'lon', 'lat'] and coord in ds.coords:
                    out_ds.coords[coord] = ds.coords[coord]
        else:
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs

    out_ds.attrs = ds.attrs
    out_ds.attrs = append_history(out_ds.attrs,
                                  f"Extracted along latitude {lat} with {num_lons} longitude points.")

    return out_ds


def extract_along_longitude(ds: xr.Dataset, lon: float, num_lats: int | None = None) -> xr.Dataset:
    """
    Extracts data along a specific longitude from a HEALPix dataset.

    Args:
        ds: Input xarray.DataArray on a HEALPix grid over time.
        lon: Longitude to extract (degrees).
        num_lats: Number of latitude points to interpolate.
    """
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'

    npix = ds.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    if num_lats is None:
        num_lats = 4 * nside

    logger.info(f"Extracting data along longitude {lon} with {num_lats} latitude points.")

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
        if cell_dim in ds[var].dims:
            da = xr.apply_ufunc(
                _interp_healpix,
                ds[var],
                kwargs={'theta': theta, 'phi': phis, 'is_nested': is_nested},
                input_core_dims=[[cell_dim]],
                output_core_dims=[['lat']],
                dask="parallelized",
                output_dtypes=[ds[var].dtype],
                dask_gufunc_kwargs={'output_sizes': {'lat': len(lats)}, 'allow_rechunk': True}
            )
            out_ds[var] = da
            out_ds[var].attrs = ds[var].attrs

            for coord in ds[var].coords:
                if coord not in [cell_dim, 'lon', 'lat'] and coord in ds.coords:
                    out_ds.coords[coord] = ds.coords[coord]
        else:
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs

    out_ds.attrs = ds.attrs
    out_ds.attrs = append_history(out_ds.attrs,
                                  f"Extracted along longitude {lon} with {num_lats} latitude points.")

    return out_ds


def _zonal_mean_block(data_block, ring_indices, n_rings):
    """
    Helper function to compute zonal mean over a block of data.
    
    Args:
        data_block: Input xarray.DataArray on a HEALPix grid over time.
        ring_indices: Ring indices for each pixel.
        n_rings: Number of rings.
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
    Computes the zonal mean of a HEALPix dataset over latitude rings.
    
    Args:
        ds: Input xarray.DataArray on a HEALPix grid over time.
    """
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'

    npix = ds.sizes[cell_dim]
    nside = hp.npix2nside(npix)
    n_rings = 4 * nside - 1

    logger.info(f"Computing zonal mean over {n_rings} latitude rings.")

    ring_indices = hp.pix2ring(nside, np.arange(npix), nest=is_nested) - 1
    theta, _ = hp.pix2ang(nside, np.arange(npix), nest=is_nested)

    sort_order = np.argsort(ring_indices, kind='stable')
    ring_first = np.searchsorted(ring_indices[sort_order], np.arange(n_rings))
    lats = 90.0 - np.rad2deg(theta[sort_order[ring_first]])

    out_ds = xr.Dataset(coords={'lat': lats})
    out_ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}

    for var in ds.data_vars:
        if cell_dim in ds[var].dims:
            da = xr.apply_ufunc(
                _zonal_mean_block,
                ds[var],
                kwargs={'ring_indices': ring_indices, 'n_rings': n_rings},
                input_core_dims=[[cell_dim]],
                output_core_dims=[['lat']],
                dask="parallelized",
                output_dtypes=[ds[var].dtype],
                dask_gufunc_kwargs={'output_sizes': {'lat': n_rings}, 'allow_rechunk': True}
            )
            out_ds[var] = da.assign_coords(lat=lats)
            out_ds[var].attrs = ds[var].attrs

            for coord in ds[var].coords:
                if coord not in [cell_dim, 'lat', 'lon'] and coord in ds.coords:
                    out_ds.coords[coord] = ds.coords[coord]
        else:
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs

    out_ds.attrs = ds.attrs
    out_ds.attrs = append_history(out_ds.attrs, "Computed zonal mean over HEALPix rings.")
    return out_ds


def extract_point(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    """
    Extracts data at a specific point from a HEALPix dataset.

    Args:
        ds: Input xarray.DataArray on a HEALPix grid over time.
        lat: Latitude of the point to extract (degrees).
        lon: Longitude of the point to extract (degrees).
    """
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'

    npix = ds.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)

    pix = hp.ang2pix(nside, theta, phi, nest=is_nested)

    logger.info(f"Extracting point data at lat={lat}, lon={lon} (Mapped to HEALPix cell {pix}).")

    out_ds = ds.isel({cell_dim: pix})

    out_ds = out_ds.assign_coords(lat=lat, lon=lon)
    out_ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    out_ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}

    out_ds.attrs = append_history(out_ds.attrs,
                                  f"Extracted point data for lat={lat}, lon={lon} (pixel {pix}).")

    return out_ds
