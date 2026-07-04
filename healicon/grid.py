import logging
import math

import healpy as hp
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# --- Canonical coordinate name constants (single source of truth) ---
LONLAT_COORD_NAMES = ("lon", "longitude", "clon", "lat", "latitude", "clat")
CELL_DIM_NAMES = ("cells", "cell", "ncells", "healpix_index", "x")


def _is_valid_npix(n: int) -> bool:
    """Return True if *n* is a valid HEALPix pixel count (12 * nside²)."""
    if n < 12:
        return False
    nside = math.isqrt(n // 12)
    return 12 * nside * nside == n and hp.isnsideok(nside, nest=True)


def get_healpix_coords(nside: int):
    """
    Generate longitude and latitude coordinates for a HEALPix grid.
    Returns:
        lon: numpy array of longitudes in degrees [0, 360]
        lat: numpy array of latitudes in degrees [-90, 90]
    """
    npix = hp.nside2npix(nside)

    # healpy pix2ang returns colatitude (theta) and longitude (phi) in radians
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # Convert colatitude to latitude (-90 to 90)
    lat = 90.0 - np.rad2deg(theta)

    # Convert longitude to degrees (0 to 360)
    lon = np.rad2deg(phi)

    return lon, lat


def create_healpix_dataset(nside: int) -> xr.Dataset:
    """
    Creates an empty xarray Dataset with HEALPix coordinates.
    """
    lon, lat = get_healpix_coords(nside)
    npix = len(lon)

    ds = xr.Dataset(
        coords={
            "cells": np.arange(npix),
        }
    )
    ds["lon"] = ("cells", lon)
    ds["lat"] = ("cells", lat)
    ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}

    return ds


def get_healpix_order(ds: xr.Dataset | xr.DataArray) -> str:
    """
    Get the HEALPix ordering from the dataset attributes.
    Returns 'nested' or 'ring'.
    """
    if isinstance(ds, xr.Dataset):
        # 1. Check for the CF grid mapping variable (most reliable)
        for name, var in ds.variables.items():
            if var.attrs.get('grid_mapping_name') == 'healpix':
                return var.attrs.get('healpix_order', 'ring').lower()
    # 2. Check global attribute
    return ds.attrs.get('healpix_scheme', 'ring').lower()


def is_healpix(ds: xr.Dataset) -> bool:
    """Return True if *ds* appears to be on a HEALPix grid.

    Checks three independent signals (any one is sufficient):
    1. A grid mapping variable with ``grid_mapping_name == 'healpix'``.
    2. A ``healpix_scheme`` global attribute.
    3. A candidate spatial dimension whose size is a valid HEALPix pixel
       count (``12 * nside²`` with ``nside`` a power of 2).
    """
    # Signal 1: grid mapping variable
    for var in ds.variables.values():
        if var.attrs.get('grid_mapping_name') == 'healpix':
            return True
    # Signal 2: global attribute
    if 'healpix_scheme' in ds.attrs:
        return True
    # Signal 3: data variable with grid_mapping = 'healpix'
    for var in ds.data_vars.values():
        if var.attrs.get('grid_mapping') == 'healpix':
            return True
    # Signal 4: spatial dim with valid HEALPix pixel count
    for dim in CELL_DIM_NAMES:
        if dim in ds.dims and _is_valid_npix(ds.sizes[dim]):
            return True
    return False


def add_healpix_grid_mapping(ds: xr.Dataset, nside: int,
                             order: str = 'ring') -> xr.Dataset:
    """Ensure *ds* carries the canonical HEALPix grid mapping metadata.

    Adds / updates:
    * The scalar ``healpix`` variable with CF grid mapping attributes.
    * ``grid_mapping = 'healpix'`` on every data variable that has the
      spatial cell dimension.
    * Global convenience attributes (``healpix_nside``, ``healpix_npix``,
      ``healpix_scheme``, ``healpix_cell_area_sr``).
    """
    order_lower = order.lower()
    npix = hp.nside2npix(nside)
    cell_area = 4.0 * np.pi / npix

    # CF grid mapping variable
    ds['healpix'] = xr.DataArray(
        np.int32(0),
        attrs={
            'grid_mapping_name': 'healpix',
            'healpix_nside': np.int32(nside),
            'healpix_order': order_lower,
        },
    )

    # Global convenience attributes
    ds.attrs['healpix_nside'] = nside
    ds.attrs['healpix_npix'] = npix
    ds.attrs['healpix_scheme'] = order_lower.upper()  # 'RING' / 'NESTED'
    ds.attrs['healpix_cell_area_sr'] = f'{cell_area:.6e}'
    _R_EARTH_KM = 6371.0  # mean Earth radius
    resolution_km = np.sqrt(cell_area) * _R_EARTH_KM
    ds.attrs['healpix_resolution_km'] = f'{resolution_km:.1f}'

    # Tag every spatial data variable
    try:
        cell_dim = get_cells_dim(ds)
    except ValueError:
        return ds

    for var in ds.data_vars:
        if var == 'healpix':
            continue
        if cell_dim in ds[var].dims:
            ds[var].attrs['grid_mapping'] = 'healpix'
            # Drop stale CDO attributes that confuse downstream tools
            ds[var].attrs.pop('CDI_grid_type', None)
            ds[var].attrs.pop('number_of_grid_in_reference', None)

    return ds


def ensure_ring(data: np.ndarray, order: str) -> np.ndarray:
    """
    Ensure the data array is in RING ordering.
    If order is 'nested', converts to 'ring'.
    Assumes HEALPix dimension is the last axis.
    """
    if order == 'nested':
        return hp.reorder(data, n2r=True)
    return data


def ensure_original_order(data: np.ndarray, original_order: str) -> np.ndarray:
    """
    Convert RING array back to its original order if it was 'nested'.
    Assumes HEALPix dimension is the last axis.
    """
    if original_order == 'nested':
        if data.ndim == 1:
            return hp.reorder(data, r2n=True)
        orig_shape = data.shape
        npix = orig_shape[-1]
        flat_data = data.reshape(-1, npix)
        reordered = np.array([hp.reorder(row, r2n=True) for row in flat_data])
        return reordered.reshape(orig_shape)
    return data


def get_cells_dim(ds: xr.Dataset | xr.DataArray) -> str:
    """Return the HEALPix spatial dimension name.

    Search order:
    1. Candidate names in ``CELL_DIM_NAMES`` whose size is a valid HEALPix
       pixel count (``12 * nside²``) — returned immediately.
    2. First candidate name that exists, even if size is unrecognised.
    3. Any dimension (regardless of name) whose size is a valid HEALPix
       pixel count — handles CDO output with unexpected dimension names.
    """
    # Pass 1 & 2: preferred canonical names
    fallback = None
    for dim in CELL_DIM_NAMES:
        if dim in ds.dims:
            if fallback is None:
                fallback = dim
            if _is_valid_npix(ds.sizes[dim]):
                return dim
    if fallback is not None:
        return fallback

    # Pass 3: scan every dimension for a valid HEALPix pixel count.
    # This catches CDO-produced files that use non-standard dim names
    # (e.g. a future CDO version that changes the naming convention).
    for dim, size in ds.sizes.items():
        if _is_valid_npix(size):
            return str(dim)

    raise ValueError(
        f"Dataset must have a HEALPix spatial dimension (one of {CELL_DIM_NAMES} "
        f"or any dim whose size equals 12·nside²). "
        f"Available dims: {dict(ds.sizes)}")


def get_cells_dim_da(da: xr.DataArray) -> str:
    """Like :func:`get_cells_dim` but accepts a :class:`DataArray`."""
    return get_cells_dim(da.to_dataset(name='__tmp__'))


def get_spatial_dims(ds: xr.Dataset) -> list[str]:
    """
    Return the list of spatial dimension names present in the dataset,
    by inspecting known lon/lat coordinate names and the cell dimension.
    """
    from .cf_coords import _find_coordinate
    spatial_dims = []

    lon_coord = _find_coordinate(ds, 'lon', raise_notfound=False)
    if lon_coord is not None:
        for dim in lon_coord.dims:
            if dim not in spatial_dims:
                spatial_dims.append(dim)

    lat_coord = _find_coordinate(ds, 'lat', raise_notfound=False)
    if lat_coord is not None:
        for dim in lat_coord.dims:
            if dim not in spatial_dims:
                spatial_dims.append(dim)

    try:
        cell_dim = get_cells_dim(ds)
        if cell_dim not in spatial_dims:
            spatial_dims.append(cell_dim)
    except ValueError:
        pass
    return spatial_dims


def append_history(ds_attrs: dict, msg: str) -> dict:
    """Return a copy of ds_attrs with msg appended to 'history'."""
    attrs = dict(ds_attrs)
    prev = attrs.get('history', '')
    attrs['history'] = f"{prev}\n{msg}" if prev else msg
    return attrs
