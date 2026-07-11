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


def _zonal_mean_block(data_block, sort_order, ring_indices, ring_boundaries, counts):
    """Compute zonal mean over a single (batch, npix) block.

    For RING-ordered data (*sort_order* is ``None``) pixels already sit in ring
    order, so ``np.add.reduceat`` gives a fully vectorised reduction with no
    data copy — ~10x faster than the bincount loop.

    For NESTED-ordered data pixels are scattered across rings; sequential
    ``np.bincount`` reads the (contiguous) data row-by-row and scatters into
    a tiny n_rings output that fits in cache, which is faster than reordering
    the full array first.

    *counts* and *ring_boundaries* are precomputed once in :func:`zonal_mean`.
    """
    orig_shape = data_block.shape
    data_2d = data_block.reshape(-1, data_block.shape[-1])
    n_rings = len(ring_boundaries)

    if sort_order is None:
        # RING ordering: pixels are already contiguous within each ring.
        ring_sums = np.add.reduceat(data_2d, ring_boundaries, axis=1)
        out = ring_sums / counts
    else:
        # NESTED ordering: sequential bincount scatter into small output.
        out = np.empty((data_2d.shape[0], n_rings), dtype=data_2d.dtype)
        for i in range(data_2d.shape[0]):
            sums = np.bincount(ring_indices, weights=data_2d[i], minlength=n_rings)
            out[i] = sums / counts

    return out.reshape(orig_shape[:-1] + (n_rings,))


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

    if is_nested:
        # NESTED: pixels are scattered across rings — must sort to find boundaries.
        sort_order = np.argsort(ring_indices, kind='stable')
        sorted_rings = ring_indices[sort_order]
    else:
        # RING: pixels are already in ring order — skip the O(npix log npix) sort.
        sort_order = None
        sorted_rings = ring_indices

    ring_boundaries = np.searchsorted(sorted_rings, np.arange(n_rings)).astype(np.intp)
    ring_ends = np.append(ring_boundaries[1:], npix)
    counts = (ring_ends - ring_boundaries)

    # Latitude of each ring: theta of the first pixel belonging to that ring.
    first_pixels = sort_order[ring_boundaries] if sort_order is not None else ring_boundaries
    theta_first, _ = hp.pix2ang(nside, first_pixels, nest=is_nested)
    lats = 90.0 - np.rad2deg(theta_first)

    out_ds = xr.Dataset(coords={'lat': lats})
    out_ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}

    # Identify geographic coordinate variables (per-pixel lon/lat arrays that
    # live in ds.data_vars rather than ds.coords).  These lose their meaning
    # once the cells dimension is gone, so we must not carry them over.
    # We check both branches (cells-dim vars AND pass-through vars).
    from .cf_coords import _find_coordinate
    _geo_var_names: set[str] = set()
    for _cf_type in ('lat', 'lon'):
        _c = _find_coordinate(ds, _cf_type, raise_notfound=False)
        if _c is not None and _c.name in ds.data_vars:
            _geo_var_names.add(_c.name)

    for var in ds.data_vars:
        # Skip geographic coordinate arrays in ALL branches — they are per-pixel
        # and have no meaning in the lat-averaged output.
        if var in _geo_var_names:
            logger.debug(f"zonal_mean: dropping geographic coordinate variable '{var}'.")
            continue

        if cell_dim in ds[var].dims:
            da = xr.apply_ufunc(
                _zonal_mean_block,
                ds[var],
                kwargs={'sort_order': sort_order, 'ring_indices': ring_indices,
                        'ring_boundaries': ring_boundaries, 'counts': counts},
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
            # Skip the HEALPix grid-mapping scalar — it describes the original
            # unstructured grid and is no longer valid after the zonal mean.
            if ds[var].dims == () and ds[var].attrs.get('grid_mapping_name') == 'healpix':
                logger.debug(f"zonal_mean: dropping HEALPix grid-mapping variable '{var}'.")
                continue
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs

    # Strip HEALPix-specific global attributes so the output is not
    # misidentified as a HEALPix dataset by is_healpix() or CDO.
    _HEALPIX_ATTRS = {'healpix_nside', 'healpix_npix', 'healpix_scheme',
                      'healpix_cell_area_sr', 'healpix_order', 'healpix_resolution_km'}
    out_attrs = {k: v for k, v in ds.attrs.items() if k not in _HEALPIX_ATTRS}
    out_ds.attrs = out_attrs
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
