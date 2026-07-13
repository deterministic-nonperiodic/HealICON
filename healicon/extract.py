import logging

import healpy as hp
import numpy as np
import xarray as xr

from .grid import get_healpix_order, get_cells_dim, is_healpix, append_history

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


def _unstructured_extract_along_longitude(
    ds: xr.Dataset, lon: float, cell_dim: str, num_lats: int | None = None
) -> xr.Dataset:
    """Extract along a meridian for non-HEALPix unstructured grids (e.g. ICON).

    For each target latitude point the nearest cell within a longitude band
    around *lon* is selected; the band width adapts to the grid resolution.
    """
    ncells = ds.sizes[cell_dim]
    cell_lat, cell_lon = _get_cell_latlon_deg(ds)

    if num_lats is None:
        num_lats = max(180, 4 * round(np.sqrt(ncells / 12)))
    target_lats = np.linspace(-90.0, 90.0, num_lats)

    # Pre-filter: keep only cells within a longitude band around *lon*.
    # Width adapts to grid resolution but is at least 5°.
    delta_lon = max(5.0, 720.0 / np.sqrt(ncells))
    lon_diff = np.abs((cell_lon - lon + 180.0) % 360.0 - 180.0)
    nearby = np.where(lon_diff < delta_lon)[0]
    if len(nearby) == 0:
        nearby = np.arange(ncells)  # fallback: search all cells

    # Cartesian unit-sphere coords for the nearby cells.
    theta_c = np.deg2rad(90.0 - cell_lat[nearby])
    phi_c   = np.deg2rad(cell_lon[nearby])
    x_c = np.sin(theta_c) * np.cos(phi_c)
    y_c = np.sin(theta_c) * np.sin(phi_c)
    z_c = np.cos(theta_c)

    # Cartesian coords of all target points (shape: num_lats).
    phi_q    = np.deg2rad(lon)
    theta_q  = np.deg2rad(90.0 - target_lats)
    xq = np.sin(theta_q) * np.cos(phi_q)
    yq = np.sin(theta_q) * np.sin(phi_q)
    zq = np.cos(theta_q)

    # (num_lats, n_nearby) squared distance; argmin gives nearest cell per lat.
    d2 = ((xq[:, None] - x_c)**2 + (yq[:, None] - y_c)**2 + (zq[:, None] - z_c)**2)
    pixel_indices = nearby[np.argmin(d2, axis=1)]   # (num_lats,)

    logger.info(
        f"Unstructured grid: extracting along lon={lon:.1f}° via nearest-cell "
        f"search ({num_lats} latitude points, band ±{delta_lon:.1f}°)."
    )

    out_ds = xr.Dataset(coords={'lat': target_lats, 'lon': lon})
    out_ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    out_ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}

    from .cf_coords import _find_coordinate
    _geo_var_names: set[str] = set()
    for _cf_type in ('lat', 'lon'):
        _c = _find_coordinate(ds, _cf_type, raise_notfound=False)
        if _c is not None and _c.name in ds.data_vars:
            _geo_var_names.add(_c.name)

    for var in ds.data_vars:
        if var in _geo_var_names:
            continue
        if cell_dim in ds[var].dims:
            da = xr.apply_ufunc(
                _select_pixels,
                ds[var],
                kwargs={'pixel_indices': pixel_indices},
                input_core_dims=[[cell_dim]],
                output_core_dims=[['lat']],
                dask='parallelized',
                output_dtypes=[ds[var].dtype],
                dask_gufunc_kwargs={'output_sizes': {'lat': num_lats}, 'allow_rechunk': True},
            )
            out_ds[var] = da.assign_coords(lat=target_lats)
            out_ds[var].attrs = ds[var].attrs
            for coord in ds[var].coords:
                if coord not in (cell_dim, 'lat', 'lon') and coord in ds.coords:
                    out_ds.coords[coord] = ds.coords[coord]
        else:
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs

    out_ds.attrs = ds.attrs
    out_ds.attrs = append_history(
        out_ds.attrs, f"Extracted along longitude {lon:.1f}° from unstructured grid."
    )
    return out_ds


def extract_along_longitude(ds: xr.Dataset, lon: float, num_lats: int | None = None) -> xr.Dataset:
    """Extract data along a specific longitude from an unstructured-grid dataset.

    For HEALPix grids, bilinear interpolation via ``healpy`` is used.
    For other unstructured grids (e.g. ICON icosahedral), a nearest-cell
    search within a lon band is performed.

    Args:
        ds: Input dataset on a HEALPix or unstructured grid.
        lon: Longitude to extract (degrees).
        num_lats: Number of latitude points in the output.
    """
    cell_dim = get_cells_dim(ds)

    if not is_healpix(ds):
        return _unstructured_extract_along_longitude(ds, lon, cell_dim, num_lats)

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


def _select_pixels(data_block, pixel_indices):
    """Index the last axis of *data_block* by *pixel_indices*."""
    return data_block[..., pixel_indices]


def _get_cell_latlon_deg(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat_deg, lon_deg) arrays for the cell-centre coordinates in *ds*.

    Works for both degrees and radians; detects the unit from CF attributes.
    """
    from .cf_coords import _find_coordinate
    lat_coord = _find_coordinate(ds, 'lat', raise_notfound=True)
    lon_coord = _find_coordinate(ds, 'lon', raise_notfound=True)
    lat = lat_coord.values.astype(np.float64)
    lon = lon_coord.values.astype(np.float64)
    for arr, coord in ((lat, lat_coord), (lon, lon_coord)):
        units = coord.attrs.get('units', '').strip().lower()
        if 'rad' in units:
            arr[:] = np.rad2deg(arr)
    return lat, lon


def _unstructured_zonal_mean_block(data_block, bin_indices, n_bins, counts):
    """Zonal mean kernel for generic unstructured grids (e.g. ICON icosahedral).

    Uses the same sequential ``np.bincount`` approach as the NESTED HEALPix
    path; *bin_indices* maps each cell to a latitude bin.
    """
    orig_shape = data_block.shape
    data_2d = data_block.reshape(-1, data_block.shape[-1])
    out = np.empty((data_2d.shape[0], n_bins), dtype=data_2d.dtype)
    
    has_nans = np.isnan(data_2d).any()
    if not has_nans:
        for i in range(data_2d.shape[0]):
            sums = np.bincount(bin_indices, weights=data_2d[i], minlength=n_bins)
            out[i] = sums / counts
    else:
        nan_mask = np.isnan(data_2d)
        clean_data = np.where(nan_mask, 0.0, data_2d)
        valid_counts = (~nan_mask).astype(clean_data.dtype)
        for i in range(data_2d.shape[0]):
            sums = np.bincount(bin_indices, weights=clean_data[i], minlength=n_bins)
            counts_i = np.bincount(bin_indices, weights=valid_counts[i], minlength=n_bins)
            counts_i = np.where(counts_i == 0, np.nan, counts_i)
            out[i] = sums / counts_i
            
    return out.reshape(orig_shape[:-1] + (n_bins,))


def _unstructured_zonal_mean(ds: xr.Dataset, cell_dim: str) -> xr.Dataset:
    """Zonal mean for generic unstructured grids with per-cell lat/lon coords.

    Bins cells into latitude bands of roughly equal width and computes the
    mean within each band. The number of bands is chosen to match the
    approximate grid spacing (``4 * round(sqrt(ncells / 12))``), mirroring
    the HEALPix ring count formula for an equivalent resolution.
    """
    ncells = ds.sizes[cell_dim]
    lat_deg, _ = _get_cell_latlon_deg(ds)

    n_bins = max(180, 4 * round(np.sqrt(ncells / 12)))
    edges = np.linspace(-90.0, 90.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    bin_indices = np.clip(np.digitize(lat_deg, edges) - 1, 0, n_bins - 1)
    counts = np.bincount(bin_indices, minlength=n_bins).astype(np.float64)
    counts = np.where(counts == 0, np.nan, counts)

    logger.info(
        f"Unstructured (non-HEALPix) grid detected (dim='{cell_dim}'). "
        f"Computing zonal mean into {n_bins} latitude bands."
    )

    out_ds = xr.Dataset(coords={'lat': centers})
    out_ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}

    from .cf_coords import _find_coordinate
    _geo_var_names: set[str] = set()
    for _cf_type in ('lat', 'lon'):
        _c = _find_coordinate(ds, _cf_type, raise_notfound=False)
        if _c is not None and _c.name in ds.data_vars:
            _geo_var_names.add(_c.name)

    for var in ds.data_vars:
        if var in _geo_var_names:
            continue
        if cell_dim in ds[var].dims:
            da = xr.apply_ufunc(
                _unstructured_zonal_mean_block,
                ds[var],
                kwargs={'bin_indices': bin_indices, 'n_bins': n_bins, 'counts': counts},
                input_core_dims=[[cell_dim]],
                output_core_dims=[['lat']],
                dask='parallelized',
                output_dtypes=[ds[var].dtype],
                dask_gufunc_kwargs={'output_sizes': {'lat': n_bins}, 'allow_rechunk': True},
            )
            out_ds[var] = da.assign_coords(lat=centers)
            out_ds[var].attrs = ds[var].attrs
            for coord in ds[var].coords:
                if coord not in (cell_dim, 'lat', 'lon') and coord in ds.coords:
                    out_ds.coords[coord] = ds.coords[coord]
        else:
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs

    out_ds.attrs = ds.attrs
    out_ds.attrs = append_history(out_ds.attrs, "Computed zonal mean over unstructured grid.")
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

    has_nans = np.isnan(data_2d).any()

    if not has_nans:
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
    else:
        nan_mask = np.isnan(data_2d)
        clean_data = np.where(nan_mask, 0.0, data_2d)
        valid_counts = (~nan_mask).astype(clean_data.dtype)

        if sort_order is None:
            # RING ordering: pixels are already contiguous within each ring.
            ring_sums = np.add.reduceat(clean_data, ring_boundaries, axis=1)
            ring_counts = np.add.reduceat(valid_counts, ring_boundaries, axis=1)
            ring_counts = np.where(ring_counts == 0, np.nan, ring_counts)
            out = ring_sums / ring_counts
        else:
            # NESTED ordering: sequential bincount scatter into small output.
            out = np.empty((data_2d.shape[0], n_rings), dtype=data_2d.dtype)
            for i in range(data_2d.shape[0]):
                sums = np.bincount(ring_indices, weights=clean_data[i], minlength=n_rings)
                counts_i = np.bincount(ring_indices, weights=valid_counts[i], minlength=n_rings)
                counts_i = np.where(counts_i == 0, np.nan, counts_i)
                out[i] = sums / counts_i

    return out.reshape(orig_shape[:-1] + (n_rings,))


def zonal_mean(ds: xr.Dataset) -> xr.Dataset:
    """Compute the zonal mean of an unstructured-grid dataset.

    For HEALPix grids the fast ring-based algorithm is used (``np.add.reduceat``
    for RING ordering, ``np.bincount`` loop for NESTED).  For any other
    unstructured grid that carries per-cell latitude / longitude coordinates
    (e.g. ICON icosahedral), cells are binned into latitude bands of roughly
    equal width.

    Args:
        ds: Input dataset on a HEALPix or unstructured grid.
    """
    cell_dim = get_cells_dim(ds)

    if not is_healpix(ds):
        return _unstructured_zonal_mean(ds, cell_dim)

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
    """Extract data at a single grid point, for HEALPix or unstructured grids.

    For HEALPix datasets the pixel index is computed analytically via
    ``hp.ang2pix``.  For other unstructured grids (e.g. ICON icosahedral)
    the nearest cell centre is found by 3-D Euclidean distance on the
    unit sphere — no external dependencies required.

    Args:
        ds: Input dataset on a HEALPix or unstructured grid.
        lat: Target latitude (degrees).
        lon: Target longitude (degrees).
    """
    cell_dim = get_cells_dim(ds)

    if not is_healpix(ds):
        cell_lat, cell_lon = _get_cell_latlon_deg(ds)
        theta_c = np.deg2rad(90.0 - cell_lat)
        phi_c   = np.deg2rad(cell_lon)
        x_c = np.sin(theta_c) * np.cos(phi_c)
        y_c = np.sin(theta_c) * np.sin(phi_c)
        z_c = np.cos(theta_c)

        theta_q = np.deg2rad(90.0 - lat)
        phi_q   = np.deg2rad(lon)
        xq = np.sin(theta_q) * np.cos(phi_q)
        yq = np.sin(theta_q) * np.sin(phi_q)
        zq = np.cos(theta_q)

        pix = int(np.argmin((x_c - xq)**2 + (y_c - yq)**2 + (z_c - zq)**2))
        logger.info(
            f"Extracting point data at lat={lat}, lon={lon} "
            f"(nearest unstructured cell index {pix}, "
            f"cell centre lat={cell_lat[pix]:.3f}°, lon={cell_lon[pix]:.3f}°)."
        )
        out_ds = ds.isel({cell_dim: pix})
        out_ds = out_ds.assign_coords(lat=lat, lon=lon)
        out_ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
        out_ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}
        out_ds.attrs = append_history(
            out_ds.attrs, f"Extracted nearest cell for lat={lat}, lon={lon} (cell {pix})."
        )
        return out_ds

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
