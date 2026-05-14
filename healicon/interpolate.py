import xarray as xr
import numpy as np
import dask.array as da
import logging
import healpy as hp
from .grid import get_healpix_coords

logger = logging.getLogger(__name__)

def lonlat_to_xyz(lon, lat):
    """Convert longitude and latitude in degrees to 3D Cartesian coordinates on unit sphere."""
    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x, y, z))

def _interp_unstructured_block(data_block, indices, weights, valid_mask):
    """
    Apply IDW interpolation to a single block of data.
    data_block: shape (..., n_source_cells)
    indices: shape (n_target_cells, k)
    weights: shape (n_target_cells, k)
    valid_mask: shape (n_target_cells,)
    """
    # Clip indices to avoid out of bounds for the invalid points
    n_source = data_block.shape[-1]
    safe_indices = np.clip(indices, 0, n_source - 1)
    
    # Normalize weights to sum to 1
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    
    # Take elements
    gathered = np.take(data_block, safe_indices, axis=-1)  # shape (..., n_target_cells, k)
    
    # Compute weighted sum
    interpolated = np.sum(gathered * weights, axis=-1)  # shape (..., n_target_cells)
    
    # Apply valid mask (where True keep, where False set to NaN)
    interpolated = np.where(valid_mask, interpolated, np.nan)
    
    return interpolated

def interpolate_unstructured(ds: xr.Dataset, nside: int, source_lon_name: str, source_lat_name: str, spatial_dim: str, use_gpu: bool = False) -> xr.Dataset:
    """
    Interpolate an unstructured grid (e.g., ICON) to a HEALPix grid using IDW (KDTree).
    """
    # Target HEALPix coordinates
    target_lon, target_lat = get_healpix_coords(nside)
    target_xyz = lonlat_to_xyz(target_lon, target_lat)
    
    # Source coordinates
    source_lon = ds[source_lon_name].values
    source_lat = ds[source_lat_name].values
        
    source_xyz = lonlat_to_xyz(source_lon, source_lat)
    
    k = 3 # Number of neighbors for IDW. Recommended for unstructured triangular meshes
    
    if use_gpu:
        try:
            import cuml
            logger.info("Using cuML for KDTree.")
            nn = cuml.neighbors.NearestNeighbors(n_neighbors=k)
            nn.fit(source_xyz)
            distances, indices = nn.kneighbors(target_xyz)
            # cuML might return cupy arrays or pandas/numpy depending on config.
            # Ensure numpy arrays for weights for now unless we do fully cupy based map_blocks.
            if hasattr(distances, 'get'):
                distances = distances.get()
                indices = indices.get()
        except ImportError:
            logger.warning("cuML not found. Falling back to SciPy cKDTree for GPU execution.")
            from scipy.spatial import cKDTree
            tree = cKDTree(source_xyz)
            distances, indices = tree.query(target_xyz, k=k)
    else:
        logger.info("Using SciPy cKDTree for interpolation.")
        from scipy.spatial import cKDTree
        tree = cKDTree(source_xyz)
        distances, indices = tree.query(target_xyz, k=k)
        
    # Distance threshold (e.g. 0.05 radians on unit sphere is ~300km)
    # If the closest neighbor is further than this, the point is outside the domain
    valid_mask = distances[:, 0] < 0.05
        
    # Compute weights (Inverse Distance Weighting)
    # Avoid division by zero
    distances = np.maximum(distances, 1e-12)
    weights = 1.0 / (distances ** 2)
    
    # Create the output dataset
    out_ds = xr.Dataset(
        coords={
            "cell": np.arange(len(target_lon)),
            "lon": ("cell", target_lon),
            "lat": ("cell", target_lat)
        }
    )
    
    for var_name, da in ds.data_vars.items():
        if spatial_dim not in da.dims:
            out_ds[var_name] = da
            continue
            
        # apply_ufunc to handle dask arrays gracefully across non-spatial dimensions
        interpolated_da = xr.apply_ufunc(
            _interp_unstructured_block,
            da,
            kwargs={'indices': indices, 'weights': weights, 'valid_mask': valid_mask},
            input_core_dims=[[spatial_dim]],
            output_core_dims=[["cell"]],
            exclude_dims=set((spatial_dim, "cell")),
            dask="parallelized",
            output_dtypes=[da.dtype],
            dask_gufunc_kwargs={'output_sizes': {'cell': len(target_lon)}}
        )
        # Assign coordinates
        interpolated_da = interpolated_da.assign_coords(cell=out_ds.cell, lon=out_ds.lon, lat=out_ds.lat)
        out_ds[var_name] = interpolated_da
        out_ds[var_name].attrs = da.attrs

    return out_ds

def _interp_regular_block(data_block, lon_coords, lat_coords, target_lon, target_lat):
    """
    Apply bilinear interpolation to a single block of data on a regular grid.
    """
    from scipy.interpolate import interpn
    
    # interpn requires the interpolation dimensions to be the FIRST dimensions.
    # apply_ufunc with input_core_dims=[[lat, lon]] puts them at the END of the block.
    # shape becomes (lat, lon, ...)
    data_reshaped = np.moveaxis(data_block, [-2, -1], [0, 1])
    
    # Prepare the target coordinates: shape (n_target_cells, 2)
    # The order must match the points tuple: (lat_coords, lon_coords)
    xi = np.column_stack((target_lat, target_lon))
    
    # Perform interpolation. Output shape: (n_target_cells, ...)
    interpolated = interpn((lat_coords, lon_coords), data_reshaped, xi, 
                           method='linear', bounds_error=False, fill_value=np.nan)
    
    # We want 'cell' to be the LAST dimension for consistency with apply_ufunc
    interpolated = np.moveaxis(interpolated, 0, -1)
    
    return interpolated

def interpolate_regular(ds: xr.Dataset, nside: int, lon_name: str, lat_name: str) -> xr.Dataset:
    """
    Interpolate a regular latitude-longitude grid to a HEALPix grid.
    Uses scipy.interpolate.interpn wrapped in apply_ufunc for memory efficiency.
    """
    target_lon, target_lat = get_healpix_coords(nside)
    
    # Align target longitudes to match source conventions if source has negative longitudes
    source_lon_min = ds[lon_name].min().item()
    if source_lon_min < 0:
        target_lon = (target_lon + 180) % 360 - 180
    
    source_lon = ds[lon_name].values
    source_lat = ds[lat_name].values
    
    # Create the output dataset
    out_ds = xr.Dataset(
        coords={
            "cell": np.arange(len(target_lon)),
            "lon": ("cell", target_lon),
            "lat": ("cell", target_lat)
        }
    )
    
    logger.info("Using SciPy interpn bilinear interpolation for regular grid.")
    for var_name, da in ds.data_vars.items():
        if lat_name not in da.dims or lon_name not in da.dims:
            out_ds[var_name] = da
            continue
            
        interpolated_da = xr.apply_ufunc(
            _interp_regular_block,
            da,
            kwargs={
                'lon_coords': source_lon, 
                'lat_coords': source_lat, 
                'target_lon': target_lon, 
                'target_lat': target_lat
            },
            input_core_dims=[[lat_name, lon_name]],
            output_core_dims=[["cell"]],
            exclude_dims=set((lat_name, lon_name)),
            dask="parallelized",
            output_dtypes=[da.dtype],
            dask_gufunc_kwargs={'output_sizes': {'cell': len(target_lon)}}
        )
        
        interpolated_da = interpolated_da.assign_coords(cell=out_ds.cell, lon=out_ds.lon, lat=out_ds.lat)
        out_ds[var_name] = interpolated_da
        out_ds[var_name].attrs = da.attrs
        
    return out_ds

def _find_cf_coordinate(ds: xr.Dataset, expected_standard: str, common_names: list) -> str:
    """Robustly find a coordinate using CF conventions or common name fallbacks."""
    # 1. CF standard_name
    for name, coord in ds.coords.items():
        if str(coord.attrs.get("standard_name", "")).lower() == expected_standard:
            return name
            
    # 2. Common names fallback
    for name in common_names:
        if name in ds.coords or name in ds.data_vars:
            return name
            
    return None

def interpolate_to_healpix(ds: xr.Dataset, nside: int, use_gpu: bool = False) -> xr.Dataset:
    """
    Determine the grid type and perform interpolation.
    """
    lon_name = _find_cf_coordinate(ds, "longitude", ["lon", "longitude", "clon"])
    lat_name = _find_cf_coordinate(ds, "latitude", ["lat", "latitude", "clat"])
            
    if not lon_name or not lat_name:
        raise ValueError("Could not automatically determine longitude/latitude coordinate names.")
        
    # Check and convert units to degrees if necessary
    lon_units = str(ds[lon_name].attrs.get('units', '')).lower()
    lat_units = str(ds[lat_name].attrs.get('units', '')).lower()
    
    is_radians = False
    if 'rad' in lon_units or 'rad' in lat_units:
        is_radians = True
    elif 'deg' not in lon_units and 'deg' not in lat_units:
        # Heuristic fallback using dask-compatible max evaluation
        lon_data = ds[lon_name].data
        lat_data = ds[lat_name].data
        lon_max = float(abs(lon_data).max().compute() if hasattr(lon_data, 'compute') else abs(lon_data).max())
        lat_max = float(abs(lat_data).max().compute() if hasattr(lat_data, 'compute') else abs(lat_data).max())
        
        if lon_max <= 2 * np.pi + 0.1 and lat_max <= np.pi / 2 + 0.1:
            is_radians = True

    if is_radians:
        logger.info(f"Coordinates {lon_name}, {lat_name} appear to be in radians. Converting to degrees.")
        new_lon = np.rad2deg(ds[lon_name])
        new_lat = np.rad2deg(ds[lat_name])
        new_lon.attrs.update(ds[lon_name].attrs)
        new_lat.attrs.update(ds[lat_name].attrs)
        new_lon.attrs['units'] = 'degrees_east'
        new_lat.attrs['units'] = 'degrees_north'
        ds = ds.assign({lon_name: new_lon, lat_name: new_lat})
        
    # Check grid structure
    # A regular grid has lat and lon as their own independent dimension coordinates
    is_regular_grid = (lon_name in ds.dims) and (lat_name in ds.dims)
    
    lon_dims = ds[lon_name].dims
    lat_dims = ds[lat_name].dims
    
    if is_regular_grid:
        logger.info(f"Detected regular latitude-longitude grid with dimensions ({lat_name}, {lon_name}).")
        out_ds = interpolate_regular(ds, nside, lon_name, lat_name)
        
    elif len(lon_dims) == 1 and lon_dims == lat_dims:
        # Unstructured grid (e.g. ICON) where coordinates share a single spatial dimension (like 'ncells')
        spatial_dim = lon_dims[0]
        logger.info(f"Detected 1D unstructured grid along spatial dimension '{spatial_dim}'.")
        out_ds = interpolate_unstructured(ds, nside, lon_name, lat_name, spatial_dim, use_gpu=use_gpu)
        
    elif len(lon_dims) == 2 and lon_dims == lat_dims:
        # Curvilinear grid (flatten it and treat as unstructured)
        logger.info(f"Detected 2D curvilinear grid with dimensions {lon_dims}. Flattening to unstructured.")
        spatial_dim = f"{lon_dims[0]}_{lon_dims[1]}"
        ds_flat = ds.stack({spatial_dim: lon_dims})
        out_ds = interpolate_unstructured(ds_flat, nside, lon_name, lat_name, spatial_dim, use_gpu=use_gpu)
        
    else:
        raise ValueError(f"Unsupported grid structure. lon dims: {lon_dims}, lat dims: {lat_dims}")
    
    # Preserve original metadata
    out_ds.attrs.update(ds.attrs)
    
    # Add HEALPix metadata
    npix = hp.nside2npix(nside)
    cell_area = 4 * np.pi / npix
    
    out_ds.attrs['healpix_nside'] = nside
    out_ds.attrs['healpix_npix'] = npix
    out_ds.attrs['healpix_scheme'] = 'RING'
    out_ds.attrs['healpix_cell_area_sr'] = f"{cell_area:.6e}"
    
    # Add CF-compliant grid mapping. Remove legacy grid attributes.
    for var in out_ds.data_vars:
        out_ds[var].attrs['grid_mapping'] = 'healpix'
        out_ds[var].attrs.pop('CDI_grid_type', None)
        out_ds[var].attrs.pop('number_of_grid_in_reference', None)
    
    history_msg = f"Interpolated to HEALPix grid (nside={nside}, scheme=RING) using HealICON."
    if 'history' in out_ds.attrs:
        out_ds.attrs['history'] = out_ds.attrs['history'] + '\n' + history_msg
    else:
        out_ds.attrs['history'] = history_msg
        
    return out_ds
