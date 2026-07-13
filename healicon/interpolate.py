import logging
import math

import numpy as np
import xarray as xr

from .grid import get_healpix_coords, append_history, add_healpix_grid_mapping

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
    n_source = data_block.shape[-1]
    safe_indices = np.clip(indices, 0, n_source - 1)
    gathered = np.take(data_block, safe_indices, axis=-1)
    interpolated = np.sum(gathered * weights, axis=-1)
    interpolated = np.where(valid_mask, interpolated, np.nan)
    return interpolated


def _interp_regular_block(data_block, lon_coords, lat_coords, target_lon, target_lat):
    """
    Apply bilinear interpolation to a single block of data on a regular grid.
    """
    from scipy.interpolate import interpn

    # Check if we need to pad longitude for periodic boundary
    if len(lon_coords) > 1:
        dx = lon_coords[1] - lon_coords[0]
        # If the grid is near-global but missing the 360 wrap point
        if (lon_coords[-1] + dx) >= 360.0 - 1e-5 and lon_coords[-1] < 360.0:
            lon_coords = np.append(lon_coords, lon_coords[-1] + dx)
            # Append the first longitude slice to the end
            data_block = np.concatenate([data_block, data_block[..., :1]], axis=-1)

    data_reshaped = np.moveaxis(data_block, [-2, -1], [0, 1])
    xi = np.column_stack((target_lat, target_lon))
    interpolated = interpn((lat_coords, lon_coords), data_reshaped, xi,
                           method='linear', bounds_error=False, fill_value=np.nan)
    interpolated = np.moveaxis(interpolated, 0, -1)
    return interpolated


class HealpixInterpolator:
    def __init__(self, nside: int | None = None, use_gpu: bool = False, order: str = 'ring'):
        self.nside = nside
        self.use_gpu = use_gpu
        self.order = order.lower()
        self._is_setup = False
        self._grid_type = None
        self._lon_name = None
        self._lat_name = None
        self._spatial_dim = None
        self._target_lon = None
        self._target_lat = None

        # For unstructured
        self._indices = None
        self._weights = None
        self._valid_mask = None

        # For regular
        self._source_lon = None
        self._source_lat = None

    def _determine_nside(self, ds: xr.Dataset):
        # Fast path: HEALPix dataset — derive nside directly from npix.
        # This works even when the file has no lon/lat coordinate variables.
        try:
            from .grid import get_cells_dim
            cell_dim = get_cells_dim(ds)
            npix = ds.sizes[cell_dim]
            nside_float = math.sqrt(npix / 12)
            if nside_float.is_integer():
                return int(nside_float)
        except ValueError:
            pass

        # Non-HEALPix source: inspect lon/lat coords to estimate grid size.
        from .cf_coords import _find_coordinate
        spatial_dims = []
        lon_coord = _find_coordinate(ds, "lon", raise_notfound=False)
        lat_coord = _find_coordinate(ds, "lat", raise_notfound=False)
        if lon_coord is not None:
            for dim in lon_coord.dims:
                if dim not in spatial_dims:
                    spatial_dims.append(dim)
        if lat_coord is not None:
            for dim in lat_coord.dims:
                if dim not in spatial_dims:
                    spatial_dims.append(dim)
        if not spatial_dims:
            raise ValueError("nside must be provided if spatial dimensions cannot be determined.")
        n_orig = 1
        for dim in spatial_dims:
            n_orig *= ds.sizes[dim]
        target_nside = math.sqrt(n_orig / 12)
        return 2 ** round(math.log2(max(1, target_nside)))

    def setup(self, ds: xr.Dataset):
        """
        Initialize the interpolation grid. Precompute KDTree for unstructured grids.
        
        Args:
            ds: Input xarray dataset with spatial coordinates.
        """
        if self._is_setup:
            return

        # Check if already HEALPix
        try:
            from .grid import get_cells_dim
            cell_dim = get_cells_dim(ds)
            npix = ds.sizes[cell_dim]
            nside_float = math.sqrt(npix / 12)
            if nside_float.is_integer():
                self._grid_type = 'healpix'
                self._spatial_dim = cell_dim
                self._current_nside = int(nside_float)
                if self.nside is None:
                    self.nside = self._current_nside
                logger.info(f"Setup complete for hp2hp grid (current nside={self._current_nside}).")
                self._is_setup = True
                return
        except ValueError:
            pass

        if self.nside is None:
            self.nside = self._determine_nside(ds)
            logger.info(f"Auto-calculated nside={self.nside}")

        from .cf_coords import _find_coordinate
        lon_coord = _find_coordinate(ds, "lon", raise_notfound=False)
        lat_coord = _find_coordinate(ds, "lat", raise_notfound=False)
        self._lon_name = getattr(lon_coord, "name", None)
        self._lat_name = getattr(lat_coord, "name", None)

        if not self._lon_name or not self._lat_name:
            raise ValueError(
                "Could not automatically determine longitude/latitude coordinate names.")

        self._target_lon, self._target_lat = get_healpix_coords(self.nside, nest=(self.order == 'nested'))

        lon_dims = ds[self._lon_name].dims
        lat_dims = ds[self._lat_name].dims

        if (self._lon_name in ds.dims) and (self._lat_name in ds.dims):
            self._grid_type = 'regular'
            source_lon_min = ds[self._lon_name].min().item()
            if source_lon_min < 0:
                self._target_lon = (self._target_lon + 180) % 360 - 180
            self._source_lon = ds[self._lon_name].values
            self._source_lat = ds[self._lat_name].values
            logger.info("Setup complete for regular grid.")
        else:
            if len(lon_dims) == 1 and lon_dims == lat_dims:
                self._grid_type = 'unstructured'
                self._spatial_dim = lon_dims[0]
            elif len(lon_dims) == 2 and lon_dims == lat_dims:
                self._grid_type = 'curvilinear'
                self._spatial_dim = f"{lon_dims[0]}_{lon_dims[1]}"
            else:
                raise ValueError(
                    f"Unsupported grid structure. lon dims: {lon_dims}, lat dims: {lat_dims}")

            # For unstructured, build KDTree
            source_lon = ds[self._lon_name].values
            source_lat = ds[self._lat_name].values

            # Convert to degrees if in radians
            lon_units = str(ds[self._lon_name].attrs.get('units', '')).lower()
            if 'rad' in lon_units:
                source_lon = np.rad2deg(source_lon)
                source_lat = np.rad2deg(source_lat)

            # Unstack if curvilinear for coords
            if self._grid_type == 'curvilinear':
                source_lon = source_lon.flatten()
                source_lat = source_lat.flatten()

            source_xyz = lonlat_to_xyz(source_lon, source_lat)
            target_xyz = lonlat_to_xyz(self._target_lon, self._target_lat)

            k = 3
            if self.use_gpu:
                try:
                    # pyrefly: ignore [missing-import]
                    import cuml
                    logger.info("Using cuML for KDTree with GPU support.")
                    nn = cuml.neighbors.NearestNeighbors(n_neighbors=k)
                    nn.fit(source_xyz)
                    distances, indices = nn.kneighbors(target_xyz)
                    if hasattr(distances, 'get'):
                        distances = distances.get()
                        indices = indices.get()
                except Exception as e:
                    logger.warning(f"GPU KDTree failed ({e}). Falling back to CPU.")
                    from scipy.spatial import cKDTree
                    tree = cKDTree(source_xyz)
                    distances, indices = tree.query(target_xyz, k=k, workers=-1)
            else:
                logger.info("Using SciPy cKDTree for interpolation.")
                from scipy.spatial import cKDTree
                tree = cKDTree(source_xyz)
                distances, indices = tree.query(target_xyz, k=k, workers=-1)

            self._valid_mask = distances[:, 0] < 0.05
            distances = np.maximum(distances, 1e-12)
            weights = 1.0 / (distances ** 2)
            # Pre-normalize: avoids redundant division on every Dask chunk
            self._weights = weights / np.sum(weights, axis=-1, keepdims=True)
            self._indices = indices
            logger.info("Setup complete for unstructured grid.")

        self._is_setup = True

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        if not self._is_setup:
            self.setup(ds)

        if self._grid_type == 'healpix':
            from .grid import get_healpix_order
            current_order = get_healpix_order(ds)
            if self.nside == self._current_nside:
                if self.order == current_order:
                    logger.info(
                        "Dataset is already on the target HEALPix grid with correct ordering. Ensuring grid mapping metadata.")
                    return add_healpix_grid_mapping(ds, self.nside, order=self.order)
                else:
                    logger.info(
                        f"Dataset is HEALPix (nside={self._current_nside}), reordering from {current_order} to {self.order}...")
                    from .grid import get_cells_dim
                    cell_dim = get_cells_dim(ds)
                    out_ds = ds.copy()
                    target_lon, target_lat = get_healpix_coords(self.nside, nest=(self.order == 'nested'))
                    out_ds['lon'] = (cell_dim, target_lon)
                    out_ds['lat'] = (cell_dim, target_lat)
                    out_ds['lon'].attrs = {"standard_name": "longitude", "units": "degrees_east"}
                    out_ds['lat'].attrs = {"standard_name": "latitude", "units": "degrees_north"}
                    for var_name, da in ds.data_vars.items():
                        if cell_dim in da.dims:
                            r2n = (self.order == 'nested')
                            n2r = (self.order == 'ring')
                            def _reorder_block(arr):
                                import healpy as hp
                                return hp.reorder(arr, r2n=r2n, n2r=n2r)
                            reordered_da = xr.apply_ufunc(
                                _reorder_block,
                                da,
                                input_core_dims=[[cell_dim]],
                                output_core_dims=[[cell_dim]],
                                dask="parallelized",
                                output_dtypes=[da.dtype],
                                keep_attrs=True
                            )
                            out_ds[var_name] = reordered_da
                    out_ds = add_healpix_grid_mapping(out_ds, self.nside, order=self.order)
                    history_msg = f"Reordered HEALPix grid (nside={self.nside}, from {current_order} to {self.order}) using HealICON."
                    out_ds.attrs = append_history(out_ds.attrs, history_msg)
                    return out_ds
            else:
                logger.info(
                    f"Dataset is HEALPix (nside={self._current_nside}, order={current_order}), but target is nside={self.nside}. Regrading resolution...")
                from .analysis import regrade_resolution
                ds_regraded = regrade_resolution(ds, new_nside=self.nside)
                if self.order == 'nested':
                    reorder_interpolator = HealpixInterpolator(nside=self.nside, use_gpu=self.use_gpu, order=self.order)
                    return reorder_interpolator(ds_regraded)
                return ds_regraded

        # Handle rad/deg conversion on the fly if needed
        lon_units = str(ds[self._lon_name].attrs.get('units', '')).lower()
        if 'rad' in lon_units:
            ds = ds.assign({self._lon_name: np.rad2deg(ds[self._lon_name]),
                            self._lat_name: np.rad2deg(ds[self._lat_name])})

        if self._grid_type == 'curvilinear':
            ds = ds.stack({self._spatial_dim: ds[self._lon_name].dims})

        out_ds = xr.Dataset(coords={"cells": np.arange(len(self._target_lon))})
        out_ds["lon"] = ("cells", self._target_lon)
        out_ds["lat"] = ("cells", self._target_lat)

        out_ds["lon"].attrs = {"standard_name": "longitude", "units": "degrees_east"}
        out_ds["lat"].attrs = {"standard_name": "latitude", "units": "degrees_north"}

        for var_name, da in ds.data_vars.items():
            if self._grid_type in ['unstructured', 'curvilinear']:
                if self._spatial_dim not in da.dims:
                    out_ds[var_name] = da
                    continue
                with xr.set_options(keep_attrs=True):
                    interpolated_da = xr.apply_ufunc(
                        _interp_unstructured_block,
                        da,
                        kwargs={'indices': self._indices, 'weights': self._weights,
                                'valid_mask': self._valid_mask},
                        input_core_dims=[[self._spatial_dim]],
                        output_core_dims=[["cells"]],
                        exclude_dims=set((self._spatial_dim, "cells")),
                        dask="parallelized",
                        output_dtypes=[da.dtype],
                        dask_gufunc_kwargs={'output_sizes': {'cells': len(self._target_lon)},
                                            'allow_rechunk': True}
                    )
            else:  # regular
                if self._lat_name not in da.dims or self._lon_name not in da.dims:
                    out_ds[var_name] = da
                    continue
                with xr.set_options(keep_attrs=True):
                    interpolated_da = xr.apply_ufunc(
                        _interp_regular_block,
                        da,
                        kwargs={
                            'lon_coords': self._source_lon,
                            'lat_coords': self._source_lat,
                            'target_lon': self._target_lon,
                            'target_lat': self._target_lat
                        },
                        input_core_dims=[[self._lat_name, self._lon_name]],
                        output_core_dims=[["cells"]],
                        exclude_dims=set((self._lat_name, self._lon_name)),
                        dask="parallelized",
                        output_dtypes=[da.dtype],
                        dask_gufunc_kwargs={'output_sizes': {'cells': len(self._target_lon)},
                                            'allow_rechunk': True}
                    )

            interpolated_da = interpolated_da.assign_coords(cells=out_ds.cells)
            out_ds[var_name] = interpolated_da
            out_ds[var_name].attrs = da.attrs

        out_ds.attrs.update(ds.attrs)
        # Explicitly copy coordinate attributes (apply_ufunc drops them for non-core coords)
        for c in ds.coords:
            if c in out_ds.coords and c not in [self._lon_name, self._lat_name, "cells"]:
                out_ds[c].attrs.update(ds[c].attrs)

        out_ds = add_healpix_grid_mapping(out_ds, self.nside, order=self.order)

        history_msg = f"Interpolated to HEALPix grid (nside={self.nside}, scheme={self.order.upper()}) using HealICON."
        out_ds.attrs = append_history(out_ds.attrs, history_msg)

        return out_ds


def interpolate_to_healpix(ds: xr.Dataset, nside: int = None, use_gpu: bool = False, order: str = 'ring') -> xr.Dataset:
    """
    Interpolate a dataset to a HEALPix grid.
    
    Args:
        ds: Input xarray dataset.
        nside: Target NSIDE for the HEALPix grid.
        use_gpu: Whether to use GPU acceleration for interpolation.
        order: Target ordering scheme ('ring' or 'nested').
    
    Returns:
        xarray.Dataset on the target HEALPix grid.
    """
    interpolator = HealpixInterpolator(nside=nside, use_gpu=use_gpu, order=order)
    return interpolator(ds)
