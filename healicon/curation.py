import logging

import healpy as hp
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def fill_healpix_gaps(ds: xr.Dataset, spatial_dim: str = 'cells',
                      time_dim: str = None) -> xr.Dataset:
    """
    Fill missing values (NaNs) in a HEALPix dataset using spatial nearest-neighbor interpolation
    and optional temporal 1D linear interpolation.
    """
    logger.info("Gap-filling dataset...")

    # We must have the spatial dim
    if spatial_dim not in ds.dims:
        raise ValueError(
            f"Spatial dimension '{spatial_dim}' not found in dataset dims: {list(ds.dims)}")

    nside = ds.attrs.get('healpix_nside')
    if nside is None:
        npix = ds.sizes[spatial_dim]
        nside = hp.npix2nside(npix)
        logger.warning(
            f"healpix_nside not found in attributes. Inferred nside={nside} from {spatial_dim} size={npix}")

    npix = hp.nside2npix(nside)
    if ds.sizes[spatial_dim] != npix:
        raise ValueError(
            f"Dimension '{spatial_dim}' size ({ds.sizes[spatial_dim]}) does not match HEALPix nside={nside} npix={npix}")

    theta, phi = hp.pix2ang(nside, np.arange(npix))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    coords = np.column_stack([x, y, z])

    # Copy dataset to avoid mutating original
    out_ds = ds.copy()

    for var_name, da in out_ds.data_vars.items():
        if spatial_dim not in da.dims:
            continue

        logger.info(f"Gap-filling variable '{var_name}'...")

        # Load into memory if not already
        data = da.values.copy()

        has_time = time_dim and time_dim in da.dims

        other_dims = [dim for dim in da.dims if dim not in [spatial_dim, time_dim]]
        other_shape = tuple(da.sizes[dim] for dim in other_dims)
        time_size = da.sizes[time_dim] if has_time else 1

        ordered_dims = other_dims + [spatial_dim]
        if has_time:
            ordered_dims.append(time_dim)

        # Transpose to put (others..., spatial, time)
        transpose_indices = [da.dims.index(dim) for dim in ordered_dims]
        data_t = np.transpose(data, transpose_indices)

        # Spatial filling
        for idx in np.ndindex(other_shape):
            if has_time:
                for t in range(time_size):
                    d = data_t[idx][:, t]
                    valid = ~np.isnan(d)
                    if not np.any(valid) or np.all(valid):
                        continue

                    tree = cKDTree(coords[valid])
                    missing = ~valid
                    _, indices = tree.query(coords[missing], workers=-1)
                    d[missing] = d[valid][indices]
            else:
                d = data_t[idx]
                valid = ~np.isnan(d)
                if not np.any(valid) or np.all(valid):
                    continue

                tree = cKDTree(coords[valid])
                missing = ~valid
                _, indices = tree.query(coords[missing], workers=-1)
                d[missing] = d[valid][indices]

        # Temporal filling
        if has_time:
            for idx in np.ndindex(other_shape):
                for p in range(npix):
                    d = data_t[idx][p, :]
                    valid = ~np.isnan(d)
                    if not np.any(valid) or np.all(valid):
                        continue

                    x_valid = np.where(valid)[0]
                    x_missing = np.where(~valid)[0]
                    # 1D linear interpolation
                    # np.interp expects increasing x. x_missing and x_valid are inherently sorted.
                    d_interp = np.interp(x_missing, x_valid, d[valid])
                    d[x_missing] = d_interp

        # Update variable
        out_ds[var_name].values = data

    return out_ds
