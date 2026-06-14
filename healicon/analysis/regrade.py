"""
HEALPix resolution regrading (upgrade/downgrade).
"""
import healpy as hp
import numpy as np
import xarray as xr

from ._common import (
    logger, _MAX_WORKERS,
    get_healpix_order, get_cells_dim, append_history,
    add_healpix_grid_mapping,
    ThreadPoolExecutor,
)


def _regrade_block(data_block, nside_out, is_nested):
    orig_shape = data_block.shape
    npix_in = orig_shape[-1]
    data_2d = data_block.reshape(-1, npix_in)
    order = 'NEST' if is_nested else 'RING'

    npix_out = hp.nside2npix(nside_out)
    out_data = np.zeros((data_2d.shape[0], npix_out), dtype=data_2d.dtype)

    def _process_slice(i):
        d = data_2d[i]
        valid_mask = ~np.isnan(d)
        if not np.any(valid_mask):
            out_data[i] = np.nan
            return
        d_unseen = np.where(valid_mask, d, hp.UNSEEN)
        regraded = hp.ud_grade(d_unseen, nside_out=nside_out, order_in=order, order_out=order)
        out_data[i] = np.where(regraded == hp.UNSEEN, np.nan, regraded)

    n = data_2d.shape[0]
    if n > 1:
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            list(pool.map(_process_slice, range(n)))
    else:
        _process_slice(0)

    out_shape = orig_shape[:-1] + (npix_out,)
    return out_data.reshape(out_shape)


def regrade_resolution(ds: xr.Dataset, new_nside: int) -> xr.Dataset:
    """
    Upgrades or downgrades the HEALPix resolution of the dataset.

    Args:
        ds: Input dataset
        new_nside: New HEALPix nside

    Returns:
        Regrraded dataset
    """
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'

    old_npix = ds.sizes[cell_dim]
    old_nside = hp.npix2nside(old_npix)
    new_npix = hp.nside2npix(new_nside)

    logger.info(f"Regrading resolution from nside={old_nside} to nside={new_nside}.")

    target_lon, target_lat = hp.pix2ang(new_nside, np.arange(new_npix), lonlat=True, nest=is_nested)

    out_ds = xr.Dataset(
        coords={
            cell_dim: np.arange(new_npix),
        }
    )
    out_ds['lon'] = (cell_dim, target_lon)
    out_ds['lat'] = (cell_dim, target_lat)

    for var in ds.data_vars:
        if cell_dim in ds[var].dims:
            da = xr.apply_ufunc(
                _regrade_block,
                ds[var],
                kwargs={'nside_out': new_nside, 'is_nested': is_nested},
                input_core_dims=[[cell_dim]],
                output_core_dims=[[cell_dim]],
                exclude_dims=set((cell_dim,)),
                dask="parallelized",
                output_dtypes=[ds[var].dtype],
                dask_gufunc_kwargs={'output_sizes': {cell_dim: new_npix}, 'allow_rechunk': True}
            )
            out_ds[var] = da.assign_coords({cell_dim: out_ds[cell_dim]})
            out_ds[var].attrs = ds[var].attrs
        else:
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs

    for coord in ds.coords:
        if coord not in [cell_dim, 'lon', 'lat'] and coord in ds.coords:
            out_ds.coords[coord] = ds.coords[coord]

    out_ds.attrs = ds.attrs
    order = get_healpix_order(ds)
    out_ds = add_healpix_grid_mapping(out_ds, new_nside, order=order)

    out_ds.attrs = append_history(out_ds.attrs, f"Regraded resolution to nside={new_nside}.")
    return out_ds
