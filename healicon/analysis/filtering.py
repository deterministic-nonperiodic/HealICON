"""
Spatial filtering: Gaussian smoothing and hard spectral low-pass cutoff.
"""
import healpy as hp
import numpy as np
import xarray as xr

from ._common import (
    logger, wavelength_to_degree,
    get_healpix_order, get_cells_dim, ensure_ring, ensure_original_order, append_history,
)


def _filter_block(data_block, fwhm_rad, lmax, is_nested):
    orig_shape = data_block.shape
    npix = orig_shape[-1]
    nside = hp.npix2nside(npix)
    data_2d = data_block.reshape(-1, npix)
    order_str = 'nested' if is_nested else 'ring'

    out_data = np.zeros_like(data_2d)

    def _process_slice(i):
        d = ensure_ring(data_2d[i], order_str)
        valid_mask = ~np.isnan(d)
        if not np.any(valid_mask):
            out_data[i] = np.nan
            return
        d_filled = np.where(valid_mask, d, np.nanmean(d))

        if fwhm_rad is not None:
            filtered = hp.smoothing(d_filled, fwhm=fwhm_rad)
        elif lmax is not None:
            alm = hp.map2alm(d_filled, lmax=lmax, iter=3)
            filtered = hp.alm2map(alm, nside=nside)

        filtered = np.where(valid_mask, filtered, np.nan)
        out_data[i] = ensure_original_order(filtered, order_str)

    n = data_2d.shape[0]
    for i in range(n):
        _process_slice(i)

    return out_data.reshape(orig_shape)


def filter_spatial(ds: xr.Dataset, fwhm_deg: float = None, lmax: int = None,
                   wavelength_km: float = None) -> xr.Dataset:
    """Filter all HEALPix variables using spherical harmonics.

    Exactly one of the three filter parameters must be given:

    fwhm_deg
        Gaussian beam smoothing with the specified full-width-at-half-maximum.
        Implemented via ``healpy.smoothing``.
    lmax
        Hard spectral low-pass: retain only degrees l ≤ lmax.
    wavelength_km
        Hard spectral low-pass expressed as a physical scale.
        Converted to ``lmax`` via ``wavelength_to_degree(wavelength_km)``.

    Variables without the HEALPix cell dimension are passed through unchanged.
    """
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'

    n_specified = sum(x is not None for x in [fwhm_deg, lmax, wavelength_km])
    if n_specified != 1:
        raise ValueError("Must specify exactly one of: fwhm_deg, lmax, or wavelength_km.")

    if wavelength_km is not None:
        lmax = int(wavelength_to_degree(wavelength_km))
        logger.info(
            f"Converting wavelength {wavelength_km} km to lmax={lmax} for hard spectral cutoff."
        )

    fwhm_rad = np.deg2rad(fwhm_deg) if fwhm_deg is not None else None

    if fwhm_deg is not None:
        logger.info(f"Applying Gaussian smoothing filter with FWHM = {fwhm_deg} degrees.")
        hist_msg = f"Filtered using Gaussian smoothing (FWHM={fwhm_deg} deg)."
    elif wavelength_km is not None:
        logger.info(f"Applying hard spectral low-pass filter at {wavelength_km} km (lmax={lmax}).")
        hist_msg = f"Filtered using hard spectral cutoff at {wavelength_km} km (lmax={lmax})."
    else:
        logger.info(f"Applying hard spectral low-pass filter with lmax = {lmax}.")
        hist_msg = f"Filtered using hard spectral cutoff (lmax={lmax})."

    out_ds = xr.Dataset(coords=ds.coords)

    for var in ds.data_vars:
        if cell_dim in ds[var].dims:
            da = xr.apply_ufunc(
                _filter_block,
                ds[var],
                kwargs={'fwhm_rad': fwhm_rad, 'lmax': lmax, 'is_nested': is_nested},
                input_core_dims=[[cell_dim]],
                output_core_dims=[[cell_dim]],
                dask="parallelized",
                output_dtypes=[ds[var].dtype],
                dask_gufunc_kwargs={'allow_rechunk': True}
            )
            out_ds[var] = da
            out_ds[var].attrs = ds[var].attrs
        else:
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs

    out_ds.attrs = ds.attrs
    out_ds.attrs = append_history(out_ds.attrs, f"{hist_msg}")
    return out_ds
