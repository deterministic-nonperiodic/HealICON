"""
Tidal analysis: directional spatial filtering, symmetric/antisymmetric decomposition,
and temporal harmonic extraction.
"""
import healpy as hp
import numpy as np
import xarray as xr

from ._common import (
    logger, _MAX_WORKERS,
    get_healpix_order, get_cells_dim, ensure_ring, ensure_original_order, append_history,
    ThreadPoolExecutor,
)


def _directional_filter_block(a_block, b_block, target_m, lmax, is_nested):
    """
    Apply a directional spatial filter to isolate specific zonal wavenumbers (m) 
    and propagation directions (eastward/westward) using Spherical Harmonics.

    Given a temporal Fourier decomposition of a field at frequency omega:
        T(t, x) = A(x) * cos(omega * t) + B(x) * sin(omega * t)
    
    We want to isolate a wave traveling in a specific direction with zonal wavenumber m.
    A pure traveling wave takes the form: cos(m * lambda +/- omega * t)
    
    Expanding this using trigonometric identities:
    - Westward (m > 0): cos(m*lambda + omega*t) = cos(m*lambda)*cos(omega*t) - sin(m*lambda)*sin(omega*t)
                         Here, A ~ cos(m*lambda) and B ~ -sin(m*lambda)
    - Eastward (m < 0):  cos(|m|*lambda - omega*t) =  cos(|m|*lambda)*cos(omega*t) + sin(|m|*lambda)*sin(omega*t)
                         Here, A ~ cos(|m|*lambda) and B ~ sin(|m|*lambda)

    In the Spherical Harmonic (ALM) domain, the basis functions are Y_l^m ~ exp(i * m * lambda).
    To extract the directional components from the full fields A and B:
    1. Transform A and B to the spherical harmonic domain: alm_a and alm_b.
    2. For a target wavenumber |m|, mask out all other wavenumbers.
    3. Apply the phase relationship between A and B to isolate the direction:
       - For m > 0 (Westward): 
           alm_a_out = 0.5 * (alm_a - i * alm_b)
           alm_b_out = i * alm_a_out
       - For m < 0 (Eastward): 
           alm_a_out = 0.5 * (alm_a + i * alm_b)
           alm_b_out = -i * alm_a_out
       - For m = 0 (Zonal Mean):
           alm_a_out = alm_a
           alm_b_out = alm_b
    4. Transform back to the spatial domain via Inverse Spherical Harmonics.

    Args:
        a_block: Cosine temporal component field.
        b_block: Sine temporal component field.
        target_m: Target zonal wavenumber (positive for westward, negative for eastward).
        lmax: Maximum spherical harmonic degree.
        is_nested: Whether the HEALPix grid uses NESTED ordering.

    Returns:
        Filtered a_block and b_block in the spatial domain.
    """
    orig_shape = a_block.shape
    npix = orig_shape[-1]
    nside = hp.npix2nside(npix)

    a_2d = a_block.reshape(-1, npix)
    b_2d = b_block.reshape(-1, npix)

    out_a = np.zeros_like(a_2d)
    out_b = np.zeros_like(b_2d)

    l_arr, m_arr = hp.Alm.getlm(lmax)
    abs_m = abs(target_m)
    mask = (m_arr == abs_m)
    order_str = 'nested' if is_nested else 'ring'

    def _process_slice(i):
        a_ring = ensure_ring(a_2d[i], order_str)
        b_ring = ensure_ring(b_2d[i], order_str)

        valid_mask = ~(np.isnan(a_ring) | np.isnan(b_ring))
        if not np.any(valid_mask):
            out_a[i] = np.nan
            out_b[i] = np.nan
            return

        a_filled = np.where(valid_mask, a_ring, 0.0)
        b_filled = np.where(valid_mask, b_ring, 0.0)

        alm_a = hp.map2alm(a_filled, lmax=lmax, iter=3)
        alm_b = hp.map2alm(b_filled, lmax=lmax, iter=3)

        if target_m > 0:
            # Positive m: Westward (cos(m*lambda + omega*t))
            alm_a_out = 0.5 * (alm_a - 1j * alm_b) * mask
            alm_b_out = 1j * alm_a_out
        elif target_m < 0:
            # Negative m: Eastward (cos(|m|*lambda - omega*t))
            alm_a_out = 0.5 * (alm_a + 1j * alm_b) * mask
            alm_b_out = -1j * alm_a_out
        else:
            alm_a_out = alm_a * mask
            alm_b_out = alm_b * mask

        a_filtered = np.where(valid_mask, hp.alm2map(alm_a_out, nside=nside), np.nan)
        b_filtered = np.where(valid_mask, hp.alm2map(alm_b_out, nside=nside), np.nan)

        out_a[i] = ensure_original_order(a_filtered, order_str)
        out_b[i] = ensure_original_order(b_filtered, order_str)

    n = a_2d.shape[0]
    if n > 1:
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            list(pool.map(_process_slice, range(n)))
    else:
        _process_slice(0)

    return out_a.reshape(orig_shape), out_b.reshape(orig_shape)


def _get_symmetric_pixels(nside, is_nested=False):
    """
    Returns an array of pixel indices that correspond to the exact reflection
    across the equator for each pixel in a HEALPix grid.

    Args:
        nside: HEALPix resolution parameter
        is_nested: Whether the HEALPix grid is in nested order

    Returns:
        Array of symmetric pixel indices
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=is_nested)
    theta_sym = np.pi - theta
    return hp.ang2pix(nside, theta_sym, phi, nest=is_nested)


def _extract_spatial_tide_components(da_cos: xr.DataArray, da_sin: xr.DataArray,
                                     m_filters: list[int] | None, cell_dim: str,
                                     sym_idx_da: xr.DataArray, phi_da: xr.DataArray,
                                     apply_filter_fn) -> dict:
    """
    Decomposes the cosine and sine tidal coefficients into symmetric/antisymmetric 
    amplitudes and phases, optionally filtering by specific wavenumbers.

    Args:
        da_cos: Cosine tidal coefficients
        da_sin: Sine tidal coefficients
        m_filters: List of spherical harmonic degrees to filter by
        cell_dim: Name of the cell dimension
        sym_idx_da: Array of symmetric pixel indices
        phi_da: Array of longitudinal angles
        apply_filter_fn: Function to apply filters to the data

    Returns:
        Dictionary containing symmetric and antisymmetric amplitudes and phases
    """
    ms = m_filters if m_filters is not None else [None]
    results = {'amp_sym': [], 'pha_sym': [], 'amp_asy': [], 'pha_asy': []}

    for m in ms:
        cos_m, sin_m = apply_filter_fn(da_cos, da_sin, m) if m is not None else (da_cos, da_sin)

        cos_sym = 0.5 * (cos_m + cos_m.isel({cell_dim: sym_idx_da}).data)
        cos_asy = 0.5 * (cos_m - cos_m.isel({cell_dim: sym_idx_da}).data)

        sin_sym = 0.5 * (sin_m + sin_m.isel({cell_dim: sym_idx_da}).data)
        sin_asy = 0.5 * (sin_m - sin_m.isel({cell_dim: sym_idx_da}).data)

        def get_phase(c, s_coef, target_m):
            if target_m is None:
                return np.arctan2(s_coef, c)
            real_part = c * np.cos(target_m * phi_da) + s_coef * np.sin(target_m * phi_da)
            imag_part = s_coef * np.cos(target_m * phi_da) - c * np.sin(target_m * phi_da)
            return np.arctan2(imag_part, real_part)

        res_m = {
            'amp_sym': np.sqrt(cos_sym ** 2 + sin_sym ** 2),
            'pha_sym': get_phase(cos_sym, sin_sym, m),
            'amp_asy': np.sqrt(cos_asy ** 2 + sin_asy ** 2),
            'pha_asy': get_phase(cos_asy, sin_asy, m)
        }

        if m is not None:
            res_m = {k: v.expand_dims(m=[m]) for k, v in res_m.items()}

        for k in results:
            results[k].append(res_m[k])

    if m_filters is not None:
        return {k: xr.concat(v, dim='m') for k, v in results.items()}
    return {k: v[0] for k, v in results.items()}


def compute_leastsquares_tidal_analysis(ds: xr.Dataset, var_name: str, periods_hours: list[float],
                           m_filters: list[int] | None = None, lmax: int | None = None,
                           time_dim: str = 'time') -> xr.Dataset:
    """
    Performs a full tidal analysis on a HEALPix dataset over time.
    
    This function processes a time-series or local solar time (LST) resolved dataset 
    to extract tidal components (e.g., diurnal, semidiurnal tides). The analysis follows these steps:
    1. Extracts the temporal harmonic coefficients for the specified periods (in hours).
    2. Optionally filters the spatial field to specific zonal wavenumbers (m_filters) 
       and propagation directions using Spherical Harmonics.
    3. Decomposes the spatial field into symmetric and antisymmetric components 
       relative to the equator.
    4. Computes Amplitude and Phase for both the symmetric and antisymmetric components.

    Args:
        ds (xr.Dataset): Input dataset containing the variable to analyze. 
            Must be in a HEALPix grid format.
        var_name (str): Name of the data variable to perform the analysis on.
        periods_hours (list[float]): List of target periods in hours for the temporal 
            Fourier extraction (e.g., [24, 12] for diurnal and semidiurnal tides).
        m_filters (list[int] | None, optional): List of zonal wavenumbers to filter. 
            Positive values denote westward propagation, negative values eastward 
            (matching Yamazaki 2023 convention). 
            If None, only the temporal extraction is performed. Defaults to None.
        lmax (int | None, optional): Maximum spherical harmonic degree to use during 
            spatial filtering. If None, calculated automatically as 3 * nside - 1. 
            Defaults to None.
        time_dim (str, optional): Name of the time or local solar time dimension. 
            Defaults to 'time'.

    Returns:
        xr.Dataset: A new dataset containing the tidal components. 
            Variables include '{var_name}_amp_sym', '{var_name}_pha_sym', 
            '{var_name}_amp_asy', and '{var_name}_pha_asy', resolved over 
            the original spatial grid and the extracted 'period' (and 'm' if filtered).
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Dataset must have a '{time_dim}' dimension for temporal tidal analysis.")

    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'
    npix = ds.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    if lmax is None:
        lmax = 3 * nside - 1

    if time_dim == 'lst':
        # LST is already in hours [0, 24).
        vals = ds[time_dim].values
        if np.issubdtype(vals.dtype, np.timedelta64):
            t_days_vals = vals / np.timedelta64(1, 'D')
        else:
            t_days_vals = vals / 24.0
    else:
        t_days = (ds[time_dim] - ds[time_dim][0]).dt.total_seconds() / 86400.0
        t_days_vals = t_days.values

    sym_idx = _get_symmetric_pixels(nside, is_nested=is_nested)
    sym_idx_da = xr.DataArray(sym_idx, dims=[cell_dim])

    # Compute phi array for phase extraction
    _, phi = hp.pix2ang(nside, np.arange(npix), nest=is_nested)
    phi_da = xr.DataArray(phi, dims=[cell_dim], coords={
        cell_dim: ds.coords[cell_dim] if cell_dim in ds.coords else np.arange(npix)})

    def apply_directional_filter(da_a, da_b, m):
        return xr.apply_ufunc(
            _directional_filter_block,
            da_a, da_b,
            kwargs={'target_m': m, 'lmax': lmax, 'is_nested': is_nested},
            input_core_dims=[[cell_dim], [cell_dim]],
            output_core_dims=[[cell_dim], [cell_dim]],
            dask="parallelized",
            output_dtypes=[da_a.dtype, da_b.dtype],
            dask_gufunc_kwargs={'allow_rechunk': True}
        )

    periods_arr = np.array(periods_hours)
    freq_cpd = 24.0 / periods_arr

    logger.info(f"Extracting temporal periods {periods_hours} hours for '{var_name}'.")

    omega = 2 * np.pi * freq_cpd[:, None]

    X = np.stack([
        np.cos(omega * t_days_vals),
        np.sin(omega * t_days_vals),
        np.ones((len(periods_hours), len(t_days_vals)))
    ], axis=-1)

    X_T = X.transpose(0, 2, 1)
    XTX = X_T @ X
    M = np.linalg.pinv(XTX) @ X_T

    M_A = xr.DataArray(M[:, 0, :], dims=['period', time_dim],
                       coords={'period': periods_hours, time_dim: ds[time_dim]})
    M_B = xr.DataArray(M[:, 1, :], dims=['period', time_dim],
                       coords={'period': periods_hours, time_dim: ds[time_dim]})

    da_cos = xr.dot(ds[var_name], M_A, dims=[time_dim])
    da_sin = xr.dot(ds[var_name], M_B, dims=[time_dim])

    spatial_res = _extract_spatial_tide_components(
        da_cos, da_sin, m_filters, cell_dim, sym_idx_da, phi_da, apply_directional_filter
    )

    out_ds = xr.Dataset(coords={c: ds.coords[c] for c in ds.coords if c != time_dim})
    out_ds.attrs = ds.attrs
    var_units = ds[var_name].attrs.get('units', '')

    # Preserve CF grid mapping variables (which are dimensionless data vars)
    for v in ds.data_vars:
        if len(ds[v].dims) == 0:
            out_ds[v] = ds[v]

    for k, combined in spatial_res.items():
        combined = combined.assign_coords({cell_dim: ds[cell_dim]})
        comp_type = 'Symmetric' if 'sym' in k else 'Antisymmetric'
        metric = 'Amplitude' if 'amp' in k else 'Phase'
        units = var_units if 'amp' in k else 'rad'
        combined.attrs = {
            'units': units,
            'grid_mapping': 'healpix',
            'long_name': f'{comp_type} {metric}'
        }
        out_ds[f'{var_name}_{k}'] = combined

    out_ds['period'].attrs = {'units': 'hours', 'long_name': 'Tidal Period'}
    if m_filters is not None:
        out_ds['m'].attrs = {'long_name': 'Zonal Wavenumber'}

    out_ds.attrs = append_history(out_ds.attrs,
                                  f"Full tidal analysis (periods: {periods_hours}h, m: {m_filters}).")

    return out_ds


# Compatibility alias
compute_tidal_analysis = compute_leastsquares_tidal_analysis

