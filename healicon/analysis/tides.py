"""
Tidal analysis: directional spatial filtering, symmetric/antisymmetric decomposition,
temporal harmonic extraction, and wavelet-based tidal decomposition.

This module is the canonical entry point for all Dataset-level tidal
analysis methods (least-squares, Fourier-wavelet, SH-wavelet).
"""

import healpy as hp
import numpy as np
import xarray as xr

from ._common import (
    logger, _MAX_WORKERS,
    get_healpix_order, get_cells_dim, ensure_ring,
    ensure_original_order, append_history, add_healpix_grid_mapping,
    ThreadPoolExecutor, get_progress_bar,
)
from ..grid import get_healpix_coords


def _directional_filter_block(a_block, b_block, target_m, lmax, is_nested):
    """
    Apply a directional spatial filter to isolate specific zonal wavenumbers (m) 
    and propagation directions (eastward/westward) using Spherical Harmonics.
    
    Transforms the temporal cosine (a_block) and sine (b_block) components to the
    spherical harmonic domain, isolates the specified wavenumber, applies phase 
    relationships to extract the desired propagation direction, and transforms back
    to the spatial domain.

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
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = [pool.submit(_process_slice, i) for i in range(n)]
            for f in get_progress_bar(as_completed(futures), desc="Directional filtering", total=n):
                f.result()
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
    hp_order = get_healpix_order(ds)
    is_nested = hp_order == 'nested'

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

    # Add metadata for the output dataset
    out_ds = add_healpix_grid_mapping(out_ds, nside, order=hp_order)

    out_ds['period'].attrs = {'units': 'hours', 'long_name': 'Tidal Period'}
    if m_filters is not None:
        out_ds['m'].attrs = {'long_name': 'Zonal Wavenumber'}

    out_ds.attrs = append_history(out_ds.attrs,
                                  f"Full tidal analysis (periods: {periods_hours}h, m: {m_filters}).")

    return out_ds


# --------------------------------------------------------------------------
# SHARED HELPERS FOR WAVELET-BASED TIDAL ANALYSIS
# --------------------------------------------------------------------------

def _infer_dt_hours(time_vals: np.ndarray, time_dim: str = 'time') -> float:
    """Infer the median time step in hours from a coordinate array.

    Handles datetime64, timedelta64 (LST-style), and plain numeric arrays.
    """
    if time_dim == 'lst':
        if np.issubdtype(time_vals.dtype, np.timedelta64):
            return float(np.median(np.diff(time_vals)) / np.timedelta64(1, 'h'))
        return float(np.median(np.diff(time_vals)))
    if np.issubdtype(time_vals.dtype, (np.datetime64, np.timedelta64)):
        return float(np.median(np.diff(time_vals)) / np.timedelta64(1, 'h'))
    return float(np.median(np.diff(time_vals)))


def _infer_t_hours_vals(time_vals: np.ndarray, time_dim: str = 'time') -> np.ndarray:
    """Build the elapsed-hours time axis used for temporal demodulation.

    This is the array fed into ``omega * t_hours_vals`` in
    :func:`_demodulate_mode`.  Shared by both
    :func:`compute_fourier_tidal_analysis` and
    :func:`compute_wavelet_tidal_analysis` so the two analyses demodulate
    against an identical time axis for the same dataset.
    """
    if time_dim == 'lst':
        if np.issubdtype(time_vals.dtype, np.timedelta64):
            return time_vals / np.timedelta64(1, 'h')
        return time_vals.astype(float)
    if np.issubdtype(time_vals.dtype, (np.datetime64, np.timedelta64)):
        return (time_vals - time_vals[0]) / np.timedelta64(1, 'h')
    return (time_vals - time_vals[0]).astype(float)


def _demodulate_mode(
        amp_vals: np.ndarray,
        pha_vals: np.ndarray,
        t_hours_vals: np.ndarray,
        period_h: float,
        want_phase: bool,
        lon_phase: np.ndarray | None = None,
) -> np.ndarray:
    """Demodulate a time-resolved amplitude/phase envelope to a single
    time-mean amplitude or phase map.

    This implements the harmonic-demodulation step shared by the
    ``temporal_mean=True`` branch of both
    :func:`compute_fourier_tidal_analysis` and
    :func:`compute_wavelet_tidal_analysis`.

    - Fourier path (``lon_phase=None``): ``pha_vals`` is already
      longitude-independent, so only time demodulation is performed.
    - Spherical-harmonic path (``lon_phase=target_m * phi``): the
      reconstructed spatial map still carries the ``m·λ`` longitude
      term, so it is folded in and removed after time-averaging.
    """
    omega = 2 * np.pi / float(period_h)
    wt = omega * t_hours_vals[:, None]

    pha_total = pha_vals + lon_phase if lon_phase is not None else pha_vals
    mw = amp_vals * np.cos(pha_total)
    mwH = -amp_vals * np.sin(pha_total)

    C_t = mw * np.cos(wt) + mwH * np.sin(wt)
    S_t = mw * np.sin(wt) - mwH * np.cos(wt)

    C_mean = C_t.mean(axis=0)
    S_mean = S_t.mean(axis=0)

    if lon_phase is not None:
        real_part = C_mean * np.cos(lon_phase) + S_mean * np.sin(lon_phase)
        imag_part = S_mean * np.cos(lon_phase) - C_mean * np.sin(lon_phase)
    else:
        real_part, imag_part = C_mean, S_mean

    if want_phase:
        return np.arctan2(imag_part, real_part)
    return np.sqrt(real_part ** 2 + imag_part ** 2)


def _build_tidal_output_dataset(
        assembled: dict,
        modes: list[dict],
        periods_hours: list[float],
        var_name: str,
        var_units: str,
        target_lon: np.ndarray,
        target_lat: np.ndarray,
        cell_dim: str,
) -> xr.Dataset:
    """Assemble the (m, period, ...) output Dataset shared by both
    wavelet-based tidal-analysis entry points.

    Stacks the per-mode ``assembled`` DataArrays into ``{var_name}_{comp}``
    variables along new ``m`` and ``period`` dimensions, then attaches the
    ``lon``/``lat`` spatial coordinates and amplitude/phase attrs.
    """
    m_vals = sorted(set(mode['m'] for mode in modes))
    period_td = xr.DataArray(
        [np.timedelta64(int(p * 3600), 's') for p in periods_hours],
        dims='period',
        attrs={'long_name': 'Tidal Period'},
    )
    m_coord = xr.DataArray(
        np.array(m_vals), dims='m',
        attrs={'long_name': 'Zonal Wavenumber',
               'description': 'positive=westward, negative=eastward'},
    )

    data_vars = {}
    for comp in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy'):
        stacked = xr.concat(
            [xr.concat([assembled[(m, p, comp)] for p in periods_hours], dim=period_td)
             for m in m_vals],
            dim=m_coord,
        )
        data_vars[f'{var_name}_{comp}'] = stacked

    out_ds = xr.Dataset(data_vars)

    lon_da = xr.DataArray(target_lon, dims=cell_dim,
                          attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    lat_da = xr.DataArray(target_lat, dims=cell_dim,
                          attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    out_ds = out_ds.assign_coords(lon=lon_da, lat=lat_da)

    label_map = {'sym': 'Symmetric', 'asy': 'Antisymmetric'}
    for sa, label in label_map.items():
        out_ds[f'{var_name}_amp_{sa}'].attrs = {
            'units': var_units, 'long_name': f'{label} Amplitude',
            'grid_mapping': 'healpix',
        }
        out_ds[f'{var_name}_pha_{sa}'].attrs = {
            'units': 'rad', 'long_name': f'{label} Phase',
            'grid_mapping': 'healpix',
        }

    return out_ds


# --------------------------------------------------------------------------
# DATASET-LEVEL FOURIER TIDAL ANALYSIS
# --------------------------------------------------------------------------

def compute_fourier_tidal_analysis(
        ds: xr.Dataset,
        var_name: str,
        periods_hours: list[float],
        m_filters: list[int] | None = None,
        time_dim: str = 'time',
        dj: float = 0.1,
        temporal_mean: bool = False,
) -> xr.Dataset:
    """Fourier-wavelet-based tidal analysis on a HEALPix Dataset.

    Extracts Fourier coefficients directly from the HEALPix isolatitude rings
    without any interpolation, runs the Fourier-wavelet analysis, decomposes
    the wavelet coefficients into symmetric/antisymmetric parts, and broadcasts
    the results exactly back to the HEALPix cells.
    """
    from .wavelet import fourier_wavelet_spectrum

    if time_dim not in ds.dims:
        raise ValueError(f"Dataset must have a '{time_dim}' dimension for Fourier tidal analysis.")

    cell_dim = get_cells_dim(ds)
    hp_order = get_healpix_order(ds)

    time_vals = ds[time_dim].values
    dt_hours = _infer_dt_hours(time_vals, time_dim)
    logger.info(f"Inferred dt = {dt_hours:.2f} hours from '{time_dim}' coordinate.")

    if m_filters is None:
        m_filters = [1]
        logger.warning("No m_filters specified; defaulting to m=[1] (DW1).")

    modes = []
    for m in m_filters:
        direction = 'westward' if m > 0 else 'eastward'
        zwn = abs(m)
        for p in periods_hours:
            modes.append({'m': m, 'zwn': zwn, 'period_h': p, 'dir': direction})

    zwn_mode_groups = {}
    for mode in modes:
        zwn_mode_groups.setdefault(mode['zwn'], []).append(mode)

    def _make_ufunc(zwn, zwn_modes):
        t_hours_vals = _infer_t_hours_vals(ds[time_dim].values, time_dim)
        periods_for_wavelet = list(set(m['period_h'] for m in zwn_modes))

        def _func(data_np):
            da_tmp = xr.DataArray(
                data_np, dims=[time_dim, cell_dim],
                coords={time_dim: ds[time_dim]},
            )
            ds_w = fourier_wavelet_spectrum(
                da_tmp, zwn=zwn, time_dim=time_dim, dt=dt_hours, dj=dj,
                periods_to_reconstruct=periods_for_wavelet,
                order=hp_order,
            )

            period_lookup = {}
            for v in ds_w.data_vars:
                parts = v.split('_')
                if len(parts) != 4:
                    continue
                _, _, direction, p_str = parts
                if direction not in ('westward', 'eastward'):
                    continue
                try:
                    actual_p = float(p_str)
                except ValueError:
                    continue
                period_lookup[(direction, p_str)] = actual_p

            outputs = []
            for mode in zwn_modes:
                direction = mode['dir']
                target_p_h = mode['period_h']

                best_p_str = min(
                    (p_str for (d, p_str) in period_lookup if d == direction),
                    key=lambda s: abs(period_lookup[(direction, s)] - target_p_h),
                )

                for comp in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy'):
                    arr = ds_w[f"{comp}_{direction}_{best_p_str}"].values
                    if temporal_mean:
                        sa = comp.split('_')[1]
                        amp_vals = ds_w[f"amp_{sa}_{direction}_{best_p_str}"].values
                        pha_vals = ds_w[f"pha_{sa}_{direction}_{best_p_str}"].values

                        final_val = _demodulate_mode(
                            amp_vals, pha_vals, t_hours_vals, target_p_h,
                            want_phase=('pha' in comp), lon_phase=None
                        )
                        outputs.append(final_val)
                    else:
                        outputs.append(arr)

            return tuple(outputs)

        return _func

    assembled = {}
    for zwn, zwn_modes in zwn_mode_groups.items():
        n_modes = len(zwn_modes)
        out_dtypes = [ds[var_name].dtype] * (n_modes * 4)

        if temporal_mean:
            out_core_dims = [[cell_dim]] * (n_modes * 4)
        else:
            out_core_dims = [[time_dim, cell_dim]] * (n_modes * 4)

        results = xr.apply_ufunc(
            _make_ufunc(zwn, zwn_modes),
            ds[var_name],
            input_core_dims=[[time_dim, cell_dim]],
            output_core_dims=out_core_dims,
            vectorize=True,
            dask='parallelized',
            output_dtypes=out_dtypes,
            dask_gufunc_kwargs={'allow_rechunk': True},
        )

        flat_modes = []
        for mode in zwn_modes:
            for comp in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy'):
                flat_modes.append((mode['m'], mode['period_h'], comp))

        if not isinstance(results, tuple):
            results = (results,)

        for i, key in enumerate(flat_modes):
            assembled[key] = results[i]

    target_lon, target_lat = get_healpix_coords(hp.npix2nside(ds.sizes[cell_dim]))
    if hp_order == 'nested':
        target_lon = ensure_original_order(target_lon, 'nested')
        target_lat = ensure_original_order(target_lat, 'nested')

    out_ds = _build_tidal_output_dataset(
        assembled, modes, periods_hours, var_name, ds[var_name].attrs.get('units', ''),
        target_lon, target_lat, cell_dim
    )

    out_ds.attrs['history'] = f"Fourier-Wavelet Tidal Analysis (dt={dt_hours}h)"
    return out_ds


# --------------------------------------------------------------------------
# DATASET-LEVEL SH-WAVELET TIDAL ANALYSIS
# --------------------------------------------------------------------------

def compute_wavelet_tidal_analysis(
        ds: xr.Dataset,
        var_name: str,
        periods_hours: list[float],
        m_filters: list[int] | None = None,
        lmax: int | None = None,
        time_dim: str = 'time',
        dj: float = 0.1,
        temporal_mean: bool = False,
        map2alm_iter: int = 3,
) -> xr.Dataset:
    """Wavelet-based tidal analysis on a HEALPix Dataset.

    This is the wavelet analogue of
    :func:`compute_leastsquares_tidal_analysis`.  It accepts a
    full Dataset with arbitrary non-core dimensions (e.g. ``height``) and
    automatically maps over them via :func:`xarray.apply_ufunc`, keeping
    peak memory at O(1 slice) regardless of the number of non-core levels.

    Pipeline (per non-core slice):
        1. For each unique ``|m|`` in *m_filters*, run
           :func:`~healicon.analysis.wavelet.spherical_harmonic_wavelet_spectrum`
           to obtain time-resolved amplitude and phase maps.
        2. Extract the requested ``periods_hours`` and propagation
           direction (westward for ``m > 0``, eastward for ``m < 0``).
        3. Optionally average over the time axis (``temporal_mean``).

    Args:
        ds:  Input Dataset on a HEALPix grid with a time dimension.
        var_name:  Name of the data variable to analyze.
        periods_hours:  Target periods in hours (e.g. ``[24, 12]``).
        m_filters:  Signed zonal wavenumbers to extract.  Positive values
            denote westward propagation, negative values eastward.
            If *None*, defaults to ``[1]``.
        lmax:  Maximum spherical harmonic degree.  If *None*, uses
            ``3 * nside - 1``.
        time_dim:  Name of the time dimension.
        dj:  Spacing between discrete wavelet scales (default 0.1).
        temporal_mean:  If *True*, average the wavelet amplitude over time
            before returning (produces output comparable to LS tides).
            Default is *False* (return the full time-resolved envelope).

    Returns:
        xr.Dataset with variables ``{var_name}_amp_sym``,
        ``{var_name}_pha_sym``, ``{var_name}_amp_asy``,
        ``{var_name}_pha_asy``.  Dimensions are
        ``(m, period, [time], *non_core_dims, cells)``.
    """
    from .wavelet import spherical_harmonic_wavelet_spectrum

    if time_dim not in ds.dims:
        raise ValueError(
            f"Dataset must have a '{time_dim}' dimension for wavelet "
            f"tidal analysis."
        )

    cell_dim = get_cells_dim(ds)
    hp_order = get_healpix_order(ds)
    da = ds[var_name]

    nside = hp.npix2nside(da.sizes[cell_dim])

    # ── Infer dt from time coordinate ────────────────────────────────
    time_vals = ds[time_dim].values
    dt_hours = _infer_dt_hours(time_vals, time_dim)
    logger.info(f"Inferred dt = {dt_hours:.2f} hours from '{time_dim}' coordinate.")

    # ── Build mode table from m_filters ──────────────────────────────
    if m_filters is None:
        m_filters = [1]
        logger.warning("No m_filters specified; defaulting to m=[1] (DW1).")

    modes = []
    for m in m_filters:
        direction = 'westward' if m > 0 else 'eastward'
        zwn = abs(m)
        for p in periods_hours:
            modes.append({'m': m, 'zwn': zwn, 'period_h': p, 'dir': direction})

    # Group modes by |m| so each wavelet call is reused
    zwn_mode_groups = {}
    for mode in modes:
        zwn_mode_groups.setdefault(mode['zwn'], []).append(mode)

    unique_zwn = sorted(zwn_mode_groups.keys())
    periods_for_wavelet = list(periods_hours)

    # ── Core function applied per non-core slice ─────────────────────

    def _make_ufunc(zwn, zwn_modes):
        """Build a ufunc for a specific zonal wavenumber."""
        n_modes = len(zwn_modes)

        t_hours_vals = _infer_t_hours_vals(ds[time_dim].values, time_dim)

        def _func(data_np):
            da_tmp = xr.DataArray(
                data_np, dims=[time_dim, cell_dim],
                coords={time_dim: ds[time_dim]},
            )
            ds_w = spherical_harmonic_wavelet_spectrum(
                da_tmp, zwn=zwn, time_dim=time_dim, dt=dt_hours, dj=dj,
                lmax=lmax, map2alm_iter=map2alm_iter,
                periods_to_reconstruct=periods_for_wavelet,
                order=hp_order,
            )

            period_lookup = {}
            for v in ds_w.data_vars:
                parts = v.split('_')
                if len(parts) != 4:
                    continue
                _, _, direction, p_str = parts
                if direction not in ('westward', 'eastward'):
                    continue
                try:
                    actual_p = float(p_str)
                except ValueError:
                    continue
                period_lookup[(direction, p_str)] = actual_p

            outputs = []
            for mode in zwn_modes:
                direction = mode['dir']
                target_p_h = mode['period_h']

                best_p_str = min(
                    (p_str for (d, p_str) in period_lookup if d == direction),
                    key=lambda s: abs(period_lookup[(direction, s)] - target_p_h),
                )

                for comp in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy'):
                    arr = ds_w[f"{comp}_{direction}_{best_p_str}"].values
                    if temporal_mean:
                        sa = comp.split('_')[1]  # 'sym' or 'asy'
                        amp_vals = ds_w[f"amp_{sa}_{direction}_{best_p_str}"].values
                        pha_vals = ds_w[f"pha_{sa}_{direction}_{best_p_str}"].values

                        nside_local = hp.npix2nside(amp_vals.shape[1])
                        target_lon_deg, _ = get_healpix_coords(nside_local)
                        if hp_order == 'nested':
                            target_lon_deg = ensure_original_order(target_lon_deg, 'nested')
                        phi_da = np.radians(target_lon_deg)
                        target_m = mode['m']

                        arr = _demodulate_mode(
                            amp_vals, pha_vals, t_hours_vals, mode['period_h'],
                            want_phase=('pha' in comp), lon_phase=target_m * phi_da,
                        )

                    outputs.append(arr)

            return tuple(outputs)

        return _func, n_modes

    # ── apply_ufunc per unique |m| ───────────────────────────────────
    logger.info(f"Applying wavelet analysis for |m| = {unique_zwn}...")

    assembled = {}  # (m_val, period_h, comp) -> DataArray

    for zwn in unique_zwn:
        zwn_modes = zwn_mode_groups[zwn]
        func, n_modes = _make_ufunc(zwn, zwn_modes)
        n_outputs = n_modes * 4

        if temporal_mean:
            out_core = [[cell_dim]] * n_outputs
        else:
            out_core = [[time_dim, cell_dim]] * n_outputs

        results = xr.apply_ufunc(
            func,
            da,
            input_core_dims=[[time_dim, cell_dim]],
            output_core_dims=out_core,
            vectorize=True,
            dask='parallelized',
            output_dtypes=[np.float64] * n_outputs,
            dask_gufunc_kwargs={'allow_rechunk': True},
        )

        # Unpack results into named DataArrays
        if n_outputs == 1:
            results = (results,)
        comps = ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy')
        for i, mode in enumerate(zwn_modes):
            for j, comp in enumerate(comps):
                da_out = results[i * 4 + j]
                assembled[(mode['m'], mode['period_h'], comp)] = da_out

    # ── Build output Dataset using xarray-native concat ─────────────────
    target_lon, target_lat = get_healpix_coords(nside)
    if hp_order == 'nested':
        target_lon = ensure_original_order(target_lon, 'nested')
        target_lat = ensure_original_order(target_lat, 'nested')

    var_units = ds[var_name].attrs.get('units', '')
    out_ds = _build_tidal_output_dataset(
        assembled, modes, periods_hours, var_name, var_units,
        target_lon, target_lat, cell_dim,
    )

    # Preserve dataset attributes and grid mapping
    out_ds.attrs = ds.attrs.copy()
    out_ds = add_healpix_grid_mapping(out_ds, nside, order=hp_order)
    out_ds.attrs = append_history(
        out_ds.attrs,
        f"Wavelet tidal analysis (periods: {periods_hours}h, "
        f"m: {m_filters}, temporal_mean={temporal_mean})."
    )

    return out_ds
