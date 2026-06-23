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
from .wavelet import (
    _sh_precompute_alm,
    _sh_reconstruct_level,
    _fourier_precompute_ring_coefs,
    _fourier_reconstruct_level,
    _get_symmetric_pixels,
    _compute_lev_batch,
    fourier_wavelet_spectrum,
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
# --------------------------------------------------------------------------
# WAVELET TIDAL ANALYSIS — per-block worker functions
# --------------------------------------------------------------------------

def _assemble_tidal_block(assembled, modes, periods_hours, var_name, var_units,
                          target_lon, target_lat, cell_dim, time_dim,
                          temporal_mean, non_core_dims, da_block):
    """Build and transpose the output Dataset from an ``assembled`` dict.

    Shared by both SH and Fourier block functions.
    """

    ds_2d = _build_tidal_output_dataset(
        assembled, modes, periods_hours, var_name, var_units,
        target_lon, target_lat, cell_dim
    )

    if non_core_dims:
        ds_2d = ds_2d.expand_dims({d: da_block.coords[d].values for d in non_core_dims})

        # Propagate auxiliary coordinates that live on non-core dims
        # (e.g. 'plev' indexed by 'lev').  expand_dims only creates the
        # index-dimension coordinate; non-dimension coords are silently
        # dropped, which causes map_blocks to raise a mismatch error
        # because the template (built from da.coords) does include them.
        aux_coords = {}
        for name, coord in da_block.coords.items():
            if name in ds_2d.coords:
                continue  # already present
            if any(d in non_core_dims for d in coord.dims):
                aux_coords[name] = coord
        if aux_coords:
            ds_2d = ds_2d.assign_coords(aux_coords)

    expected_dims = ['m', 'period']
    if not temporal_mean:
        expected_dims.append(time_dim)
    expected_dims.extend(non_core_dims)
    expected_dims.append(cell_dim)
    return ds_2d.transpose(*expected_dims)


def _wavelet_sh_analysis_block(
        da_block, modes, zwn_mode_groups, time_dim, cell_dim, hp_order,
        dt_hours, dj, temporal_mean, periods_hours, var_name, var_units,
        target_lon, target_lat, spectrum_kwargs,
):
    """SH block: CWT + spatial reconstruction using pre-computed A_lm/B_lm.

    ``spectrum_kwargs`` must contain ``'_alm_cache'`` (dict returned by
    ``_sh_precompute_alm``) and ``'_da_ref'`` (the original full DataArray,
    used to map block coordinates to linear level indices).
    """
    non_core_dims = [d for d in da_block.dims if d not in (time_dim, cell_dim)]

    alm_cache = spectrum_kwargs['_alm_cache']
    A_lm_all = alm_cache['A_lm']  # {abs_m: (n_time, n_extra, n_l)}
    B_lm_all = alm_cache['B_lm']
    nside_sh = alm_cache['nside']
    lmax_sh = alm_cache['lmax']
    is_nested_sh = alm_cache['is_nested']

    # Map this block's coordinate value(s) to a linear index in A_lm.
    # Use argmin (nearest-match) to be robust against float rounding.
    if non_core_dims:
        da_full = spectrum_kwargs['_da_ref']
        flat_idx = 0
        stride = 1
        for d in reversed(non_core_dims):
            coord_val = float(da_block[d].values.flat[0])
            full_coords = np.asarray(da_full.coords[d].values, dtype=float)
            local_idx = int(np.argmin(np.abs(full_coords - coord_val)))
            flat_idx += local_idx * stride
            stride *= len(full_coords)
    else:
        flat_idx = 0

    assembled = {}
    for zwn, zwn_modes in zwn_mode_groups.items():
        abs_m = abs(zwn)
        A_lm_lev = A_lm_all[abs_m][:, flat_idx, :]  # (n_time, n_l)
        B_lm_lev = B_lm_all[abs_m][:, flat_idx, :]

        rec = _sh_reconstruct_level(
            A_lm_lev, B_lm_lev,
            dt=dt_hours, dj=dj,
            nside=nside_sh, lmax=lmax_sh, abs_m=abs_m,
            periods_to_reconstruct=list(periods_hours),
            is_nested=is_nested_sh,
        )
        period_arr = rec['period']

        for mode in zwn_modes:
            direction = mode['dir']
            actual_p = float(period_arr[np.argmin(np.abs(period_arr - mode['period_h']))])

            for comp in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy'):
                sa = comp.split('_')[1]  # 'sym' / 'asy'
                ap = comp.split('_')[0]  # 'amp' / 'pha'
                arr = rec[(actual_p, direction, sa, ap)]  # (n_time, npix)

                if temporal_mean:
                    nside_local = hp.npix2nside(arr.shape[-1])
                    lon_deg, _ = get_healpix_coords(nside_local)
                    if hp_order == 'nested':
                        lon_deg = ensure_original_order(lon_deg, 'nested')
                    lon_phase = mode['m'] * np.radians(lon_deg)
                    t_hrs = _infer_t_hours_vals(da_block[time_dim].values, time_dim)
                    arr = _demodulate_mode(
                        rec[(actual_p, direction, sa, 'amp')],
                        rec[(actual_p, direction, sa, 'pha')],
                        t_hrs, mode['period_h'],
                        want_phase=(ap == 'pha'), lon_phase=lon_phase,
                    )
                    _cell_coords = ({cell_dim: da_block.coords[cell_dim]}
                                    if cell_dim in da_block.coords else {})
                    da_out = xr.DataArray(arr, dims=[cell_dim], coords=_cell_coords)
                else:
                    _cell_coords = ({cell_dim: da_block.coords[cell_dim]}
                                    if cell_dim in da_block.coords else {})
                    da_out = xr.DataArray(
                        arr, dims=[time_dim, cell_dim],
                        coords={time_dim: da_block.coords[time_dim], **_cell_coords},
                    )
                assembled[(mode['m'], mode['period_h'], comp)] = da_out

    return _assemble_tidal_block(
        assembled, modes, periods_hours, var_name, var_units,
        target_lon, target_lat, cell_dim, time_dim,
        temporal_mean, non_core_dims, da_block,
    )


def _wavelet_fourier_analysis_block(
        da_block, modes, zwn_mode_groups, time_dim, cell_dim, hp_order,
        dt_hours, dj, temporal_mean, periods_hours, var_name, var_units,
        target_lon, target_lat, spectrum_kwargs,
):
    """Fourier block: CWT + spatial reconstruction using pre-computed ring series.

    When ``spectrum_kwargs`` contains ``'_ring_cache'`` (dict returned by
    :func:`_fourier_precompute_ring_coefs`), the block reads the pre-computed
    ``(Ck, Sk)`` ring Fourier coefficients for its level and runs CWT +
    ``assign_to_cells`` directly — no pixel loads inside the block.

    Falls back to :func:`fourier_wavelet_spectrum` if ``'_ring_cache'`` is
    absent (e.g. when called from user code without the precompute step).
    """

    non_core_dims = [d for d in da_block.dims if d not in (time_dim, cell_dim)]
    da_2d = da_block.isel({d: 0 for d in non_core_dims})
    t_hours_vals = _infer_t_hours_vals(da_2d[time_dim].values, time_dim)

    ring_cache = spectrum_kwargs.get('_ring_cache')

    if ring_cache is None:
        # ── Fallback: call full fourier_wavelet_spectrum (old path) ──────
        assembled = {}
        for zwn, zwn_modes in zwn_mode_groups.items():
            ds_w = fourier_wavelet_spectrum(
                da_2d, zwn=zwn, time_dim=time_dim, dt=dt_hours, dj=dj,
                periods_to_reconstruct=list(periods_hours),
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
                    period_lookup[(direction, p_str)] = float(p_str)
                except ValueError:
                    continue

            for mode in zwn_modes:
                direction = mode['dir']
                best_p_str = min(
                    (p_str for (d, p_str) in period_lookup if d == direction),
                    key=lambda s: abs(period_lookup[(direction, s)] - mode['period_h']),
                )
                for comp in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy'):
                    arr = ds_w[f"{comp}_{direction}_{best_p_str}"].values
                    if temporal_mean:
                        sa = comp.split('_')[1]
                        arr = _demodulate_mode(
                            ds_w[f"amp_{sa}_{direction}_{best_p_str}"].values,
                            ds_w[f"pha_{sa}_{direction}_{best_p_str}"].values,
                            t_hours_vals, mode['period_h'],
                            want_phase=('pha' in comp), lon_phase=None,
                        )
                    _cell_c = ({cell_dim: da_2d.coords[cell_dim]}
                               if cell_dim in da_2d.coords else {})
                    if temporal_mean:
                        da_out = xr.DataArray(arr, dims=[cell_dim], coords=_cell_c)
                    else:
                        da_out = xr.DataArray(
                            arr, dims=[time_dim, cell_dim],
                            coords={time_dim: da_2d[time_dim], **_cell_c},
                        )
                    assembled[(mode['m'], mode['period_h'], comp)] = da_out

        return _assemble_tidal_block(
            assembled, modes, periods_hours, var_name, var_units,
            target_lon, target_lat, cell_dim, time_dim,
            temporal_mean, non_core_dims, da_block,
        )

    # ── Fast path: use precomputed ring Fourier coefficients ─────────────
    # Resolve level index using the same flat_idx logic as the SH block.
    da_ref = spectrum_kwargs['_da_ref']
    if non_core_dims:
        flat_idx = 0
        stride = 1
        for d in reversed(non_core_dims):
            coord_val = float(da_block[d].values.flat[0])
            full_coords = np.asarray(da_ref.coords[d].values, dtype=float)
            local_idx = int(np.argmin(np.abs(full_coords - coord_val)))
            flat_idx += local_idx * stride
            stride *= len(full_coords)
    else:
        flat_idx = 0

    # Helper: expand ring values to pixel space.
    npix = da_block.sizes[cell_dim]
    nside = hp.npix2nside(npix)
    is_nested = hp_order == 'nested'

    assembled = {}
    for zwn, zwn_modes in zwn_mode_groups.items():
        cache = ring_cache[zwn]
        Ck_lev = cache['Ck'][:, flat_idx, :]  # (n_time, n_rings)
        Sk_lev = cache['Sk'][:, flat_idx, :]
        ring_pix = cache['ring_pix']

        rec = _fourier_reconstruct_level(
            Ck_lev, Sk_lev, ring_pix,
            dt_hours, dj, nside, is_nested,
            periods_to_reconstruct=list(periods_hours),
        )
        period_arr = rec['period']

        for mode in zwn_modes:
            direction = mode['dir']
            actual_p = float(period_arr[np.argmin(np.abs(period_arr - mode['period_h']))])

            for comp in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy'):
                ap, sa = comp.split('_')
                arr = rec[(actual_p, direction, sa, ap)]

                if temporal_mean:
                    arr = _demodulate_mode(
                        rec[(actual_p, direction, sa, 'amp')],
                        rec[(actual_p, direction, sa, 'pha')],
                        t_hours_vals, mode['period_h'],
                        want_phase=(ap == 'pha'), lon_phase=None,
                    )

                _cell_c = ({cell_dim: da_2d.coords[cell_dim]}
                           if cell_dim in da_2d.coords else {})
                if temporal_mean:
                    da_out = xr.DataArray(arr, dims=[cell_dim], coords=_cell_c)
                else:
                    da_out = xr.DataArray(
                        arr, dims=[time_dim, cell_dim],
                        coords={time_dim: da_2d[time_dim], **_cell_c},
                    )
                assembled[(mode['m'], mode['period_h'], comp)] = da_out

    return _assemble_tidal_block(
        assembled, modes, periods_hours, var_name, var_units,
        target_lon, target_lat, cell_dim, time_dim,
        temporal_mean, non_core_dims, da_block,
    )


def _recommend_dask_scheduler(
        bytes_per_block: int,
        budget_fraction: float = 0.40,
) -> tuple[str, int]:
    """Return a safe ``(scheduler, n_workers)`` pair for Dask block execution.

    Estimates how many blocks can run concurrently without exhausting RAM,
    using *budget_fraction* of currently available system memory.  The result
    is capped at ``n_cpu // 2`` so NumPy's internal BLAS / OpenMP threads
    retain enough CPU cores and memory bandwidth.

    Args:
        bytes_per_block: Peak RAM per Dask block in bytes.  Should include
            ALL arrays that coexist during one block's execution (output maps
            for every ZWN group and period).
        budget_fraction: Fraction of available RAM allowed for concurrent
            blocks.  Default 0.40 leaves headroom for the spectral cache,
            OS buffers, and NumPy thread pools.

    Returns:
        ``('synchronous', 1)`` when only one block fits, otherwise
        ``('threads', n_workers)``.
    """
    import os as _os
    try:
        import psutil as _psutil
        available_bytes = _psutil.virtual_memory().available
    except ImportError:
        available_bytes = 16 * 1024 ** 3  # conservative 16 GB fallback

    n_cpu = _os.cpu_count() or 4
    n_workers = max(1, min(
        n_cpu // 2,
        int(available_bytes * budget_fraction / bytes_per_block),
    ))
    scheduler = 'synchronous' if n_workers == 1 else 'threads'
    logger.info(
        f"Adaptive scheduler: {n_workers} worker(s) "
        f"(block_peak={bytes_per_block / 1e9:.1f} GB, "
        f"budget={available_bytes * budget_fraction / 1e9:.0f} GB of "
        f"{available_bytes / 1e9:.0f} GB available)"
    )
    return scheduler, n_workers


def _iterate_tidal_analysis(
        da, ds, modes, zwn_mode_groups, time_dim, cell_dim, hp_order,
        dt_hours, dj, temporal_mean, method, periods_hours, var_name,
        var_units, target_lon, target_lat, spectrum_kwargs
):
    """Execute wavelet tidal analysis via map_blocks.

    Uses chunk=1 per non-core dimension (one level per block) to keep
    peak memory bounded.  For the 'sh' method, map2alm has already been
    pre-computed across all levels by ``compute_wavelet_tidal_analysis``
    and is stored in ``spectrum_kwargs['_alm_cache']``.
    """
    non_core_dims = [d for d in da.dims if d not in (time_dim, cell_dim)]

    block_args = (modes, zwn_mode_groups, time_dim, cell_dim, hp_order,
                  dt_hours, dj, temporal_mean, periods_hours,
                  var_name, var_units, target_lon, target_lat, spectrum_kwargs)

    # Both methods use map_blocks — one level per block, streams to disk
    # without accumulating all levels in memory simultaneously.
    # For the SH method the caller should use scheduler='synchronous' to
    # avoid N_threads × block_memory concurrent allocations; see cli.py.
    rechunk_dict = {time_dim: -1, cell_dim: -1}
    for d in non_core_dims:
        rechunk_dict[d] = 1
    da_chunked = da.chunk(rechunk_dict)

    dummy_assembled = {}
    for mode in modes:
        for comp in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy'):
            key = (mode['m'], mode['period_h'], comp)
            dummy_assembled[key] = (da_chunked.isel({time_dim: 0}, drop=True)
                                    if temporal_mean else da_chunked)

    template = _build_tidal_output_dataset(
        dummy_assembled, modes, periods_hours, var_name, var_units,
        target_lon, target_lat, cell_dim
    ).chunk({'m': -1, 'period': -1})

    block_fn = _wavelet_sh_analysis_block if method == 'sh' else _wavelet_fourier_analysis_block
    out_ds = xr.map_blocks(
        block_fn,
        da_chunked,
        args=block_args,
        template=template,
    )

    # Re-attach auxiliary coordinates from da (e.g. pressure levels indexed by
    # a height dimension).  These are excluded from map_blocks blocks to avoid
    # template-mismatch errors, so we add them back here once, outside the graph.
    if non_core_dims:
        allowed_dims = set(non_core_dims) | {cell_dim}
        if not temporal_mean:
            allowed_dims.add(time_dim)
        aux_coords = {
            name: coord
            for name, coord in da.coords.items()
            if (name not in out_ds.coords
                and name not in non_core_dims
                and name != time_dim
                and name != cell_dim
                and all(d in allowed_dims for d in coord.dims))
        }
        if aux_coords:
            out_ds = out_ds.assign_coords(aux_coords)

    # ── Adaptive scheduler recommendation ────────────────────────────────
    # Block peak = 8 output maps × n_periods × n_zwn_groups × n_time × n_cell × 8 B.
    n_periods_out = len(periods_hours)
    n_zwn_groups = len(zwn_mode_groups)
    bytes_per_block = (
            8 * n_periods_out * n_zwn_groups * da.sizes[time_dim] * da.sizes[cell_dim] * 8
    )
    scheduler, n_workers = _recommend_dask_scheduler(bytes_per_block, budget_fraction=0.8)
    out_ds.attrs['_recommended_dask_scheduler'] = scheduler
    out_ds.attrs['_recommended_dask_num_workers'] = n_workers

    return out_ds


# --------------------------------------------------------------------------
# DATASET-LEVEL WAVELET TIDAL ANALYSIS
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
        method: str = 'sh',
) -> xr.Dataset:
    """Wavelet-based tidal analysis on a HEALPix Dataset.

    Supports both spherical harmonics ('sh') and Fourier ('fourier') wavelet methods.
    Non-core dimensions (e.g. height) are processed in parallel/lazy mode level-by-level
    via xr.map_blocks to keep memory usage bounded.

    Args:
        ds:  Input Dataset on a HEALPix grid with a time dimension.
        var_name:  Name of the data variable to analyze.
        periods_hours:  Target periods in hours (e.g. ``[24, 12]``).
        m_filters:  Signed zonal wavenumbers to extract.  Positive values
            denote westward propagation, negative values eastward.
            If *None*, defaults to ``[1]``.
        lmax:  Maximum spherical harmonic degree (SH method only).
            If *None*, uses ``3 * nside - 1``.
        time_dim:  Name of the time dimension.
        dj:  Spacing between discrete wavelet scales (default 0.1).
        temporal_mean:  If *True*, average the wavelet amplitude over time
            before returning (produces output comparable to LS tides).
            Default is *False* (return the full time-resolved envelope).
        map2alm_iter:  Number of iterations for map2alm (SH method only).
        method:  The spectral method to use: 'sh' (spherical harmonics CWT) or
            'fourier' (ring Fourier CWT).

    Returns:
        xr.Dataset with variables ``{var_name}_amp_sym``,
        ``{var_name}_pha_sym``, ``{var_name}_amp_asy``,
        ``{var_name}_pha_asy``.  Dimensions are
        ``(m, period, [time], *non_core_dims, cells)``.
    """
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

    target_lon, target_lat = get_healpix_coords(nside)
    if hp_order == 'nested':
        target_lon = ensure_original_order(target_lon, 'nested')
        target_lat = ensure_original_order(target_lat, 'nested')

    if method == 'sh':
        if lmax is None:
            lmax = 3 * nside - 1

        # Pre-compute A_lm/B_lm for ALL levels in batched map2alm calls.
        # This is the only place map2alm runs; each map_blocks block uses
        # the pre-computed coefficients without touching pixel data again.
        zwn_list = list({mode['zwn'] for mode in modes})
        non_core = [d for d in da.dims if d not in (time_dim, cell_dim)]
        n_extra = max(1, int(np.prod([da.sizes[d] for d in non_core])))
        lev_batch = _compute_lev_batch(da.sizes[time_dim], da.sizes[cell_dim], n_extra=n_extra)
        logger.info(
            f"Pre-computing SH coefficients for {len(zwn_list)} |m| group(s) "
            f"across all levels (lev_batch={lev_batch})..."
        )
        alm_cache = _sh_precompute_alm(
            da, zwn_list=zwn_list, time_dim=time_dim,
            lmax=lmax, map2alm_iter=map2alm_iter,
            order=hp_order, lev_batch=lev_batch,
        )
        spectrum_kwargs = {
            '_alm_cache': alm_cache,
            '_da_ref': da,  # used to map block coords to linear indices
        }
    elif method == 'fourier':
        # Pre-compute ring Fourier coefficients (Ck, Sk) for ALL levels.
        # Memory cost: n_zwn × n_time × n_extra × n_rings × 16 bytes ≈ trivial
        # compared to pixel data.  Allows each map_blocks block to run the
        # CWT + assign_to_cells without loading any pixel data.
        zwn_list = list({mode['zwn'] for mode in modes})
        logger.info(
            f"Pre-computing ring Fourier coefficients for "
            f"{len(zwn_list)} ZWN group(s) across all levels..."
        )
        ring_cache = _fourier_precompute_ring_coefs(
            da, zwn_list=zwn_list, time_dim=time_dim, order=hp_order,
        )
        spectrum_kwargs = {
            '_ring_cache': ring_cache,
            '_da_ref': da,  # used to map block coords to linear indices
        }
    else:
        raise ValueError(f"Unknown wavelet method: {method}")

    # Compute tidal analysis on each ZWN group
    out_ds = _iterate_tidal_analysis(
        da, ds, modes, zwn_mode_groups, time_dim, cell_dim, hp_order,
        dt_hours, dj, temporal_mean, method, periods_hours, var_name,
        ds[var_name].attrs.get('units', ''), target_lon, target_lat,
        spectrum_kwargs
    )

    # _iterate_tidal_analysis sets '_recommended_dask_scheduler' on out_ds;
    # save it before the attrs overwrite below wipes it.
    scheduler = out_ds.attrs.pop('_recommended_dask_scheduler', 'synchronous')
    n_workers = out_ds.attrs.pop('_recommended_dask_num_workers', 1)

    # Preserve dataset attributes and grid mapping
    out_ds.attrs = ds.attrs.copy()
    out_ds = add_healpix_grid_mapping(out_ds, nside, order=hp_order)
    out_ds.attrs = append_history(
        out_ds.attrs,
        f"Wavelet tidal analysis (method: {method}, periods: {periods_hours}h, "
        f"m: {m_filters}, temporal_mean={temporal_mean})."
    )
    out_ds.attrs['_recommended_dask_scheduler'] = scheduler
    out_ds.attrs['_recommended_dask_num_workers'] = n_workers

    return out_ds
