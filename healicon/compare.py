"""Statistical comparison of two atmospheric datasets.

Public API
----------
compute_stats(arr_ref, arr_cmp)        → dict of scalar metrics
compare(ds_ref, ds_cmp, variables, …)  → pd.DataFrame
print_table(df, fmt, precision, …)     → None (writes to stdout / stderr)
"""
from __future__ import annotations

import logging
import re
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ── ANSI helpers ────────────────────────────────────────────────────────────
_RESET = '\033[0m'
_GREEN = '\033[92m'
_YELLOW = '\033[93m'
_RED = '\033[91m'

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub('', s)


# ── Core statistics (pure numpy, no xarray dependency) ──────────────────────

def compute_stats(arr_ref: np.ndarray, arr_cmp: np.ndarray) -> dict:
    """Compute comparison statistics over flat arrays of jointly valid values.

    Parameters
    ----------
    arr_ref, arr_cmp : array-like
        Flattened values from the reference and comparison datasets.
        NaN and Inf are excluded before computation.

    Returns
    -------
    dict with keys: N, Bias, RMSE, cRMSE, MAE, r, σ_ref, σ_cmp, Skill
    """
    arr_ref = np.asarray(arr_ref, dtype=np.float64).ravel()
    arr_cmp = np.asarray(arr_cmp, dtype=np.float64).ravel()
    valid = np.isfinite(arr_ref) & np.isfinite(arr_cmp)
    ref = arr_ref[valid]
    cmp = arr_cmp[valid]
    n = int(valid.sum())
    nan = float('nan')

    if n < 2:
        return dict(N=n, Bias=nan, RMSE=nan, cRMSE=nan, MAE=nan, r=nan,
                    σ_ref=nan, σ_cmp=nan, Skill=nan)

    # BLAS-backed dot products replace elementwise square+mean passes, and the
    # |diff| pass reuses the diff buffer -- ~1.5x faster, fewer temporaries.
    diff = cmp - ref
    s_d = float(diff.sum())
    s_d2 = float(np.dot(diff, diff))
    bias = s_d / n
    rmse = float(np.sqrt(s_d2 / n))
    crmse = float(np.sqrt(max(0.0, rmse ** 2 - bias ** 2)))
    mae = float(np.abs(diff, out=diff).sum()) / n  # diff no longer needed

    ref_a = ref - ref.mean()
    cmp_a = cmp - cmp.mean()
    sxx = float(np.dot(ref_a, ref_a))
    syy = float(np.dot(cmp_a, cmp_a))
    sxy = float(np.dot(ref_a, cmp_a))
    denom = float(np.sqrt(sxx * syy))
    r = sxy / denom if denom > 0 else nan

    sigma_ref = float(np.sqrt(sxx / (n - 1)))
    sigma_cmp = float(np.sqrt(syy / (n - 1)))
    skill = 1.0 - rmse ** 2 / sigma_ref ** 2 if sigma_ref > 0 else nan

    return dict(N=n, Bias=float(bias), RMSE=rmse, cRMSE=crmse, MAE=float(mae),
                r=float(r), σ_ref=sigma_ref, σ_cmp=sigma_cmp, Skill=float(skill))


def compute_vector_stats(
        u_ref: np.ndarray, v_ref: np.ndarray,
        u_cmp: np.ndarray, v_cmp: np.ndarray,
) -> dict:
    """Vector statistics for horizontal wind (u, v).

    Returns the same key schema as :func:`compute_stats` so the formatter
    can render both scalar and vector rows identically.

    Vector correlation follows Crosby et al. (1993):
        r_v = Σ(uc·ur + vc·vr) / √(Σ(uc²+vc²) · Σ(ur²+vr²))

    Bias is the magnitude of the mean-difference vector;
    RMSE is the RMS of the 2-D difference vectors;
    σ_ref / σ_cmp are the vector standard deviations √(σ_u² + σ_v²).
    """
    u_ref = np.asarray(u_ref, dtype=np.float64).ravel()
    v_ref = np.asarray(v_ref, dtype=np.float64).ravel()
    u_cmp = np.asarray(u_cmp, dtype=np.float64).ravel()
    v_cmp = np.asarray(v_cmp, dtype=np.float64).ravel()

    valid = (np.isfinite(u_ref) & np.isfinite(v_ref)
             & np.isfinite(u_cmp) & np.isfinite(v_cmp))
    ur, vr = u_ref[valid], v_ref[valid]
    uc, vc = u_cmp[valid], v_cmp[valid]
    n = int(valid.sum())
    nan = float('nan')

    if n < 2:
        return dict(N=n, Bias=nan, RMSE=nan, cRMSE=nan, MAE=nan, r=nan,
                    σ_ref=nan, σ_cmp=nan, Skill=nan)

    # Crosby et al. (1993) vector correlation coefficient (BLAS dot products)
    dot = float(np.dot(uc, ur) + np.dot(vc, vr))
    mag_cmp_sq = float(np.dot(uc, uc) + np.dot(vc, vc))
    mag_ref_sq = float(np.dot(ur, ur) + np.dot(vr, vr))
    r = (dot / np.sqrt(mag_cmp_sq * mag_ref_sq)
         if mag_cmp_sq > 0 and mag_ref_sq > 0 else nan)

    du, dv = uc - ur, vc - vr
    mag2 = du * du
    mag2 += dv * dv  # in-place: du^2+dv^2
    rmse = float(np.sqrt(mag2.mean()))
    bias = float(np.sqrt(du.mean() ** 2 + dv.mean() ** 2))
    crmse = float(np.sqrt(max(0.0, rmse ** 2 - bias ** 2)))
    mae = float(np.sqrt(mag2, out=mag2).mean())  # reuse buffer

    sigma_ref = float(np.sqrt(np.std(ur, ddof=1) ** 2 + np.std(vr, ddof=1) ** 2))
    sigma_cmp = float(np.sqrt(np.std(uc, ddof=1) ** 2 + np.std(vc, ddof=1) ** 2))
    skill = float(1.0 - rmse ** 2 / sigma_ref ** 2) if sigma_ref > 0 else nan

    return dict(N=n, Bias=bias, RMSE=rmse, cRMSE=crmse, MAE=mae, r=r,
                σ_ref=sigma_ref, σ_cmp=sigma_cmp, Skill=skill)


# ── Vectorized per-level engine ──────────────────────────────────────────────

# Per-level slices smaller than this use the vectorized kernel (Python/numpy
# call overhead dominates there, measured ~14x faster); larger slices keep the
# cache-friendly per-slice loop (measured faster than full-size temporaries).
_VEC_LEVEL_THRESHOLD = 4096


def _stats_by_level(arr_ref, arr_cmp, lev_axis, cmp_lev_axis, indices) -> list[dict]:
    """Per-level compute_stats over `indices`, hybrid loop/vectorized dispatch.

    Returns one stats dict per index, identical to calling
    ``compute_stats(np.take(arr_ref, i, lev_axis), np.take(arr_cmp, i, cmp_lev_axis))``.
    """
    indices = list(indices)
    n_per_level = arr_ref.size // max(arr_ref.shape[lev_axis], 1)
    if n_per_level > _VEC_LEVEL_THRESHOLD:
        return [compute_stats(np.take(arr_ref, i, axis=lev_axis),
                              np.take(arr_cmp, i, axis=cmp_lev_axis))
                for i in indices]
    return _stats_by_level_vec(arr_ref, arr_cmp, lev_axis, cmp_lev_axis, indices)


def _stats_by_level_vec(arr_ref, arr_cmp, lev_axis, cmp_lev_axis, indices) -> list[dict]:
    """All requested levels in a handful of vectorized reductions (no transpose
    copy: reduces over every axis except the level axis in place)."""
    R = np.asarray(arr_ref, dtype=np.float64)
    C = np.asarray(arr_cmp, dtype=np.float64)
    if cmp_lev_axis != lev_axis:
        C = np.moveaxis(C, cmp_lev_axis, lev_axis)  # view
    idx = np.asarray(indices, dtype=int)
    R = np.take(R, idx, axis=lev_axis)
    C = np.take(C, idx, axis=lev_axis)
    axes = tuple(i for i in range(R.ndim) if i != lev_axis)

    with np.errstate(invalid='ignore', divide='ignore'):
        V = np.isfinite(R) & np.isfinite(C)
        n = V.sum(axis=axes)
        n_f = np.where(n > 1, n, np.nan).astype(np.float64)
        Rz = np.where(V, R, 0.0)
        Cz = np.where(V, C, 0.0)
        D = Cz - Rz
        bias = D.sum(axis=axes) / n_f
        rmse = np.sqrt((D * D).sum(axis=axes) / n_f)
        crmse = np.sqrt(np.maximum(0.0, rmse ** 2 - bias ** 2))
        mae = np.abs(D).sum(axis=axes) / n_f
        mr = Rz.sum(axis=axes) / n_f
        mc = Cz.sum(axis=axes) / n_f
        sh = [1] * R.ndim
        sh[lev_axis] = -1
        Ra = np.where(V, R - mr.reshape(sh), 0.0)
        Ca = np.where(V, C - mc.reshape(sh), 0.0)
        sxx = (Ra * Ra).sum(axis=axes)
        syy = (Ca * Ca).sum(axis=axes)
        sxy = (Ra * Ca).sum(axis=axes)
        denom = np.sqrt(sxx * syy)
        r = np.where(denom > 0, sxy / denom, np.nan)
        sig_r = np.sqrt(sxx / (n_f - 1))
        sig_c = np.sqrt(syy / (n_f - 1))
        skill = np.where(sig_r > 0, 1.0 - rmse ** 2 / sig_r ** 2, np.nan)

    keys = ('Bias', 'RMSE', 'cRMSE', 'MAE', 'r', 'σ_ref', 'σ_cmp', 'Skill')
    vals = (bias, rmse, crmse, mae, r, sig_r, sig_c, skill)
    return [dict(N=int(n[i]), **{k: float(v[i]) for k, v in zip(keys, vals)})
            for i in range(len(idx))]


# ── Alignment helpers ────────────────────────────────────────────────────────

def _align_coords(da_ref, da_cmp) -> tuple:
    """Align all shared 1-D coordinates after spatial reduction.

    For each dimension that appears in both arrays but has different coordinate
    values (e.g. global vs. regional lat grids after zonal mean, or different
    vertical-level sets), this function:

    1. Finds the overlapping value range.
    2. Interpolates both arrays to the coarser / smaller grid within that range.

    Time is never touched here (already aligned upstream).
    Raises ``ValueError`` when a shared dimension has no overlap.
    """
    interp_kwargs: dict = {}

    for dim in da_ref.dims:
        if dim == 'time':
            continue
        if dim not in da_cmp.dims:
            continue
        if dim not in da_ref.coords or dim not in da_cmp.coords:
            continue

        c_ref = da_ref[dim].values.astype(float)
        c_cmp = da_cmp[dim].values.astype(float)

        if np.array_equal(c_ref, c_cmp):
            continue  # already identical — nothing to do

        lo = max(float(c_ref.min()), float(c_cmp.min()))
        hi = min(float(c_ref.max()), float(c_cmp.max()))

        if lo > hi:
            raise ValueError(
                f"No overlap in dimension '{dim}': "
                f"REF [{c_ref.min():.4g}, {c_ref.max():.4g}] vs "
                f"CMP [{c_cmp.min():.4g}, {c_cmp.max():.4g}]."
            )

        in_ref = c_ref[(c_ref >= lo) & (c_ref <= hi)]
        in_cmp = c_cmp[(c_cmp >= lo) & (c_cmp <= hi)]
        # Use the coarser (fewer-point) grid as the common target.
        target = in_ref if len(in_ref) <= len(in_cmp) else in_cmp

        logger.info(
            f"Aligning '{dim}': [{lo:.4g}, {hi:.4g}], "
            f"{len(in_ref)} vs {len(in_cmp)} pts → {len(target)} common pts."
        )
        interp_kwargs[dim] = target

    if interp_kwargs:
        # sel is much cheaper than interp when the target values already exist
        # exactly on an array's grid (the common case for the coarser array).
        def _regrid(da):
            sel_kw, int_kw = {}, {}
            for dim, target in interp_kwargs.items():
                own = da[dim].values.astype(float)
                if np.isin(target, own).all():
                    sel_kw[dim] = target
                else:
                    int_kw[dim] = target
            if sel_kw:
                da = da.sel(sel_kw)
            if int_kw:
                da = da.interp(int_kw, method='linear')
            return da

        da_ref = _regrid(da_ref)
        da_cmp = _regrid(da_cmp)

    return da_ref, da_cmp


def _align_time(ds1, ds2):
    """Return (ds1, ds2) restricted to their common time steps."""
    if 'time' not in ds1.dims or 'time' not in ds2.dims:
        return ds1, ds2

    t1 = ds1['time'].values
    t2 = ds2['time'].values
    common = np.intersect1d(t1, t2)

    if len(common) == 0:
        raise ValueError("REF and CMP share no common time steps.")

    n_max = max(len(t1), len(t2))
    if len(common) < n_max:
        logger.warning(
            f"Time ranges differ ({len(t1)} vs {len(t2)} steps). "
            f"Comparing over {len(common)} common steps."
        )

    return ds1.sel(time=common), ds2.sel(time=common)


def _common_variables(ds1, ds2, requested=None) -> list[str]:
    """Return sorted list of variables present in both datasets."""
    common = set(ds1.data_vars) & set(ds2.data_vars)

    if requested is not None:
        missing = set(requested) - common
        if missing:
            logger.warning(f"Variables absent from one or both datasets: {sorted(missing)}")
        common &= set(requested)

    if not common:
        raise ValueError("No common variables found between REF and CMP.")

    return sorted(common)


def _apply_spatial_reduce(ds, mode: str, lat_range: tuple[float, float]):
    """Reduce spatial dimensions; clip to lat_range afterwards."""

    if mode == 'none':
        return ds

    from .extract import zonal_mean as _zonal_mean
    from .grid import get_cells_dim

    # Reduce to zonal mean (handles HEALPix, ICON, and lat-lon automatically).
    try:
        cell_dim = get_cells_dim(ds)
        has_cells = cell_dim in ds.dims
    except Exception:
        has_cells = False

    if has_cells:
        ds = _zonal_mean(ds)
    else:
        # Regular lat-lon: average over longitude
        for lon_name in ('lon', 'longitude'):
            if lon_name in ds.dims:
                ds = ds.mean(lon_name)
                break

    # Clip latitude window
    if 'lat' in ds.dims:
        lat = ds['lat']
        ds = ds.isel(lat=((lat >= lat_range[0]) & (lat <= lat_range[1])).values)

    if mode == 'global':
        if 'lat' in ds.dims:
            weights = np.cos(np.deg2rad(ds['lat']))
            ds = ds.weighted(weights).mean('lat')

    return ds


# ── Orchestrator ─────────────────────────────────────────────────────────────

def compare(
        ds_ref,
        ds_cmp,
        variables: list[str] | None = None,
        by_level: bool = False,
        select_levels: list[float] | None = None,
        reduce: str = 'zonal-mean',
        lat_range: tuple[float, float] = (-90., 90.),
        level_range: tuple[float, float] | None = None,
        vector: bool = False,
        lmax: int | None = None,
        wavelength_km: float | None = None,
) -> 'pd.DataFrame':
    """Compare two datasets and return a DataFrame of statistics.

    Parameters
    ----------
    ds_ref, ds_cmp : xr.Dataset
        Reference and comparison datasets.
    variables : list of str, optional
        Variables to compare. Defaults to all variables common to both.
    by_level : bool
        If True, emit one statistics row per vertical level plus a GLOBAL row.
    select_levels : list of float, optional
        When given (together with ``by_level=True``), only emit rows for the
        nearest available level to each requested value.  Values are in native
        units (m for height grids, hPa for pressure grids).
    reduce : {'zonal-mean', 'global', 'none'}
        Spatial reduction applied before comparison.
        ``'none'`` requires identical spatial grids.
    lat_range : (float, float)
        Latitude window in degrees (default: full globe).
    level_range : (float, float), optional
        Restrict vertical range. Values in native coordinate units
        (m for height grids, hPa for pressure grids).
    vector : bool
        If True and both ``u`` and ``v`` are present, append a ``wind`` row
        with vector correlation statistics (Crosby et al., 1993).
    lmax : int, optional
        Hard spectral low-pass cutoff: retain only degrees l ≤ lmax.
        Both datasets must be on HEALPix grids.
    wavelength_km : float, optional
        Hard spectral low-pass cutoff expressed as a physical scale (km).
        Converted to ``lmax`` internally.  Both datasets must be on HEALPix.

    Returns
    -------
    pd.DataFrame
        One row per (variable, level) pair. Internal columns ``is_pres`` and
        ``is_meter`` carry coordinate metadata for the formatter.
        ``df.attrs['actual_lat_range']`` and ``df.attrs['actual_level_range']``
        hold the actual coordinate ranges used in the comparison.
    """
    import pandas as pd

    ds_ref, ds_cmp = _align_time(ds_ref, ds_cmp)
    variables = _common_variables(ds_ref, ds_cmp, variables)

    logger.info(f"Comparing {len(variables)} variable(s): {', '.join(variables)}")

    actual_lat_range = None
    actual_level_range = None

    def _subset_dataset(ds, vars_list):
        vars_to_keep = list(vars_list)
        for name, var in ds.variables.items():
            if var.attrs.get('grid_mapping_name') == 'healpix':
                vars_to_keep.append(name)
            elif var.attrs.get('grid_mapping') in ds.variables:
                vars_to_keep.append(var.attrs['grid_mapping'])
        seen = set()
        unique_vars = [v for v in vars_to_keep if
                       v in ds.variables and not (v in seen or seen.add(v))]
        return ds[unique_vars]

    # ── Optional low-pass spectral filter (before spatial reduction) ──────
    if lmax is not None or wavelength_km is not None:
        from .grid import is_healpix as _is_healpix
        if not (_is_healpix(ds_ref) and _is_healpix(ds_cmp)):
            raise ValueError(
                "Low-pass filtering (--lmax / --wavelength) requires both "
                "datasets to be on HEALPix grids."
            )
        from .analysis import filter_spatial as _filter_spatial
        filt_kw = dict(lmax=lmax) if lmax is not None else dict(wavelength_km=wavelength_km)
        logger.info(f"Applying spectral low-pass filter: {filt_kw}")
        ds_ref = _filter_spatial(_subset_dataset(ds_ref, variables), **filt_kw)
        ds_cmp = _filter_spatial(_subset_dataset(ds_cmp, variables), **filt_kw)
    else:
        ds_ref = _subset_dataset(ds_ref, variables)
        ds_cmp = _subset_dataset(ds_cmp, variables)

    ds_ref = _apply_spatial_reduce(ds_ref, reduce, lat_range)
    ds_cmp = _apply_spatial_reduce(ds_cmp, reduce, lat_range)

    from .cf_coords import _find_coordinate, _coord_is_meter, _is_pressure_coord
    from .visualize.common import VARIABLE_ATTRS

    _skip_dims = {'time', 'lat', 'latitude', 'lon', 'longitude'}

    def _find_lev_dim(da, var_name):
        """Return the vertical dimension name, with CF detection and a fallback."""
        try:
            coord = _find_coordinate(da.to_dataset(name=var_name), 'level',
                                     raise_notfound=False)
            if coord is not None and coord.name in da.dims:
                return coord.name
        except Exception:
            pass
        # Fallback: first dim that is not a spatiotemporal dim
        for _d in da.dims:
            if _d.lower() not in _skip_dims:
                logger.debug("'%s': level dim '%s' found by fallback (dims=%s).",
                             var_name, _d, list(da.dims))
                return _d
        return None

    DISPLAY_MAPPING = {
        'temp': 'Temperature',
        'ta': 'Temperature',
        'temperature': 'Temperature',
        'u': 'Zonal wind',
        'ua': 'Zonal wind',
        'v': 'Meridional wind',
        'va': 'Meridional wind',
        'w': 'Vertical wind',
        'wa': 'Vertical wind',
        'wind': 'Wind speed',
        'wind_speed': 'Wind speed',
        'pres': 'Pressure',
        'p': 'Pressure',
        'pressure': 'Pressure',
        'qv': 'Specific humidity',
        'hus': 'Specific humidity',
        'zg': 'Geopotential height',
        'tke': 'TKE',
    }

    records = []
    _vector_cache: dict[str, dict] = {}

    for var in variables:
        da_ref = ds_ref[var]
        da_cmp = ds_cmp[var]

        display_name = DISPLAY_MAPPING.get(var.lower())
        if not display_name:
            display_name = VARIABLE_ATTRS.get(var, {}).get('label')
        if not display_name and var in ds_ref:
            long_name = ds_ref[var].attrs.get('long_name')
            if long_name and len(long_name) <= 20:
                display_name = long_name
        if not display_name:
            display_name = var

        attrs = VARIABLE_ATTRS.get(var, {})
        factor = attrs.get('factor', 1.0)
        units = attrs.get('units', da_ref.attrs.get('units', ''))

        lev_dim = _find_lev_dim(da_ref, var)
        logger.debug("'%s': lev_dim=%r  da_ref.dims=%s  da_cmp.dims=%s",
                     var, lev_dim, list(da_ref.dims), list(da_cmp.dims))

        is_meter = (_coord_is_meter(da_ref[lev_dim])
                    if lev_dim and lev_dim in da_ref.coords else False)
        is_pres = (_is_pressure_coord(lev_dim, da_ref.coords) if lev_dim else False)

        # Apply level range filter
        if level_range is not None and lev_dim is not None and lev_dim in da_ref.coords:
            lv = da_ref[lev_dim]
            mask = (lv >= level_range[0]) & (lv <= level_range[1])
            da_ref = da_ref.isel({lev_dim: mask.values})
            if lev_dim in da_cmp.dims:
                if lev_dim in da_cmp.coords:
                    lv_cmp = da_cmp[lev_dim]
                    mask_cmp = (lv_cmp >= level_range[0]) & (lv_cmp <= level_range[1])
                    da_cmp = da_cmp.isel({lev_dim: mask_cmp.values})
                else:
                    # da_cmp shares the dim name but has no coordinate values;
                    # reuse the same positional index mask from da_ref.
                    da_cmp = da_cmp.isel({lev_dim: mask.values})

        # Capture actual level range AFTER filtering (first variable wins).
        if actual_level_range is None and lev_dim and lev_dim in da_ref.coords:
            lv_vals = da_ref[lev_dim].values.astype(float)
            if len(lv_vals):
                actual_level_range = [float(lv_vals.min()), float(lv_vals.max())]

        # Align remaining coordinates (lat, level, …) to the overlapping range.
        da_ref, da_cmp = _align_coords(da_ref, da_cmp)

        # If both arrays share the same dimension names but in a different order
        # (e.g. da_ref=(time,lat,z_mc) vs da_cmp=(time,z_mc,lat) after
        # unstructured zonal mean), transpose da_cmp to match da_ref so that
        # element-wise flattening (GLOBAL) and per-level slicing (by_level)
        # operate on corresponding grid points.
        if (set(da_ref.dims) == set(da_cmp.dims)
                and list(da_ref.dims) != list(da_cmp.dims)):
            da_cmp = da_cmp.transpose(*da_ref.dims)

        # Capture actual lat range AFTER alignment (first variable wins).
        if actual_lat_range is None and 'lat' in da_ref.coords:
            lat_vals = da_ref['lat'].values.astype(float)
            if len(lat_vals):
                actual_lat_range = [float(lat_vals.min()), float(lat_vals.max())]

        # Eagerly compute (triggers dask if chunked)
        logger.info(f"  Loading '{var}' …")
        arr_ref = np.asarray(da_ref)
        arr_cmp = np.asarray(da_cmp)
        if factor != 1.0:  # avoid a full copy when a no-op
            arr_ref = arr_ref * factor
            arr_cmp = arr_cmp * factor

        # Level axis info — computed unconditionally (used by by_level and vector).
        lev_axis = cmp_lev_axis = lev_vals = None
        if lev_dim is not None and lev_dim in da_ref.dims:
            lev_axis = list(da_ref.dims).index(lev_dim)
            if lev_dim in da_cmp.dims:
                cmp_lev_axis = list(da_cmp.dims).index(lev_dim)
            else:
                cmp_lev_axis = next(
                    (list(da_cmp.dims).index(_d)
                     for _d in da_cmp.dims if _d.lower() not in _skip_dims),
                    lev_axis,
                )
            lev_vals = da_ref[lev_dim].values if lev_dim in da_ref.coords else None

        if vector and var in ('u', 'v'):
            _vector_cache[var] = dict(
                arr_ref=arr_ref, arr_cmp=arr_cmp,
                lev_dim=lev_dim, lev_axis=lev_axis,
                cmp_lev_axis=cmp_lev_axis, lev_vals=lev_vals,
                is_pres=is_pres, is_meter=is_meter, units=units,
            )

        _skip_scalar = vector and var in ('u', 'v')

        if by_level and lev_axis is not None:
            if lev_vals is None:
                logger.warning("'%s': level dim '%s' has no coordinate values; "
                               "skipping per-level rows.", var, lev_dim)
            else:
                n_lev = min(arr_ref.shape[lev_axis], arr_cmp.shape[cmp_lev_axis])

                # Build the list of (array_index, coordinate_value) pairs to emit.
                if select_levels is not None:
                    # Nearest-level match for each requested value; deduplicate.
                    seen: set[int] = set()
                    indices = []
                    for sv in select_levels:
                        idx = int(np.argmin(np.abs(lev_vals - sv)))
                        if idx < n_lev and idx not in seen:
                            indices.append(idx)
                            seen.add(idx)
                    indices.sort()
                else:
                    indices = list(range(n_lev))

                if not _skip_scalar:
                    stats_list = _stats_by_level(arr_ref, arr_cmp,
                                                 lev_axis, cmp_lev_axis, indices)
                    for i, stats in zip(indices, stats_list):
                        lev_f = float(lev_vals[i])
                        lev_display = lev_f / 1000.0 if is_meter else lev_f
                        records.append({'Variable': display_name, 'Units': units,
                                        'is_pres': is_pres, 'is_meter': is_meter,
                                        'Level': lev_display, **stats})

        if not _skip_scalar:
            # GLOBAL row: flatten all remaining dimensions
            stats_g = compute_stats(arr_ref, arr_cmp)
            records.append({'Variable': display_name, 'Units': units,
                            'is_pres': is_pres, 'is_meter': is_meter,
                            'Level': None, **stats_g})

    # ── Vector wind statistics ────────────────────────────────────────────────
    if vector:
        if 'u' in _vector_cache and 'v' in _vector_cache:
            cu, cv = _vector_cache['u'], _vector_cache['v']
            vec_pres = cu['is_pres']
            vec_m = cu['is_meter']
            vec_u = cu['units'] or 'm s⁻¹'
            u_r, u_c = cu['arr_ref'], cu['arr_cmp']
            v_r, v_c = cv['arr_ref'], cv['arr_cmp']

            if by_level and cu['lev_axis'] is not None and cu['lev_vals'] is not None:
                lv_vals = cu['lev_vals']
                la_u, la_v = cu['lev_axis'], cv['lev_axis']
                cla_u, cla_v = cu['cmp_lev_axis'], cv['cmp_lev_axis']
                n_lev = min(u_r.shape[la_u], u_c.shape[cla_u],
                            v_r.shape[la_v], v_c.shape[cla_v])

                if select_levels is not None:
                    seen_v: set[int] = set()
                    vec_indices = []
                    for sv in select_levels:
                        idx = int(np.argmin(np.abs(lv_vals - sv)))
                        if idx < n_lev and idx not in seen_v:
                            vec_indices.append(idx)
                            seen_v.add(idx)
                    vec_indices.sort()
                else:
                    vec_indices = list(range(n_lev))

                for i in vec_indices:
                    stats = compute_vector_stats(
                        np.take(u_r, i, axis=la_u), np.take(v_r, i, axis=la_v),
                        np.take(u_c, i, axis=cla_u), np.take(v_c, i, axis=cla_v),
                    )
                    lev_f = float(lv_vals[i])
                    records.append({'Variable': 'Winds', 'Units': vec_u,
                                    'is_pres': vec_pres, 'is_meter': vec_m,
                                    'Level': lev_f / 1000.0 if vec_m else lev_f,
                                    **stats})

            stats_vec = compute_vector_stats(u_r, v_r, u_c, v_c)
            records.append({'Variable': 'Winds', 'Units': vec_u,
                            'is_pres': vec_pres, 'is_meter': vec_m,
                            'Level': None, **stats_vec})
        else:
            logger.warning(
                "--vector: both 'u' and 'v' must be present in both datasets "
                "(found in cache: %s).", sorted(_vector_cache))

    df = pd.DataFrame(records)
    df.attrs['actual_lat_range'] = actual_lat_range
    df.attrs['actual_level_range'] = actual_level_range
    return df


# ── Formatter ────────────────────────────────────────────────────────────────

def _col_widths(headers: list[str], rows: list[list[str]]) -> list[int]:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(_strip_ansi(cell)))
    return widths


def _fmt_row(cells: list[str], widths: list[int], left_cols: int = 1) -> str:
    parts = []
    for i, (cell, w) in enumerate(zip(cells, widths)):
        pad = w - len(_strip_ansi(cell))
        if i < left_cols:
            parts.append(cell + ' ' * pad)
        else:
            parts.append(' ' * pad + cell)
    return '  '.join(parts)


def _fmt_num(val, precision: int, sign: bool = False) -> str:
    if not isinstance(val, (int, float)) or (isinstance(val, float) and np.isnan(val)):
        return '—'
    fmt = f'+.{precision}f' if sign else f'.{precision}f'
    return format(val, fmt)


def print_table(
        df,
        fmt: str = 'table',
        precision: int = 3,
        no_color: bool = False,
        meta: dict | None = None,
        output_file: str | None = None,
) -> None:
    """Print the comparison DataFrame as a formatted table, CSV, or Markdown.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compare`.
    fmt : {'table', 'csv', 'markdown', 'latex'}
    precision : int
        Decimal places for floating-point columns.
    no_color : bool
        Suppress ANSI color even when stdout is a TTY.
    meta : dict, optional
        Key-value pairs printed as a header line to stderr.
    output_file : str, optional
        Write CSV, Markdown, or LaTeX output to this path instead of stdout.
    """
    if meta:
        header_str = '  ·  '.join(f'{k}: {v}' for k, v in meta.items())
        print(f'\n  {header_str}\n', file=sys.stderr)

    if fmt == 'csv':
        if output_file:
            _export_df(df, precision).to_csv(output_file, index=False)
            logger.info(f"CSV written to {output_file}")
        else:
            _export_df(df, precision).to_csv(sys.stdout, index=False)
        return

    if fmt == 'markdown':
        export = _export_df(df, precision)
        if 'Variable' in export.columns:
            export.loc[export['Variable'].duplicated(), 'Variable'] = ''
        try:
            md_text = export.to_markdown(index=False)
        except ImportError:
            from tabulate import tabulate as _tab
            md_text = _tab(export, headers='keys', tablefmt='pipe', showindex=False)
        if output_file:
            with open(output_file, 'w') as fh:
                fh.write(md_text + '\n')
            logger.info(f"Markdown written to {output_file}")
        else:
            print(md_text)
        return

    if fmt == 'latex':
        latex_text = _to_latex(df, precision)
        if output_file:
            with open(output_file, 'w') as fh:
                fh.write(latex_text + '\n')
            logger.info(f"LaTeX written to {output_file}")
        else:
            print(latex_text)
        return

    _print_table(df, precision, no_color)


def _export_df(df, precision: int):
    """Build a clean DataFrame suitable for CSV / Markdown export."""
    import pandas as pd
    rows = []
    for _, row in df.iterrows():
        lev = row['Level']
        lev_str = 'GLOBAL' if lev is None or (isinstance(lev, float) and np.isnan(lev)) else (
            f'{lev:.1f}' if not row['is_pres'] else f'{lev:.2g}')
        rows.append({
            'Variable': row['Variable'],
            'Level': lev_str,
            'N': int(row['N']),
            'Bias': round(row['Bias'], precision),
            'RMSE': round(row['RMSE'], precision),
            'cRMSE': round(row['cRMSE'], precision),
            'MAE': round(row['MAE'], precision),
            'r': round(row['r'], precision),
            'σ_ref': round(row['σ_ref'], precision),
            'σ_cmp': round(row['σ_cmp'], precision),
            'Skill': round(row['Skill'], precision),
        })
    return pd.DataFrame(rows)


def _to_latex(df, precision: int) -> str:
    import numpy as np

    p = precision
    multi_var = df['Variable'].nunique() > 1
    is_pres = bool(df['is_pres'].any())
    is_meter = bool(df['is_meter'].any())
    lev_hdr = 'Pres (hPa)' if is_pres else ('Alt (km)' if is_meter else 'Level')

    def escape_latex(text: str) -> str:
        if not isinstance(text, str):
            return text
        if '$' in text:
            return text.replace('%', '\\%')

        text = text.replace('\\', '\\textbackslash{}')
        for char in '&%$#_':
            text = text.replace(char, f'\\{char}')
        text = text.replace('{', '\\{').replace('}', '\\}')

        text = text.replace('⁻¹', '\\textsuperscript{-1}')
        text = text.replace('⁻²', '\\textsuperscript{-2}')
        text = text.replace('²', '\\textsuperscript{2}')
        text = text.replace('³', '\\textsuperscript{3}')
        text = text.replace('°', '\\textdegree{}')

        if '^' in text or '**' in text:
            text = text.replace('^-1', '\\textsuperscript{-1}')
            text = text.replace('**-1', '\\textsuperscript{-1}')
            text = text.replace('^-2', '\\textsuperscript{-2}')
            text = text.replace('**-2', '\\textsuperscript{-2}')
            text = text.replace('^2', '\\textsuperscript{2}')
            text = text.replace('**2', '\\textsuperscript{2}')
            text = text.replace('^3', '\\textsuperscript{3}')
            text = text.replace('**3', '\\textsuperscript{3}')

        return text

    def make_latex_headers(lev_hdr: str, units: str) -> list[str]:
        u = f' ({escape_latex(units)})' if units else ''
        stat_hdrs = [
            escape_latex(lev_hdr),
            'N',
            f'Bias{u}',
            f'RMSE{u}',
            f'cRMSE{u}',
            f'MAE{u}',
            'r',
            f'$\\sigma_{{\\text{{ref}}}}${u}',
            f'$\\sigma_{{\\text{{cmp}}}}${u}',
            'Skill'
        ]
        if multi_var:
            return ['Variable'] + stat_hdrs
        return stat_hdrs

    var_units: dict[str, str] = {}
    for _, row in df.iterrows():
        v = row['Variable']
        if v not in var_units:
            var_units[v] = row['Units'] or ''

    first_units = next(iter(var_units.values()), '')
    align = ('l' if multi_var else '') + 'l' + 'r' * 9

    lines = []
    lines.append(f'\\begin{{tabular}}{{{align}}}')
    lines.append('\\toprule')

    headers = make_latex_headers(lev_hdr, first_units)
    lines.append(' & '.join(headers) + ' \\\\')
    lines.append('\\midrule')

    prev_var = None
    prev_units = first_units

    var_has_level_rows: dict[str, bool] = {}
    for _, row in df.iterrows():
        v = row['Variable']
        if v not in var_has_level_rows:
            var_has_level_rows[v] = False
        lv = row['Level']
        if lv is not None and not (isinstance(lv, float) and np.isnan(lv)):
            var_has_level_rows[v] = True

    for _, row in df.iterrows():
        cur_var = row['Variable']
        cur_units = var_units.get(cur_var, '')

        if multi_var and prev_var is not None and cur_var != prev_var:
            if cur_units != prev_units:
                new_hdrs = make_latex_headers(lev_hdr, cur_units)
                lines.append('\\midrule')
                lines.append(' & '.join(new_hdrs) + ' \\\\')
                lines.append('\\midrule')
            else:
                lines.append('\\midrule')

        lev = row['Level']
        is_global = lev is None or (isinstance(lev, float) and np.isnan(lev))
        lev_cell = 'GLOBAL' if is_global else (f'{lev:.2g}' if is_pres else f'{lev:.1f}')

        if is_global and var_has_level_rows.get(cur_var, False):
            lines.append('\\midrule')

        def fmt_val(val, precision: int, sign: bool = False) -> str:
            if not isinstance(val, (int, float)) or (isinstance(val, float) and np.isnan(val)):
                return '---'
            fmt = f'+.{precision}f' if sign else f'.{precision}f'
            return format(val, fmt)

        cells = [
            escape_latex(lev_cell),
            f'{int(row["N"]):,}',
            fmt_val(row['Bias'], p, sign=True),
            fmt_val(row['RMSE'], p),
            fmt_val(row['cRMSE'], p),
            fmt_val(row['MAE'], p),
            fmt_val(row['r'], p),
            fmt_val(row['σ_ref'], p),
            fmt_val(row['σ_cmp'], p),
            fmt_val(row['Skill'], p),
        ]
        if multi_var:
            v_cell = cur_var if cur_var != prev_var else ""
            cells = [escape_latex(v_cell)] + cells

        lines.append(' & '.join(cells) + ' \\\\')
        prev_var = cur_var
        prev_units = cur_units

    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    return '\n'.join(lines)


def _make_stat_headers(lev_hdr: str, units: str) -> list[str]:
    """Build the stat column headers with an optional unit suffix."""
    u = f' ({units})' if units else ''
    return [lev_hdr, 'N',
            f'Bias{u}', f'RMSE{u}', f'cRMSE{u}', f'MAE{u}',
            'r', f'σ_ref{u}', f'σ_cmp{u}', 'Skill']


def _print_table(df, precision: int, no_color: bool) -> None:
    isatty = sys.stdout.isatty()
    do_color = not no_color and isatty
    p = precision

    multi_var = df['Variable'].nunique() > 1
    is_pres = bool(df['is_pres'].any())
    is_meter = bool(df['is_meter'].any())
    lev_hdr = 'Pres (hPa)' if is_pres else ('Alt (km)' if is_meter else 'Level')

    # Per-variable unit lookup (preserves insertion order)
    var_units: dict[str, str] = {}
    for _, row in df.iterrows():
        v = row['Variable']
        if v not in var_units:
            var_units[v] = row['Units'] or ''

    # Initial header uses the first variable's units
    first_units = next(iter(var_units.values()), '')
    stat_headers = _make_stat_headers(lev_hdr, first_units)
    headers = (['Variable'] + stat_headers) if multi_var else stat_headers

    # Build string rows
    str_rows = []
    prev_v_prep = None
    for _, row in df.iterrows():
        lev = row['Level']
        lev_cell = ('GLOBAL' if lev is None or (isinstance(lev, float) and np.isnan(lev))
                    else (f'{lev:.2g}' if is_pres else f'{lev:.1f}'))

        r_val = row['r']
        r_str = _fmt_num(r_val, p)
        if do_color and isinstance(r_val, float) and not np.isnan(r_val):
            code = _GREEN if r_val >= 0.99 else (_YELLOW if r_val >= 0.95 else _RED)
            r_str = f'{code}{r_str}{_RESET}'

        cells = [
            lev_cell,
            f'{int(row["N"]):,}',
            _fmt_num(row['Bias'], p, sign=True),
            _fmt_num(row['RMSE'], p),
            _fmt_num(row['cRMSE'], p),
            _fmt_num(row['MAE'], p),
            r_str,
            _fmt_num(row['σ_ref'], p),
            _fmt_num(row['σ_cmp'], p),
            _fmt_num(row['Skill'], p),
        ]
        if multi_var:
            v_cell = row['Variable'] if row['Variable'] != prev_v_prep else ""
            prev_v_prep = row['Variable']
            cells = [v_cell] + cells
        str_rows.append(cells)

    left_cols = 2 if multi_var else 1
    widths = _col_widths(headers, str_rows)

    # If units differ between variables we will re-emit headers; compute
    # widths against ALL possible header variants so columns stay stable.
    if multi_var and len(set(var_units.values())) > 1:
        for vu in var_units.values():
            alt_headers = (['Variable'] + _make_stat_headers(lev_hdr, vu))
            for i, h in enumerate(alt_headers):
                widths[i] = max(widths[i], len(h))

    sep = '  '.join('─' * w for w in widths)
    thin = '  '.join('╌' * w for w in widths)

    print(_fmt_row(headers, widths, left_cols=left_cols))
    print(sep)

    prev_var = None
    prev_units = first_units
    var_has_level_rows: dict[str, bool] = {}
    for _, row in df.iterrows():
        v = row['Variable']
        if v not in var_has_level_rows:
            var_has_level_rows[v] = False
        lv = row['Level']
        if lv is not None and not (isinstance(lv, float) and np.isnan(lv)):
            var_has_level_rows[v] = True

    for cells, (_, row) in zip(str_rows, df.iterrows()):
        cur_var = row['Variable']
        cur_units = var_units.get(cur_var, '')

        if multi_var and prev_var is not None and cur_var != prev_var:
            if cur_units != prev_units:
                # Units changed — re-emit header with new unit suffix
                new_stat = _make_stat_headers(lev_hdr, cur_units)
                new_hdr = (['Variable'] + new_stat)
                print(sep)
                print()
                print(_fmt_row(new_hdr, widths, left_cols=left_cols))
                print(sep)
            else:
                print()

        prev_var = cur_var
        prev_units = cur_units

        _lv = row['Level']
        if (_lv is None or (isinstance(_lv, float) and np.isnan(_lv))) and var_has_level_rows.get(
                cur_var, False):
            print(thin)
        print(_fmt_row(cells, widths, left_cols=left_cols))

    print(sep)
