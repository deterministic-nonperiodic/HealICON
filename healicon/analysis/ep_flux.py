"""
Eliassen-Palm Flux Analysis
===========================

Coordinate-aware EP flux with separate HEIGHT and PRESSURE branches, each
self-consistent in its vertical derivative and ρ₀ treatment.

  HEIGHT coords (metres) — full primitive-equation TEM:
      Uses the full θ-gradient stream function Ψ (Hardiman et al. / Iwasaki 1989):

      Ψ = a cosφ · (ρ₀[v'θ']·θ_z − ρ₀[w'θ']·(θ_φ/a)) / ((θ_φ/a)² + θ_z²)

      F^(phi) = -rho0 a cosphi [u'v']  +  Ψ · du_bar/dz
      F^(z)   = -rho0 a cosphi [u'w']  +  Ψ · f_hat
      div = (a cosphi)^-1 d(F_phi cosphi)/dphi + dF_z/dz
      a_EP = div / (rho0 a cosphi)

      When θ_φ → 0 (horizontal θ-surfaces), Ψ → a cosφ ρ₀ [v'θ']/θ_z and the
      formulas reduce exactly to Andrews, Holton & Leovy (1987), eq. 3.5.3:
          F^(phi) = rho0 a cosphi ( u_bar_z [v'th']/th_z - [u'v'] )
          F^(z)   = rho0 a cosphi ( f_hat   [v'th']/th_z - [u'w'] )

      Also outputs: Ψ [kg/s], v* and w* (TEM residual velocities).

  PRESSURE coords (isobaric): Edmon/Hoskins/McIntyre (1980) form
      F^(phi) = a cosphi ( u_bar_p [v'th']/th_p - [u'v'] )
      F^(p)   = a cosphi ( f_hat   [v'th']/th_p - [u'omega] )
      div = (a cosphi)^-1 d(F_phi cosphi)/dphi + dF_p/dp
      a_EP = div / (a cosphi)                       # NO rho0

f_hat = f - (a cosphi)^-1 d(u_bar cosphi)/dphi = f + zeta_bar  is the absolute
vorticity factor and is coordinate-independent.

The isobaric vertical eddy flux needs omega = dp/dt, not w. When only w (m/s) is
available it is converted hydrostatically: omega' ~ -rho0 g w', hence
    [u'omega] ~ -rho0 g [u'w'] .
This is the one place density re-enters the pressure branch; it is an excellent
approximation for the zonal mean even where the instantaneous flow is
non-hydrostatic, because the mean state stays close to hydrostatic balance.

Validity: above ~100-120 km molecular diffusion and ion drag dominate the
momentum budget and the EP-flux framework itself (not the coordinate choice)
loses meaning. compute_ep_divergence records a documented validity floor in the
output attrs; it does NOT mask by default (that is the plotting layer's job).

References:
    Andrews, D. G., Holton, J. R., & Leovy, C. B. (1987).
    Middle Atmosphere Dynamics. Academic Press.

    Edmon , H. J., B. J. Hoskins, and M. E. McIntyre, 1980:
    Eliassen-Palm Cross-Sections for the Troposphere. J. Atmos. Sci., 37, 2600-2616,
    https://doi.org/10.1175/1520-0469(1980)037<2600:EPCSFT>2.0.CO;2.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import xarray as xr

from ..cf_coords import _cf_guess, _find_coordinate, _is_z, convert_units
from ..extract import zonal_mean as _zonal_mean
from ..grid import append_history, get_cells_dim

logger = logging.getLogger(__name__)

# ── Physical constants ────────────────────────────────────────────────────────
_RD = 287.05  # J kg⁻¹ K⁻¹  dry-air gas constant
_CP = 1004.0  # J kg⁻¹ K⁻¹  specific heat at constant pressure
_G = 9.80665  # m s⁻²       standard gravity
_OMEGA = 7.2921e-5  # rad s⁻¹     Earth rotation rate
_A = 6.371e6  # m            Earth mean radius
_KAPPA = _RD / _CP  # ≈ 0.2857
_P0 = 1.0e5  # Pa           reference pressure for θ

_SECS_PER_DAY = 86400.0

# Standard isobaric levels (Pa) used when interpolating from model levels.
_STD_PRES_LEVELS = np.array([
    100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000,
    15000, 10000, 7000, 5000, 3000, 2000, 1000, 700, 500, 300,
    200, 100, 70, 50, 30, 20, 10, 1, 0.1, 0.01,
    0.001, 0.0001, 0.00001], dtype=float)  # Pa

# ── Variable lookup ───────────────────────────────────────────────────────────

# Fallback name patterns for each physical quantity.
_VAR_NAME_ALIASES = {
    'u': ('u', 'ua', 'U'),
    'v': ('v', 'va', 'V'),
    'theta': ('theta', 'theta_zm', 'pot_temp', 'pt'),
    'temperature': ('temp', 'temp_zm', 'ta', 'temperature', 't'),
    'pressure': ('pres', 'pres_zm', 'pressure', 'pfull', 'p'),
    'w': ('w', 'wa', 'wap'),  # omega (Pa/s) excluded — different physical quantity
    'density': ('rho', 'rho0', 'density'),
}


def _find_var(ds: xr.Dataset, target: str) -> str | None:
    """Find the actual variable name for a physical quantity in *ds*.

    Priority:
    1. Well-known name aliases (unambiguous, e.g. 'temp' → temperature).
    2. CF ``standard_name`` match (reliable, e.g. ``air_temperature``).
    3. CF units-only fallback (ambiguous — only for vars without standard_name).

    Returns ``None`` when no candidate is found.
    """
    # 1. Name-alias match (most specific, no false positives)
    for alias in _VAR_NAME_ALIASES.get(target, ()):
        if alias in ds:
            return alias
    # 2. CF-aware guess (standard_name, then units-only fallback)
    cf_hit = _cf_guess(ds, target)
    if cf_hit is not None:
        return cf_hit
    return None


def _parse_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Validate and normalise an input dataset for the EP flux pipeline.

    1. Resolve variable names to **canonical** forms (``u``, ``v``, ``w``,
       ``temp``, ``theta``, ``pres``) using CF conventions + common aliases.
    2. Derive missing thermodynamic fields where possible:
       * θ from T + p:  ``theta = temp * (P0/pres)^kappa``
       * θ from T + z:  ``theta ≈ temp * exp(κ z / H)``  (analytical)
       * p from θ + T:  ``pres = P0 * (T/θ)^(1/κ)``
    3. Validate that the minimum required set (u, v, and at least one
       thermodynamic path to θ) is present; raise ``ValueError`` otherwise.

    Returns
    -------
    xr.Dataset
        Copy with canonical variable names and any derived fields added.
    """
    # ── 1. Resolve variable names ────────────────────────────────────────────
    var_map: dict[str, str | None] = {
        'u': _find_var(ds, 'u') or ('u' if 'u' in ds else None),
        'v': _find_var(ds, 'v') or ('v' if 'v' in ds else None),
        'w': _find_var(ds, 'w'),
        'temp': _find_var(ds, 'temperature'),
        'theta': _find_var(ds, 'theta'),
        'pres': _find_var(ds, 'pressure'),
    }

    # Required: u and v
    missing = [k for k in ('u', 'v') if var_map[k] is None]
    if missing:
        raise ValueError(
            f"Dataset missing required wind variables: {missing}. "
            f"Available: {list(ds.data_vars)}"
        )

    # ── 2. Rename to canonical names ─────────────────────────────────────────
    rename = {}
    for canonical, actual in var_map.items():
        if actual is not None and actual != canonical and actual in ds.data_vars:
            rename[actual] = canonical
    if rename:
        # Avoid clobbering: only rename if the canonical name is free
        safe_rename = {k: v for k, v in rename.items() if v not in ds.data_vars}
        if safe_rename:
            logger.debug(f"Renaming variables to canonical names: {safe_rename}")
            ds = ds.rename_vars(safe_rename)

    # ── 2.5 Unit Verification & Auto-Conversion ──────────────────────────────
    from ..cf_coords import check_convert_units
    logger.debug("Validating and auto-converting physical units...")
    ds = check_convert_units(ds)

    # ── 3. Derive θ if not present ───────────────────────────────────────────
    has_theta = 'theta' in ds
    has_temp = 'temp' in ds
    has_pres = 'pres' in ds

    if not has_theta:
        if has_temp and has_pres:
            logger.info("Deriving θ = T·(P₀/p)^κ from 'temp' and 'pres'.")
            ds['theta'] = ds['temp'] * (_P0 / ds['pres']) ** _KAPPA
            ds['theta'].attrs = {'long_name': 'Potential temperature', 'units': 'K'}
        elif has_temp:
            # Try multiple paths to derive θ from T alone
            from ..cf_coords import _coord_is_meter
            try:
                alt_name = _find_alt_name(ds)
                z = ds[alt_name]
            except ValueError:
                z = None

            if z is not None and _is_pressure_coord(alt_name, ds):
                # Vertical coordinate IS pressure → θ = T·(P₀/p)^κ
                p_coord_units = str(z.attrs.get('units', '')) or 'Pa'
                p_coord = convert_units(z.copy(), p_coord_units, 'Pa')
                logger.info(f"Deriving θ from T and pressure coordinate '{alt_name}'.")
                ds['theta'] = ds['temp'] * (_P0 / p_coord) ** _KAPPA
                ds['theta'].attrs = {'long_name': 'Potential temperature', 'units': 'K'}
                # Also promote the pressure coordinate to a proper variable (in Pa)
                if 'pres' not in ds:
                    ds['pres'] = p_coord.broadcast_like(ds['temp'])
                    ds['pres'].attrs = {'long_name': 'Pressure', 'units': 'Pa',
                                        'standard_name': 'air_pressure'}
            elif z is not None and _coord_is_meter(z):
                logger.warning("No pressure field; estimating θ from T via "
                               "local scale height (analytical).")
                H = (_RD * ds['temp']) / _G  # H = Rd·T/g
                ds['theta'] = ds['temp'] * np.exp(_KAPPA * z / H)
                ds['theta'].attrs = {'long_name': 'Potential temperature (estimated)',
                                     'units': 'K'}
            else:
                raise ValueError(
                    "Cannot compute potential temperature: no θ, no pressure, "
                    "and the vertical coordinate is not in metres or pressure. "
                    f"Available variables: {list(ds.data_vars)}"
                )
        else:
            raise ValueError(
                "Cannot compute potential temperature: dataset has neither θ "
                f"nor T. Available: {list(ds.data_vars)}. "
                "Provide at least: theta (θ), or temp (T) + pres (p)."
            )

    # ── 4. Derive pressure from θ + T if missing ─────────────────────────────
    if 'pres' not in ds and 'temp' in ds and 'theta' in ds:
        logger.info("Deriving p = P₀·(T/θ)^(1/κ) from 'temp' and 'theta'.")
        ds['pres'] = _P0 * (ds['temp'] / ds['theta']) ** (1.0 / _KAPPA)
        ds['pres'].attrs = {'long_name': 'Pressure (derived)', 'units': 'Pa',
                            'standard_name': 'air_pressure'}

    logger.info(
        f"Parsed dataset: "
        f"u={'u' in ds}, v={'v' in ds}, w={'w' in ds}, "
        f"temp={'temp' in ds}, theta={'theta' in ds}, pres={'pres' in ds}"
    )

    # ── 5. Drop fully-NaN vertical levels ───────────────────────────────────
    try:
        alt_name = _find_alt_name(ds)
    except ValueError:
        alt_name = None
    if alt_name is not None and alt_name in ds.dims:
        # Use u as the reference field (always present after step 1)
        ref_var = ds['u']
        non_alt_dims = [d for d in ref_var.dims if d != alt_name]
        if non_alt_dims:
            all_nan = ref_var.isnull().all(dim=non_alt_dims)
            n_drop = int(all_nan.sum())
            if n_drop > 0:
                valid_mask = ~all_nan
                ds = ds.sel({alt_name: valid_mask})
                logger.info(
                    f"Dropped {n_drop} fully-NaN levels along '{alt_name}' "
                    f"({ds.sizes[alt_name]} levels remain)."
                )

    return ds


# ── Pressure-level interpolation ─────────────────────────────────────────────

def _log_interp_vertical(
        data_np: np.ndarray,
        pres_np: np.ndarray,
        p_tgt: np.ndarray,
        lev_axis: int,
) -> np.ndarray:
    """Vectorised log-linear interpolation from varying to constant pressure levels.

    Parameters
    ----------
    data_np : ndarray  (..., N, ...)
        Data on model levels; vertical axis at *lev_axis*.
    pres_np : ndarray, same shape
        Pressure at each model level and location (Pa).
    p_tgt : ndarray (P,)
        Target pressure levels in Pa, any order (sorted descending internally).
    lev_axis : int
        Axis index of the model-level dimension.

    Returns
    -------
    ndarray  (..., P, ...)
    """
    # Move level axis to last for simpler indexing
    data = np.moveaxis(data_np, lev_axis, -1).astype(float, copy=False)  # (..., N)
    pres = np.moveaxis(pres_np, lev_axis, -1).astype(float, copy=False)  # (..., N)
    # Target pressure is already sorted ascending by the caller
    log_p_tgt = np.log(p_tgt)  # (P,)
    log_pres = np.log(np.maximum(pres, 1e-10))  # (..., N)

    prefix = data.shape[:-1]
    N = data.shape[-1]
    P = len(p_tgt)
    M = int(np.prod(prefix)) if prefix else 1

    flat_lp = log_pres.reshape(M, N)  # (M, N)
    flat_d = data.reshape(M, N)  # (M, N)

    # Sort every column ascending in log-pressure
    sort_idx = np.argsort(flat_lp, axis=1)
    flat_lp_s = np.take_along_axis(flat_lp, sort_idx, axis=1)  # (M, N) ascending
    flat_d_s = np.take_along_axis(flat_d, sort_idx, axis=1)  # (M, N)

    m_idx = np.arange(M)
    out = np.empty((M, P), dtype=float)

    for i, lp_val in enumerate(log_p_tgt):
        # Count how many sorted levels are <= lp_val per column  → bracket index
        j = (flat_lp_s <= lp_val).sum(axis=1)  # (M,)
        j_above = np.clip(j, 1, N - 1)
        j_below = j_above - 1

        x0 = flat_lp_s[m_idx, j_below]
        x1 = flat_lp_s[m_idx, j_above]
        y0 = flat_d_s[m_idx, j_below]
        y1 = flat_d_s[m_idx, j_above]

        dx = x1 - x0
        # Allow linear extrapolation for points outside the pressure column
        # (e.g. subterranean) to prevent sharp NaN gradients near the surface.
        frac = np.where(np.abs(dx) > 1e-30, (lp_val - x0) / dx, 0.5)
        out[:, i] = y0 + frac * (y1 - y0)

    out = out.reshape(prefix + (P,))
    return np.moveaxis(out, -1, lev_axis)


def _interp_to_pressure_levels(
        ds: xr.Dataset,
        lev_name: str,
        pres_key: str = 'pres',
        p_levels: np.ndarray | None = None,
) -> xr.Dataset:
    """Remap *ds* from model levels to constant isobaric levels.

    Uses the 4-D pressure field *pres_key* (Pa) in *ds* to perform
    log-linear vertical interpolation via :func:`xr.apply_ufunc`.
    xarray aligns non-core dimensions automatically, so variables whose
    non-level dimensions differ from *pres_key* (e.g. coordinate bounds)
    are forwarded unchanged rather than causing a reshape error.

    Parameters
    ----------
    ds : xr.Dataset
    lev_name : str
        Name of the model-level dimension.
    pres_key : str
        Name of the pressure variable in *ds* (default ``'pres'``).
    p_levels : ndarray, optional
        Target pressure levels in Pa.  Defaults to :data:`_STD_PRES_LEVELS`
        filtered to the pressure range present in *ds*.

    Returns
    -------
    xr.Dataset
        On a ``plev`` dimension with ``standard_name='air_pressure'``.
    """
    pres_da = ds[pres_key]
    n_lev = ds.sizes[lev_name]

    # Default: standard levels within the model pressure range
    if p_levels is None:
        p_min = float(pres_da.min())
        p_max = float(pres_da.max())
        p_levels = _STD_PRES_LEVELS[
            (_STD_PRES_LEVELS >= p_min * 0.5) & (_STD_PRES_LEVELS <= p_max * 1.1)
            ]
        if len(p_levels) == 0:
            p_levels = np.geomspace(p_max, p_min, n_lev)

    p_levels = np.sort(np.asarray(p_levels, dtype=float))  # ascending
    P = len(p_levels)

    logger.info(
        f"Interpolating {n_lev} model levels -> {P} isobaric levels "
        f"[{p_levels[0] / 100:.1f}-{p_levels[-1] / 100:.1f} hPa] using '{pres_key}'."
    )

    plev_attrs = {
        'standard_name': 'air_pressure',
        'long_name': 'Pressure',
        'units': 'Pa',
        'axis': 'Z',
        'positive': 'down',
    }
    plev_coord = xr.DataArray(p_levels, dims=['plev'], attrs=plev_attrs)

    # Kernel called by apply_ufunc: level dim is already last (moved by xarray).
    def _kernel(data_nd, pres_nd):
        return _log_interp_vertical(data_nd, pres_nd, p_levels, lev_axis=-1)

    # Non-level dims of the pressure variable - used to filter interpolatable vars.
    pres_non_lev = set(pres_da.dims) - {lev_name}

    new_vars: dict = {}
    for name, da in ds.data_vars.items():
        # Skip non-level variables.
        if lev_name not in da.dims:
            if lev_name in da.coords:
                da = da.drop_vars(lev_name)
            new_vars[name] = da
            continue

        if name == pres_key:
            # Log-linear interpolation for P vs log(P) produces artifacts.
            # We explicitly skip interpolating it and will set it exactly to plev later.
            continue

        # Drop variables whose non-level dims are incompatible with pres_da
        # (e.g. coordinate bounds like height_bnds with a 'bnds=2' dimension).
        da_non_lev = set(da.dims) - {lev_name}
        if not da_non_lev.issubset(pres_non_lev | {'time'}):
            logger.debug(
                f"Dropping '{name}': non-level dims {da_non_lev} not "
                f"compatible with pressure dims {pres_non_lev}."
            )
            continue  # omit from output entirely

        interp_da = xr.apply_ufunc(
            _kernel,
            da,
            pres_da,
            input_core_dims=[[lev_name], [lev_name]],
            output_core_dims=[['plev']],
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={'output_sizes': {'plev': P}, 'allow_rechunk': True},
        )
        interp_da = interp_da.assign_coords(plev=plev_coord)
        interp_da.attrs.update(da.attrs)
        new_vars[name] = interp_da

    out = xr.Dataset(new_vars, attrs=ds.attrs)
    out = out.assign_coords(plev=plev_coord)

    if pres_key in ds.data_vars:
        # Re-create pressure variable perfectly matching plev.
        # This replaces the mathematically inaccurate log-linear interpolation 
        # of P vs log(P) with the exact definition of the isobaric surface.
        out[pres_key] = out['plev']

    # Final safety: remove any residual reference to the old level coordinate.
    if lev_name in out.coords and lev_name != 'plev':
        out = out.drop_vars(lev_name)
    logger.debug(
        f"[_interp_to_pressure_levels] output dims={dict(out.sizes)} "
        f"coords={list(out.coords)} "
        f"data_vars={list(out.data_vars)}"
    )
    return out


def _preprocess_vertical(ds: xr.Dataset) -> xr.Dataset:
    """Ensure the dataset carries a usable vertical coordinate before EP flux.

    Cases, in priority order:

    1. Real height coordinate (metres via :func:`_is_z`) → no-op.
    2. Pressure already a dim/coord                      → no-op.
    3. 4-D pressure *variable* (``pres`` / ``pressure``) → interpolate to
       :data:`_STD_PRES_LEVELS` via :func:`_interp_to_pressure_levels`.
    4. Nothing useful found → warn and return unchanged.
    """
    coord = _find_coordinate(ds, 'level', raise_notfound=False)

    # Case 1: real height coordinate — but guard against integer model-level
    # indices that carry height-like CF attributes (e.g. ICON sets axis='Z'
    # on its 1-based 'height' dimension even though values are 1, 2, … 90).
    if coord is not None and _is_z(coord.name, ds.coords) and not _is_level_index(coord):
        return ds

    # Case 2: pressure already a dim/coord (by name or CF attributes)
    for pname in ('pres', 'pres_zm', 'plev', 'pressure'):
        if pname in ds.coords or pname in ds.dims:
            return ds
    # Also check by CF attributes (e.g. 'level' with standard_name='air_pressure')
    if coord is not None and _is_pressure_coord(coord.name, ds):
        return ds

    # Case 3: 4-D pressure variable available → interpolate
    pres_key = next((p for p in ('pres', 'pressure') if p in ds.data_vars), None)
    if pres_key is not None:
        # Determine model-level dim: the dim of pres that isn't time/spatial
        spatial_like = {'time', 'lat', 'lon', 'latitude', 'longitude',
                        'ncells', 'cells', 'cell', 'ncell'}
        lev_name = (
            coord.name if coord is not None
            else next(
                (d for d in ds[pres_key].dims if d.lower() not in spatial_like),
                None,
            )
        )
        if lev_name is not None:
            return _interp_to_pressure_levels(ds, lev_name, pres_key=pres_key)
        logger.warning("Could not identify model-level dimension for pressure interpolation.")
        return ds

    # Case 4: nothing useful
    logger.warning(
        "No height-in-metres coordinate and no pressure variable found. "
        "EP flux will be computed on raw model-level coordinates."
    )
    return ds


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_alt_name(ds: xr.Dataset | xr.DataArray) -> str:
    """Return the name of the vertical coordinate using CF-convention inference.

    Strategy
    --------
    1. Ask :func:`~healicon.cf_coords._find_coordinate` for a candidate.
    2. Validate it with :func:`~healicon.cf_coords._is_z`: the coordinate must
       carry meter-like units (CF height/altitude).  Plain integer model-level
       indices (e.g. ICON ``height`` = 1 … 90) fail this test.
    3. If the candidate is not real height in metres, search for a pressure
       coordinate (``pres``, ``pres_zm``, ``plev``, ``pressure``).
    4. If nothing better is found, fall back to the original candidate.
    """
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name='_tmp')

    coord = _find_coordinate(ds, 'level', raise_notfound=False)

    # Happy path: found a real height coordinate (metres).
    if coord is not None and _is_z(coord.name, ds.coords):
        return str(coord.name)

    # Candidate is a model-level index (or absent) → prefer pressure.
    # First: check if the candidate itself is a pressure coord (CF attributes).
    if coord is not None and _is_pressure_coord(coord.name, ds):
        logger.debug(
            f"Vertical coord '{coord.name}' identified as pressure via CF attributes."
        )
        return str(coord.name)

    # Second: search by well-known pressure coordinate names.
    for pres_name in ('pres', 'pres_zm', 'plev', 'pressure'):
        if pres_name in ds.coords or pres_name in ds.dims:
            logger.debug(
                f"Vertical coord '{getattr(coord, 'name', None)}' is not height "
                f"in metres; using pressure coordinate '{pres_name}'."
            )
            return pres_name

    # Third: scan ALL coordinates for pressure-like CF attributes.
    for cname in ds.coords:
        if cname in ds.dims and _is_pressure_coord(cname, ds):
            logger.debug(f"Found pressure coordinate '{cname}' via CF scan.")
            return cname

    # Last resort: use whatever _find_coordinate found (or raise).
    if coord is not None:
        logger.warning(
            f"Vertical coord '{coord.name}' is not height in metres and no "
            "pressure coordinate found; using it as-is."
        )
        return str(coord.name)

    raise ValueError(
        "No vertical coordinate found. "
        f"Available dims: {list(ds.dims)}, coords: {list(ds.coords)}"
    )


def _coriolis(lat_rad: xr.DataArray) -> xr.DataArray:
    """Coriolis parameter f = 2Ω sinφ."""
    return 2.0 * _OMEGA * np.sin(lat_rad)


def _is_level_index(coord: xr.DataArray) -> bool:
    """Return True if coord looks like integer model-level indices (not metres)."""
    vals = np.asarray(coord.values, dtype=float)
    if vals.max() > 1000:
        return False
    return bool(np.all(np.abs(vals - np.round(vals)) < 0.01) and vals.max() <= 500)


def _resolve_scale_height(ds_zm: xr.Dataset) -> xr.DataArray:
    """Return locally-varying scale height H(z, φ) in metres.

    Priority:
    1. Compute H = Rd T̄ / g from the zonal-mean temperature.
    2. Fit H as a constant via linear regression of log(p̄) vs z.
    3. Return 7 km as a hard-coded fallback.

    Accepts variable names 'temp' or 'temp_zm', 'pres' or 'pres_zm'.
    """
    alt_name = _find_alt_name(ds_zm)
    temp_key = 'temp' if 'temp' in ds_zm else ('temp_zm' if 'temp_zm' in ds_zm else None)
    pres_key = 'pres' if 'pres' in ds_zm else ('pres_zm' if 'pres_zm' in ds_zm else None)

    if temp_key is not None:
        T_bar = ds_zm[temp_key]
        H = (_RD * T_bar) / _G
        H.attrs = {'long_name': 'Scale height', 'units': 'm'}
        logger.debug("Scale height H computed from T̄ = Rd T̄/g.")
        return H

    if pres_key is not None:
        p_bar = ds_zm[pres_key].mean('lat') if 'lat' in ds_zm[pres_key].dims else ds_zm[pres_key]
        z_vals = ds_zm[alt_name].values.astype(float)
        log_p = np.log(np.maximum(p_bar.values.astype(float), 1e-30))
        log_p_1d = log_p.reshape(-1, len(z_vals)).mean(axis=0)
        coeffs = np.polyfit(z_vals, log_p_1d, 1)
        H_const = float(-1.0 / coeffs[0]) if coeffs[0] != 0 else 7000.0
        H_const = max(H_const, 1000.0)
        logger.debug(f"Scale height H fitted from log(p̄) vs z: H = {H_const / 1e3:.2f} km")
        return xr.DataArray(H_const, attrs={'long_name': 'Scale height', 'units': 'm'})

    logger.warning("Neither temp nor pres available; using H = 7 km.")
    return xr.DataArray(7000.0, attrs={'long_name': 'Scale height', 'units': 'm'})


def _resolve_density(ds_zm: xr.Dataset, H: xr.DataArray) -> xr.DataArray:
    """Return ρ₀(z, φ) in kg m⁻³.

    Priority:
    1. pres + temp present   → ρ₀ = p̄ / (Rd T̄)
    2. temp only, no pres    → hydrostatic integration to get p̄, then ρ₀ = p̄/(Rd T̄)
    3. neither               → exponential ρ_surf exp(−z/H)

    Accepts variable names 'temp'/'temp_zm', 'pres'/'pres_zm'.
    """
    alt_name = _find_alt_name(ds_zm)
    alt = ds_zm[alt_name]
    temp_key = 'temp' if 'temp' in ds_zm else ('temp_zm' if 'temp_zm' in ds_zm else None)
    pres_key = 'pres' if 'pres' in ds_zm else ('pres_zm' if 'pres_zm' in ds_zm else None)

    # Derive pressure from theta + temp if both exist but pres does not:
    # p = P0 * (T / theta)^(1/kappa)
    if pres_key is None and temp_key is not None:
        theta_key = 'theta' if 'theta' in ds_zm else ('theta_zm' if 'theta_zm' in ds_zm else None)
        if theta_key is not None:
            logger.info("Density: deriving p̄ from θ̄ and T̄ via p = P₀·(T/θ)^(1/κ).")
            ratio = ds_zm[temp_key] / ds_zm[theta_key]
            p_derived = _P0 * ratio ** (1.0 / _KAPPA)
            rho = p_derived / (_RD * ds_zm[temp_key])
            rho.attrs = {'long_name': 'Reference density (from θ and T)', 'units': 'kg m-3'}
            return rho

    if pres_key is not None and temp_key is not None:
        logger.info("Density: using ρ₀ = p̄/(Rd T̄) (exact ideal gas).")
        rho = ds_zm[pres_key] / (_RD * ds_zm[temp_key])
        rho.attrs = {'long_name': 'Reference density', 'units': 'kg m-3'}
        return rho

    if temp_key is not None:
        logger.info("Density: reconstructing p̄ from hydrostatic integration.")
        T_bar = ds_zm[temp_key].sortby(alt_name)
        alt_ax = T_bar.dims.index(alt_name)
        z = T_bar[alt_name].values.astype(float)
        T_np = np.moveaxis(T_bar.values.astype(float), alt_ax, -1)  # (..., n_alt)
        integrand = _G / (_RD * T_np)
        dz = np.diff(z)
        n_alt = len(z)
        cum_int = np.zeros_like(T_np)
        for i in range(n_alt - 2, -1, -1):
            trap = 0.5 * (integrand[..., i] + integrand[..., i + 1]) * dz[i]
            cum_int[..., i] = cum_int[..., i + 1] + trap
        p_toa = 1.0
        p_bar_np = np.moveaxis(p_toa * np.exp(cum_int), -1, alt_ax)  # restore axis
        p_bar = xr.DataArray(p_bar_np, coords=T_bar.coords, dims=T_bar.dims)
        rho = p_bar / (_RD * T_bar)
        rho.attrs = {'long_name': 'Reference density (hydrostatic)', 'units': 'kg m-3'}
        return rho

    logger.warning(
        "Density: neither pres nor temp — using exponential profile ρ₀=ρ_surf·exp(−z/H).")
    rho_surf = 1.225
    H_vals = H.values if hasattr(H, 'values') else float(H)
    z_vals = alt.values.astype(float)
    rho_np = rho_surf * np.exp(-z_vals / H_vals)
    rho = xr.DataArray(rho_np, coords={alt_name: alt}, dims=[alt_name],
                       attrs={'long_name': 'Reference density (exponential)', 'units': 'kg m-3'})
    return rho


def _align_w_to_full_levels(ds: xr.Dataset) -> xr.Dataset:
    """Interpolate staggered vertical wind w to the full model levels.

    ICON stores w on half-levels (e.g. ``height_2`` with n+1 levels) while u, v,
    and temp are on full levels (``height`` with n levels).  Detects that stagger
    pattern and interpolates w to the full-level coordinate so all fields share
    a common vertical dimension for eddy-flux computation.

    Strategy (in priority order):
    1. ``xr.interp`` against the full-level coordinate values — exact and correct
       for non-uniform level spacing.  Used whenever both the half-level and
       full-level coordinates carry numeric values.
    2. Arithmetic mean of adjacent half-levels — only when coordinate values are
       absent.  This is equivalent to assuming the full level sits exactly at the
       midpoint of two consecutive interface heights (uniform-spacing assumption).
       A warning is emitted so the caller knows a less accurate path was taken.
    """
    if 'w' not in ds:
        return ds

    try:
        alt_name = _find_alt_name(ds)
    except ValueError:
        return ds

    w_dims = ds['w'].dims

    # Already on full levels — nothing to do.
    if alt_name in w_dims:
        return ds

    full_size = ds.sizes.get(alt_name)

    # Detect the half-level dimension.
    half_dim = None
    for dim in w_dims:
        if full_size and ds.sizes.get(dim, 0) == full_size + 1:
            half_dim = dim
            break
        if dim.endswith('_2') or 'half' in dim or 'bnds' in dim:
            half_dim = dim
            break

    if half_dim is None:
        return ds

    # ── Strategy 1: xr.interp (correct for non-uniform level spacing) ────────
    if half_dim in ds.coords and alt_name in ds.coords:
        logger.info(
            f"Destaggering w: interpolating from '{half_dim}' to full levels '{alt_name}'."
        )
        w_interp = ds['w'].interp(
            {half_dim: ds[alt_name]},
            method='linear',
            kwargs={'fill_value': 'extrapolate'},
        )
        if half_dim in w_interp.dims and half_dim != alt_name:
            w_interp = w_interp.rename({half_dim: alt_name})
        w_interp.attrs = ds['w'].attrs
        ds = ds.assign({'w': w_interp})
        if half_dim in ds.dims:
            ds = ds.drop_dims(half_dim)
        return ds

    # ── Strategy 2: arithmetic mean (fallback — assumes uniform spacing) ──────
    half_n = ds.sizes.get(half_dim, 0)
    if half_n == full_size + 1:
        logger.warning(
            f"Destaggering w from '{half_dim}' to '{alt_name}': no coordinate values "
            f"available — falling back to arithmetic mean (assumes uniform level spacing)."
        )
        w_data = ds['w'].data
        axis = list(w_dims).index(half_dim)
        sl = [slice(None)] * ds['w'].ndim
        su = [slice(None)] * ds['w'].ndim
        sl[axis] = slice(1, None)
        su[axis] = slice(0, -1)
        w_interp_data = 0.5 * (w_data[tuple(sl)] + w_data[tuple(su)])

        new_dims = [d if d != half_dim else alt_name for d in w_dims]
        new_coords = {k: v for k, v in ds['w'].coords.items() if k != half_dim}
        if alt_name in ds.coords:
            new_coords[alt_name] = ds[alt_name]

        w_interp = xr.DataArray(
            w_interp_data, dims=new_dims, coords=new_coords, attrs=ds['w'].attrs
        )
        ds = ds.assign({'w': w_interp})
        if half_dim in ds.dims:
            ds = ds.drop_dims(half_dim)
        return ds

    return ds


# ── Stage 1: Eddy covariances ─────────────────────────────────────────────────

def compute_eddy_fluxes(
        ds: xr.Dataset,
        preprocess: bool = False,
        lmax: int | None = None,
) -> xr.Dataset:
    """Compute zonal-mean eddy covariances from a HEALPix dataset.

    Eddy quantities are defined as the departure from the zonal mean:
    q' = q − [q], where [·] denotes the zonal average over HEALPix rings.

    Covariances always computed: [u'v'], [v'θ'].
    Computed when *w* is present: [u'w'], [w'θ'].

    Parameters
    ----------
    ds : xr.Dataset
        HEALPix dataset with canonical variable names (u, v required;
        theta required; w, temp, pres optional).
    preprocess : bool, default False
        Run :func:`_preprocess_vertical` before computing covariances.
        Keep False when calling from :func:`eliassen_palm` (it preprocesses
        once and passes the result here).
    lmax : int or None
        Band-limit all fields to spherical-harmonic degree *lmax* before
        computing covariances.  Useful for scale-selective diagnostics and
        for suppressing sub-grid noise in the θ-gradients used by Ψ.

    Returns
    -------
    xr.Dataset
        Zonal-mean fields: u_zm, v_zm, theta_zm, upvp_zm [m² s⁻²],
        vptp_zm [K m s⁻¹], and, if w is present, upwp_zm [m² s⁻²],
        wptp_zm [K m s⁻¹].  Also passes through temp_zm and pres_zm.
    """
    # _parse_dataset guarantees canonical names (u, v, theta, temp, pres, w).
    # Vertical-coordinate preprocessing is owned by the driver (eliassen_palm).
    # Only run it here if explicitly asked (standalone use).
    if preprocess:
        ds = _parse_dataset(ds)
        ds = _align_w_to_full_levels(ds)
        ds = _preprocess_vertical(ds)

    if lmax is not None:
        from .filtering import filter_spatial
        logger.info(f"Pre-filtering input fields to lmax={lmax} before eddy flux computation.")
        ds = filter_spatial(ds, lmax=lmax)

    for v in ('u', 'v'):
        if v not in ds:
            raise ValueError(f"Dataset missing required variable '{v}' "
                             f"(after parsing). Available: {list(ds.data_vars)}")
    if 'theta' not in ds:
        raise ValueError("Dataset missing 'theta' after parsing. "
                         "Run _parse_dataset or eliassen_palm as the entry point.")

    # ── Zonal means ──────────────────────────────────────────────────────────
    logger.info("Computing zonal means for EP flux eddy covariances.")
    ds_zm = _zonal_mean(ds)

    has_temp = 'temp' in ds
    has_w = 'w' in ds
    has_pres = 'pres' in ds

    # θ is guaranteed present by _parse_dataset
    theta = ds['theta']
    theta_zm = _zonal_mean(xr.Dataset({'theta': theta}))['theta']
    theta_zm.attrs = {'long_name': 'Zonal-mean potential temperature', 'units': 'K'}

    # ── Broadcast zonal means to HEALPix pixels via healpy ring look-up ──────
    import healpy as hp
    cell_dim = get_cells_dim(ds)
    npix = ds.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    from ..grid import get_healpix_order
    is_nested = get_healpix_order(ds) == 'nested'
    theta_ring, _phi_ring = hp.pix2ang(nside, np.arange(npix), nest=is_nested)
    lats_pixels = 90.0 - np.rad2deg(theta_ring)
    lats_zm = ds_zm['lat'].values

    ring_idx = np.abs(lats_zm[:, None] - lats_pixels[None, :]).argmin(axis=0)

    def _broadcast_to_pixels(da_zm):
        """(…, lat) zonal-mean DataArray -> (…, cells) by ring/pixel mapping."""
        vals = da_zm.values
        lat_ax = da_zm.dims.index('lat')
        vals = np.moveaxis(vals, lat_ax, -1)
        pxl = vals[..., ring_idx]
        new_dims = [d if d != 'lat' else cell_dim for d in da_zm.dims]
        new_coords = {k: v for k, v in da_zm.coords.items() if k != 'lat'}
        new_coords[cell_dim] = np.arange(npix)
        return xr.DataArray(pxl, dims=new_dims, coords=new_coords)

    u_zm_px = _broadcast_to_pixels(ds_zm['u'])
    v_zm_px = _broadcast_to_pixels(ds_zm['v'])

    u_prime = ds['u'] - u_zm_px
    v_prime = ds['v'] - v_zm_px

    logger.info("Computing eddy covariances [u'v'] and [v'theta'].")
    upvp_zm = _zonal_mean(xr.Dataset({'upvp': u_prime * v_prime}))['upvp']
    upvp_zm.attrs = {'long_name': "Zonal-mean eddy momentum flux [u'v']", 'units': 'm2 s-2'}

    out = xr.Dataset()
    out['u_zm'] = ds_zm['u']
    out['v_zm'] = ds_zm['v']
    out['upvp_zm'] = upvp_zm

    # θ is always available
    out['theta_zm'] = theta_zm
    theta_zm_px = _broadcast_to_pixels(theta_zm)
    theta_prime = theta - theta_zm_px
    vptp_zm = _zonal_mean(xr.Dataset({'vptp': v_prime * theta_prime}))['vptp']
    vptp_zm.attrs = {'long_name': "Zonal-mean eddy heat flux [v'theta']",
                     'units': 'K m s-1'}
    out['vptp_zm'] = vptp_zm

    if has_temp:
        out['temp_zm'] = ds_zm['temp']

    if has_pres:
        out['pres_zm'] = ds_zm['pres']

    if has_w:
        w_zm_px = _broadcast_to_pixels(ds_zm['w'])
        w_prime = ds['w'] - w_zm_px
        out['w_zm'] = ds_zm['w'].assign_attrs(
            {'long_name': 'Zonal-mean vertical velocity', 'units': 'm s-1'})
        upwp_zm = _zonal_mean(xr.Dataset({'upwp': u_prime * w_prime}))['upwp']
        upwp_zm.attrs = {'long_name': "Zonal-mean eddy vertical flux [u'w']", 'units': 'm2 s-2'}
        out['upwp_zm'] = upwp_zm
        wptp_zm = _zonal_mean(xr.Dataset({'wptp': w_prime * theta_prime}))['wptp']
        wptp_zm.attrs = {'long_name': "Zonal-mean eddy vertical heat flux [w'theta']",
                         'units': 'K m s-1'}
        out['wptp_zm'] = wptp_zm

    out.attrs = ds.attrs
    return out


# ── Stage 2-3: EP flux components ────────────────────────────────────────────

def _is_pressure_coord(alt_name, ds):
    """True if the vertical coordinate `alt_name` is a pressure coordinate."""
    if alt_name in ('plev', 'pres', 'pres_zm', 'pressure'):
        return True
    coord = ds[alt_name] if alt_name in ds.coords else None
    if coord is not None:
        units = str(coord.attrs.get('units', '')).lower()
        if units in ('pa', 'hpa', 'mb', 'mbar', 'millibar', 'hectopascal'):
            return True
        if str(coord.attrs.get('standard_name', '')).lower() == 'air_pressure':
            return True
    return False


def _diff_safe(da, dim):
    """2nd-order centred finite difference along *dim*, NaN-masked at boundaries.

    Chunks along *dim* are consolidated before differentiating so the stencil
    is not split across chunks.  Points whose immediate neighbour is NaN are
    also masked, suppressing spurious large values at data boundaries.
    """
    nan_mask = da.isnull()
    if da.chunks is not None:
        da = da.chunk({dim: -1})
    result = da.differentiate(dim, edge_order=2)
    # Re-apply NaN mask: any point that was NaN in the input, or whose
    # immediate neighbour along `dim` was NaN, should remain NaN.
    # shift() fills with NaN (float), so convert back to bool with fillna.
    shifted_fwd = nan_mask.shift({dim: 1}).fillna(False).astype(bool)
    shifted_bwd = nan_mask.shift({dim: -1}).fillna(False).astype(bool)
    expanded_mask = nan_mask | shifted_fwd | shifted_bwd
    return result.where(~expanded_mask)


def _stability(theta_bar, alt_name):
    """dθ̄/d(coord) [K/m or K/Pa] with a sign-preserving floor.

    The floor (±ε = ±1e-10) avoids dividing by zero in near-neutral layers
    (mesopause minimum, isothermal stratopause) while preserving the sign so
    the EP flux components do not flip spuriously.
    """
    tz = _diff_safe(theta_bar, alt_name)
    eps = 1e-10
    tz_floored = xr.where(np.abs(tz) > eps, tz, eps * xr.where(tz >= 0, 1.0, -1.0))
    return tz_floored.where(tz.notnull())


def _theta_stream_function(
        rho0:      xr.DataArray,
        vptp:      xr.DataArray,
        wptp:      xr.DataArray,
        theta_bar: xr.DataArray,
        alt_name:  str,
        cos_phi:   xr.DataArray,
        eps:       float = 1e-30,
) -> xr.DataArray:
    """TEM mass stream function Ψ [kg s⁻¹] from the full θ-gradient projection.

    Ψ = a cosφ · (ρ₀[v'θ']·θ_z  −  ρ₀[w'θ']·(θ_φ/a)) / ((θ_φ/a)² + θ_z²)

    Both gradient components are in K m⁻¹.  In the limit θ_φ → 0 (horizontal
    θ-surfaces):
        Ψ → a cosφ ρ₀ [v'θ'] / θ_z
    which is the Iwasaki (1989) / AHL87 eq. 3.5.3 limit.

    The denominator is regularised to *eps* to guard against near-zero
    θ-gradients (isothermal layers, mesopause minimum, polar vortex core).
    """
    theta_z = _stability(theta_bar, alt_name)                          # dθ̄/dz [K/m]
    # _diff_safe differentiates w.r.t. lat in degrees; ×(180/π) converts to
    # radians, then /a gives the meridional K/m gradient component.
    theta_phi_over_a = _diff_safe(theta_bar, 'lat') * (180.0 / np.pi) / _A  # [K/m]

    grad_sq = theta_phi_over_a ** 2 + theta_z ** 2
    grad_sq = xr.where(grad_sq < eps, eps, grad_sq)

    # Log the max tilt ratio so users can gauge when the θ_φ correction matters.
    # Values > 0.1 (i.e. θ_φ/a > ~30 % of θ_z) indicate strongly tilted θ-surfaces.
    try:
        tilt_ratio = float(
            (np.abs(theta_phi_over_a) / (np.abs(theta_z) + 1e-30)).max()
        )
        logger.debug(
            f"θ-surface tilt: max |θ_φ/a| / |θ_z| = {tilt_ratio:.3g}  "
            f"({'significant' if tilt_ratio > 0.1 else 'small'} Ψ correction)"
        )
    except Exception:
        pass

    numer = rho0 * vptp * theta_z - rho0 * wptp * theta_phi_over_a   # [kg K²/(m³ s)]
    Psi_unscaled = numer / grad_sq                                     # [kg/(m·s)]
    return _A * cos_phi * Psi_unscaled                                 # [kg/s]


def compute_ep_flux(eddy_ds, mode="auto"):
    """Compute F_phi and F_z from zonal-mean eddy covariances.

    Branches on the vertical coordinate detected in *eddy_ds*:

    Height coords (mode='full'):
        F^φ = −ρ₀ a cosφ [u'v'] + Ψ · dū/dz       [kg s⁻²]
        F^z = −ρ₀ a cosφ [u'w'] + Ψ · f̂            [kg s⁻²]
        where Ψ is the TEM stream function from :func:`_theta_stream_function`.

    Pressure coords (mode='full', Edmon et al. 1980 eq. 3.1):
        F^φ = a cosφ (ū_p [v'θ']/θ̄_p − [u'v'])    [m³ s⁻²]
        F^z = a cosφ (f̂   [v'θ']/θ̄_p − [u'ω'])    [Pa m² s⁻²]

    QG limit (mode='qg'): Ψ = 0, f̂ → f.

    mode='auto' uses 'full' when [u'w'] is available, 'qg' otherwise.
    """
    has_upwp = 'upwp_zm' in eddy_ds
    if mode == 'full' and not has_upwp:
        raise ValueError("mode='full' requires 'upwp_zm' (needs w in the input).")
    use_full = has_upwp if mode == 'auto' else (mode == 'full')
    mode_used = 'full' if use_full else 'qg'

    alt_name = _find_alt_name(eddy_ds)
    is_pres = _is_pressure_coord(alt_name, eddy_ds)
    coord_kind = 'pressure' if is_pres else 'height'
    logger.info(f"Computing EP flux ({mode_used}, {coord_kind} coordinate).")

    rho0 = _resolve_density(eddy_ds, _resolve_scale_height(eddy_ds))

    lat_rad = np.deg2rad(eddy_ds['lat'])
    cos_phi = np.cos(lat_rad)
    f = _coriolis(lat_rad)

    upvp = eddy_ds['upvp_zm']
    has_theta = 'theta_zm' in eddy_ds

    if has_theta:
        theta_s = _stability(eddy_ds['theta_zm'], alt_name)  # d(theta)/dz or /dp
        vptp = eddy_ds.get('vptp_zm', xr.zeros_like(upvp))
    else:
        logger.warning("theta_zm not available; setting [v'theta'] = 0.")
        theta_s = xr.DataArray(1.0)
        vptp = xr.zeros_like(upvp)

    if use_full:
        u_bar = eddy_ds['u_zm']
        u_shear = _diff_safe(u_bar, alt_name)  # du_bar/dz or /dp

        # absolute vorticity factor f_hat = f + zeta_bar (coordinate-independent)
        ubar_cos = u_bar * cos_phi
        dubar_cos_dphi = _diff_safe(ubar_cos, 'lat') * (180.0 / np.pi)
        f_hat = f - dubar_cos_dphi / (_A * cos_phi)

        if is_pres:
            # isobaric: no rho0 prefactor; vertical eddy flux is [u'omega]
            upwp = eddy_ds['upwp_zm']
            upomega = -rho0 * _G * upwp  # w -> omega (hydrostatic)
            F_phi = _A * cos_phi * (u_shear * vptp / theta_s - upvp)
            F_vert = _A * cos_phi * (f_hat * vptp / theta_s - upomega)
        else:
            # Height coords: full primitive-equation TEM via Ψ stream function.
            # Ψ uses the full θ-gradient (both θ_z and θ_φ/a); when [w'θ'] is
            # absent (no w in input) wptp is zero and Ψ → a cosφ ρ₀[v'θ']/θ_z,
            # recovering the AHL87 formula exactly.
            upwp = eddy_ds['upwp_zm']
            wptp = eddy_ds.get('wptp_zm', xr.zeros_like(vptp))
            Psi = _theta_stream_function(
                rho0, vptp, wptp, eddy_ds['theta_zm'], alt_name, cos_phi
            )
            F_phi  = -rho0 * _A * cos_phi * upvp  +  Psi * u_shear
            F_vert = -rho0 * _A * cos_phi * upwp  +  Psi * f_hat
    else:
        if is_pres:
            F_phi = -_A * cos_phi * upvp
            F_vert = _A * cos_phi * (f * vptp / theta_s)
        else:
            F_phi = -rho0 * _A * cos_phi * upvp
            F_vert = rho0 * _A * cos_phi * (f * vptp / theta_s)

    # Height:   F_phi = rho0 a cosphi [m2/s2]  → kg/s2  (≡ Pa m = N m-1)
    # Pressure: F_phi = a cosphi [m2/s2]        → m3/s2
    #           F_vert = a cosphi [Pa m/s2]      → Pa m2/s2  (omega carries Pa/s)
    F_phi.attrs = {'long_name': 'EP flux, meridional component',
                   'units': 'kg s-2' if not is_pres else 'm3 s-2'}
    F_vert.attrs = {'long_name': f'EP flux, vertical component ({coord_kind})',
                    'units': 'kg s-2' if not is_pres else 'Pa m2 s-2'}

    out = xr.Dataset({'F_phi': F_phi, 'F_z': F_vert, 'rho0': rho0})

    # Ψ and residual velocities — height coords only (Ψ is undefined for isobaric)
    if not is_pres and use_full:
        out['Psi'] = Psi.assign_attrs(
            {'long_name': 'TEM mass stream function', 'units': 'kg s-1'})

        if 'v_zm' in eddy_ds and 'w_zm' in eddy_ds:
            dPsi_dz   = _diff_safe(Psi, alt_name)
            # cosφ floor avoids 1/cosφ blow-up at the poles
            cos_phi_safe = xr.where(np.abs(cos_phi) > 1e-6, cos_phi, 1e-6)
            dPsiC_dphi = _diff_safe(Psi * cos_phi, 'lat') * (180.0 / np.pi)
            v_star = eddy_ds['v_zm'] - dPsi_dz   / (rho0 * _A * cos_phi_safe)
            w_star = eddy_ds['w_zm'] + dPsiC_dphi / (rho0 * _A ** 2 * cos_phi_safe)
            out['v_star'] = v_star.assign_attrs(
                {'long_name': 'TEM residual meridional velocity', 'units': 'm s-1'})
            out['w_star'] = w_star.assign_attrs(
                {'long_name': 'TEM residual vertical velocity', 'units': 'm s-1'})

    for passthrough in ('u_zm', 'v_zm', 'w_zm', 'pres_zm', 'temp_zm', 'wptp_zm'):
        if passthrough in eddy_ds:
            out[passthrough] = eddy_ds[passthrough]
    out.attrs['ep_flux_mode'] = mode_used
    out.attrs['ep_flux_coord'] = coord_kind  # consumed by the divergence
    return out


def compute_ep_divergence(ep_ds, valid_pressure_min_pa=1e-1,
                          valid_height_max_m=1.2e5, mask_invalid=False):
    """Compute EP flux divergence and the residual zonal acceleration a_EP.

    div F = (a cosφ)⁻¹ ∂(F_φ cosφ)/∂φ + ∂F_z/∂z    (height, AHL87 eq. 3.5.6)
    div F = (a cosφ)⁻¹ ∂(F_φ cosφ)/∂φ + ∂F_z/∂p    (pressure, Edmon+80 eq. 3.3)

    a_EP = div F / (ρ₀ a cosφ)   [m s⁻¹ day⁻¹]  (height)
    a_EP = div F / (a cosφ)      [m s⁻¹ day⁻¹]  (pressure)

    Parameters
    ----------
    valid_pressure_min_pa, valid_height_max_m : float
        Documented validity ceiling recorded in ``a_EP.attrs``. Above ~100-120 km
        molecular diffusion and ion drag dominate and the EP-flux framework loses
        meaning regardless of coordinate choice.
    mask_invalid : bool
        If True, NaN a_EP beyond the validity ceiling. Default False leaves the
        values in place and lets the plotting layer decide what to mask.
    """
    alt_name = _find_alt_name(ep_ds)
    is_pres = (ep_ds.attrs.get('ep_flux_coord', None) == 'pressure') \
              or _is_pressure_coord(alt_name, ep_ds)

    lat_rad = np.deg2rad(ep_ds['lat'])
    cos_phi = np.cos(lat_rad)
    F_phi, F_z, rho0 = ep_ds['F_phi'], ep_ds['F_z'], ep_ds['rho0']

    # Compute divergence of EP flux
    dFphi_cos_dphi = _diff_safe(F_phi * cos_phi, 'lat') * (180.0 / np.pi)
    dFz_dvert = _diff_safe(F_z, alt_name)  # d/dp or d/dz
    div_F = (1.0 / (_A * cos_phi)) * dFphi_cos_dphi + dFz_dvert

    # acceleration: divide by (rho0 a cosphi) in height, (a cosphi) in pressure
    denom = (_A * cos_phi) if is_pres else (rho0 * _A * cos_phi)
    a_EP = div_F / denom * _SECS_PER_DAY

    # Suppress boundary-layer/topographic edge artifacts where QG theory breaks down.
    # We expand the subterranean NaN mask by 1 grid point to shave off the jagged rim.
    nan_mask = a_EP.isnull()
    edge_mask = (
        nan_mask.shift({alt_name: 1}).fillna(False).astype(bool) |
        nan_mask.shift({alt_name: -1}).fillna(False).astype(bool)
    )
    if 'lat' in nan_mask.dims:
        edge_mask = edge_mask | (
            nan_mask.shift({'lat': 1}).fillna(False).astype(bool) |
            nan_mask.shift({'lat': -1}).fillna(False).astype(bool)
        )
    a_EP = a_EP.where(~edge_mask)

    # div_F = (1/(a cosphi)) d(F_phi cosphi)/dphi + dF_vert/dvert
    # Height: [kg/s2 / m] = kg/(m s2) = Pa/m → / (rho0 a cosphi [kg/m2]) = m/s2
    # Pressure: [m3/s2 / m + Pa m2/s2 / Pa] = m2/s2  → / (a cosphi [m]) = m/s2
    div_F.attrs = {'long_name': 'EP flux divergence',
                   'units': 'Pa' if not is_pres else 'm2 s-2',
                   'ep_flux_coord': 'pressure' if is_pres else 'height'}
    a_EP.attrs = {
        'long_name': 'EP-flux wave forcing on zonal-mean wind',
        'units': 'm s-1 day-1',
        'valid_pressure_min_pa': valid_pressure_min_pa,
        'valid_height_max_m': valid_height_max_m,
        'validity_note': ('EP-flux framework degrades above ~100-120 km '
                          '(molecular diffusion, ion drag); interpret with care.'),
    }

    if mask_invalid and alt_name in a_EP.coords:
        coord = a_EP[alt_name]
        if is_pres:
            a_EP = a_EP.where(coord >= valid_pressure_min_pa)
        else:
            a_EP = a_EP.where(coord <= valid_height_max_m)

    out = ep_ds.copy()
    out['div_F'] = div_F
    out['a_EP'] = a_EP
    out.attrs = append_history(
        ep_ds.attrs, f"Computed EP flux divergence ({'pressure' if is_pres else 'height'}).")
    return out


# ── Output helpers ────────────────────────────────────────────────────────────

def _reorder_output_dims(ds: xr.Dataset) -> xr.Dataset:
    """Transpose all output variables to a canonical dimension order.

    Preferred order: ``time → level/height/pres → lat → <remaining>``.
    Dimensions absent from a particular variable are simply skipped.
    """
    try:
        alt_name = _find_alt_name(ds)
    except ValueError:
        alt_name = None

    # Build the preferred leading-dim sequence.
    priority: list[str] = []
    if 'time' in ds.dims:
        priority.append('time')
    if alt_name and alt_name in ds.dims:
        priority.append(alt_name)
    if 'lat' in ds.dims:
        priority.append('lat')

    if not priority:
        return ds

    new_vars: dict[str, xr.DataArray] = {}
    for vname, da in ds.data_vars.items():
        leading = [d for d in priority if d in da.dims]
        trailing = [d for d in da.dims if d not in priority]
        new_order = leading + trailing
        if list(da.dims) != new_order:
            da = da.transpose(*new_order)
        new_vars[vname] = da

    return ds.assign(new_vars)


# ── Top-level convenience wrapper ─────────────────────────────────────────────

def eliassen_palm(
        ds: xr.Dataset,
        mode: Literal['auto', 'full', 'qg'] = 'auto',
        time_mean: bool = False,
        vertical: Literal['auto', 'native'] = 'auto',
        lmax: int | None = None,
) -> xr.Dataset:
    """Full EP-flux pipeline: HEALPix dataset → F, div F, a_EP.

    Pipeline stages:
    1. :func:`_parse_dataset`       — canonicalise variable names, derive θ/p.
    2. :func:`_align_w_to_full_levels` — destagger w from half-levels if needed.
    3. :func:`_preprocess_vertical` — optionally interpolate to isobaric levels.
    4. :func:`compute_eddy_fluxes`  — zonal-mean eddy covariances.
    5. :func:`compute_ep_flux`      — F_phi, F_z (branches on coord type).
    6. :func:`compute_ep_divergence`— div F and a_EP [m s⁻¹ day⁻¹].

    Parameters
    ----------
    mode : {'auto', 'full', 'qg'}
        'auto': full TEM when [u'w'] is available, QG otherwise.
        'full': full primitive-equation TEM (requires w in input).
        'qg':  quasi-geostrophic limit (no Ψ stream function correction).
    time_mean : bool
        Average output over the time dimension after computing EP flux.
    vertical : {'auto', 'native'}
        'auto'   : run :func:`_preprocess_vertical`, which may interpolate to
                   standard isobaric levels when a 4-D pressure field is present
                   but no real height coordinate.
        'native' : skip preprocessing; compute on the coordinate already in the
                   dataset (use for HL output with height-in-metres already set).
    lmax : int or None
        Band-limit inputs to this spherical-harmonic degree before computing
        eddy covariances; passed to :func:`compute_eddy_fluxes`.

    Returns
    -------
    xr.Dataset
        Contains: F_phi, F_z, rho0, div_F, a_EP, u_zm.
        Height-coord full-TEM also adds: Psi, v_star, w_star.
    """
    # ---- Validate & normalise variable names FIRST -------------------------
    ds = _parse_dataset(ds)

    # ---- Align staggered w -------------------------------------------------
    ds = _align_w_to_full_levels(ds)

    # ---- ONE coordinate decision, here, visible in the log -----------------
    if vertical == 'native':
        logger.info("vertical='native': skipping _preprocess_vertical (no interpolation).")
    else:
        ds = _preprocess_vertical(ds)

    alt_name = _find_alt_name(ds)
    coord_kind = 'pressure' if _is_pressure_coord(alt_name, ds) else 'height'
    logger.info(f"EP flux vertical coordinate: '{alt_name}' ({coord_kind}).")

    # ---- stages inherit the coordinate; none of them re-decide -------------
    eddy_ds = compute_eddy_fluxes(ds, lmax=lmax)  # canonical names guaranteed by _parse_dataset
    ep_ds = compute_ep_flux(eddy_ds, mode=mode)  # branches on coord_kind
    out = compute_ep_divergence(ep_ds)  # branches on coord_kind

    if time_mean and 'time' in out.dims:
        logger.info("Averaging EP flux output over time.")
        out = out.mean(dim='time', keep_attrs=True)

    out = _reorder_output_dims(out)
    out.attrs = append_history(
        ds.attrs,
        f"Computed Eliassen-Palm flux (mode={ep_ds.attrs.get('ep_flux_mode', mode)}, "
        f"coord={coord_kind}) using HealICON.",
    )
    return out
