"""
Eliassen-Palm Flux Analysis
===========================

Implements the EP flux vector **F** and its divergence ∇·**F** following the
Transformed Eulerian Mean (TEM) framework of Andrews, Holton & Leovy (1987).

Full TEM formulation in height coordinates (§3.5.2):

  F^(φ) = ρ₀ a cosφ · [ f [v′θ′]/θ̄_z − [u′v′] ]
  F^(z) = ρ₀ a cosφ · [ [v′θ′] ū_z / θ̄_z − [u′w′] ]

QG approximation (used when vertical wind w is absent):

  F^(φ) = −ρ₀ a cosφ · [u′v′]
  F^(z) =  ρ₀ a cosφ · f [v′θ′] / θ̄_z

∇·F = (a cosφ)⁻¹ ∂(F^(φ) cosφ)/∂φ + ∂F^(z)/∂z

Density ρ₀ is resolved via a priority cascade:
  1. `pres` present          → ρ₀ = p̄ / (Rd T̄)
  2. `temp` present, no pres → hydrostatic integration → p̄ → ρ₀
  3. Neither                 → exponential profile ρ_surf exp(−z/H)

Scale height H = Rd T̄(z,φ)/g (locally varying); fitted from log-pressure if
temperature is unavailable.

References:
    Andrews, D. G., Holton, J. R., & Leovy, C. B. (1987).
    Middle Atmosphere Dynamics. Academic Press.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import xarray as xr

from ..grid import append_history, get_cells_dim
from ..extract import zonal_mean as _zonal_mean

logger = logging.getLogger(__name__)

# ── Physical constants ────────────────────────────────────────────────────────
_RD = 287.05        # J kg⁻¹ K⁻¹  dry-air gas constant
_CP = 1004.0        # J kg⁻¹ K⁻¹  specific heat at constant pressure
_G = 9.80665        # m s⁻²       standard gravity
_OMEGA = 7.2921e-5  # rad s⁻¹     Earth rotation rate
_A = 6.371e6        # m            Earth mean radius
_KAPPA = _RD / _CP  # ≈ 0.2857
_P0 = 1.0e5         # Pa           reference pressure for θ

_SECS_PER_DAY = 86400.0

# ── Helpers ───────────────────────────────────────────────────────────────────

def _coriolis(lat_rad: xr.DataArray) -> xr.DataArray:
    """Coriolis parameter f = 2Ω sinφ."""
    return 2.0 * _OMEGA * np.sin(lat_rad)


def _lat_rad(ds_zm: xr.Dataset) -> xr.DataArray:
    """Return latitude in radians from the 'lat' coordinate."""
    return np.deg2rad(ds_zm['lat'])


def _resolve_scale_height(ds_zm: xr.Dataset) -> xr.DataArray:
    """Return locally-varying scale height H(z, φ) in metres.

    Priority:
    1. Compute H = Rd T̄ / g from the zonal-mean temperature.
    2. Fit H as a constant via linear regression of log(p̄) vs z.
    3. Return 7 km as a hard-coded fallback.
    """
    alt = ds_zm['alt'] if 'alt' in ds_zm.coords else ds_zm['altitude']

    if 'temp' in ds_zm:
        T_bar = ds_zm['temp']  # (lat, alt) or (time, lat, alt)
        H = (_RD * T_bar) / _G
        H.attrs = {'long_name': 'Scale height', 'units': 'm'}
        logger.debug("Scale height H computed from T̄ = Rd T̄/g.")
        return H

    if 'pres' in ds_zm:
        # Fit H from log(p̄) vs z using the zonal-mean pressure profile
        p_bar = ds_zm['pres'].mean('lat') if 'lat' in ds_zm['pres'].dims else ds_zm['pres']
        z_vals = alt.values.astype(float)
        log_p = np.log(p_bar.values.astype(float))
        # Fit per time step if needed; otherwise treat as 1D
        log_p_1d = log_p.reshape(-1, len(z_vals)).mean(axis=0)
        coeffs = np.polyfit(z_vals, log_p_1d, 1)   # slope = −1/H
        H_const = float(-1.0 / coeffs[0]) if coeffs[0] != 0 else 7000.0
        H_const = max(H_const, 1000.0)  # sanity floor 1 km
        logger.debug(f"Scale height H fitted from log(p̄) vs z: H = {H_const/1e3:.2f} km")
        return xr.DataArray(H_const, attrs={'long_name': 'Scale height', 'units': 'm'})

    logger.warning("Neither temp nor pres available; using H = 7 km.")
    return xr.DataArray(7000.0, attrs={'long_name': 'Scale height', 'units': 'm'})


def _resolve_density(ds_zm: xr.Dataset, H: xr.DataArray) -> xr.DataArray:
    """Return ρ₀(z, φ) in kg m⁻³.

    Priority:
    1. pres present         → ρ₀ = p̄ / (Rd T̄)
    2. temp only, no pres   → hydrostatic integration to get p̄, then ρ₀ = p̄/(Rd T̄)
    3. neither              → exponential ρ_surf exp(−z/H)
    """
    alt = ds_zm['alt'] if 'alt' in ds_zm.coords else ds_zm['altitude']

    if 'pres' in ds_zm and 'temp' in ds_zm:
        logger.info("Density: using ρ₀ = p̄/(Rd T̄) (exact ideal gas).")
        rho = ds_zm['pres'] / (_RD * ds_zm['temp'])
        rho.attrs = {'long_name': 'Reference density', 'units': 'kg m-3'}
        return rho

    if 'temp' in ds_zm:
        logger.info("Density: reconstructing p̄ from hydrostatic integration.")
        T_bar = ds_zm['temp']   # (... lat, alt) with alt in metres
        # Sort altitude ascending for integration
        T_bar = T_bar.sortby('alt')
        z = T_bar['alt'].values.astype(float)

        # Integrate from TOA downward: p(z) = p_toa exp(∫_{z}^{z_top} g/(Rd T) dz)
        # Discretise with trapezoid rule
        T_np = T_bar.values.astype(float)   # shape (..., n_alt)
        integrand = _G / (_RD * T_np)       # 1/H(z)

        dz = np.diff(z)                      # (n_alt-1,)
        # cumulative integral from top (index 0 in sorted-ascending is bottom)
        # We integrate from top (TOA) downward to get p at each level.
        # p(z_i) = p_TOA * exp( integral_{z_i}^{z_TOA} g/(Rd T) dz )
        # = p_TOA * exp( sum_{j=i}^{N-2} 0.5*(integrand[j]+integrand[j+1])*dz[j] )
        n_alt = len(z)
        # Build the cumulative sum from the top
        cum_int = np.zeros_like(T_np)
        for i in range(n_alt - 2, -1, -1):
            trap = 0.5 * (integrand[..., i] + integrand[..., i + 1]) * dz[i]
            cum_int[..., i] = cum_int[..., i + 1] + trap

        # Assume p_TOA = 1 Pa (any reference — we only need shape for ρ₀)
        p_toa = 1.0
        p_bar_np = p_toa * np.exp(cum_int)
        p_bar = xr.DataArray(p_bar_np, coords=T_bar.coords, dims=T_bar.dims)

        rho = p_bar / (_RD * T_bar)
        rho.attrs = {'long_name': 'Reference density (hydrostatic)', 'units': 'kg m-3'}
        return rho

    # Fallback: exponential profile
    logger.warning("Density: neither pres nor temp — using exponential profile ρ₀=ρ_surf·exp(−z/H).")
    rho_surf = 1.225  # kg m⁻³ at sea level (ISA)
    H_vals = H.values if hasattr(H, 'values') else float(H)
    z_vals = alt.values.astype(float)
    rho_np = rho_surf * np.exp(-z_vals / H_vals)
    rho = xr.DataArray(rho_np, coords={'alt': alt}, dims=['alt'],
                        attrs={'long_name': 'Reference density (exponential)', 'units': 'kg m-3'})
    return rho


# ── Stage 1: Eddy covariances ─────────────────────────────────────────────────

def compute_eddy_fluxes(ds: xr.Dataset) -> xr.Dataset:
    """Compute zonal-mean eddy covariances from a HEALPix dataset.

    Parameters
    ----------
    ds : xr.Dataset
        HEALPix dataset.  Must contain ``u`` and ``v``.  ``temp`` and ``pres``
        are strongly recommended.  ``w`` enables full TEM computation.

    Returns
    -------
    xr.Dataset
        Zonal-mean dataset with variables:
        ``u_zm``, ``v_zm``, ``theta_zm`` (if temp present),
        ``upvp_zm`` ([u′v′]),
        ``vptp_zm`` ([v′θ′], if temp present),
        ``upwp_zm`` ([u′w′], if w present),
        ``pres_zm``, ``temp_zm`` (if available).
    """
    required = [v for v in ('u', 'v') if v not in ds]
    if required:
        raise ValueError(f"Dataset missing required variables: {required}")

    # ── Zonal means ────────────────────────────────────────────────────────────
    logger.info("Computing zonal means for EP flux eddy covariances.")
    ds_zm = _zonal_mean(ds)

    # ── Potential temperature θ on HEALPix pixels ────────────────────────────
    has_temp = 'temp' in ds and 'temp' in ds_zm
    has_w = 'w' in ds
    has_pres = 'pres' in ds

    if has_temp and has_pres:
        theta = ds['temp'] * (_P0 / ds['pres']) ** _KAPPA
        theta_zm = ds_zm['temp'] * (_P0 / ds_zm['pres']) ** _KAPPA
        theta.attrs = {'long_name': 'Potential temperature', 'units': 'K'}
        theta_zm.attrs = {'long_name': 'Zonal-mean potential temperature', 'units': 'K'}
    elif has_temp:
        # Approximate θ ≈ T if no pressure available
        logger.warning("pres not available; approximating θ ≈ T.")
        theta = ds['temp']
        theta_zm = ds_zm['temp']
    else:
        theta = None
        theta_zm = None

    # ── Broadcast zonal means to HEALPix pixels via healpy ring look-up ──────
    # The zonal_mean output has a 'lat' dimension that corresponds to HEALPix rings.
    # We need to map each pixel back to its ring latitude.
    import healpy as hp
    cell_dim = get_cells_dim(ds)
    npix = ds.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    from ..grid import get_healpix_order
    is_nested = get_healpix_order(ds) == 'nested'
    theta_ring, phi_ring = hp.pix2ang(nside, np.arange(npix), nest=is_nested)
    lats_pixels = 90.0 - np.rad2deg(theta_ring)  # (npix,) in degrees
    lats_zm = ds_zm['lat'].values  # ring latitudes

    # For each pixel find the nearest ring index
    ring_idx = np.searchsorted(np.sort(lats_zm), lats_pixels)
    ring_idx = np.clip(ring_idx, 0, len(lats_zm) - 1)
    # Use the sorted index map — lats_zm may be in any order
    sorted_idx = np.argsort(lats_zm)
    ring_idx = sorted_idx[ring_idx]

    def _broadcast_to_pixels(da_zm):
        """Take a (…, lat) DataArray from zm and return (…, cells) by pixel mapping."""
        vals = da_zm.values  # shape: (... × n_lat)
        # Move lat axis to last position
        lat_ax = da_zm.dims.index('lat')
        vals = np.moveaxis(vals, lat_ax, -1)   # (..., n_lat)
        out = vals[..., ring_idx]              # (..., npix)
        new_dims = [d if d != 'lat' else cell_dim for d in da_zm.dims]
        new_coords = {k: v for k, v in da_zm.coords.items() if k != 'lat'}
        new_coords[cell_dim] = np.arange(npix)
        return xr.DataArray(out, dims=new_dims, coords=new_coords)

    u_zm_px = _broadcast_to_pixels(ds_zm['u'])
    v_zm_px = _broadcast_to_pixels(ds_zm['v'])

    # ── Perturbations ─────────────────────────────────────────────────────────
    u_prime = ds['u'] - u_zm_px
    v_prime = ds['v'] - v_zm_px

    # ── Eddy covariances → zonal means ────────────────────────────────────────
    logger.info("Computing eddy covariances [u′v′] and [v′θ′].")
    upvp_ds = _zonal_mean(xr.Dataset({'upvp': u_prime * v_prime}))
    upvp_zm = upvp_ds['upvp']
    upvp_zm.attrs = {'long_name': "Zonal-mean eddy momentum flux [u'v']", 'units': 'm2 s-2'}

    out = xr.Dataset()
    out['u_zm'] = ds_zm['u']
    out['v_zm'] = ds_zm['v']
    out['upvp_zm'] = upvp_zm

    if has_temp:
        out['temp_zm'] = ds_zm['temp']
        theta_zm_arr = theta_zm if theta_zm is not None else ds_zm['temp']
        out['theta_zm'] = theta_zm_arr

        theta_zm_px = _broadcast_to_pixels(theta_zm_arr)
        theta_prime = theta - theta_zm_px
        vptp_ds = _zonal_mean(xr.Dataset({'vptp': v_prime * theta_prime}))
        vptp_zm = vptp_ds['vptp']
        vptp_zm.attrs = {'long_name': "Zonal-mean eddy heat flux [v'θ']", 'units': 'K m s-1'}
        out['vptp_zm'] = vptp_zm

    if has_pres:
        out['pres_zm'] = ds_zm['pres']

    if has_w:
        w_zm_px = _broadcast_to_pixels(ds_zm['w'])
        w_prime = ds['w'] - w_zm_px
        upwp_ds = _zonal_mean(xr.Dataset({'upwp': u_prime * w_prime}))
        upwp_zm = upwp_ds['upwp']
        upwp_zm.attrs = {'long_name': "Zonal-mean eddy vertical flux [u'w']", 'units': 'm2 s-2'}
        out['upwp_zm'] = upwp_zm

    out.attrs = ds.attrs
    return out


# ── Stage 2-3: EP flux components ────────────────────────────────────────────

def compute_ep_flux(
        eddy_ds: xr.Dataset,
        mode: Literal['auto', 'full', 'qg'] = 'auto',
) -> xr.Dataset:
    """Compute EP flux components F^(φ) and F^(z).

    Parameters
    ----------
    eddy_ds : xr.Dataset
        Output of :func:`compute_eddy_fluxes`.
    mode : {'auto', 'full', 'qg'}
        ``'auto'`` uses full TEM when ``upwp_zm`` is present, else falls back
        to QG.  ``'full'`` raises if ``upwp_zm`` is missing.

    Returns
    -------
    xr.Dataset
        Contains ``F_phi``, ``F_z`` (dims: lat × alt [× time]),
        ``rho0``, ``theta_zm_z``, and the mode used.
    """
    has_upwp = 'upwp_zm' in eddy_ds
    if mode == 'full' and not has_upwp:
        raise ValueError("mode='full' requires upwp_zm (w in input dataset).")
    use_full = has_upwp if mode == 'auto' else (mode == 'full')
    mode_used = 'full' if use_full else 'qg'
    logger.info(f"Computing EP flux in {mode_used.upper()} mode.")

    alt_name = 'alt' if 'alt' in eddy_ds.coords else 'altitude'

    H = _resolve_scale_height(eddy_ds)
    rho0 = _resolve_density(eddy_ds, H)

    lat_rad = np.deg2rad(eddy_ds['lat'])       # (lat,)
    cos_phi = np.cos(lat_rad)
    f = _coriolis(lat_rad)

    upvp = eddy_ds['upvp_zm']
    has_theta = 'theta_zm' in eddy_ds

    if has_theta:
        theta_bar = eddy_ds['theta_zm']
        # ∂θ̄/∂z; avoid division by zero at the top
        theta_z = theta_bar.differentiate(alt_name)  # K m⁻¹
        theta_z = theta_z.where(np.abs(theta_z) > 1e-10, other=1e-10)
        vptp = eddy_ds.get('vptp_zm', xr.zeros_like(upvp))
    else:
        logger.warning("theta_zm not available; setting [v′θ′] = 0.")
        theta_z = xr.DataArray(1.0)      # placeholder — terms using it will vanish
        vptp = xr.zeros_like(upvp)

    if use_full:
        u_bar = eddy_ds['u_zm']
        u_z = u_bar.differentiate(alt_name)   # m s⁻¹ m⁻¹
        upwp = eddy_ds['upwp_zm']

        F_phi = rho0 * _A * cos_phi * (f * vptp / theta_z - upvp)
        F_z = rho0 * _A * cos_phi * (vptp * u_z / theta_z - upwp)
    else:
        F_phi = -rho0 * _A * cos_phi * upvp
        F_z = rho0 * _A * cos_phi * (f * vptp / theta_z)

    F_phi.attrs = {'long_name': 'EP flux meridional component F^(phi)',
                   'units': 'kg s-2'}
    F_z.attrs = {'long_name': 'EP flux vertical component F^(z)',
                 'units': 'kg m-1 s-2'}

    out = xr.Dataset({'F_phi': F_phi, 'F_z': F_z, 'rho0': rho0})
    if has_theta:
        out['theta_zm'] = eddy_ds['theta_zm']
        out['theta_zm_z'] = theta_z
    out.attrs['ep_flux_mode'] = mode_used
    return out


# ── Stage 4: Divergence ───────────────────────────────────────────────────────

def compute_ep_divergence(ep_ds: xr.Dataset) -> xr.Dataset:
    """Compute ∇·F and the EP-flux-induced zonal-mean acceleration.

    Parameters
    ----------
    ep_ds : xr.Dataset
        Output of :func:`compute_ep_flux`.  Must contain ``F_phi`` and ``F_z``.

    Returns
    -------
    xr.Dataset
        Adds ``div_F`` (W m⁻³ equivalent) and ``a_EP`` (m s⁻¹ day⁻¹).
    """
    alt_name = 'alt' if 'alt' in ep_ds.coords else 'altitude'
    lat_rad = np.deg2rad(ep_ds['lat'])
    cos_phi = np.cos(lat_rad)

    F_phi = ep_ds['F_phi']
    F_z = ep_ds['F_z']
    rho0 = ep_ds['rho0']

    # ∂(F^(φ) cosφ)/∂φ  in spherical coords (φ in radians)
    Fphi_cos = F_phi * cos_phi
    # DataArray.differentiate works on a coordinate; we use lat (degrees) and
    # convert the derivative: d/dφ = (180/π) · d/d(lat_deg)
    dFphi_cos_dphi = Fphi_cos.differentiate('lat') * (180.0 / np.pi)

    # ∂F^(z)/∂z
    dFz_dz = F_z.differentiate(alt_name)

    # ∇·F
    div_F = (1.0 / (_A * cos_phi)) * dFphi_cos_dphi + dFz_dz
    div_F.attrs = {'long_name': 'EP flux divergence ∇·F',
                   'units': 'kg m-1 s-2 m-1'}

    # Zonal-mean acceleration a_EP = ∇·F / (ρ₀ a cosφ)  in m s⁻² → m s⁻¹ day⁻¹
    a_EP = div_F / (rho0 * _A * cos_phi) * _SECS_PER_DAY
    a_EP.attrs = {'long_name': 'EP-flux wave forcing on zonal-mean wind',
                  'units': 'm s-1 day-1'}

    out = ep_ds.copy()
    out['div_F'] = div_F
    out['a_EP'] = a_EP
    out.attrs = append_history(ep_ds.attrs, "Computed Eliassen-Palm flux divergence ∇·F.")
    return out


# ── Top-level convenience wrapper ─────────────────────────────────────────────

def eliassen_palm(
        ds: xr.Dataset,
        mode: Literal['auto', 'full', 'qg'] = 'auto',
        time_mean: bool = False,
) -> xr.Dataset:
    """Full EP-flux pipeline: HEALPix dataset → F, ∇·F, and acceleration.

    Parameters
    ----------
    ds : xr.Dataset
        Input HEALPix dataset.  Must contain at minimum ``u`` and ``v``.
        ``temp`` and ``pres`` are strongly recommended; ``w`` enables full TEM.
    mode : {'auto', 'full', 'qg'}
        Passed to :func:`compute_ep_flux`.
    time_mean : bool
        If True, average over the time dimension before returning.

    Returns
    -------
    xr.Dataset
        Zonal-mean dataset with ``F_phi``, ``F_z``, ``div_F``, ``a_EP`` and
        diagnostic fields.
    """
    eddy_ds = compute_eddy_fluxes(ds)
    ep_ds = compute_ep_flux(eddy_ds, mode=mode)
    out = compute_ep_divergence(ep_ds)

    if time_mean and 'time' in out.dims:
        logger.info("Averaging EP flux output over time.")
        out = out.mean(dim='time', keep_attrs=True)

    out.attrs = append_history(
        ds.attrs,
        f"Computed Eliassen-Palm flux (mode={ep_ds.attrs.get('ep_flux_mode', mode)}) "
        f"using HealICON.",
    )
    return out
