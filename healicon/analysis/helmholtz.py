"""
Helmholtz decomposition, vorticity/divergence computation,
and wind reconstruction from vorticity/divergence.
"""
import healpy as hp
import numpy as np
import xarray as xr

from ._common import (
    logger, EARTH_RADIUS_KM, _EARTH_RADIUS_M, _MAX_WORKERS,
    get_healpix_order, get_cells_dim, ensure_ring, ensure_original_order, append_history,
    ThreadPoolExecutor, get_progress_bar,
)

# Earth radius in metres (used to scale streamfunction / velocity potential to m²/s)
_EARTH_RADIUS_M = EARTH_RADIUS_KM * 1e3


def _helmholtz_block(u_block, v_block, lmax, nside, is_nested):
    """
    Helmholtz decomposition of a single block of wind data.

    Returns 6 maps, always in this order:
        u_rot, v_rot  - rotational (non-divergent) wind  [m/s]
        u_div, v_div  - divergent  (irrotational)  wind  [m/s]
        psi           - streamfunction                   [m²/s]
        chi           - velocity potential               [m²/s]

    Method (healpy spin-1 SHT):
      1. map2alm_spin([-v, u], spin=1) → almE (divergent mode), almB (rotational mode)
      2. Rotational wind  ← alm2map_spin([0,   almB], spin=1)
         Divergent  wind  ← alm2map_spin([almE, 0  ], spin=1)
      3. ψ_lm = -almB_lm / √[l(l+1)]   → alm2map x a
         χ_lm =  almE_lm / √[l(l+1)]   → alm2map x a
    """
    orig_shape = u_block.shape
    npix = orig_shape[-1]

    u_2d = u_block.reshape(-1, npix)
    v_2d = v_block.reshape(-1, npix)
    n = u_2d.shape[0]

    u_rot = np.zeros_like(u_2d)
    v_rot = np.zeros_like(u_2d)
    u_div = np.zeros_like(u_2d)
    v_div = np.zeros_like(u_2d)
    psi = np.zeros_like(u_2d)
    chi = np.zeros_like(u_2d)

    l_arr, _ = hp.Alm.getlm(lmax)
    fl = np.sqrt(l_arr * (l_arr + 1.0))
    # Safe denominator: avoid division by zero at l=0 (monopole, physically meaningless for wind)
    fl_safe = np.where(l_arr > 0, fl, 1.0)
    n_alm = len(l_arr)
    order_str = 'nested' if is_nested else 'ring'

    def _process_slice(i):
        u_ring = ensure_ring(u_2d[i], order_str)
        v_ring = ensure_ring(v_2d[i], order_str)

        valid_mask = ~(np.isnan(u_ring) | np.isnan(v_ring))
        if not np.any(valid_mask):
            u_rot[i] = v_rot[i] = u_div[i] = v_div[i] = psi[i] = chi[i] = np.nan
            return

        u_filled = np.where(valid_mask, u_ring, 0.0)
        v_filled = np.where(valid_mask, v_ring, 0.0)

        v_theta = -v_filled
        v_phi = u_filled

        almE, almB = hp.map2alm_spin([v_theta, v_phi], spin=1, lmax=lmax)

        # Rotational wind: keep only B-mode
        zeros = np.zeros(n_alm, dtype=np.complex128)
        m_rot_maps = hp.alm2map_spin([zeros, almB], nside, 1, lmax=lmax)
        u_rot_ring = np.where(valid_mask, m_rot_maps[1], np.nan)
        v_rot_ring = np.where(valid_mask, -m_rot_maps[0], np.nan)

        # Divergent wind: keep only E-mode
        zeros2 = np.zeros(n_alm, dtype=np.complex128)
        m_div_maps = hp.alm2map_spin([almE, zeros2], nside, 1, lmax=lmax)
        u_div_ring = np.where(valid_mask, m_div_maps[1], np.nan)
        v_div_ring = np.where(valid_mask, -m_div_maps[0], np.nan)

        u_rot[i] = ensure_original_order(u_rot_ring, order_str)
        v_rot[i] = ensure_original_order(v_rot_ring, order_str)
        u_div[i] = ensure_original_order(u_div_ring, order_str)
        v_div[i] = ensure_original_order(v_div_ring, order_str)

        # Streamfunction ψ: ζ = ∇²ψ  →  ψ_lm = -almB_lm / fl  (× a for m²/s)
        psi_alm = np.where(l_arr > 0, -almB / fl_safe, 0.0 + 0.0j)
        psi_map = np.where(valid_mask, hp.alm2map(psi_alm, nside, lmax=lmax) * _EARTH_RADIUS_M,
                           np.nan)
        psi[i] = ensure_original_order(psi_map, order_str)

        # Velocity potential χ: D = ∇²χ  →  χ_lm = almE_lm / fl  (× a for m²/s)
        chi_alm = np.where(l_arr > 0, almE / fl_safe, 0.0 + 0.0j)
        chi_map = np.where(valid_mask, hp.alm2map(chi_alm, nside, lmax=lmax) * _EARTH_RADIUS_M,
                           np.nan)
        chi[i] = ensure_original_order(chi_map, order_str)

    if n > 1:
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = [pool.submit(_process_slice, i) for i in range(n)]
            for f in get_progress_bar(as_completed(futures), desc="Helmholtz decomposition",
                                      total=n):
                f.result()
    else:
        _process_slice(0)

    s = orig_shape
    return (u_rot.reshape(s), v_rot.reshape(s),
            u_div.reshape(s), v_div.reshape(s),
            psi.reshape(s), chi.reshape(s))


def compute_helmholtz(ds: xr.Dataset, u_var: str, v_var: str,
                      lmax: int | None = None,
                      include_psi: bool = True,
                      include_chi: bool = True) -> xr.Dataset:
    """
    Helmholtz decomposition of horizontal wind (u, v) on a HEALPix sphere.

    The wind is split into:
        rotational (non-divergent) component  →  u_rot, v_rot
        divergent  (irrotational)  component  →  u_div, v_div

    Optionally:
        streamfunction    ψ  [m²/s]  (include_psi=True)
        velocity potential χ  [m²/s]  (include_chi=True)

    Args:
        ds         : HEALPix xr.Dataset with a 'cell' dimension.
        u_var      : Name of the eastward  wind variable.
        v_var      : Name of the northward wind variable.
        lmax       : Maximum spherical harmonic degree (default: 3*nside-1).
        include_psi: Include streamfunction in the output.
        include_chi: Include velocity potential in the output.

    Returns:
        xr.Dataset containing u_rot, v_rot, u_div, v_div, and optionally ψ, χ.
    """
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'

    npix = ds.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    if lmax is None:
        lmax = 3 * nside - 1

    logger.info(
        f"Computing Helmholtz decomposition from '{u_var}' and '{v_var}' "
        f"(lmax={lmax}, psi={include_psi}, chi={include_chi})."
    )

    dtype = ds[u_var].dtype
    u_rot, v_rot, u_div, v_div, psi, chi = xr.apply_ufunc(
        _helmholtz_block,
        ds[u_var], ds[v_var],
        kwargs={'lmax': lmax, 'nside': nside, 'is_nested': is_nested},
        input_core_dims=[[cell_dim], [cell_dim]],
        output_core_dims=[[cell_dim]] * 6,
        dask="parallelized",
        output_dtypes=[dtype] * 6,
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

    coords = ds.coords
    wind_attrs_base = {'units': ds[u_var].attrs.get('units', 'm s-1'), 'grid_mapping': 'healpix'}

    out_ds = xr.Dataset(coords=coords)
    out_ds['u_rot'] = u_rot.assign_coords({cell_dim: ds[cell_dim]})
    out_ds['u_rot'].attrs = {**wind_attrs_base,
                             'long_name': 'Rotational (non-divergent) eastward wind'}
    out_ds['v_rot'] = v_rot.assign_coords({cell_dim: ds[cell_dim]})
    out_ds['v_rot'].attrs = {**wind_attrs_base,
                             'long_name': 'Rotational (non-divergent) northward wind'}
    out_ds['u_div'] = u_div.assign_coords({cell_dim: ds[cell_dim]})
    out_ds['u_div'].attrs = {**wind_attrs_base,
                             'long_name': 'Divergent (irrotational) eastward wind'}
    out_ds['v_div'] = v_div.assign_coords({cell_dim: ds[cell_dim]})
    out_ds['v_div'].attrs = {**wind_attrs_base,
                             'long_name': 'Divergent (irrotational) northward wind'}

    if include_psi:
        out_ds['psi'] = psi.assign_coords({cell_dim: ds[cell_dim]})
        out_ds['psi'].attrs = {
            'standard_name': 'atmosphere_horizontal_streamfunction',
            'long_name': 'Streamfunction',
            'units': 'm2 s-1',
            'grid_mapping': 'healpix',
        }

    if include_chi:
        out_ds['chi'] = chi.assign_coords({cell_dim: ds[cell_dim]})
        out_ds['chi'].attrs = {
            'standard_name': 'atmosphere_horizontal_velocity_potential',
            'long_name': 'Velocity potential',
            'units': 'm2 s-1',
            'grid_mapping': 'healpix',
        }

    # Pass through any other variables unchanged
    for var in ds.data_vars:
        if var not in [u_var, v_var]:
            out_ds[var] = ds[var]

    out_ds.attrs = ds.attrs
    out_ds.attrs = append_history(out_ds.attrs,
                                  f"Helmholtz decomposition of ({u_var}, {v_var}), lmax={lmax}.")
    return out_ds


def _vorticity_divergence_block(u_block, v_block, lmax, nside, is_nested):
    """
    Computes horizontal vorticity and divergence from U and V wind components.
    
    Args:
        u_block: U wind component data
        v_block: V wind component data
        lmax: Maximum spherical harmonic degree
        nside: HEALPix nside
        is_nested: Whether data is in nested order

    Returns:
        Tuple of (divergence, vorticity)
    """
    # u_block, v_block shape (..., npix)
    orig_shape = u_block.shape
    npix = orig_shape[-1]

    u_2d = u_block.reshape(-1, npix)
    v_2d = v_block.reshape(-1, npix)

    div_out = np.zeros_like(u_2d)
    vor_out = np.zeros_like(v_2d)

    l, m = hp.Alm.getlm(lmax)
    # Prefactors for vector fields (using positive curl convention)
    fl = np.sqrt(l * (l + 1))
    R = EARTH_RADIUS_KM * 1000.0
    order_str = 'nested' if is_nested else 'ring'

    def _process_slice(i):
        u_ring = ensure_ring(u_2d[i], order_str)
        v_ring = ensure_ring(v_2d[i], order_str)

        valid_mask = ~(np.isnan(u_ring) | np.isnan(v_ring))
        if not np.any(valid_mask):
            div_out[i] = np.nan
            vor_out[i] = np.nan
            return

        u_filled = np.where(valid_mask, u_ring, 0.0)
        v_filled = np.where(valid_mask, v_ring, 0.0)

        v_theta = -v_filled
        v_phi = u_filled

        # map2alm_spin returns E and B modes for spin=1
        almE, almB = hp.map2alm_spin([v_theta, v_phi], spin=1, lmax=lmax)

        div_alm = -fl * almE / R
        vor_alm = fl * almB / R

        # Transform back to map space
        div_map = np.where(valid_mask, hp.alm2map(div_alm, nside, lmax=lmax), np.nan)
        vor_map = np.where(valid_mask, hp.alm2map(vor_alm, nside, lmax=lmax), np.nan)

        div_out[i] = ensure_original_order(div_map, order_str)
        vor_out[i] = ensure_original_order(vor_map, order_str)

    n = u_2d.shape[0]
    if n > 1:
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = [pool.submit(_process_slice, i) for i in range(n)]
            for f in get_progress_bar(as_completed(futures), desc="Computing vorticity/divergence",
                                      total=n):
                f.result()
    else:
        _process_slice(0)

    out_shape = orig_shape
    return div_out.reshape(out_shape), vor_out.reshape(out_shape)


def compute_vorticity_divergence(ds: xr.Dataset, u_var: str, v_var: str,
                                 lmax: int | None = None) -> xr.Dataset:
    """
    Computes horizontal vorticity and divergence from U and V wind components.
    
    Args:
        ds: Dataset containing U and V wind components
        u_var: Name of the U wind variable
        v_var: Name of the V wind variable
        lmax: Maximum spherical harmonic degree (optional)
        
    Returns:
        Dataset containing vorticity and divergence
    """
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'

    npix = ds.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    if lmax is None:
        lmax = 3 * nside - 1

    logger.info(f"Computing vorticity and divergence from '{u_var}' and '{v_var}' (lmax={lmax}).")

    div, vor = xr.apply_ufunc(
        _vorticity_divergence_block,
        ds[u_var], ds[v_var],
        kwargs={'lmax': lmax, 'nside': nside, 'is_nested': is_nested},
        input_core_dims=[[cell_dim], [cell_dim]],
        output_core_dims=[[cell_dim], [cell_dim]],
        dask="parallelized",
        output_dtypes=[ds[u_var].dtype, ds[v_var].dtype],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

    out_ds = xr.Dataset(coords=ds.coords)
    out_ds['divergence'] = div.assign_coords({cell_dim: ds[cell_dim]})
    out_ds['divergence'].attrs = {'standard_name': 'divergence_of_wind', 'units': 's-1',
                                  'grid_mapping': 'healpix'}

    out_ds['vorticity'] = vor.assign_coords({cell_dim: ds[cell_dim]})
    out_ds['vorticity'].attrs = {'standard_name': 'atmosphere_relative_vorticity', 'units': 's-1',
                                 'grid_mapping': 'healpix'}

    for var in ds.data_vars:
        if var not in [u_var, v_var]:
            out_ds[var] = ds[var]

    out_ds.attrs = ds.attrs
    out_ds.attrs = append_history(out_ds.attrs, f"Computed vorticity and divergence.")

    return out_ds


def _uv_from_vorticity_divergence_block(div_block, vor_block, lmax, nside, is_nested):
    """
    Reconstructs U and V wind components from vorticity and divergence.
    
    Args:
        div_block: Divergence data (potentially chunked)
        vor_block: Vorticity data (potentially chunked)
        lmax: Maximum spherical harmonic degree
        nside: HEALPix nside
        is_nested: Whether data is in nested order

    Returns:
        Tuple of (u_reconstructed, v_reconstructed)
    """
    orig_shape = div_block.shape
    npix = orig_shape[-1]

    div_2d = div_block.reshape(-1, npix)
    vor_2d = vor_block.reshape(-1, npix)

    u_out = np.zeros_like(div_2d)
    v_out = np.zeros_like(vor_2d)

    l, m = hp.Alm.getlm(lmax)
    fl = np.sqrt(l * (l + 1))
    fl_safe = np.where(l > 0, fl, 1.0)
    R = EARTH_RADIUS_KM * 1000.0
    order_str = 'nested' if is_nested else 'ring'

    def _process_slice(i):
        div_ring = ensure_ring(div_2d[i], order_str)
        vor_ring = ensure_ring(vor_2d[i], order_str)

        valid_mask = ~(np.isnan(div_ring) | np.isnan(vor_ring))
        if not np.any(valid_mask):
            u_out[i] = np.nan
            v_out[i] = np.nan
            return

        div_filled = np.where(valid_mask, div_ring, 0.0)
        vor_filled = np.where(valid_mask, vor_ring, 0.0)

        # Convert to alm
        div_alm = hp.map2alm(div_filled, lmax=lmax)
        vor_alm = hp.map2alm(vor_filled, lmax=lmax)

        # Get E and B modes
        # Div = -fl * E / R -> E = -Div * R / fl
        # Vor = fl * B / R -> B = Vor * R / fl
        almE = np.where(l > 0, -div_alm * R / fl_safe, 0.0 + 0.0j)
        almB = np.where(l > 0, vor_alm * R / fl_safe, 0.0 + 0.0j)

        # Transform back to map space with spin=1
        m_spin = hp.alm2map_spin([almE, almB], nside, spin=1, lmax=lmax)
        v_theta = m_spin[0]
        v_phi = m_spin[1]

        # u is Eastward -> v_phi
        # v is Northward -> -v_theta
        u_ring = np.where(valid_mask, v_phi, np.nan)
        v_ring = np.where(valid_mask, -v_theta, np.nan)

        u_out[i] = ensure_original_order(u_ring, order_str)
        v_out[i] = ensure_original_order(v_ring, order_str)

    n = div_2d.shape[0]
    if n > 1:
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = [pool.submit(_process_slice, i) for i in range(n)]
            for f in get_progress_bar(as_completed(futures), desc="Reconstructing wind components",
                                      total=n):
                f.result()
    else:
        _process_slice(0)

    out_shape = orig_shape
    return u_out.reshape(out_shape), v_out.reshape(out_shape)


def compute_uv_from_vorticity_divergence(ds: xr.Dataset, div_var: str, vor_var: str,
                                         lmax: int | None = None) -> xr.Dataset:
    """
    Computes U and V wind components from horizontal divergence and vorticity.
    
    Args:
        ds: Dataset containing divergence and vorticity
        div_var: Name of the divergence variable
        vor_var: Name of the vorticity variable
        lmax: Maximum spherical harmonic degree (optional)
        
    Returns:
        Dataset containing U and V wind components
    """
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'

    npix = ds.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    if lmax is None:
        lmax = 3 * nside - 1

    logger.info(f"Computing U and V from '{div_var}' and '{vor_var}' (lmax={lmax}).")

    u, v = xr.apply_ufunc(
        _uv_from_vorticity_divergence_block,
        ds[div_var], ds[vor_var],
        kwargs={'lmax': lmax, 'nside': nside, 'is_nested': is_nested},
        input_core_dims=[[cell_dim], [cell_dim]],
        output_core_dims=[[cell_dim], [cell_dim]],
        dask="parallelized",
        output_dtypes=[ds[div_var].dtype, ds[vor_var].dtype],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

    out_ds = xr.Dataset(coords=ds.coords)
    out_ds['u'] = u.assign_coords({cell_dim: ds[cell_dim]})
    out_ds['u'].attrs = {'standard_name': 'eastward_wind', 'long_name': 'Zonal wind',
                         'units': 'm s-1',
                         'grid_mapping': 'healpix'}

    out_ds['v'] = v.assign_coords({cell_dim: ds[cell_dim]})
    out_ds['v'].attrs = {'standard_name': 'northward_wind', 'long_name': 'Meridional wind',
                         'units': 'm s-1',
                         'grid_mapping': 'healpix'}

    for var in ds.data_vars:
        if var not in [div_var, vor_var]:
            out_ds[var] = ds[var]

    out_ds.attrs = ds.attrs
    out_ds.attrs = append_history(out_ds.attrs, f"Computed U and V from vorticity and divergence.")

    return out_ds
