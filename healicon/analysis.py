import logging

import healpy as hp
import numpy as np
import xarray as xr

from .grid import get_healpix_order, get_cells_dim, ensure_ring, ensure_original_order

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.229


def degree_to_wavelength(l, radius=EARTH_RADIUS_KM):
    """
    Converts spherical harmonic degree l to characteristic wavelength (scale).
    
    Args:
        l: Spherical harmonic degree (scalar or array)
        radius: Radius of the sphere (defaults to Earth radius 6371.229 km)
        
    Returns:
        Wavelength matching the units of radius (km).
    """
    # Avoid division by zero for l=0
    l_safe = np.maximum(l, 1e-10)
    return (2 * np.pi * radius) / np.sqrt(l_safe * (l_safe + 1))


def wavelength_to_degree(wavelength, radius=EARTH_RADIUS_KM):
    """
    Converts characteristic wavelength (scale) to spherical harmonic degree l.
    
    Args:
        wavelength: Characteristic wavelength in same units as radius
        radius: Radius of the sphere (defaults to Earth radius 6371.229 km)
        
    Returns:
        Spherical harmonic degree l (float)
    """
    val = (2 * np.pi * radius) / wavelength
    # Solve l^2 + l - val^2 = 0
    return (-1.0 + np.sqrt(1.0 + 4.0 * val ** 2)) / 2.0


def _anafast_block(data_block, lmax, is_nested):
    orig_shape = data_block.shape
    npix = orig_shape[-1]
    data_2d = data_block.reshape(-1, npix)

    n_l = lmax + 1
    out_data = np.zeros((data_2d.shape[0], n_l), dtype=data_2d.dtype)

    for i in range(data_2d.shape[0]):
        # anafast returns cl array of shape (lmax+1,)
        d = ensure_ring(data_2d[i], 'nested' if is_nested else 'ring')
        out_data[i] = hp.anafast(d, lmax=lmax)

    out_shape = orig_shape[:-1] + (n_l,)
    return out_data.reshape(out_shape)


def compute_spectrum(ds: xr.Dataset, var_name: str, lmax: int = None) -> xr.Dataset:
    """
    Computes the angular power spectrum (Cl) of a variable using spherical harmonics.
    """
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'
    npix = ds.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    if lmax is None:
        lmax = 3 * nside - 1

    logger.info(f"Computing power spectrum for '{var_name}' up to lmax={lmax}.")

    l_coords = np.arange(lmax + 1)

    da = xr.apply_ufunc(
        _anafast_block,
        ds[var_name],
        kwargs={'lmax': lmax, 'is_nested': is_nested},
        input_core_dims=[[cell_dim]],
        output_core_dims=[['l']],
        dask="parallelized",
        output_dtypes=[ds[var_name].dtype],
        dask_gufunc_kwargs={'output_sizes': {'l': len(l_coords)}, 'allow_rechunk': True}
    )

    out_ds = xr.Dataset(
        data_vars={f"{var_name}_cl": da},
        coords={'l': l_coords}
    )

    # Attach wavelength as a secondary coordinate on l for convenience
    wavelength_km = degree_to_wavelength(np.maximum(l_coords, 1))  # avoid l=0 singularity
    out_ds = out_ds.assign_coords(wavelength_km=('l', wavelength_km))
    out_ds['wavelength_km'].attrs = {
        'long_name': 'Equivalent wavelength',
        'units': 'km',
    }

    for coord in ds[var_name].coords:
        if coord not in [cell_dim, 'l', 'lat', 'lon'] and coord in ds.coords:
            out_ds.coords[coord] = ds.coords[coord]

    out_ds.l.attrs = {"long_name": "Spherical harmonic degree (l)"}
    out_ds.attrs = ds.attrs
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs[
        'history'] = f"{history}{sep}Computed angular power spectrum (Cl) up to lmax={lmax} for {var_name}."
    return out_ds


def _filter_block(data_block, fwhm_rad, lmax, is_nested):
    orig_shape = data_block.shape
    npix = orig_shape[-1]
    # Cache nside once per block rather than re-computing per iteration
    nside = hp.npix2nside(npix)
    data_2d = data_block.reshape(-1, npix)

    out_data = np.zeros_like(data_2d)

    for i in range(data_2d.shape[0]):
        d = ensure_ring(data_2d[i], 'nested' if is_nested else 'ring')
        if fwhm_rad is not None:
            filtered = hp.smoothing(d, fwhm=fwhm_rad)
        elif lmax is not None:
            alm = hp.map2alm(d, lmax=lmax, iter=3)
            filtered = hp.alm2map(alm, nside=nside)
        out_data[i] = ensure_original_order(filtered, 'nested' if is_nested else 'ring')

    return out_data.reshape(orig_shape)


def filter_spatial(ds: xr.Dataset, fwhm_deg: float = None, lmax: int = None,
                   wavelength_km: float = None) -> xr.Dataset:
    """
    Filters spatial data using spherical harmonics.

    Specify exactly one of:
        fwhm_deg     : Full-width at half-maximum (degrees) for a Gaussian beam.
        lmax         : Hard low-pass cutoff — retain only spherical harmonic degrees <= lmax.
        wavelength_km: Hard low-pass cutoff expressed as a physical scale. Equivalent to
                       passing lmax=int(wavelength_to_degree(wavelength_km)).
    """
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'

    n_specified = sum(x is not None for x in [fwhm_deg, lmax, wavelength_km])
    if n_specified != 1:
        raise ValueError("Must specify exactly one of: fwhm_deg, lmax, or wavelength_km.")

    # Convert wavelength_km → lmax so the rest of the code is unchanged
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
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs['history'] = f"{history}{sep}{hist_msg}"
    return out_ds


def _regrade_block(data_block, nside_out, is_nested):
    orig_shape = data_block.shape
    npix_in = orig_shape[-1]
    data_2d = data_block.reshape(-1, npix_in)

    npix_out = hp.nside2npix(nside_out)
    out_data = np.zeros((data_2d.shape[0], npix_out), dtype=data_2d.dtype)

    for i in range(data_2d.shape[0]):
        # ud_grade preserves sum(map)/npix.
        order = 'NEST' if is_nested else 'RING'
        out_data[i] = hp.ud_grade(data_2d[i], nside_out=nside_out, order_in=order, order_out=order)

    out_shape = orig_shape[:-1] + (npix_out,)
    return out_data.reshape(out_shape)


def regrade_resolution(ds: xr.Dataset, new_nside: int) -> xr.Dataset:
    """
    Upgrades or downgrades the HEALPix resolution of the dataset.
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
    out_ds.attrs['healpix_nside'] = new_nside
    out_ds.attrs['healpix_npix'] = new_npix
    out_ds.attrs['healpix_cell_area_sr'] = f"{4 * np.pi / new_npix:.6e}"

    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs['history'] = f"{history}{sep}Regraded resolution to nside={new_nside}."
    return out_ds


# Earth radius in metres (used to scale streamfunction / velocity potential to m²/s)
_EARTH_RADIUS_M = EARTH_RADIUS_KM * 1e3


def _helmholtz_block(u_block, v_block, lmax, nside, is_nested):
    """
    Helmholtz decomposition of a single block of wind data.

    Returns 6 maps, always in this order:
        u_rot, v_rot  – rotational (non-divergent) wind  [m/s]
        u_div, v_div  – divergent  (irrotational)  wind  [m/s]
        psi           – streamfunction                   [m²/s]
        chi           – velocity potential               [m²/s]

    Method (healpy spin-1 SHT):
      1. map2alm_spin([−v, u], spin=1) → almE (divergent mode), almB (rotational mode)
      2. Rotational wind  ← alm2map_spin([0,   almB], spin=1)
         Divergent  wind  ← alm2map_spin([almE, 0  ], spin=1)
      3. ψ_lm = −almB_lm / √[l(l+1)]   → alm2map → × a
         χ_lm =  almE_lm / √[l(l+1)]   → alm2map → × a
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
    zeros = np.zeros(len(l_arr), dtype=np.complex128)

    for i in range(n):
        u_ring = ensure_ring(u_2d[i], 'nested' if is_nested else 'ring')
        v_ring = ensure_ring(v_2d[i], 'nested' if is_nested else 'ring')

        v_theta = -v_ring
        v_phi = u_ring

        almE, almB = hp.map2alm_spin([v_theta, v_phi], spin=1, lmax=lmax)

        # Rotational wind: keep only B-mode
        m_rot = hp.alm2map_spin([zeros.copy(), almB], nside, 1, lmax=lmax)
        u_rot_ring = m_rot[1]
        v_rot_ring = -m_rot[0]

        # Divergent wind: keep only E-mode
        m_div = hp.alm2map_spin([almE, zeros.copy()], nside, 1, lmax=lmax)
        u_div_ring = m_div[1]
        v_div_ring = -m_div[0]

        u_rot[i] = ensure_original_order(u_rot_ring, 'nested' if is_nested else 'ring')
        v_rot[i] = ensure_original_order(v_rot_ring, 'nested' if is_nested else 'ring')
        u_div[i] = ensure_original_order(u_div_ring, 'nested' if is_nested else 'ring')
        v_div[i] = ensure_original_order(v_div_ring, 'nested' if is_nested else 'ring')

        # Streamfunction ψ: ζ = ∇²ψ  →  ψ_lm = -almB_lm / fl  (× a for m²/s)
        psi_alm = np.where(l_arr > 0, -almB / fl_safe, 0.0 + 0.0j)
        psi[i] = ensure_original_order(hp.alm2map(psi_alm, nside, lmax=lmax) * _EARTH_RADIUS_M, 'nested' if is_nested else 'ring')

        # Velocity potential χ: D = ∇²χ  →  χ_lm = almE_lm / fl  (× a for m²/s)
        chi_alm = np.where(l_arr > 0, almE / fl_safe, 0.0 + 0.0j)
        chi[i] = ensure_original_order(hp.alm2map(chi_alm, nside, lmax=lmax) * _EARTH_RADIUS_M, 'nested' if is_nested else 'ring')

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
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs['history'] = (
        f"{history}{sep}Helmholtz decomposition of ({u_var}, {v_var}), lmax={lmax}."
    )
    return out_ds


def _vorticity_divergence_block(u_block, v_block, lmax, nside, is_nested):
    # u_block, v_block shape (..., npix)
    orig_shape = u_block.shape
    npix = orig_shape[-1]

    u_2d = u_block.reshape(-1, npix)
    v_2d = v_block.reshape(-1, npix)

    div_out = np.zeros_like(u_2d)
    vor_out = np.zeros_like(v_2d)

    l, m = hp.Alm.getlm(lmax)
    # The prefactor for vector fields: 
    # Div = -sqrt(l(l+1)) * E
    # Vor = sqrt(l(l+1)) * B (signs may vary by convention, typical meteorological is positive curl)
    fl = np.sqrt(l * (l + 1))

    for i in range(u_2d.shape[0]):
        u_ring = ensure_ring(u_2d[i], 'nested' if is_nested else 'ring')
        v_ring = ensure_ring(v_2d[i], 'nested' if is_nested else 'ring')
        v_theta = -v_ring
        v_phi = u_ring

        # map2alm_spin returns E and B modes for spin=1
        almE, almB = hp.map2alm_spin([v_theta, v_phi], spin=1, lmax=lmax)

        div_alm = -fl * almE
        vor_alm = fl * almB

        # Transform back to map space
        div_out[i] = ensure_original_order(hp.alm2map(div_alm, nside, lmax=lmax), 'nested' if is_nested else 'ring')
        vor_out[i] = ensure_original_order(hp.alm2map(vor_alm, nside, lmax=lmax), 'nested' if is_nested else 'ring')

    out_shape = orig_shape
    return div_out.reshape(out_shape), vor_out.reshape(out_shape)


def compute_vorticity_divergence(ds: xr.Dataset, u_var: str, v_var: str,
                                 lmax: int | None = None) -> xr.Dataset:
    """
    Computes horizontal vorticity and divergence from U and V wind components.
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
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs['history'] = f"{history}{sep}Computed vorticity and divergence."

    return out_ds


def _uv_from_vorticity_divergence_block(div_block, vor_block, lmax, nside, is_nested):
    orig_shape = div_block.shape
    npix = orig_shape[-1]

    div_2d = div_block.reshape(-1, npix)
    vor_2d = vor_block.reshape(-1, npix)

    u_out = np.zeros_like(div_2d)
    v_out = np.zeros_like(vor_2d)

    l, m = hp.Alm.getlm(lmax)
    fl = np.sqrt(l * (l + 1))
    fl_safe = np.where(l > 0, fl, 1.0)

    for i in range(div_2d.shape[0]):
        div_ring = ensure_ring(div_2d[i], 'nested' if is_nested else 'ring')
        vor_ring = ensure_ring(vor_2d[i], 'nested' if is_nested else 'ring')

        # Convert to alm
        div_alm = hp.map2alm(div_ring, lmax=lmax)
        vor_alm = hp.map2alm(vor_ring, lmax=lmax)

        # Get E and B modes
        # Div = -fl * E -> E = -Div / fl
        # Vor = fl * B -> B = Vor / fl
        almE = np.where(l > 0, -div_alm / fl_safe, 0.0 + 0.0j)
        almB = np.where(l > 0, vor_alm / fl_safe, 0.0 + 0.0j)

        # Transform back to map space with spin=1
        m_spin = hp.alm2map_spin([almE, almB], nside, spin=1, lmax=lmax)
        v_theta = m_spin[0]
        v_phi = m_spin[1]

        # u is Eastward -> v_phi
        # v is Northward -> -v_theta
        u_ring = v_phi
        v_ring = -v_theta

        u_out[i] = ensure_original_order(u_ring, 'nested' if is_nested else 'ring')
        v_out[i] = ensure_original_order(v_ring, 'nested' if is_nested else 'ring')

    out_shape = orig_shape
    return u_out.reshape(out_shape), v_out.reshape(out_shape)


def compute_uv_from_vorticity_divergence(ds: xr.Dataset, div_var: str, vor_var: str,
                                         lmax: int | None = None) -> xr.Dataset:
    """
    Computes U and V wind components from horizontal divergence and vorticity.
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
    out_ds['u'].attrs = {'standard_name': 'eastward_wind', 'long_name': 'Zonal wind', 'units': 'm s-1',
                         'grid_mapping': 'healpix'}

    out_ds['v'] = v.assign_coords({cell_dim: ds[cell_dim]})
    out_ds['v'].attrs = {'standard_name': 'northward_wind', 'long_name': 'Meridional wind', 'units': 'm s-1',
                         'grid_mapping': 'healpix'}

    for var in ds.data_vars:
        if var not in [div_var, vor_var]:
            out_ds[var] = ds[var]

    out_ds.attrs = ds.attrs
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs['history'] = f"{history}{sep}Computed U and V from vorticity and divergence."

    return out_ds


def _directional_filter_block(a_block, b_block, target_m, lmax, is_nested):
    print(f"DEBUG: _directional_filter_block called with lmax={lmax}")
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
    
    for i in range(a_2d.shape[0]):
        a_ring = ensure_ring(a_2d[i], 'nested' if is_nested else 'ring')
        b_ring = ensure_ring(b_2d[i], 'nested' if is_nested else 'ring')
        
        alm_a = hp.map2alm(a_ring, lmax=lmax, iter=3)
        alm_b = hp.map2alm(b_ring, lmax=lmax, iter=3)
        
        if target_m > 0:
            alm_a_out = 0.5 * (alm_a + 1j * alm_b) * mask
            alm_b_out = -1j * alm_a_out
        elif target_m < 0:
            alm_a_out = 0.5 * (alm_a - 1j * alm_b) * mask
            alm_b_out = 1j * alm_a_out
        else:
            alm_a_out = alm_a * mask
            alm_b_out = alm_b * mask
            
        a_filtered = hp.alm2map(alm_a_out, nside=nside)
        b_filtered = hp.alm2map(alm_b_out, nside=nside)
        
        out_a[i] = ensure_original_order(a_filtered, 'nested' if is_nested else 'ring')
        out_b[i] = ensure_original_order(b_filtered, 'nested' if is_nested else 'ring')
        
    return out_a.reshape(orig_shape), out_b.reshape(orig_shape)


def _get_symmetric_pixels(nside, is_nested=False):
    """
    Returns an array of pixel indices that correspond to the exact reflection
    across the equator for each pixel in a HEALPix grid.
    """
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=is_nested)
    theta_sym = np.pi - theta
    return hp.ang2pix(nside, theta_sym, phi, nest=is_nested)


def _extract_spatial_tide_components(da_cos: xr.DataArray, da_sin: xr.DataArray,
                                     m_filters: list[int] | None, cell_dim: str,
                                     sym_idx_da: xr.DataArray, apply_filter_fn) -> dict:
    """
    Decomposes the cosine and sine tidal coefficients into symmetric/antisymmetric 
    amplitudes and phases, optionally filtering by specific wavenumbers.
    """
    ms = m_filters if m_filters is not None else [None]
    results = {'amp_sym': [], 'pha_sym': [], 'amp_asy': [], 'pha_asy': []}
    
    for m in ms:
        cos_m, sin_m = apply_filter_fn(da_cos, da_sin, m) if m is not None else (da_cos, da_sin)
            
        cos_sym = 0.5 * (cos_m + cos_m.isel({cell_dim: sym_idx_da}))
        cos_asy = 0.5 * (cos_m - cos_m.isel({cell_dim: sym_idx_da}))
        
        sin_sym = 0.5 * (sin_m + sin_m.isel({cell_dim: sym_idx_da}))
        sin_asy = 0.5 * (sin_m - sin_m.isel({cell_dim: sym_idx_da}))
        
        res_m = {
            'amp_sym': np.sqrt(cos_sym**2 + sin_sym**2),
            'pha_sym': np.arctan2(sin_sym, cos_sym),
            'amp_asy': np.sqrt(cos_asy**2 + sin_asy**2),
            'pha_asy': np.arctan2(sin_asy, cos_asy)
        }
        
        if m is not None:
            res_m = {k: v.expand_dims(m=[m]) for k, v in res_m.items()}
            
        for k in results:
            results[k].append(res_m[k])
            
    if m_filters is not None:
        return {k: xr.concat(v, dim='m') for k, v in results.items()}
    return {k: v[0] for k, v in results.items()}



def compute_tidal_analysis(ds: xr.Dataset, var_name: str, periods_hours: list[float], 
                           m_filters: list[int] | None = None, lmax: int | None = None) -> xr.Dataset:
    """
    Performs a full tidal analysis on a HEALPix dataset over time.
    
    1. Extracts the temporal harmonic coefficients for the specified periods (in hours).
    2. Optionally filters the spatial field to specific zonal wavenumbers (m_filters).
    3. Decomposes the spatial field into symmetric and antisymmetric components relative to the equator.
    4. Computes Amplitude and Phase for symmetric and antisymmetric components.
    """
    if 'time' not in ds.dims:
        raise ValueError("Dataset must have a 'time' dimension for temporal tidal analysis.")
        
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'
    npix = ds.sizes[cell_dim]
    nside = hp.npix2nside(npix)
    
    if lmax is None:
        lmax = 3 * nside - 1
        
    t_days = (ds['time'] - ds['time'][0]).dt.total_seconds() / 86400.0
    t_days_vals = t_days.values
    
    sym_idx = _get_symmetric_pixels(nside, is_nested=is_nested)
    sym_idx_da = xr.DataArray(sym_idx, dims=[cell_dim])
    
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
    
    M_A = xr.DataArray(M[:, 0, :], dims=['period', 'time'], coords={'period': periods_hours, 'time': ds['time']})
    M_B = xr.DataArray(M[:, 1, :], dims=['period', 'time'], coords={'period': periods_hours, 'time': ds['time']})
    
    da_cos = xr.dot(ds[var_name], M_A, dims=['time'])
    da_sin = xr.dot(ds[var_name], M_B, dims=['time'])
    
    spatial_res = _extract_spatial_tide_components(
        da_cos, da_sin, m_filters, cell_dim, sym_idx_da, apply_directional_filter
    )
            
    out_ds = xr.Dataset(coords={c: ds.coords[c] for c in ds.coords if c != 'time'})
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
        
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs['history'] = f"{history}{sep}Full tidal analysis (periods: {periods_hours}h, m: {m_filters})."
    
    return out_ds
