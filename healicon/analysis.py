import logging

import healpy as hp
import numpy as np
import xarray as xr

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


def _anafast_block(data_block, lmax):
    orig_shape = data_block.shape
    npix = orig_shape[-1]
    data_2d = data_block.reshape(-1, npix)

    n_l = lmax + 1
    out_data = np.zeros((data_2d.shape[0], n_l), dtype=data_2d.dtype)

    for i in range(data_2d.shape[0]):
        # anafast returns cl array of shape (lmax+1,)
        out_data[i] = hp.anafast(data_2d[i], lmax=lmax)

    out_shape = orig_shape[:-1] + (n_l,)
    return out_data.reshape(out_shape)


def compute_spectrum(ds: xr.Dataset, var_name: str, lmax: int = None) -> xr.Dataset:
    """
    Computes the angular power spectrum (Cl) of a variable using spherical harmonics.
    """
    if 'cell' not in ds.dims:
        raise ValueError("Dataset must have a 'cell' dimension.")

    npix = ds.sizes['cell']
    nside = hp.npix2nside(npix)

    if lmax is None:
        lmax = 3 * nside - 1

    logger.info(f"Computing power spectrum for '{var_name}' up to lmax={lmax}.")

    l_coords = np.arange(lmax + 1)

    da = xr.apply_ufunc(
        _anafast_block,
        ds[var_name],
        kwargs={'lmax': lmax},
        input_core_dims=[['cell']],
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
        if coord not in ['cell', 'l', 'lat', 'lon'] and coord in ds.coords:
            out_ds.coords[coord] = ds.coords[coord]

    out_ds.l.attrs = {"long_name": "Spherical harmonic degree (l)"}
    out_ds.attrs = ds.attrs
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs[
        'history'] = f"{history}{sep}Computed angular power spectrum (Cl) up to lmax={lmax} for {var_name}."
    return out_ds


def _filter_block(data_block, fwhm_rad, lmax):
    orig_shape = data_block.shape
    npix = orig_shape[-1]
    # Cache nside once per block rather than re-computing per iteration
    nside = hp.npix2nside(npix)
    data_2d = data_block.reshape(-1, npix)

    out_data = np.zeros_like(data_2d)

    for i in range(data_2d.shape[0]):
        if fwhm_rad is not None:
            out_data[i] = hp.smoothing(data_2d[i], fwhm=fwhm_rad)
        elif lmax is not None:
            alm = hp.map2alm(data_2d[i], lmax=lmax, iter=1)
            out_data[i] = hp.alm2map(alm, nside=nside)

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
    if 'cell' not in ds.dims:
        raise ValueError("Dataset must have a 'cell' dimension.")

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
        if 'cell' in ds[var].dims:
            da = xr.apply_ufunc(
                _filter_block,
                ds[var],
                kwargs={'fwhm_rad': fwhm_rad, 'lmax': lmax},
                input_core_dims=[['cell']],
                output_core_dims=[['cell']],
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


def _regrade_block(data_block, nside_out):
    orig_shape = data_block.shape
    npix_in = orig_shape[-1]
    data_2d = data_block.reshape(-1, npix_in)

    npix_out = hp.nside2npix(nside_out)
    out_data = np.zeros((data_2d.shape[0], npix_out), dtype=data_2d.dtype)

    for i in range(data_2d.shape[0]):
        # ud_grade preserves sum(map)/npix.
        out_data[i] = hp.ud_grade(data_2d[i], nside_out=nside_out)

    out_shape = orig_shape[:-1] + (npix_out,)
    return out_data.reshape(out_shape)


def regrade_resolution(ds: xr.Dataset, new_nside: int) -> xr.Dataset:
    """
    Upgrades or downgrades the HEALPix resolution of the dataset.
    """
    if 'cell' not in ds.dims:
        raise ValueError("Dataset must have a 'cell' dimension.")

    old_npix = ds.sizes['cell']
    old_nside = hp.npix2nside(old_npix)
    new_npix = hp.nside2npix(new_nside)

    logger.info(f"Regrading resolution from nside={old_nside} to nside={new_nside}.")

    target_lon, target_lat = hp.pix2ang(new_nside, np.arange(new_npix), lonlat=True)

    out_ds = xr.Dataset(
        coords={
            "cell": np.arange(new_npix),
            "lon": ("cell", target_lon),
            "lat": ("cell", target_lat)
        }
    )

    for var in ds.data_vars:
        if 'cell' in ds[var].dims:
            da = xr.apply_ufunc(
                _regrade_block,
                ds[var],
                kwargs={'nside_out': new_nside},
                input_core_dims=[['cell']],
                output_core_dims=[['cell']],
                exclude_dims=set(('cell',)),
                dask="parallelized",
                output_dtypes=[ds[var].dtype],
                dask_gufunc_kwargs={'output_sizes': {'cell': new_npix}, 'allow_rechunk': True}
            )
            out_ds[var] = da.assign_coords(cell=out_ds.cell, lon=out_ds.lon, lat=out_ds.lat)
            out_ds[var].attrs = ds[var].attrs
        else:
            out_ds[var] = ds[var]
            out_ds[var].attrs = ds[var].attrs

    for coord in ds.coords:
        if coord not in ['cell', 'lon', 'lat'] and coord in ds.coords:
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


def _helmholtz_block(u_block, v_block, lmax, nside):
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
        v_theta = -v_2d[i]  # healpy: theta points South  →  v points North  → v_theta = -v
        v_phi = u_2d[i]  # healpy: phi   points East   →  u points East   → v_phi   =  u

        almE, almB = hp.map2alm_spin([v_theta, v_phi], spin=1, lmax=lmax)

        # Rotational wind: keep only B-mode
        m_rot = hp.alm2map_spin([zeros.copy(), almB], nside, 1, lmax=lmax)
        u_rot[i] = m_rot[1]  # v_phi   →  u
        v_rot[i] = -m_rot[0]  # -v_theta →  v

        # Divergent wind: keep only E-mode
        m_div = hp.alm2map_spin([almE, zeros.copy()], nside, 1, lmax=lmax)
        u_div[i] = m_div[1]
        v_div[i] = -m_div[0]

        # Streamfunction ψ: ζ = ∇²ψ  →  ψ_lm = -almB_lm / fl  (× a for m²/s)
        psi_alm = np.where(l_arr > 0, -almB / fl_safe, 0.0 + 0.0j)
        psi[i] = hp.alm2map(psi_alm, nside, lmax=lmax) * _EARTH_RADIUS_M

        # Velocity potential χ: D = ∇²χ  →  χ_lm = almE_lm / fl  (× a for m²/s)
        chi_alm = np.where(l_arr > 0, almE / fl_safe, 0.0 + 0.0j)
        chi[i] = hp.alm2map(chi_alm, nside, lmax=lmax) * _EARTH_RADIUS_M

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
    if 'cell' not in ds.dims:
        raise ValueError("Dataset must have a 'cell' dimension.")

    npix = ds.sizes['cell']
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
        kwargs={'lmax': lmax, 'nside': nside},
        input_core_dims=[['cell'], ['cell']],
        output_core_dims=[['cell']] * 6,
        dask="parallelized",
        output_dtypes=[dtype] * 6,
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

    coords = ds.coords
    wind_attrs_base = {'units': ds[u_var].attrs.get('units', 'm s-1'), 'grid_mapping': 'healpix'}

    out_ds = xr.Dataset(coords=coords)
    out_ds['u_rot'] = u_rot.assign_coords(cell=ds.cell)
    out_ds['u_rot'].attrs = {**wind_attrs_base,
                             'long_name': 'Rotational (non-divergent) eastward wind'}
    out_ds['v_rot'] = v_rot.assign_coords(cell=ds.cell)
    out_ds['v_rot'].attrs = {**wind_attrs_base,
                             'long_name': 'Rotational (non-divergent) northward wind'}
    out_ds['u_div'] = u_div.assign_coords(cell=ds.cell)
    out_ds['u_div'].attrs = {**wind_attrs_base,
                             'long_name': 'Divergent (irrotational) eastward wind'}
    out_ds['v_div'] = v_div.assign_coords(cell=ds.cell)
    out_ds['v_div'].attrs = {**wind_attrs_base,
                             'long_name': 'Divergent (irrotational) northward wind'}

    if include_psi:
        out_ds['psi'] = psi.assign_coords(cell=ds.cell)
        out_ds['psi'].attrs = {
            'standard_name': 'atmosphere_horizontal_streamfunction',
            'long_name': 'Streamfunction',
            'units': 'm2 s-1',
            'grid_mapping': 'healpix',
        }

    if include_chi:
        out_ds['chi'] = chi.assign_coords(cell=ds.cell)
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


def _vorticity_divergence_block(u_block, v_block, lmax, nside):
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
        # HEALPix convention: theta points South, phi points East
        # u is Eastward -> v_phi
        # v is Northward -> -v_theta
        v_theta = -v_2d[i]
        v_phi = u_2d[i]

        # map2alm_spin returns E and B modes for spin=1
        almE, almB = hp.map2alm_spin([v_theta, v_phi], spin=1, lmax=lmax)

        div_alm = -fl * almE
        vor_alm = fl * almB

        # Transform back to map space
        div_out[i] = hp.alm2map(div_alm, nside, lmax=lmax)
        vor_out[i] = hp.alm2map(vor_alm, nside, lmax=lmax)

    out_shape = orig_shape
    return div_out.reshape(out_shape), vor_out.reshape(out_shape)


def compute_vorticity_divergence(ds: xr.Dataset, u_var: str, v_var: str,
                                 lmax: int | None = None) -> xr.Dataset:
    """
    Computes horizontal vorticity and divergence from U and V wind components.
    """
    if 'cell' not in ds.dims:
        raise ValueError("Dataset must have a 'cell' dimension.")

    npix = ds.sizes['cell']
    nside = hp.npix2nside(npix)

    if lmax is None:
        lmax = 3 * nside - 1

    logger.info(f"Computing vorticity and divergence from '{u_var}' and '{v_var}' (lmax={lmax}).")

    div, vor = xr.apply_ufunc(
        _vorticity_divergence_block,
        ds[u_var], ds[v_var],
        kwargs={'lmax': lmax, 'nside': nside},
        input_core_dims=[['cell'], ['cell']],
        output_core_dims=[['cell'], ['cell']],
        dask="parallelized",
        output_dtypes=[ds[u_var].dtype, ds[v_var].dtype],
        dask_gufunc_kwargs={'allow_rechunk': True}
    )

    out_ds = xr.Dataset(coords=ds.coords)
    out_ds['divergence'] = div.assign_coords(cell=ds.cell)
    out_ds['divergence'].attrs = {'standard_name': 'divergence_of_wind', 'units': 's-1',
                                  'grid_mapping': 'healpix'}

    out_ds['vorticity'] = vor.assign_coords(cell=ds.cell)
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
