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


def _anafast_block(*args, lmax=None, is_nested=False, op_type='power'):
    # Extract data arrays from args (kwargs are passed separately)
    data_blocks = args

    orig_shape = data_blocks[0].shape
    npix = orig_shape[-1]

    # Reshape all inputs to 2D
    data_2ds = [block.reshape(-1, npix) for block in data_blocks]

    n_l = lmax + 1
    out_data = np.zeros((data_2ds[0].shape[0], n_l), dtype=data_2ds[0].dtype)

    for i in range(data_2ds[0].shape[0]):
        # Fill missing values and ensure ring ordering for all maps
        d_filled_list = []
        all_valid = True
        for data_2d in data_2ds:
            d = ensure_ring(data_2d[i], 'nested' if is_nested else 'ring')
            valid_mask = ~np.isnan(d)
            if not np.any(valid_mask):
                all_valid = False
                break
            d_filled_list.append(np.where(valid_mask, d, np.nanmean(d)))

        if not all_valid:
            out_data[i] = np.nan
            continue

        if op_type == 'power':
            out_data[i] = hp.anafast(d_filled_list[0], lmax=lmax)
        elif op_type == 'cross':
            out_data[i] = hp.anafast(d_filled_list[0], map2=d_filled_list[1], lmax=lmax)
        elif op_type == 'kinetic-spin1':
            # Spin-1 transform for u and v wind components
            alm_E, alm_B = hp.map2alm_spin([d_filled_list[0], d_filled_list[1]], spin=1, lmax=lmax)
            cl_E = hp.alm2cl(alm_E)
            cl_B = hp.alm2cl(alm_B)
            # Kinetic energy spectrum
            out_data[i] = 0.5 * (cl_E + cl_B)
        elif op_type == 'kinetic-scalar':
            # Scalar power spectra for divergence and vorticity
            cl_D = hp.anafast(d_filled_list[0], lmax=lmax)
            cl_Z = hp.anafast(d_filled_list[1], lmax=lmax)

            l = np.arange(lmax + 1)
            # Avoid division by zero at l=0
            l_fac = np.zeros_like(l, dtype=float)
            l_fac[1:] = 1.0 / (l[1:] * (l[1:] + 1))

            # Earth radius squared (in meters)
            R_sq = (EARTH_RADIUS_KM * 1000) ** 2

            out_data[i] = 0.5 * R_sq * l_fac * (cl_D + cl_Z)
        else:
            raise ValueError(f"Unknown operation type: {op_type}")

    out_shape = orig_shape[:-1] + (n_l,)
    return out_data.reshape(out_shape)


def compute_spectrum(ds: xr.Dataset, var_name: str | list[str] | None = None,
                     lmax: int | None = None, spectrum_type: str = 'power') -> xr.Dataset:
    """
    Computes the angular power spectrum (Cl) of one or more variables using spherical harmonics.
    If var_name is None, computes the spectrum for all data variables in the dataset that are defined on the HEALPix grid.
    Supported spectrum types: 'power' (default), 'cross', 'kinetic'
    """
    cell_dim = get_cells_dim(ds)
    is_nested = get_healpix_order(ds) == 'nested'
    npix = ds.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    if lmax is None:
        lmax = 3 * nside - 1

    var_names = []
    if var_name is not None:
        if isinstance(var_name, str):
            var_names = [var_name]
        else:
            var_names = list(var_name)

    # Auto-detect variables for kinetic energy if not explicitly provided
    if spectrum_type == 'kinetic' and not var_names:
        if 'u' in ds and 'v' in ds:
            var_names = ['u', 'v']
            logger.info("Auto-detected wind components 'u' and 'v' for kinetic energy spectrum.")
        elif 'divergence' in ds and 'vorticity' in ds:
            var_names = ['divergence', 'vorticity']
            logger.info("Auto-detected 'divergence' and 'vorticity' for kinetic energy spectrum.")
        elif 'div' in ds and 'vor' in ds:
            var_names = ['div', 'vor']
            logger.info("Auto-detected 'div' and 'vor' for kinetic energy spectrum.")
        else:
            raise ValueError(
                "Could not auto-detect variables for kinetic energy spectrum. Please specify using --var.")

    if not var_names and spectrum_type == 'power':
        exclude = {'lon', 'lat', 'clon', 'clat'}
        for v in ds.data_vars:
            if cell_dim in ds[v].dims:
                v_str = str(v)
                if v_str not in exclude and not v_str.endswith('_bnds') and not v_str.endswith(
                        '_bounds'):
                    var_names.append(v_str)
        if not var_names:
            raise ValueError(
                f"No variables found with spatial dimension '{cell_dim}' to compute spectrum.")

    if spectrum_type in ('cross', 'kinetic') and len(var_names) != 2:
        raise ValueError(
            f"Spectrum type '{spectrum_type}' requires exactly 2 variables, but got {len(var_names)}: {var_names}")

    l_coords = np.arange(lmax + 1)
    out_ds = xr.Dataset(coords={'l': l_coords})

    if spectrum_type == 'power':
        for v in var_names:
            logger.info(f"Computing power spectrum for '{v}' up to lmax={lmax}.")
            da = xr.apply_ufunc(
                _anafast_block,
                ds[v],
                kwargs={'lmax': lmax, 'is_nested': is_nested, 'op_type': 'power'},
                input_core_dims=[[cell_dim]],
                output_core_dims=[['l']],
                dask="parallelized",
                output_dtypes=[ds[v].dtype],
                dask_gufunc_kwargs={'output_sizes': {'l': len(l_coords)}, 'allow_rechunk': True}
            )
            out_ds[f"{v}_cl"] = da
    else:
        v1, v2 = var_names
        if spectrum_type == 'cross':
            logger.info(f"Computing cross-spectrum for '{v1}' and '{v2}' up to lmax={lmax}.")
            op_type = 'cross'
            out_name = f"{v1}_{v2}_cl"
        else:  # kinetic
            logger.info(
                f"Computing kinetic energy spectrum using '{v1}' and '{v2}' up to lmax={lmax}.")
            # Check units to decide between wind components (spin-1) and div/vor (scalar)
            units1 = str(ds[v1].attrs.get('units', '')).strip().lower()
            if units1 in ('s-1', 's^-1', '1/s') or v1 in ('divergence', 'div', 'vorticity', 'vor'):
                op_type = 'kinetic-scalar'
            else:
                op_type = 'kinetic-spin1'
            out_name = "kinetic_energy_cl"

        da = xr.apply_ufunc(
            _anafast_block,
            ds[v1], ds[v2],
            kwargs={'lmax': lmax, 'is_nested': is_nested, 'op_type': op_type},
            input_core_dims=[[cell_dim], [cell_dim]],
            output_core_dims=[['l']],
            dask="parallelized",
            output_dtypes=[ds[v1].dtype],
            dask_gufunc_kwargs={'output_sizes': {'l': len(l_coords)}, 'allow_rechunk': True}
        )
        out_ds[out_name] = da
        if spectrum_type == 'kinetic':
            out_ds[out_name].attrs['units'] = 'm2 s-2'
            out_ds[out_name].attrs['long_name'] = 'Kinetic Energy Spectrum'

    # Attach wavelength as a secondary coordinate on l for convenience
    wavelength_km = degree_to_wavelength(np.maximum(l_coords, 1))  # avoid l=0 singularity
    out_ds = out_ds.assign_coords(wavelength_km=('l', wavelength_km))
    out_ds['wavelength_km'].attrs = {
        'long_name': 'Equivalent wavelength',
        'units': 'km',
    }

    first_var = var_names[0]
    for coord in ds[first_var].coords:
        if coord not in [cell_dim, 'l', 'lat', 'lon'] and coord in ds.coords:
            out_ds.coords[coord] = ds.coords[coord]

    out_ds.l.attrs = {"long_name": "Spherical harmonic degree (l)"}
    out_ds.attrs = ds.attrs
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""

    if spectrum_type == 'power':
        desc = f"power spectrum for {', '.join(var_names)}"
    elif spectrum_type == 'cross':
        desc = f"cross-spectrum for {var_names[0]} and {var_names[1]}"
    else:
        desc = f"kinetic energy spectrum from {var_names[0]} and {var_names[1]}"

    out_ds.attrs['history'] = f"{history}{sep}Computed {desc} up to lmax={lmax}."
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
        valid_mask = ~np.isnan(d)
        if not np.any(valid_mask):
            out_data[i] = np.nan
            continue
        d_filled = np.where(valid_mask, d, np.nanmean(d))

        if fwhm_rad is not None:
            filtered = hp.smoothing(d_filled, fwhm=fwhm_rad)
        elif lmax is not None:
            alm = hp.map2alm(d_filled, lmax=lmax, iter=3)
            filtered = hp.alm2map(alm, nside=nside)

        filtered = np.where(valid_mask, filtered, np.nan)
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
        d = data_2d[i]
        valid_mask = ~np.isnan(d)
        if not np.any(valid_mask):
            out_data[i] = np.nan
            continue
        d_unseen = np.where(valid_mask, d, hp.UNSEEN)
        regraded = hp.ud_grade(d_unseen, nside_out=nside_out, order_in=order, order_out=order)
        out_data[i] = np.where(regraded == hp.UNSEEN, np.nan, regraded)

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

        valid_mask = ~(np.isnan(u_ring) | np.isnan(v_ring))
        if not np.any(valid_mask):
            u_rot[i] = np.nan
            v_rot[i] = np.nan
            u_div[i] = np.nan
            v_div[i] = np.nan
            psi[i] = np.nan
            chi[i] = np.nan
            continue

        u_filled = np.where(valid_mask, u_ring, 0.0)
        v_filled = np.where(valid_mask, v_ring, 0.0)

        v_theta = -v_filled
        v_phi = u_filled

        almE, almB = hp.map2alm_spin([v_theta, v_phi], spin=1, lmax=lmax)

        # Rotational wind: keep only B-mode
        m_rot = hp.alm2map_spin([zeros.copy(), almB], nside, 1, lmax=lmax)
        u_rot_ring = np.where(valid_mask, m_rot[1], np.nan)
        v_rot_ring = np.where(valid_mask, -m_rot[0], np.nan)

        # Divergent wind: keep only E-mode
        m_div = hp.alm2map_spin([almE, zeros.copy()], nside, 1, lmax=lmax)
        u_div_ring = np.where(valid_mask, m_div[1], np.nan)
        v_div_ring = np.where(valid_mask, -m_div[0], np.nan)

        u_rot[i] = ensure_original_order(u_rot_ring, 'nested' if is_nested else 'ring')
        v_rot[i] = ensure_original_order(v_rot_ring, 'nested' if is_nested else 'ring')
        u_div[i] = ensure_original_order(u_div_ring, 'nested' if is_nested else 'ring')
        v_div[i] = ensure_original_order(v_div_ring, 'nested' if is_nested else 'ring')

        # Streamfunction ψ: ζ = ∇²ψ  →  ψ_lm = -almB_lm / fl  (× a for m²/s)
        psi_alm = np.where(l_arr > 0, -almB / fl_safe, 0.0 + 0.0j)
        psi_map = np.where(valid_mask, hp.alm2map(psi_alm, nside, lmax=lmax) * _EARTH_RADIUS_M,
                           np.nan)
        psi[i] = ensure_original_order(psi_map, 'nested' if is_nested else 'ring')

        # Velocity potential χ: D = ∇²χ  →  χ_lm = almE_lm / fl  (× a for m²/s)
        chi_alm = np.where(l_arr > 0, almE / fl_safe, 0.0 + 0.0j)
        chi_map = np.where(valid_mask, hp.alm2map(chi_alm, nside, lmax=lmax) * _EARTH_RADIUS_M,
                           np.nan)
        chi[i] = ensure_original_order(chi_map, 'nested' if is_nested else 'ring')

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

        valid_mask = ~(np.isnan(u_ring) | np.isnan(v_ring))
        if not np.any(valid_mask):
            div_out[i] = np.nan
            vor_out[i] = np.nan
            continue

        u_filled = np.where(valid_mask, u_ring, 0.0)
        v_filled = np.where(valid_mask, v_ring, 0.0)

        v_theta = -v_filled
        v_phi = u_filled

        # map2alm_spin returns E and B modes for spin=1
        almE, almB = hp.map2alm_spin([v_theta, v_phi], spin=1, lmax=lmax)

        R = EARTH_RADIUS_KM * 1000.0
        div_alm = -fl * almE / R
        vor_alm = fl * almB / R

        # Transform back to map space
        div_map = np.where(valid_mask, hp.alm2map(div_alm, nside, lmax=lmax), np.nan)
        vor_map = np.where(valid_mask, hp.alm2map(vor_alm, nside, lmax=lmax), np.nan)

        div_out[i] = ensure_original_order(div_map, 'nested' if is_nested else 'ring')
        vor_out[i] = ensure_original_order(vor_map, 'nested' if is_nested else 'ring')

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

        valid_mask = ~(np.isnan(div_ring) | np.isnan(vor_ring))
        if not np.any(valid_mask):
            u_out[i] = np.nan
            v_out[i] = np.nan
            continue

        div_filled = np.where(valid_mask, div_ring, 0.0)
        vor_filled = np.where(valid_mask, vor_ring, 0.0)

        # Convert to alm
        div_alm = hp.map2alm(div_filled, lmax=lmax)
        vor_alm = hp.map2alm(vor_filled, lmax=lmax)

        # Get E and B modes
        # Div = -fl * E / R -> E = -Div * R / fl
        # Vor = fl * B / R -> B = Vor * R / fl
        R = EARTH_RADIUS_KM * 1000.0
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
    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs['history'] = f"{history}{sep}Computed U and V from vorticity and divergence."

    return out_ds


def _directional_filter_block(a_block, b_block, target_m, lmax, is_nested):
    """
    Apply a directional spatial filter to isolate specific zonal wavenumbers (m) 
    and propagation directions (eastward/westward) using Spherical Harmonics.

    Mathematical Formulation:
    Given a temporal Fourier decomposition of a field at frequency omega:
        T(t, x) = A(x) * cos(omega * t) + B(x) * sin(omega * t)
    
    We want to isolate a wave traveling in a specific direction with zonal wavenumber m.
    A pure traveling wave takes the form: cos(m * lambda +/- omega * t)
    
    Expanding this using trigonometric identities:
    - Westward (m > 0):  cos(m*lambda - omega*t) =  cos(m*lambda)*cos(omega*t) + sin(m*lambda)*sin(omega*t)
                         Here, A ~ cos(m*lambda) and B ~ sin(m*lambda)
    - Eastward (m < 0): cos(|m|*lambda + omega*t) = cos(|m|*lambda)*cos(omega*t) - sin(|m|*lambda)*sin(omega*t)
                         Here, A ~ cos(|m|*lambda) and B ~ -sin(|m|*lambda)

    In the Spherical Harmonic (ALM) domain, the basis functions are Y_l^m ~ exp(i * m * lambda).
    To extract the directional components from the full fields A and B:
    1. Transform A and B to the spherical harmonic domain: alm_a and alm_b.
    2. For a target wavenumber |m|, mask out all other wavenumbers.
    3. Apply the phase relationship between A and B to isolate the direction:
       - For m > 0 (Westward): 
           alm_a_out = 0.5 * (alm_a + i * alm_b)
           alm_b_out = -i * alm_a_out
       - For m < 0 (Eastward): 
           alm_a_out = 0.5 * (alm_a - i * alm_b)
           alm_b_out = i * alm_a_out
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

    for i in range(a_2d.shape[0]):
        a_ring = ensure_ring(a_2d[i], 'nested' if is_nested else 'ring')
        b_ring = ensure_ring(b_2d[i], 'nested' if is_nested else 'ring')

        valid_mask = ~(np.isnan(a_ring) | np.isnan(b_ring))
        if not np.any(valid_mask):
            out_a[i] = np.nan
            out_b[i] = np.nan
            continue

        a_filled = np.where(valid_mask, a_ring, 0.0)
        b_filled = np.where(valid_mask, b_ring, 0.0)

        alm_a = hp.map2alm(a_filled, lmax=lmax, iter=3)
        alm_b = hp.map2alm(b_filled, lmax=lmax, iter=3)

        if target_m > 0:
            alm_a_out = 0.5 * (alm_a + 1j * alm_b) * mask
            alm_b_out = -1j * alm_a_out
        elif target_m < 0:
            alm_a_out = 0.5 * (alm_a - 1j * alm_b) * mask
            alm_b_out = 1j * alm_a_out
        else:
            alm_a_out = alm_a * mask
            alm_b_out = alm_b * mask

        a_filtered = np.where(valid_mask, hp.alm2map(alm_a_out, nside=nside), np.nan)
        b_filtered = np.where(valid_mask, hp.alm2map(alm_b_out, nside=nside), np.nan)

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
                                     sym_idx_da: xr.DataArray, phi_da: xr.DataArray,
                                     apply_filter_fn) -> dict:
    """
    Decomposes the cosine and sine tidal coefficients into symmetric/antisymmetric 
    amplitudes and phases, optionally filtering by specific wavenumbers.
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


def compute_tidal_analysis(ds: xr.Dataset, var_name: str, periods_hours: list[float],
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
            Positive values denote westward propagation, negative values eastward. 
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

    history = ds.attrs.get('history', '')
    sep = "\n" if history else ""
    out_ds.attrs[
        'history'] = f"{history}{sep}Full tidal analysis (periods: {periods_hours}h, m: {m_filters})."

    return out_ds
