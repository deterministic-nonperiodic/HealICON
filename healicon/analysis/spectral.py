"""
Spectral analysis: power spectra, cross-spectra, and kinetic energy spectra
using spherical harmonic transforms.
"""

import healpy as hp
import numpy as np
import xarray as xr

from ._common import (
    logger, EARTH_RADIUS_KM, _parse_units, _MAX_WORKERS,
    degree_to_wavelength, get_healpix_order, get_cells_dim, ensure_ring, append_history,
    ThreadPoolExecutor, get_progress_bar,
)
from ..cf_coords import _cf_guess


def _anafast_block(*args, lmax=None, is_nested=False, op_type='power'):
    """
    Internal helper function to compute the angular power spectrum (Cl) of one or more variables
    using spherical harmonics.
    
    Args:
        *args: One or more arrays of data (should be )
        lmax: Maximum spherical harmonic degree (optional)
        is_nested: Whether the input data is in nested order
        op_type: Type of spectrum to compute ('power', 'cross', 'kinetic')

    Returns:
        Dataset containing the power spectrum
    """
    data_blocks = args

    orig_shape = data_blocks[0].shape
    npix = orig_shape[-1]

    import warnings

    # Vectorize pre-processing (ordering, NaN checks, and mean filling) across all slices
    prepared_2ds = []
    valid_masks = []
    for block in data_blocks:
        d_2d = block.reshape(-1, npix)
        d_2d_ring = ensure_ring(d_2d, 'nested' if is_nested else 'ring')

        valid_mask = ~np.isnan(d_2d_ring)

        # Calculate mean for each map, ignoring all-NaN slice warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            means = np.nanmean(d_2d_ring, axis=-1, keepdims=True)

        d_filled = np.where(valid_mask, d_2d_ring, means)
        prepared_2ds.append(d_filled)
        valid_masks.append(valid_mask)

    # A slice is valid only if it has at least one valid pixel in every variable
    is_slice_valid = np.all([np.any(mask, axis=-1) for mask in valid_masks], axis=0)

    n_l = lmax + 1
    out_data = np.zeros((prepared_2ds[0].shape[0], n_l), dtype=prepared_2ds[0].dtype)

    # Pre-compute scalar kinetic factors once (outside thread loop)
    if op_type == 'kinetic-scalar':
        l = np.arange(lmax + 1)
        l_fac = np.zeros_like(l, dtype=float)
        l_fac[1:] = 1.0 / (l[1:] * (l[1:] + 1))
        R_sq = (EARTH_RADIUS_KM * 1000) ** 2

    def _process_slice(i):
        if not is_slice_valid[i]:
            out_data[i] = np.nan
            return
        d_filled_list = [d[i] for d in prepared_2ds]
        if op_type == 'power':
            out_data[i] = hp.anafast(d_filled_list[0], lmax=lmax)
        elif op_type == 'cross':
            out_data[i] = hp.anafast(d_filled_list[0], map2=d_filled_list[1], lmax=lmax)
        elif op_type == 'kinetic-spin1':
            alm_E, alm_B = hp.map2alm_spin([d_filled_list[0], d_filled_list[1]], spin=1, lmax=lmax)
            out_data[i] = 0.5 * (hp.alm2cl(alm_E) + hp.alm2cl(alm_B))
        elif op_type == 'kinetic-scalar':
            cl_D = hp.anafast(d_filled_list[0], lmax=lmax)
            cl_Z = hp.anafast(d_filled_list[1], lmax=lmax)
            out_data[i] = 0.5 * R_sq * l_fac * (cl_D + cl_Z)

    n = prepared_2ds[0].shape[0]
    if n > 1:
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = [pool.submit(_process_slice, i) for i in range(n)]
            for f in get_progress_bar(as_completed(futures), desc="Spectral analysis", total=n):
                f.result()
    else:
        _process_slice(0)

    out_shape = orig_shape[:-1] + (n_l,)
    return out_data.reshape(out_shape)


def compute_spectrum(ds: xr.Dataset, var_name: str | list[str] | None = None,
                     lmax: int | None = None, spectrum_type: str = 'power') -> xr.Dataset:
    """
    Computes the angular power spectrum (Cl) of one or more variables using spherical harmonics.
    If var_name is None, computes the spectrum for all data variables in the dataset that are defined
    on the HEALPix grid. Supported spectrum types: 'power' (default), 'cross', 'kinetic'

    Note:
        For computing kinetic energy spectra, using `spectrum_type="kinetic"` is superior to
        computing the "power" spectrum on both velocity components individually and averaging them
        via `(u_pow + v_pow) / 2`. While the difference may be small at small scales (high spherical
        harmonic degrees l), it is significant at large scales (low l). The reason is that
        velocity components u and v constitute a tangent vector field on the sphere rather than
        scalar fields. Treating them as independent scalars ignores the spherical geometry and coordinate
        singularities at the poles. The vector field must instead be decomposed using spin-weighted
        (spin-1) spherical harmonics to resolve it into coordinate-invariant gradient (E-mode/divergence)
        and curl (B-mode/vorticity) components. The true, coordinate-invariant kinetic energy spectrum
        is then given by the sum of these components: KE(l) = 0.5 * (C_l^E + C_l^B).

    Args:
        ds: Input dataset
        var_name: Name of the variable to compute the spectrum for (optional)
        lmax: Maximum spherical harmonic degree (optional)
        spectrum_type: Type of spectrum to compute ('power', 'cross', 'kinetic')

    Returns:
        Dataset containing the power spectrum
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
        u_name = _cf_guess(ds, "u")
        v_name = _cf_guess(ds, "v")
        div_name = _cf_guess(ds, "divergence")
        vor_name = _cf_guess(ds, "vorticity")

        if u_name and v_name:
            var_names = [u_name, v_name]
            logger.info(
                f"Auto-detected wind components '{u_name}' and '{v_name}' for kinetic energy spectrum.")
        elif div_name and vor_name:
            var_names = [div_name, vor_name]
            logger.info(f"Auto-detected '{div_name}' and '{vor_name}' for kinetic energy spectrum.")
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
            out_ds[f"{v}_cl"].attrs = ds[v].attrs.copy()
            orig_units = ds[v].attrs.get('units', '').strip()
            if orig_units:
                try:
                    new_units = str((_parse_units(orig_units) ** 2).units)
                except Exception:
                    # Fallback if pint fails to parse
                    new_units = f"({orig_units})2" if any(
                        c in orig_units for c in [' ', '/', '^', '-']) else f"{orig_units}2"
                out_ds[f"{v}_cl"].attrs['units'] = new_units

            orig_name = ds[v].attrs.get('long_name', ds[v].attrs.get('standard_name', v))
            out_ds[f"{v}_cl"].attrs['long_name'] = f"Power spectrum of {orig_name}"
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
                # Assume divergence and vorticity in s^-1
                op_type = 'kinetic-scalar'
            else:
                # Assume vector wind components (u, v) in m/s
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

    if spectrum_type == 'power':
        desc = f"power spectrum for {', '.join(var_names)}"
    elif spectrum_type == 'cross':
        desc = f"cross-spectrum for {var_names[0]} and {var_names[1]}"
    else:
        desc = f"kinetic energy spectrum from {var_names[0]} and {var_names[1]}"

    out_ds.attrs = append_history(out_ds.attrs, f"Computed {desc} up to lmax={lmax}.")
    return out_ds
