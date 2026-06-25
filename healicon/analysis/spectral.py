"""
Spectral analysis: power spectra, cross-spectra, and kinetic energy spectra
using spherical harmonic transforms.
"""

import healpy as hp
import numpy as np
import xarray as xr

from ._common import (
    logger, EARTH_RADIUS_KM, _parse_units, degree_to_wavelength, get_healpix_order, get_cells_dim,
    ensure_ring, append_history,
)
from ..cf_coords import _cf_guess


def _anafast_block(*args, lmax=None, is_nested=False, op_type='power'):
    """
    Internal helper function to compute the angular power spectrum (Cl) of one or more variables
    using spherical harmonics.

    Args:
        *args: One or more arrays of data (should be HEALPix maps)
        lmax: Maximum spherical harmonic degree (optional)
        is_nested: Whether the input data is in nested order
        op_type: Type of spectrum to compute ('power', 'cross', 'kinetic-spin1', 'kinetic-scalar')

    Returns:
        Array of shape (*leading_dims, lmax+1) containing the power spectrum.
    """
    data_blocks = args

    orig_shape = data_blocks[0].shape
    npix = orig_shape[-1]
    n_l = lmax + 1

    # --- Pre-processing: ordering, NaN fill, and validity mask ---
    prepared_2ds = []
    valid_masks = []
    for block in data_blocks:
        d_2d = block.reshape(-1, npix)
        d_2d_ring = ensure_ring(d_2d, 'nested' if is_nested else 'ring')

        valid_mask = ~np.isnan(d_2d_ring)  # shape (n, npix)

        # Fill NaN pixels with the per-slice mean.
        # Use nansum/count instead of nanmean to avoid RuntimeWarning on all-NaN slices.
        # All-NaN rows get fill=0.0, which is harmless: they are already flagged invalid below.
        count = np.maximum(valid_mask.sum(axis=-1, keepdims=True), 1)
        means = np.nansum(d_2d_ring, axis=-1, keepdims=True) / count

        prepared_2ds.append(np.where(valid_mask, d_2d_ring, means))
        valid_masks.append(valid_mask)

    n = prepared_2ds[0].shape[0]

    # Vectorised validity check: a slice is valid when every variable has ≥1 finite pixel
    # valid_masks[k] has shape (n, npix); any(axis=-1) → (n,); stack + all(axis=0) → (n,)
    is_slice_valid = np.all(
        np.stack([m.any(axis=-1) for m in valid_masks], axis=0), axis=0
    )  # shape (n,)

    # Output is always float64 – spectral coefficients require full precision
    out_data = np.empty((n, n_l), dtype=np.float64)
    out_data[~is_slice_valid] = np.nan

    # Pre-compute l-dependent kinetic-scalar factor once per call
    if op_type == 'kinetic-scalar':
        l_arr = np.arange(n_l, dtype=np.float64)
        l_fac = np.zeros(n_l, dtype=np.float64)
        l_fac[1:] = 1.0 / (l_arr[1:] * (l_arr[1:] + 1.0))
        R_sq = (EARTH_RADIUS_KM * 1_000.0) ** 2

    def _process_slice(i):
        slices = [d[i] for d in prepared_2ds]
        if op_type == 'power':
            out_data[i] = hp.anafast(slices[0], lmax=lmax)
        elif op_type == 'cross':
            out_data[i] = hp.anafast(slices[0], map2=slices[1], lmax=lmax)
        elif op_type == 'kinetic-spin1':
            alm_E, alm_B = hp.map2alm_spin([slices[0], slices[1]], spin=1, lmax=lmax)
            out_data[i] = 0.5 * (hp.alm2cl(alm_E) + hp.alm2cl(alm_B))
        elif op_type == 'kinetic-scalar':
            cl_D = hp.anafast(slices[0], lmax=lmax)
            cl_Z = hp.anafast(slices[1], lmax=lmax)
            out_data[i] = 0.5 * R_sq * l_fac * (cl_D + cl_Z)

    valid_indices = np.where(is_slice_valid)[0]

    if len(valid_indices) == 0:
        pass  # all slices invalid – out_data already filled with NaN
    else:
        for i in valid_indices:
            _process_slice(i)

    return out_data.reshape(orig_shape[:-1] + (n_l,))


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

    # --- Resolve variable names ---
    if var_name is None:
        var_names = []
    elif isinstance(var_name, str):
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
        var_names = [
            str(v) for v in ds.data_vars
            if cell_dim in ds[v].dims
               and str(v) not in exclude
               and not str(v).endswith('_bnds')
               and not str(v).endswith('_bounds')
        ]
        if not var_names:
            raise ValueError(
                f"No variables found with spatial dimension '{cell_dim}' to compute spectrum.")

    if spectrum_type in ('cross', 'kinetic') and len(var_names) != 2:
        raise ValueError(
            f"Spectrum type '{spectrum_type}' requires exactly 2 variables, "
            f"but got {len(var_names)}: {var_names}"
        )

    l_coords = np.arange(lmax + 1)
    out_ds = xr.Dataset(coords={'l': l_coords})

    # Shared apply_ufunc kwargs
    _ufunc_kw = dict(
        dask="parallelized",
        output_core_dims=[['l']],
        dask_gufunc_kwargs={'output_sizes': {'l': len(l_coords)}, 'allow_rechunk': True},
    )

    if spectrum_type == 'power':
        for v in var_names:
            logger.info(f"Computing power spectrum for '{v}' up to lmax={lmax}.")
            da = xr.apply_ufunc(
                _anafast_block,
                ds[v],
                kwargs={'lmax': lmax, 'is_nested': is_nested, 'op_type': 'power'},
                input_core_dims=[[cell_dim]],
                output_dtypes=[np.float64],
                **_ufunc_kw,
            )
            cl_name = f"{v}_cl"
            out_ds[cl_name] = da
            attrs = ds[v].attrs.copy()
            orig_units = attrs.get('units', '').strip()
            if orig_units:
                try:
                    attrs['units'] = str((_parse_units(orig_units) ** 2).units)
                except Exception:
                    has_special = any(c in orig_units for c in (' ', '/', '^', '-'))
                    attrs['units'] = f"({orig_units})2" if has_special else f"{orig_units}2"
            orig_name = attrs.get('long_name', attrs.get('standard_name', v))
            attrs['long_name'] = f"Power spectrum of {orig_name}"
            out_ds[cl_name].attrs = attrs

    else:
        v1, v2 = var_names
        if spectrum_type == 'cross':
            logger.info(f"Computing cross-spectrum for '{v1}' and '{v2}' up to lmax={lmax}.")
            op_type = 'cross'
            out_name = f"{v1}_{v2}_cl"
        else:  # kinetic
            logger.info(
                f"Computing kinetic energy spectrum using '{v1}' and '{v2}' up to lmax={lmax}.")
            units1 = str(ds[v1].attrs.get('units', '')).strip().lower()
            op_type = (
                'kinetic-scalar'
                if units1 in ('s-1', 's^-1', '1/s') or v1 in ('divergence', 'div', 'vorticity',
                                                              'vor')
                else 'kinetic-spin1'
            )
            out_name = "kinetic_energy_cl"

        da = xr.apply_ufunc(
            _anafast_block,
            ds[v1], ds[v2],
            kwargs={'lmax': lmax, 'is_nested': is_nested, 'op_type': op_type},
            input_core_dims=[[cell_dim], [cell_dim]],
            output_dtypes=[np.float64],
            **_ufunc_kw,
        )
        out_ds[out_name] = da
        if spectrum_type == 'kinetic':
            out_ds[out_name].attrs.update(
                {'units': 'm2 s-2', 'long_name': 'Kinetic Energy Spectrum'})

    # Attach wavelength as a secondary coordinate on l for convenience
    wavelength_km = degree_to_wavelength(np.maximum(l_coords, 1))  # avoid l=0 singularity
    out_ds = out_ds.assign_coords(wavelength_km=('l', wavelength_km))
    out_ds['wavelength_km'].attrs = {
        'long_name': 'Equivalent wavelength',
        'units': 'km',
    }

    # Forward non-spatial coordinates from the first variable.
    # Exclude HEALPix grid coordinates — the output lives in l-space, not pixel-space.
    first_var = var_names[0]
    _healpix_coord_prefixes = ('healpix_', 'grid_mapping')
    coords_to_skip = {cell_dim, 'l', 'lat', 'lon'}
    coords_to_skip.update(
        c for c in ds[first_var].coords
        if any(c.startswith(p) for p in _healpix_coord_prefixes)
    )
    for coord in ds[first_var].coords:
        if coord not in coords_to_skip and coord in ds.coords:
            out_ds.coords[coord] = ds.coords[coord]

    out_ds.l.attrs = {"long_name": "Spherical harmonic degree (l)"}

    # Copy global attrs but drop any HEALPix / grid-mapping references that are
    # no longer meaningful for an l-space dataset.
    _healpix_attr_prefixes = ('healpix_', 'grid_mapping')
    out_ds.attrs = {
        k: v for k, v in ds.attrs.items()
        if not any(k.startswith(p) for p in _healpix_attr_prefixes)
    }

    if spectrum_type == 'power':
        desc = f"power spectrum for {', '.join(var_names)}"
    elif spectrum_type == 'cross':
        desc = f"cross-spectrum for {var_names[0]} and {var_names[1]}"
    else:
        desc = f"kinetic energy spectrum from {var_names[0]} and {var_names[1]}"

    out_ds.attrs = append_history(out_ds.attrs, f"Computed {desc} up to lmax={lmax}.")
    return out_ds
