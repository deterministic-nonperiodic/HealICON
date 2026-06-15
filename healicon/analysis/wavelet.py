"""
HealICON Wavelet Analysis Module
Contains an optimized, xarray-native implementation of the Fourier-Wavelet spectrum,
based on Torrence & Compo (1998) and Y. Yamazaki (2023).

References:
Torrence, C. and G. P. Compo, 1998: A Practical Guide to Wavelet Analysis. 
Bull. Amer. Meteor. Soc., 79, 61--78.

Yamazaki, Y., 2023: A method to derive Fourier--wavelet spectra for the 
characterization of global-scale waves in the mesosphere and lower thermosphere.
Geosci. Model Dev., 16, 4749--4766.
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import healpy as hp
import numpy as np
import xarray as xr
from scipy.optimize import fminbound
from scipy.special import gamma, gammainc

from healicon.analysis import _get_symmetric_pixels
from healicon.grid import (get_cells_dim, get_healpix_order,
                           append_history, add_healpix_grid_mapping,
                           get_healpix_coords, ensure_original_order)

logger = logging.getLogger(__name__)
logging.getLogger('healpy').setLevel(logging.WARNING)


@lru_cache(maxsize=16)
def _get_basis_maps(nside: int, lmax: int, abs_m: int):
    """Precompute cos/sin spatial basis maps for a single zonal wavenumber.
 
    Returns (basis_cos, basis_sin) each of shape (n_l, npix), where n_l is
    the number of SH degrees with order m: n_l = lmax - abs_m + 1.
 
    The result is cached so that repeated calls with the same
    (nside, lmax, abs_m) are free.
    """
    npix = hp.nside2npix(nside)
    n_alm_total = hp.Alm.getsize(lmax)
    _, m_arr = hp.Alm.getlm(lmax)
    idx_m = np.where(m_arr == abs_m)[0]
    n_l = len(idx_m)

    basis_cos = np.zeros((n_l, npix))
    basis_sin = np.zeros((n_l, npix))

    def _synth_one(k):
        alm_c = np.zeros(n_alm_total, dtype=complex)
        alm_c[idx_m[k]] = 1.0
        cos_map = hp.alm2map(alm_c, nside=nside)

        alm_s = np.zeros(n_alm_total, dtype=complex)
        alm_s[idx_m[k]] = 1j
        sin_map = hp.alm2map(alm_s, nside=nside)
        return k, cos_map, sin_map

    n_workers = min(32, os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for k, cos_map, sin_map in pool.map(_synth_one, range(n_l)):
            basis_cos[k] = cos_map
            basis_sin[k] = sin_map

    return basis_cos, basis_sin


# --------------------------------------------------------------------------
# CORE WAVELET FUNCTIONS
# --------------------------------------------------------------------------

def wave_bases(mother, k, scale, param):
    """
    Computes the wavelet function as a function of Fourier frequency.
    
    Args:
        mother: Name of the mother wavelet ('MORLET', 'PAUL', or 'DOG').
        k: Array of Fourier frequencies.
        scale: Scale parameter (can be scalar or array).
        param: Additional parameter for the wavelet.
    
    Returns:
        Complex-valued wavelet function and degrees of freedom minimum.
    """
    n = len(k)
    kplus = np.array(k > 0., dtype=float)
    scale_is_scalar = np.isscalar(scale)
    scale = np.atleast_1d(scale)

    if mother == 'MORLET':
        if param == -1: param = 6.
        k0 = np.copy(param)
        expnt = -(scale[:, None] * k[None, :] - k0) ** 2 / 2. * kplus[None, :]
        norm = np.sqrt(scale * k[1]) * (np.pi ** (-0.25)) * np.sqrt(n)
        daughter = norm[:, None] * np.exp(expnt) * kplus[None, :]
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
        coi = fourier_factor / np.sqrt(2)
        dofmin = 2
    elif mother == 'PAUL':
        if param == -1: param = 4.
        m = param
        expnt = -scale[:, None] * k[None, :] * kplus[None, :]
        norm_bottom = np.sqrt(m * np.prod(np.arange(1, (2 * m))))
        norm = np.sqrt(scale * k[1]) * (2 ** m / norm_bottom) * np.sqrt(n)
        daughter = norm[:, None] * ((scale[:, None] * k[None, :]) ** m) * np.exp(expnt) * kplus[
            None, :]
        fourier_factor = 4 * np.pi / (2 * m + 1)
        coi = fourier_factor * np.sqrt(2)
        dofmin = 2
    elif mother == 'DOG':
        if param == -1: param = 2.
        m = param
        expnt = -(scale[:, None] * k[None, :]) ** 2 / 2.0
        norm = np.sqrt(scale * k[1] / gamma(m + 0.5)) * np.sqrt(n)
        daughter = -norm[:, None] * (1j ** m) * ((scale[:, None] * k[None, :]) ** m) * np.exp(expnt)
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
        coi = fourier_factor / np.sqrt(2)
        dofmin = 1
    else:
        raise ValueError('Mother must be one of MORLET, PAUL, DOG')

    if scale_is_scalar:
        daughter = daughter[0]

    return daughter, fourier_factor, coi, dofmin


def chisquare_solve(XGUESS, P, V):
    PGUESS = gammainc(V / 2, V * XGUESS / 2)
    PDIFF = np.abs(PGUESS - P)
    TOL = 1E-4
    if PGUESS >= 1 - TOL:
        PDIFF = XGUESS
    return PDIFF


def chisquare_inv(P, V):
    if (1 - P) < 1E-4:
        raise ValueError('P must be < 0.9999')
    if P == 0.95 and V == 2:
        return 5.9915
    MINN = 0.01
    MAXX = 1.0
    X = 1.0
    TOLERANCE = 1E-4
    while (X + TOLERANCE) >= MAXX:
        MAXX = MAXX * 10.
        X = fminbound(chisquare_solve, MINN, MAXX, args=(P, V), xtol=TOLERANCE)
        MINN = MAXX
    return X * V


def wave_signif(Y, dt, scale, sigtest=0, lag1=0.0, siglvl=0.95,
                mother='MORLET', param=-1, gws=None, Y_is_var=False):
    """
    Compute the significance level for the wavelet power spectrum.
    
    Args:
        Y: Input time series (or precomputed variance array if Y_is_var=True).
        dt: Time step.
        scale: Array of scales.
        sigtest: Significance test flag (0 = chi-square, 1 = Gaussian).
        lag1: Autocorrelation coefficient for red noise.
        siglvl: Significance level (e.g., 0.95 for 95% confidence).
        mother: Name of the mother wavelet.
        param: Additional parameter for the wavelet.
        gws: Global wavelet spectrum.
        Y_is_var: If True, Y is treated directly as precomputed variance.
    
    Returns:
        Significance levels and confidence interval.
    """
    if Y_is_var:
        variance = Y
    else:
        n1 = len(np.atleast_1d(Y))
        if n1 == 1:
            variance = Y
        else:
            variance = np.std(Y) ** 2

    if mother == 'MORLET':
        empir = ([2., -1, -1, -1])
        if param == -1:
            param = 6.
            empir[1:] = ([0.776, 2.32, 0.60])
        fourier_factor = (4 * np.pi) / (param + np.sqrt(2 + param ** 2))
    elif mother == 'PAUL':
        empir = ([2, -1, -1, -1])
        if param == -1:
            param = 4
            empir[1:] = ([1.132, 1.17, 1.5])
        fourier_factor = (4 * np.pi) / (2 * param + 1)
    elif mother == 'DOG':
        empir = ([1., -1, -1, -1])
        if param == -1:
            param = 2.
            empir[1:] = ([3.541, 1.43, 1.4])
        elif param == 6:
            empir[1:] = ([1.966, 1.37, 0.97])
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * param + 1))
    else:
        raise ValueError('Mother must be one of MORLET, PAUL, DOG')

    period = scale * fourier_factor
    dofmin = empir[0]

    freq = dt / period

    if gws is not None:
        fft_theory = gws
    else:
        fft_theory = (1 - lag1 ** 2) / (1 - 2 * lag1 * np.cos(freq * 2 * np.pi) + lag1 ** 2)
        if isinstance(variance, np.ndarray) and variance.ndim > 0:
            fft_theory = variance[..., np.newaxis] * fft_theory
        else:
            fft_theory = variance * fft_theory

    if sigtest == 0:
        dof = dofmin
        chi_square = chisquare_inv(siglvl, dof) / dof
        significance = fft_theory * chi_square
    else:
        raise NotImplementedError('Only sigtest=0 is fully ported for vectorized usage.')

    return significance


def wavelet(Y, dt, pad=1, dj=-1, s0=-1, J1=-1, mother='MORLET', param=-1, axis=-1):
    """
    Computes the 1D Wavelet transform along a specified axis of a multidimensional array.
    
    Args:
        Y: Input data array.
        dt: Time step.
        pad: Whether to pad the data.
        dj: Scale parameter.
        s0: Initial scale.
        J1: Final scale.
        mother: Name of the mother wavelet.
        param: Additional parameter for the wavelet.
        axis: Axis along which to compute the transform.
    
    Returns:
        Complex-valued wavelet transform.
    """
    n1 = Y.shape[axis]

    if s0 == -1: s0 = 2 * dt
    if dj == -1: dj = 1. / 4.
    if J1 == -1: J1 = int(np.fix((np.log(n1 * dt / s0) / np.log(2)) / dj))

    Y_work = np.moveaxis(Y, axis, -1)
    x = Y_work - np.mean(Y_work, axis=-1, keepdims=True)

    if pad == 1:
        base2 = np.fix(np.log(n1) / np.log(2) + 0.4999)
        nzeroes = int(2 ** (base2 + 1) - n1)
        pad_shape = list(x.shape)
        pad_shape[-1] = nzeroes
        x = np.concatenate((x, np.zeros(pad_shape, dtype=x.dtype)), axis=-1)

    n = x.shape[-1]

    k_plus = np.arange(1, int(n / 2) + 1)
    k_plus = (k_plus * 2 * np.pi / (n * dt))
    k_minus = np.arange(1, int((n - 1) / 2) + 1)
    k_minus = np.sort((-k_minus * 2 * np.pi / (n * dt)))
    k = np.concatenate(([0.], k_plus, k_minus))

    f = np.fft.fft(x, axis=-1)

    if mother == 'MORLET':
        p = 6. if param == -1 else param
        fourier_factor = 4 * np.pi / (p + np.sqrt(2 + p ** 2))
    elif mother == 'PAUL':
        p = 4. if param == -1 else param
        fourier_factor = 4 * np.pi / (2 * p + 1)
    elif mother == 'DOG':
        p = 2. if param == -1 else param
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * p + 1))
    else:
        raise ValueError("Invalid mother wavelet")

    j = np.arange(0, J1 + 1)
    scale = s0 * 2. ** (j * dj)
    period = scale * fourier_factor

    daughter, _, coi, _ = wave_bases(mother, k, scale, param)
    wave = np.fft.ifft(f[..., np.newaxis, :] * daughter, axis=-1)

    coi_out = coi * dt * np.concatenate((
        np.insert(np.arange(int((n1 + 1) / 2) - 1), [0], [1E-5]),
        np.insert(np.flipud(np.arange(0, int(n1 / 2) - 1)), [-1], [1E-5])
    ))

    # Trim padding and move axes
    wave = wave[..., :n1]

    # Original order was (..., n_scales, time_dim) where time_dim was at -1
    # Move scales axis to front
    wave = np.moveaxis(wave, -2, 0)
    # Move time_dim back to its original axis (+1 because of new scales axis at front)
    wave = np.moveaxis(wave, -1, axis + 1)

    return wave, period, scale, coi_out


# --------------------------------------------------------------------------
# XARRAY-NATIVE FOURIER-WAVELET API
# --------------------------------------------------------------------------

def fourier_wavelet_spectrum(da: xr.DataArray, zwn: int,
                             lon_dim: str = 'lon', time_dim: str = 'time',
                             dt: float = 1.0, dj: float = 0.1,
                             min_t: float = -1, max_t: float = np.inf) -> xr.Dataset:
    """
    Computes the Fourier-Wavelet spectrum of a spatiotemporal DataArray.
    
    Args:
        da: Input xarray.DataArray
        zwn: Zonal wavenumber
        lon_dim: Name of the longitude dimension
        time_dim: Name of the time dimension
        dt: Sampling time in your preferred time units (default 1.0)
        dj: Spacing between discrete scales (default 0.1)
        min_t: Minimum period to include
        max_t: Maximum period to include
        
    Returns:
        xr.Dataset containing amplitudes, phases, and significance metrics
    """
    if lon_dim not in da.dims:
        raise ValueError(f"Longitude dimension '{lon_dim}' not found in DataArray")
    if time_dim not in da.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in DataArray")

    # Check for NaN
    if da.isnull().any():
        logger.warning(
            "NaN values found in data. Using standard FFT may produce NaNs. Consider interpolating first.")

    # 1. Fourier Analysis in Longitude — xarray-native rfft then isel
    L = da.sizes[lon_dim]
    f_da = xr.apply_ufunc(
        np.fft.fft, da, kwargs={'axis': da.dims.index(lon_dim)},
        input_core_dims=[[lon_dim]], output_core_dims=[[lon_dim]],
    ) / L
    f_zwn = f_da.isel({lon_dim: zwn}, drop=True)

    if zwn == 0:
        Ck_da = f_zwn.real
        Sk_da = xr.zeros_like(Ck_da)
    else:
        Ck_da = 2 * f_zwn.real
        Sk_da = -2 * f_zwn.imag

    # Keep numpy arrays for the wavelet call; resolve dim index after lon removal
    Ck = Ck_da.values
    Sk = Sk_da.values
    remaining_dims = [d for d in da.dims if d != lon_dim]
    time_axis = remaining_dims.index(time_dim)

    n_time = da.sizes[time_dim]

    # 2. Wavelet Analysis in Time
    param = 6
    pad = 1

    Ck_w0, period, scale, coi = wavelet(Ck, dt, pad, dj, 2 * dt, mother='MORLET', param=param,
                                        axis=time_axis)
    Sk_w0, _, _, _ = wavelet(Sk, dt, pad, dj, 2 * dt, mother='MORLET', param=param, axis=time_axis)

    # Scale normalization: broadcast scale over all non-scale axes
    scale_factor = np.sqrt(dt / scale).reshape((-1,) + (1,) * (Ck_w0.ndim - 1))

    Ck_w = Ck_w0 * scale_factor
    Sk_w = Sk_w0 * scale_factor

    # 3. Calculation of Amplitude and Phase
    Ak = np.real(Ck_w)
    Bk = -np.imag(Ck_w)
    del Ck_w
    ak = np.real(Sk_w)
    bk = -np.imag(Sk_w)
    del Sk_w

    # Filter by period min/max
    period_mask = (period >= min_t) & (period <= max_t)

    period = period[period_mask]
    scale = scale[period_mask]

    # Filter arrays first to avoid computing Rw/Re/Pw/Pe on discarded scales
    Ak = Ak[period_mask, ...]
    Bk = Bk[period_mask, ...]
    ak = ak[period_mask, ...]
    bk = bk[period_mask, ...]

    # Compute amplitude and phase only on filtered scales
    Rw = 0.5 * np.sqrt((Ak - bk) ** 2 + (Bk + ak) ** 2)
    Re = 0.5 * np.sqrt((Ak + bk) ** 2 + (Bk - ak) ** 2)
    Pw = np.arctan2(Bk + ak, Ak - bk)
    Pe = np.arctan2(Bk - ak, Ak + bk)

    # Significance levels — compute variance with xarray, then vectorized signif
    var_Ck = Ck_da.var(dim=time_dim)
    var_Sk = Sk_da.var(dim=time_dim)
    max_var = np.maximum(var_Ck.values, var_Sk.values)

    # Vectorized significance calculation (no loops over spatial dimensions)
    sig = wave_signif(max_var, dt, scale, lag1=0.72, siglvl=0.95, mother='MORLET', param=param,
                      Y_is_var=True)
    # sig shape: (*max_var.shape, n_scales)
    sig = sig * (0.5 * np.sqrt(dt / scale))

    if np.ndim(max_var) == 0:
        signif_arr = sig
    else:
        signif_arr = np.moveaxis(sig, -1, 0)  # shape: (n_scales, *max_var.shape)

    # Broadcast significance over the time axis using xarray
    # Build a DataArray without time, then broadcast to full output shape via expand_dims
    non_time_dims = [d for d in remaining_dims if d != time_dim]
    signif_coords = {d: da.coords[d] for d in non_time_dims if d in da.coords}
    signif_coords['period'] = period
    sig_da = xr.DataArray(
        signif_arr,
        dims=['period'] + non_time_dims,
        coords=signif_coords,
    ).expand_dims({time_dim: da.sizes[time_dim]},
                  axis=list(['period'] + non_time_dims).index(time_dim) if time_dim in [
                      'period'] + non_time_dims else 1)
    signif = sig_da.values

    # Construct output Dataset — drop lon, add period coordinate natively
    out_coords = {k: v for k, v in da.coords.items() if k != lon_dim}
    out_coords['period'] = period

    # Build the dims tuple: ('period', dim1, dim2, ...) matching the new shape
    out_dims = ['period'] + [d for d in da.dims if d != lon_dim]

    ds = xr.Dataset(
        data_vars={
            'amplitude_westward': (out_dims, Rw),
            'amplitude_eastward': (out_dims, Re),
            'phase_westward': (out_dims, Pw),
            'phase_eastward': (out_dims, Pe),
            'significance_95': (out_dims, signif),
            'coi': ((time_dim,), coi)
        },
        coords=out_coords
    )

    ds.attrs = da.attrs
    ds.attrs['zonal_wavenumber'] = zwn
    ds.attrs['dj'] = dj
    ds.attrs['dt'] = dt

    return ds


# --------------------------------------------------------------------------
# SPHERICAL-HARMONIC WAVELET API
# --------------------------------------------------------------------------

def spherical_harmonic_wavelet_spectrum(da: xr.DataArray, zwn: int,
                                        time_dim: str = 'time',
                                        dt: float = 1.0, dj: float = 0.1,
                                        min_t: float = -1, max_t: float = np.inf,
                                        lmax: int | None = None,
                                        map2alm_iter: int = 3,
                                        periods_to_reconstruct: list[float] | None = None,
                                        order: str | None = None) -> xr.Dataset:
    """
    Computes the Spherical-Harmonic Wavelet spectrum of a HEALPix DataArray.

    This is the SH analogue of ``fourier_wavelet_spectrum``.  Instead of
    a longitude FFT, the cos(m·λ) and sin(m·λ) spectral coefficients are
    extracted via Spherical Harmonic analysis, correctly handling the
    latitude-dependent associated Legendre functions.

    Pipeline:
        1. For each time step, compute map2alm → keep only m = |zwn|.
           Extract real spectral coefficients:
             A_l(t) = 2·Re(a_lm(t))   — cos(m·λ) coefficient for each degree l
             B_l(t) = -2·Im(a_lm(t))  — sin(m·λ) coefficient for each degree l
        2. Apply CWT along the time axis to each A_l(t), B_l(t).
        3. Separate westward / eastward components using the Yamazaki (2023)
           analytic-signal formulation.
        4. Compute global-mean amplitude via Parseval's theorem:
             Amp_global = sqrt(Σ_l (Rw_l² or Re_l²) / 2)
        5. Optionally reconstruct full spatial maps at selected periods using
           basis-map synthesis, then decompose into symmetric/antisymmetric.

    Args:
        da: Input xarray.DataArray on a HEALPix grid with a time dimension.
        zwn: Target zonal wavenumber (positive integer).
        time_dim: Name of the time dimension.
        dt: Sampling interval in the units of the time coordinate.
        dj: Spacing between discrete scales (default 0.1).
        min_t: Minimum period to include in output.
        max_t: Maximum period to include in output.
        lmax: Maximum spherical harmonic degree. If None, uses 3·nside - 1.
        map2alm_iter: Number of iterative corrections in ``hp.map2alm``
            (default 3).  For full-sky data without NaN gaps, ``iter=0``
            is exact and ~4x faster.
        periods_to_reconstruct: Periods (in ``dt`` units) for which to
            reconstruct full spatial amplitude/phase maps.  If None, only
            the global-mean spectrum is returned.
        order: HEALPix pixel ordering of the input data: ``'ring'`` or
            ``'nested'``.  If *None* (default), the ordering is detected
            from ``da.attrs['grid_mapping']`` and its parent dataset's
            ``healpix`` variable, falling back to ``'ring'``.

    Returns:
        xr.Dataset with global_amplitude_westward/eastward (period x time),
        coi (time), and optionally per-period spatial maps.
    """
    if time_dim not in da.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in DataArray")

    cell_dim = get_cells_dim(da)

    # Determine pixel ordering
    if order is not None:
        is_nested = order.lower() == 'nested'
    else:
        # Auto-detect from DataArray attrs (avoids materializing a Dataset).
        _order = da.attrs.get('healpix_scheme', 'ring').lower()
        is_nested = _order == 'nested'
    logger.info(f"HEALPix ordering: {'nested' if is_nested else 'ring'}")
    npix = da.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    if lmax is None:
        lmax = 3 * nside - 1

    n_time = da.sizes[time_dim]

    # ------------------------------------------------------------------
    # Spectral decomposition — extract A_l(t), B_l(t)
    # ------------------------------------------------------------------
    # Thread-parallel map2alm: healpy releases the GIL during the C-level
    # SH transform, so threading gives real speedup (~4× with 4 workers).
    l_arr, m_arr = hp.Alm.getlm(lmax)
    abs_m = abs(zwn)
    m_mask = (m_arr == abs_m)
    n_l = int(np.sum(m_mask))
    order_str = 'nested' if is_nested else 'ring'

    # Ensure data is in memory and in RING ordering
    da_np = da.values
    cell_axis = list(da.dims).index(cell_dim)
    if cell_axis != da_np.ndim - 1:
        da_np = np.moveaxis(da_np, cell_axis, -1)
    if is_nested:
        da_np = hp.reorder(da_np, n2r=True)

    # A_lm, B_lm have shape (n_time, n_l) — spectral coefficients per degree
    A_lm = np.zeros((n_time, n_l), dtype=np.float64)
    B_lm = np.zeros((n_time, n_l), dtype=np.float64)

    logger.info(f"Computing SH spectral coefficients for m={abs_m} "
                f"over {n_time} time steps ({n_l} degrees)...")

    # Fast NaN check
    has_nan = np.isnan(np.sum(da_np))

    if has_nan:
        # Pre-compute NaN masks outside the thread pool.
        # np.isnan and np.where hold the GIL; doing them here lets every
        # worker thread enter hp.map2alm (GIL-releasing C layer) immediately.
        nan_masks = np.isnan(da_np)  # (n_time, npix)  bool
        filled_maps = np.where(nan_masks, 0.0, da_np)  # (n_time, npix) float64
        all_nan = nan_masks.all(axis=-1)  # (n_time,) bool

        def _map2alm_one(i):
            """Process a single time step (thread-safe, GIL-releasing)."""
            if all_nan[i]:
                return i, None
            alm = hp.map2alm(filled_maps[i], lmax=lmax, iter=map2alm_iter)
            return i, alm[m_mask]
    else:
        # If no NaNs, we can auto-optimize map2alm_iter=0 for full-sky exact transform
        if map2alm_iter == 3:
            logger.info("No NaN values detected. Setting map2alm_iter=0 for ~4x speedup.")
            map2alm_iter = 0

        def _map2alm_one(i):
            """Process a single time step (thread-safe, GIL-releasing)."""
            alm = hp.map2alm(da_np[i], lmax=lmax, iter=map2alm_iter)
            return i, alm[m_mask]

    hp_logger = logging.getLogger('healpy')
    old_hp_level = hp_logger.level
    hp_logger.setLevel(logging.WARNING)
    try:
        n_workers = min(32, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for i, alm_m in pool.map(_map2alm_one, range(n_time)):
                if alm_m is None:
                    continue
                if abs_m == 0:
                    A_lm[i, :] = np.real(alm_m)
                else:
                    A_lm[i, :] = 2.0 * np.real(alm_m)
                    B_lm[i, :] = -2.0 * np.imag(alm_m)
    finally:
        hp_logger.setLevel(old_hp_level)

    # ------------------------------------------------------------------
    # CWT along the time axis for each spectral coefficient
    # ------------------------------------------------------------------
    logger.info("Applying Continuous Wavelet Transform...")
    param = 6
    pad = 1

    # W_A shape: (n_scales, n_time, n_l)
    W_A, period, scale, coi = wavelet(A_lm, dt, pad, dj, 2 * dt,
                                      mother='MORLET', param=param, axis=0)
    W_B, _, _, _ = wavelet(B_lm, dt, pad, dj, 2 * dt,
                           mother='MORLET', param=param, axis=0)

    # Scale normalization (Torrence & Compo) — broadcast over all trailing axes
    scale_factor = np.sqrt(dt / scale).reshape((-1,) + (1,) * (W_A.ndim - 1))
    W_A *= scale_factor
    W_B *= scale_factor

    # ------------------------------------------------------------------
    # E/W separation — same analytic signal as fourier_wavelet
    # ------------------------------------------------------------------
    Ak = np.real(W_A)  # (n_scales, n_time, n_l)
    Bk = -np.imag(W_A)
    del W_A
    ak = np.real(W_B)
    bk = -np.imag(W_B)
    del W_B

    # Filter by period — apply mask to ALL arrays atomically so that
    # idx = argmin(|period - target_p|) always indexes the same scale
    # across period, Rw_l/Re_l and the coefficient arrays Ak/Bk/ak/bk.
    period_mask = (period >= min_t) & (period <= max_t)
    period = period[period_mask]
    
    # Filter arrays first to avoid computing Rw_l/Re_l on discarded scales
    Ak = Ak[period_mask, ...]
    Bk = Bk[period_mask, ...]
    ak = ak[period_mask, ...]
    bk = bk[period_mask, ...]

    # Per-degree westward/eastward amplitude computed only on filtered scales
    Rw_l = 0.5 * np.sqrt((Ak - bk) ** 2 + (Bk + ak) ** 2)
    Re_l = 0.5 * np.sqrt((Ak + bk) ** 2 + (Bk - ak) ** 2)

    # ------------------------------------------------------------------
    #  Global-mean amplitude via Parseval
    # ------------------------------------------------------------------
    # Parseval normalization: for healpy's orthonormal SH convention,
    # Amp_physical = sqrt(sum_l Rw_l^2 / (2*pi)).
    # See docs/spectral_methods.tex §4.3 for the full derivation.
    norm = 1.0 / np.sqrt(2.0 * np.pi)
    global_amp_w = norm * np.sqrt(np.sum(Rw_l ** 2, axis=-1))
    global_amp_e = norm * np.sqrt(np.sum(Re_l ** 2, axis=-1))

    out_coords = {k: v for k, v in da.coords.items() if k != cell_dim}
    out_coords['period'] = period

    out_vars = {
        'global_amplitude_westward': (['period', time_dim], global_amp_w),
        'global_amplitude_eastward': (['period', time_dim], global_amp_e),
        'coi': ([time_dim], coi)
    }

    # ------------------------------------------------------------------
    # Spatial reconstruction for specific periods
    # ------------------------------------------------------------------
    # To avoid holding the full (n_time, npix) matrices for all eight
    # output variables simultaneously, the reconstruction is performed
    # in time-chunks.  Peak memory is bounded at
    #   O(TIME_CHUNK × npix)  instead of  O(n_time × npix).
    TIME_CHUNK = 32

    # Collect spatial arrays in a plain dict keyed by
    # (direction, sym_or_asy, amp_or_pha) so that the output-building
    # step can look them up without parsing string-encoded periods.
    spatial_arrays = {}  # (p_idx, direction, sa, ap) -> (n_time, npix) array

    if periods_to_reconstruct is not None:
        # Basis maps from alm2map are ALWAYS in RING ordering, so the
        # symmetric-pixel index must also be computed in RING.
        sym_idx = _get_symmetric_pixels(nside, is_nested=False)

        # Basis maps are cached across calls for the same (nside, lmax, m).
        # They are in RING ordering (healpy default).
        basis_cos, basis_sin = _get_basis_maps(nside, lmax, abs_m)

        # Hoist lon_rad: pix2ang is the same for every period (item 4).
        _, lon_rad = hp.pix2ang(nside, np.arange(npix))
        abs_m_lon = abs_m * lon_rad  # pre-scale; reused in every chunk

        for p_idx, target_p in enumerate(periods_to_reconstruct):
            idx = np.argmin(np.abs(period - target_p))
            actual_p = period[idx]

            # E/W separated spectral coefficients in A_lm/B_lm space.
            # Ak, Bk, ak, bk: (n_filtered_scales, n_time, n_l)
            # Selecting idx → (n_time, n_l)
            cw_l = 0.5 * (Ak[idx] - bk[idx])  # westward Re(alm) × 2
            sw_l = 0.5 * (Bk[idx] + ak[idx])  # westward Im(alm) × 2 (negated)
            ce_l = 0.5 * (Ak[idx] + bk[idx])  # eastward Re(alm) × 2
            se_l = 0.5 * (Bk[idx] - ak[idx])  # eastward Im(alm) × 2 (negated)

            # Convert E/W A_lm/B_lm to Re(alm)/Im(alm) for synthesis.
            # The factor-of-2 is absorbed: A_lm = 2·Re(alm), B_lm = -2·Im(alm)
            # so Re(alm) = cw_l/2, Im(alm) = sw_l/2 (sign from B convention).
            re_w = cw_l * 0.5  # (n_time, n_l)
            im_w = sw_l * 0.5
            re_e = ce_l * 0.5
            im_e = se_l * 0.5

            # Pre-allocate eight output arrays (item 3: one allocation block
            # per period, written into by the fused chunk loop below).
            shape = (n_time, npix)
            amp_w_sym = np.empty(shape, dtype=np.float64)
            amp_w_asy = np.empty(shape, dtype=np.float64)
            amp_e_sym = np.empty(shape, dtype=np.float64)
            amp_e_asy = np.empty(shape, dtype=np.float64)
            pha_w_sym = np.empty(shape, dtype=np.float64)
            pha_w_asy = np.empty(shape, dtype=np.float64)
            pha_e_sym = np.empty(shape, dtype=np.float64)
            pha_e_asy = np.empty(shape, dtype=np.float64)

            # Process in time-chunks to cap intermediate memory at
            # O(TIME_CHUNK × npix) instead of O(n_time × npix).
            # All W and E sym/asy quantities are computed in one fused
            # pass per chunk to avoid re-reading re_w/im_w/re_e/im_e.
            for t0 in range(0, n_time, TIME_CHUNK):
                t1 = min(t0 + TIME_CHUNK, n_time)
                sl = slice(t0, t1)

                # Vectorized spatial synthesis via basis maps:
                #   map(x) = Σ_k Re(alm_k)·basis_cos_k(x) + Im(alm_k)·basis_sin_k(x)
                #   map_H(x) = Σ_k [-Im(alm_k)·basis_cos_k(x) + Re(alm_k)·basis_sin_k(x)]
                # (chunk, n_l) @ (n_l, npix) → (chunk, npix)
                mw = re_w[sl] @ basis_cos + im_w[sl] @ basis_sin
                mwH = (-im_w[sl]) @ basis_cos + re_w[sl] @ basis_sin
                me = re_e[sl] @ basis_cos + im_e[sl] @ basis_sin
                meH = (-im_e[sl]) @ basis_cos + re_e[sl] @ basis_sin

                # Symmetric / Antisymmetric decomposition on the real
                # spatial fields (matching the tides module convention).
                mw_s = 0.5 * (mw + mw[:, sym_idx])
                mwH_s = 0.5 * (mwH + mwH[:, sym_idx])
                mw_a = 0.5 * (mw - mw[:, sym_idx])
                mwH_a = 0.5 * (mwH - mwH[:, sym_idx])

                # Phase: arctan2(-H, map) then remove zonal-wavenumber lon phase.
                # np.mod(...+ pi, 2pi) - pi wraps to (-pi, pi].
                phase_offset = abs_m_lon[None, :]  # (1, npix) broadcast
                amp_w_sym[sl] = np.sqrt(mw_s ** 2 + mwH_s ** 2)
                pha_w_sym[sl] = np.mod(np.arctan2(-mwH_s, mw_s) - phase_offset + np.pi,
                                       2 * np.pi) - np.pi
                amp_w_asy[sl] = np.sqrt(mw_a ** 2 + mwH_a ** 2)
                pha_w_asy[sl] = np.mod(np.arctan2(-mwH_a, mw_a) - phase_offset + np.pi,
                                       2 * np.pi) - np.pi

                me_s = 0.5 * (me + me[:, sym_idx])
                meH_s = 0.5 * (meH + meH[:, sym_idx])
                me_a = 0.5 * (me - me[:, sym_idx])
                meH_a = 0.5 * (meH - meH[:, sym_idx])

                amp_e_sym[sl] = np.sqrt(me_s ** 2 + meH_s ** 2)
                pha_e_sym[sl] = np.mod(np.arctan2(-meH_s, me_s) - phase_offset + np.pi,
                                       2 * np.pi) - np.pi
                amp_e_asy[sl] = np.sqrt(me_a ** 2 + meH_a ** 2)
                pha_e_asy[sl] = np.mod(np.arctan2(-meH_a, me_a) - phase_offset + np.pi,
                                       2 * np.pi) - np.pi

            spatial_arrays[(p_idx, 'westward', 'sym', 'amp')] = (actual_p, amp_w_sym)
            spatial_arrays[(p_idx, 'westward', 'sym', 'pha')] = (actual_p, pha_w_sym)
            spatial_arrays[(p_idx, 'westward', 'asy', 'amp')] = (actual_p, amp_w_asy)
            spatial_arrays[(p_idx, 'westward', 'asy', 'pha')] = (actual_p, pha_w_asy)
            spatial_arrays[(p_idx, 'eastward', 'sym', 'amp')] = (actual_p, amp_e_sym)
            spatial_arrays[(p_idx, 'eastward', 'sym', 'pha')] = (actual_p, pha_e_sym)
            spatial_arrays[(p_idx, 'eastward', 'asy', 'amp')] = (actual_p, amp_e_asy)
            spatial_arrays[(p_idx, 'eastward', 'asy', 'pha')] = (actual_p, pha_e_asy)

    # Build output with proper HEALPix cell coordinates
    target_lon, target_lat = get_healpix_coords(nside)
    out_coords[cell_dim] = np.arange(npix)

    if is_nested:
        # Item 6: reorder spatial arrays directly — no dtype/len duck-typing.
        for key, (actual_p, arr) in spatial_arrays.items():
            spatial_arrays[key] = (actual_p, ensure_original_order(arr, 'nested'))
        target_lon = ensure_original_order(target_lon, 'nested')
        target_lat = ensure_original_order(target_lat, 'nested')

    # Populate out_vars from spatial_arrays using the original string-key
    # convention so downstream Dataset construction is unchanged.
    for (p_idx, direction, sa, ap), (actual_p, arr) in spatial_arrays.items():
        p_str = f"{actual_p:.1f}"
        var_name = f"{ap}_{sa}_{direction}_{p_str}"
        out_vars[var_name] = ([time_dim, cell_dim], arr)

    ds = xr.Dataset(data_vars=out_vars, coords=out_coords)
    lon_da = xr.DataArray(target_lon, dims=cell_dim,
                          attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    lat_da = xr.DataArray(target_lat, dims=cell_dim,
                          attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    ds = ds.assign_coords(lon=lon_da, lat=lat_da)

    ds.attrs['zonal_wavenumber'] = zwn
    ds.attrs['dj'] = dj
    ds.attrs['dt'] = dt

    # Attach CF-compliant HEALPix grid mapping
    ds = add_healpix_grid_mapping(ds, nside, order=order_str)

    return ds


def _infer_dt_hours(time_vals: np.ndarray, time_dim: str = 'time') -> float:
    """Infer the median time step in hours from a coordinate array.

    Handles datetime64, timedelta64 (LST-style), and plain numeric arrays.
    Mirrors the logic used in ``compute_tidal_analysis`` so that both
    functions produce identical ``dt`` values for the same dataset.
    """
    if time_dim == 'lst':
        if np.issubdtype(time_vals.dtype, np.timedelta64):
            return float(np.median(np.diff(time_vals)) / np.timedelta64(1, 'h'))
        return float(np.median(np.diff(time_vals)))
    if np.issubdtype(time_vals.dtype, (np.datetime64, np.timedelta64)):
        return float(np.median(np.diff(time_vals)) / np.timedelta64(1, 'h'))
    return float(np.median(np.diff(time_vals)))


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
) -> xr.Dataset:
    """Wavelet-based tidal analysis on a HEALPix Dataset.

    This is the wavelet analogue of
    :func:`~healicon.analysis.tides.compute_tidal_analysis`.  It accepts a
    full Dataset with arbitrary non-core dimensions (e.g. ``height``) and
    automatically maps over them via :func:`xarray.apply_ufunc`, keeping
    peak memory at O(1 slice) regardless of the number of non-core levels.

    Pipeline (per non-core slice):
        1. For each unique ``|m|`` in *m_filters*, run
           :func:`spherical_harmonic_wavelet_spectrum` to obtain
           time-resolved amplitude and phase maps.
        2. Extract the requested ``periods_hours`` and propagation
           direction (westward for ``m > 0``, eastward for ``m < 0``).
        3. Optionally average over the time axis (``temporal_mean``).

    Args:
        ds:  Input Dataset on a HEALPix grid with a time dimension.
        var_name:  Name of the data variable to analyse.
        periods_hours:  Target periods in hours (e.g. ``[24, 12]``).
        m_filters:  Signed zonal wavenumbers to extract.  Positive values
            denote westward propagation, negative values eastward
            (matching :func:`compute_tidal_analysis`).  If *None*, all
            wavenumbers up to ``lmax`` are kept (no directional filter).
        lmax:  Maximum spherical harmonic degree.  If *None*, uses
            ``3 * nside - 1``.
        time_dim:  Name of the time dimension.
        dj:  Spacing between discrete wavelet scales (default 0.1)
        temporal_mean:  If *True*, average the wavelet amplitude over time
            before returning (produces output comparable to LS tides).
            Default is *False* (return the full time-resolved envelope).

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

    unique_zwn = sorted(zwn_mode_groups.keys())
    periods_for_wavelet = list(periods_hours)

    # ── Core function applied per non-core slice ─────────────────────
    # Returns a flat tuple of numpy arrays, one per (mode, component).
    # Order: for each mode in modes_for_this_zwn, 4 arrays
    #   (amp_sym, amp_asy, pha_sym, pha_asy)

    def _make_ufunc(zwn, zwn_modes):
        """Build a ufunc for a specific zonal wavenumber."""
        n_modes = len(zwn_modes)

        # Compute t_hours_vals once per ufunc build, not per mode
        # per slice.  The time coordinate is the same for every call.
        _time_vals = ds[time_dim].values
        if time_dim == 'lst':
            if np.issubdtype(_time_vals.dtype, np.timedelta64):
                t_hours_vals = _time_vals / np.timedelta64(1, 'h')
            else:
                t_hours_vals = _time_vals.astype(float)
        else:
            if np.issubdtype(_time_vals.dtype, (np.datetime64, np.timedelta64)):
                t_hours_vals = (_time_vals - _time_vals[0]) / np.timedelta64(1, 'h')
            else:
                t_hours_vals = (_time_vals - _time_vals[0]).astype(float)

        def _func(data_np):
            # data_np: (time, cells) numpy array
            da_tmp = xr.DataArray(
                data_np, dims=[time_dim, cell_dim],
                coords={time_dim: ds[time_dim]},
            )
            ds_w = spherical_harmonic_wavelet_spectrum(
                da_tmp, zwn=zwn, time_dim=time_dim, dt=dt_hours, dj=dj,
                lmax=lmax, map2alm_iter=map2alm_iter,
                periods_to_reconstruct=periods_for_wavelet,
                order=hp_order,
            )

            # Build an explicit (direction, p_str) → actual_p_str
            # lookup from the Dataset variables so we never do prefix scans
            # that could collide on period substrings (e.g. "12.0" vs "120.0").
            # The var names follow: {ap}_{sa}_{direction}_{p_str}
            # e.g. amp_sym_westward_12.0
            period_lookup = {}  # (direction, target_p_h) -> p_str
            for v in ds_w.data_vars:
                parts = v.split('_')
                # Expected parts: [ap, sa, direction, p_str]
                # (amp|pha, sym|asy, westward|eastward, float_str)
                if len(parts) != 4:
                    continue
                _, _, direction, p_str = parts
                if direction not in ('westward', 'eastward'):
                    continue
                try:
                    actual_p = float(p_str)
                except ValueError:
                    continue
                period_lookup[(direction, p_str)] = actual_p

            outputs = []
            for mode in zwn_modes:
                direction = mode['dir']
                target_p_h = mode['period_h']

                # Find the p_str whose actual period is nearest to target_p_h
                # using only entries for this direction — no prefix scan.
                best_p_str = min(
                    (p_str for (d, p_str) in period_lookup if d == direction),
                    key=lambda s: abs(period_lookup[(direction, s)] - target_p_h),
                )

                for comp in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy'):
                    arr = ds_w[f"{comp}_{direction}_{best_p_str}"].values
                    if temporal_mean:
                        sa = comp.split('_')[1]  # 'sym' or 'asy'
                        amp_vals = ds_w[f"amp_{sa}_{direction}_{best_p_str}"].values
                        pha_vals = ds_w[f"pha_{sa}_{direction}_{best_p_str}"].values

                        nside_local = hp.npix2nside(amp_vals.shape[1])
                        target_lon_deg, _ = get_healpix_coords(nside_local)
                        if hp_order == 'nested':
                            target_lon_deg = ensure_original_order(target_lon_deg, 'nested')
                        phi_da = np.radians(target_lon_deg)
                        target_m = mode['m']

                        omega = 2 * np.pi / float(mode['period_h'])
                        wt = omega * t_hours_vals[:, None]

                        # Reconstruct spatial wave (negative phase convention
                        # of Morlet CWT — must match spherical_harmonic_wavelet_spectrum).
                        pha_total = pha_vals + target_m * phi_da
                        mw = amp_vals * np.cos(pha_total)
                        mwH = -amp_vals * np.sin(pha_total)

                        # Demodulate time
                        C_t = mw * np.cos(wt) + mwH * np.sin(wt)
                        S_t = mw * np.sin(wt) - mwH * np.cos(wt)

                        C_mean = C_t.mean(axis=0)
                        S_mean = S_t.mean(axis=0)

                        # Demodulate space (remove longitude dependence)
                        real_part = C_mean * np.cos(target_m * phi_da) + S_mean * np.sin(
                            target_m * phi_da)
                        imag_part = S_mean * np.cos(target_m * phi_da) - C_mean * np.sin(
                            target_m * phi_da)

                        arr = np.arctan2(imag_part, real_part) if 'pha' in comp \
                            else np.sqrt(real_part ** 2 + imag_part ** 2)

                    outputs.append(arr)

            return tuple(outputs)

        return _func, n_modes

    # ── apply_ufunc per unique |m| ───────────────────────────────────
    logger.info(f"Applying wavelet analysis for |m| = {unique_zwn} "
                f"via apply_ufunc...")

    assembled = {}  # (m_val, period_h, comp) -> DataArray

    for zwn in unique_zwn:
        zwn_modes = zwn_mode_groups[zwn]
        func, n_modes = _make_ufunc(zwn, zwn_modes)
        n_outputs = n_modes * 4

        if temporal_mean:
            out_core = [[cell_dim]] * n_outputs
        else:
            out_core = [[time_dim, cell_dim]] * n_outputs

        results = xr.apply_ufunc(
            func,
            da,
            input_core_dims=[[time_dim, cell_dim]],
            output_core_dims=out_core,
            vectorize=True,
            dask='parallelized',
            output_dtypes=[np.float64] * n_outputs,
            dask_gufunc_kwargs={'allow_rechunk': True},
        )

        # Unpack results into named DataArrays
        if n_outputs == 1:
            results = (results,)
        comps = ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy')
        for i, mode in enumerate(zwn_modes):
            for j, comp in enumerate(comps):
                da_out = results[i * 4 + j]
                assembled[(mode['m'], mode['period_h'], comp)] = da_out

    # ── Build output Dataset using xarray-native concat ─────────────────
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

    var_units = ds[var_name].attrs.get('units', '')
    data_vars = {}

    for comp in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy'):
        stacked = xr.concat(
            [xr.concat([assembled[(m, p, comp)] for p in periods_hours], dim=period_td)
             for m in m_vals],
            dim=m_coord,
        )
        data_vars[f'{var_name}_{comp}'] = stacked

    out_ds = xr.Dataset(data_vars)

    # Spatial coordinates - assign via assign_coords for a single fluent call
    target_lon, target_lat = get_healpix_coords(nside)
    if hp_order == 'nested':
        target_lon = ensure_original_order(target_lon, 'nested')
        target_lat = ensure_original_order(target_lat, 'nested')
    lon_da = xr.DataArray(target_lon, dims=cell_dim,
                          attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    lat_da = xr.DataArray(target_lat, dims=cell_dim,
                          attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    out_ds = out_ds.assign_coords(lon=lon_da, lat=lat_da)

    # Variable metadata - keys follow the {var_name}_{comp} pattern produced
    # by the concat block above, e.g. "u_amp_sym", "u_pha_asy".
    label_map = {'sym': 'Symmetric', 'asy': 'Antisymmetric'}
    for sa in ('sym', 'asy'):
        label = label_map[sa]
        out_ds[f'{var_name}_amp_{sa}'].attrs = {
            'units': var_units, 'long_name': f'{label} Amplitude',
            'grid_mapping': 'healpix',
        }
        out_ds[f'{var_name}_pha_{sa}'].attrs = {
            'units': 'rad', 'long_name': f'{label} Phase',
            'grid_mapping': 'healpix',
        }

    # Preserve dataset attributes and grid mapping
    out_ds.attrs = ds.attrs.copy()
    out_ds = add_healpix_grid_mapping(out_ds, nside, order=hp_order)
    out_ds.attrs = append_history(
        out_ds.attrs,
        f"Wavelet tidal analysis (periods: {periods_hours}h, "
        f"m: {m_filters}, temporal_mean={temporal_mean})."
    )

    return out_ds


# --------------------------------------------------------------------------
# DATASET-LEVEL FOURIER TIDAL ANALYSIS
# --------------------------------------------------------------------------

def compute_fourier_tidal_analysis(
        ds: xr.Dataset,
        var_name: str,
        periods_hours: list[float],
        m_filters: list[int] | None = None,
        time_dim: str = 'time',
        dj: float = 0.1,
        temporal_mean: bool = False,
) -> xr.Dataset:
    """Fourier-wavelet-based tidal analysis on a HEALPix Dataset.

    This is the Fourier analogue of :func:`compute_wavelet_tidal_analysis`.
    It interpolates the HEALPix grid to a regular lat-lon grid, runs the
    Fourier-wavelet analysis, decomposes the wavelet coefficients into
    symmetric/antisymmetric parts, and interpolates the results back to
    the HEALPix cells.

    Args:
        ds: Input Dataset on a HEALPix grid.
        var_name: Name of the variable to analyze.
        periods_hours: Target periods in hours.
        m_filters: Signed zonal wavenumbers to extract.
        time_dim: Name of the time dimension.
        dj: Spacing between discrete wavelet scales.
        temporal_mean: If True, average the wavelet amplitude over time.

    Returns:
        xr.Dataset containing symmetric/antisymmetric amplitudes and phases.
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Dataset must have a '{time_dim}' dimension for Fourier tidal analysis.")

    cell_dim = get_cells_dim(ds)
    hp_order = get_healpix_order(ds)
    is_nested = (hp_order == 'nested')
    da = ds[var_name]

    npix = da.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    # Infer dt from time coordinate
    time_vals = ds[time_dim].values
    dt_hours = _infer_dt_hours(time_vals, time_dim)
    logger.info(f"Inferred dt = {dt_hours:.2f} hours from '{time_dim}' coordinate.")

    if m_filters is None:
        m_filters = [1]
        logger.warning("No m_filters specified; defaulting to m=[1] (DW1).")

    # Define target regular grid resolution
    n_lats = max(180, 4 * nside)
    n_lons = max(360, 8 * nside)

    lats = np.linspace(-90.0, 90.0, n_lats)
    lons = np.linspace(-180.0, 180.0, n_lons, endpoint=False)

    theta_mesh, phi_mesh = np.meshgrid(np.deg2rad(90.0 - lats), np.deg2rad(lons), indexing='ij')

    target_lon, target_lat = get_healpix_coords(nside)
    if is_nested:
        target_lon = ensure_original_order(target_lon, 'nested')
        target_lat = ensure_original_order(target_lat, 'nested')

    # Core function for apply_ufunc
    def _fourier_tide_ufunc(data_np):
        # data_np: shape (..., time, npix)
        orig_shape = data_np.shape
        batch_shape = orig_shape[:-2]
        n_time = orig_shape[-2]

        data_2d = data_np.reshape(-1, n_time, npix)
        n_batch = data_2d.shape[0]

        # We will collect outputs for all modes
        # Output shapes: (n_modes, 4, n_batch, [n_time], npix)
        # We will reorder them afterwards
        n_modes = len(m_filters) * len(periods_hours)
        if temporal_mean:
            out_shape = (n_modes, 4, n_batch, npix)
        else:
            out_shape = (n_modes, 4, n_batch, n_time, npix)

        out_amp_sym = np.zeros(out_shape[2:], dtype=np.float64)
        out_amp_asy = np.zeros(out_shape[2:], dtype=np.float64)
        out_pha_sym = np.zeros(out_shape[2:], dtype=np.float64)
        out_pha_asy = np.zeros(out_shape[2:], dtype=np.float64)

        # Allocate final list of arrays
        out_list = []
        for _ in range(n_modes * 4):
            out_list.append(np.zeros(out_shape[2:], dtype=np.float64))

        param = 6
        pad = 1
        t_hours_vals = np.arange(n_time, dtype=float) * dt_hours

        for b in range(n_batch):
            # 1. HEALPix to regular lat-lon
            grid_reg = np.zeros((n_time, n_lats, n_lons), dtype=data_np.dtype)
            for t_idx in range(n_time):
                grid_reg[t_idx] = hp.get_interp_val(
                    data_2d[b, t_idx], theta_mesh.ravel(), phi_mesh.ravel(), nest=is_nested
                ).reshape(theta_mesh.shape)

            # 2. FFT along longitude (axis 2)
            f_grid = np.fft.fft(grid_reg, axis=2) / n_lons

            # Process each mode
            mode_idx = 0
            for m in m_filters:
                direction = 'westward' if m > 0 else 'eastward'
                zwn = abs(m)

                f_zwn = f_grid[:, :, zwn]
                if zwn == 0:
                    Ck = f_zwn.real
                    Sk = np.zeros_like(Ck)
                else:
                    Ck = 2 * f_zwn.real
                    Sk = -2 * f_zwn.imag

                # CWT in time (axis 0)
                Ck_w0, period_arr, scale_arr, coi_arr = wavelet(
                    Ck, dt_hours, pad, dj, 2 * dt_hours, mother='MORLET', param=param, axis=0
                )
                Sk_w0, _, _, _ = wavelet(
                    Sk, dt_hours, pad, dj, 2 * dt_hours, mother='MORLET', param=param, axis=0
                )

                scale_factor = np.sqrt(dt_hours / scale_arr)[:, None, None]
                Ck_w = Ck_w0 * scale_factor
                Sk_w = Sk_w0 * scale_factor

                Ak = np.real(Ck_w)
                Bk = -np.imag(Ck_w)
                ak = np.real(Sk_w)
                bk = -np.imag(Sk_w)

                # Symmetry decomposition along lat (axis 2 of Ak, which is shape (n_scales, n_time, n_lats))
                Ak_sym = 0.5 * (Ak + np.flip(Ak, axis=2))
                Ak_asy = 0.5 * (Ak - np.flip(Ak, axis=2))
                Bk_sym = 0.5 * (Bk + np.flip(Bk, axis=2))
                Bk_asy = 0.5 * (Bk - np.flip(Bk, axis=2))
                ak_sym = 0.5 * (ak + np.flip(ak, axis=2))
                ak_asy = 0.5 * (ak - np.flip(ak, axis=2))
                bk_sym = 0.5 * (bk + np.flip(bk, axis=2))
                bk_asy = 0.5 * (bk - np.flip(bk, axis=2))

                for target_p in periods_hours:
                    scale_idx = np.argmin(np.abs(period_arr - target_p))

                    for comp_idx, comp in enumerate(('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy')):
                        sa = comp.split('_')[1]
                        is_pha = 'pha' in comp

                        if sa == 'sym':
                            Ak_c, Bk_c, ak_c, bk_c = Ak_sym[scale_idx], Bk_sym[scale_idx], ak_sym[scale_idx], bk_sym[scale_idx]
                        else:
                            Ak_c, Bk_c, ak_c, bk_c = Ak_asy[scale_idx], Bk_asy[scale_idx], ak_asy[scale_idx], bk_asy[scale_idx]

                        if direction == 'westward':
                            amp = 0.5 * np.sqrt((Ak_c - bk_c) ** 2 + (Bk_c + ak_c) ** 2)
                            pha = np.arctan2(Bk_c + ak_c, Ak_c - bk_c)
                        else:
                            amp = 0.5 * np.sqrt((Ak_c + bk_c) ** 2 + (Bk_c - ak_c) ** 2)
                            pha = np.arctan2(Bk_c - ak_c, Ak_c + bk_c)

                        if temporal_mean:
                            mw = amp * np.cos(pha)
                            mwH = -amp * np.sin(pha)
                            omega = 2 * np.pi / target_p
                            wt = omega * t_hours_vals[:, None]
                            C_t = mw * np.cos(wt) + mwH * np.sin(wt)
                            S_t = mw * np.sin(wt) - mwH * np.cos(wt)
                            C_mean = C_t.mean(axis=0)
                            S_mean = S_t.mean(axis=0)

                            final_val = np.arctan2(S_mean, C_mean) if is_pha else np.sqrt(C_mean ** 2 + S_mean ** 2)
                        else:
                            final_val = pha if is_pha else amp

                        # Interpolate back to HEALPix cell latitudes
                        if temporal_mean:
                            hp_val = np.interp(target_lat, lats, final_val)
                            out_list[mode_idx * 4 + comp_idx][b] = hp_val
                        else:
                            hp_val = np.zeros((n_time, npix))
                            for t_idx in range(n_time):
                                hp_val[t_idx] = np.interp(target_lat, lats, final_val[t_idx])
                            out_list[mode_idx * 4 + comp_idx][b] = hp_val

                    mode_idx += 1

        # Reshape back to batch shape
        if temporal_mean:
            out_shape_final = batch_shape + (npix,)
        else:
            out_shape_final = batch_shape + (n_time, npix)

        return tuple(arr.reshape(out_shape_final) for arr in out_list)

    # Output details for apply_ufunc
    n_modes = len(m_filters) * len(periods_hours)
    output_dtypes = [np.float64] * (n_modes * 4)

    if temporal_mean:
        output_core_dims = [[cell_dim]] * (n_modes * 4)
    else:
        output_core_dims = [[time_dim, cell_dim]] * (n_modes * 4)

    dask_gufunc_kwargs = {'allow_rechunk': True, 'output_sizes': {cell_dim: npix}}
    if not temporal_mean:
        dask_gufunc_kwargs['output_sizes'][time_dim] = da.sizes[time_dim]

    logger.info("Running parallelized Fourier-wavelet analysis...")
    res_tuple = xr.apply_ufunc(
        _fourier_tide_ufunc,
        da,
        input_core_dims=[[time_dim, cell_dim]],
        output_core_dims=output_core_dims,
        exclude_dims=set((time_dim, cell_dim)),
        dask="parallelized",
        output_dtypes=output_dtypes,
        dask_gufunc_kwargs=dask_gufunc_kwargs
    )

    # 5. Pack outputs into a dataset
    # We rebuild the structured dataset coordinates
    out_coords = {k: v for k, v in ds.coords.items() if k not in (time_dim, cell_dim)}
    if not temporal_mean:
        out_coords[time_dim] = ds[time_dim]
    out_coords[cell_dim] = np.arange(npix)

    # Add m and period dimensions
    p_timedeltas = [np.timedelta64(int(p), 'h') for p in periods_hours]
    out_coords['m'] = m_filters
    out_coords['period'] = p_timedeltas

    out_dims = ['m', 'period']
    if not temporal_mean:
        out_dims.append(time_dim)
    # Add non-core dims
    non_core_dims = [d for d in da.dims if d not in (time_dim, cell_dim)]
    out_dims.extend(non_core_dims)
    out_dims.append(cell_dim)

    # Combine tuple elements into xarray DataArrays
    out_ds = xr.Dataset(coords=out_coords)
    var_units = da.attrs.get('units', '')

    mode_idx = 0
    # Group results by variable component
    comp_data = {c: [] for c in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy')}

    # Retrieve output arrays and organize them
    #apply_ufunc returns a single DataArray if n_outputs=1, else a tuple.
    # Since we have n_modes * 4 outputs, it is a list/tuple of DataArrays.
    if n_modes * 4 == 1:
        res_tuple = (res_tuple,)

    for m in m_filters:
        m_list = {c: [] for c in comp_data}
        for p in periods_hours:
            for comp_idx, comp in enumerate(('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy')):
                da_res = res_tuple[mode_idx * 4 + comp_idx]
                m_list[comp].append(da_res)
            mode_idx += 1
        for comp in comp_data:
            # Concatenate along period dimension
            comp_data[comp].append(xr.concat(m_list[comp], dim='period'))

    for comp in comp_data:
        # Concatenate along m dimension
        da_comb = xr.concat(comp_data[comp], dim='m')
        # Ensure dimensions order
        da_comb = da_comb.transpose(*out_dims)
        comp_type = 'Symmetric' if 'sym' in comp else 'Antisymmetric'
        metric = 'Amplitude' if 'amp' in comp else 'Phase'
        units = var_units if 'amp' in comp else 'rad'

        da_comb.attrs = {
            'units': units,
            'grid_mapping': 'healpix',
            'long_name': f'{comp_type} {metric}'
        }
        out_ds[f'{var_name}_{comp}'] = da_comb

    # Set coordinate attributes
    out_ds['period'].attrs = {'long_name': 'Tidal Period'}
    out_ds['m'].attrs = {'long_name': 'Zonal Wavenumber'}

    lon_da = xr.DataArray(target_lon, dims=cell_dim,
                          attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    lat_da = xr.DataArray(target_lat, dims=cell_dim,
                          attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    out_ds = out_ds.assign_coords(lon=lon_da, lat=lat_da)

    out_ds.attrs = ds.attrs.copy()
    out_ds = add_healpix_grid_mapping(out_ds, nside, order=hp_order)
    out_ds.attrs = append_history(
        out_ds.attrs,
        f"Fourier tidal analysis (periods: {periods_hours}h, "
        f"m: {m_filters}, temporal_mean={temporal_mean})."
    )

    return out_ds

