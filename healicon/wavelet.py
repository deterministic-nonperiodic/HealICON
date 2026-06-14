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
from healicon.grid import get_cells_dim, add_healpix_grid_mapping

logger = logging.getLogger(__name__)


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

    # Suppress healpy INFO logs during synthesis
    hp_logger = logging.getLogger('healpy')
    old_level = hp_logger.level
    hp_logger.setLevel(logging.WARNING)

    def _synth_one(k):
        alm_c = np.zeros(n_alm_total, dtype=complex)
        alm_c[idx_m[k]] = 1.0
        cos_map = hp.alm2map(alm_c, nside=nside)

        alm_s = np.zeros(n_alm_total, dtype=complex)
        alm_s[idx_m[k]] = 1j
        sin_map = hp.alm2map(alm_s, nside=nside)
        return k, cos_map, sin_map

    try:
        n_workers = min(32, os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for k, cos_map, sin_map in pool.map(_synth_one, range(n_l)):
                basis_cos[k] = cos_map
                basis_sin[k] = sin_map
    finally:
        hp_logger.setLevel(old_level)

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

    # 1. Fourier Analysis in Longitude
    lon_axis = da.dims.index(lon_dim)
    L = da.sizes[lon_dim]

    # Vectorized FFT
    f_data = np.fft.fft(da.values, axis=lon_axis) / L
    f_zwn = np.take(f_data, zwn, axis=lon_axis)

    if zwn == 0:
        Ck = np.real(f_zwn)
        Sk = np.zeros_like(Ck)
    else:
        Ck = 2 * np.real(f_zwn)
        Sk = -2 * np.imag(f_zwn)

    # Find new time axis index after extracting longitude
    time_axis = list(da.dims).index(time_dim)
    if time_axis > lon_axis:
        time_axis -= 1

    n_time = da.sizes[time_dim]

    # 2. Wavelet Analysis in Time
    param = 6
    pad = 1

    Ck_w0, period, scale, coi = wavelet(Ck, dt, pad, dj, 2 * dt, mother='MORLET', param=param,
                                        axis=time_axis)
    Sk_w0, _, _, _ = wavelet(Sk, dt, pad, dj, 2 * dt, mother='MORLET', param=param, axis=time_axis)

    # Ck_w0 has shape (n_scales, ..., time, ...)
    # Create broadcastable sqrt factor
    # scale is (n_scales,)
    scale_factor = np.sqrt(dt / scale)

    # Broadcast scale_factor along the first axis (scales)
    bcast_shape = [1] * Ck_w0.ndim
    bcast_shape[0] = len(scale)
    scale_factor_bcast = scale_factor.reshape(bcast_shape)

    Ck_w = Ck_w0 * scale_factor_bcast
    Sk_w = Sk_w0 * scale_factor_bcast

    # 3. Calculation of Amplitude and Phase
    Ak = np.real(Ck_w)
    Bk = -np.imag(Ck_w)
    ak = np.real(Sk_w)
    bk = -np.imag(Sk_w)

    Rw = 0.5 * np.sqrt((Ak - bk) ** 2 + (Bk + ak) ** 2)
    Re = 0.5 * np.sqrt((Ak + bk) ** 2 + (Bk - ak) ** 2)
    Pw = np.arctan2(Bk + ak, Ak - bk)
    Pe = np.arctan2(Bk - ak, Ak + bk)

    # Filter by period min/max
    period_mask = (period >= min_t) & (period <= max_t)

    period = period[period_mask]
    scale = scale[period_mask]

    # We slice along the scales axis (axis 0)
    Rw = Rw[period_mask, ...]
    Re = Re[period_mask, ...]
    Pw = Pw[period_mask, ...]
    Pe = Pe[period_mask, ...]

    # Significance levels (we calculate variance over the time axis)
    var_Ck = np.var(Ck, axis=time_axis)
    var_Sk = np.var(Sk, axis=time_axis)
    max_var = np.maximum(var_Ck, var_Sk)

    # Vectorized significance calculation (no loops over spatial dimensions)
    sig = wave_signif(max_var, dt, scale, lag1=0.72, siglvl=0.95, mother='MORLET', param=param,
                      Y_is_var=True)
    # sig shape: (*max_var.shape, n_scales)
    sig = sig * (0.5 * np.sqrt(dt / scale))

    if np.ndim(max_var) == 0:
        signif_arr = sig
    else:
        signif_arr = np.moveaxis(sig, -1, 0)  # shape: (n_scales, *max_var.shape)

    # Broadcast significance over time axis
    # The output should match Rw shape: (n_scales, ..., time_dim, ...)
    # signif_arr lacks the time dimension. We need to expand dims at time_axis + 1.
    signif = np.expand_dims(signif_arr, axis=time_axis + 1)
    signif = np.repeat(signif, n_time, axis=time_axis + 1)

    # Construct output Dataset
    # Drop lon_dim from original coords
    out_coords = {k: v for k, v in da.coords.items() if k != lon_dim}
    out_coords['period'] = period

    # Build the dims tuple: ('period', dim1, dim2, ...) matching the new shape
    out_dims = ['period']
    for d in da.dims:
        if d != lon_dim:
            out_dims.append(d)

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

    def _map2alm_one(i):
        """Process a single time step (thread-safe, GIL-releasing)."""
        ring_map = da_np[i]
        valid = ~np.isnan(ring_map)
        if not np.any(valid):
            return i, None
        filled = np.where(valid, ring_map, 0.0)
        alm = hp.map2alm(filled, lmax=lmax, iter=map2alm_iter)
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

    # Scale normalization (Torrence & Compo)
    scale_factor = np.sqrt(dt / scale)
    bcast_shape = [len(scale)] + [1] * (W_A.ndim - 1)
    W_A *= scale_factor.reshape(bcast_shape)
    W_B *= scale_factor.reshape(bcast_shape)

    # ------------------------------------------------------------------
    # E/W separation — same analytic signal as fourier_wavelet
    # ------------------------------------------------------------------
    Ak = np.real(W_A)  # (n_scales, n_time, n_l)
    Bk = -np.imag(W_A)
    ak = np.real(W_B)
    bk = -np.imag(W_B)

    # Per-degree westward/eastward amplitude
    Rw_l = 0.5 * np.sqrt((Ak - bk) ** 2 + (Bk + ak) ** 2)
    Re_l = 0.5 * np.sqrt((Ak + bk) ** 2 + (Bk - ak) ** 2)

    # Filter by period
    period_mask = (period >= min_t) & (period <= max_t)
    period = period[period_mask]
    Rw_l = Rw_l[period_mask, ...]
    Re_l = Re_l[period_mask, ...]
    Ak = Ak[period_mask, ...]
    Bk = Bk[period_mask, ...]
    ak = ak[period_mask, ...]
    bk = bk[period_mask, ...]

    # ------------------------------------------------------------------
    #  Global-mean amplitude via Parseval
    # ------------------------------------------------------------------
    # Parseval normalization: for healpy's orthonormal SH convention,
    # Amp_physical = sqrt(sum_l Rw_l^2 / (2*pi)).
    # See docs/spectral_methods.tex §4.3 for the full derivation.
    norm = 1.0 / np.sqrt(2.0 * np.pi)
    global_amp_w = norm * np.sqrt(np.sum(Rw_l ** 2, axis=-1))
    global_amp_e = norm * np.sqrt(np.sum(Re_l ** 2, axis=-1))

    out_coords = {k: v for k, v in da.coords.items() if k not in [cell_dim]}
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

    if periods_to_reconstruct is not None:
        # Basis maps from alm2map are ALWAYS in RING ordering, so the
        # symmetric-pixel index must also be computed in RING.
        sym_idx = _get_symmetric_pixels(nside, is_nested=False)

        # Basis maps are cached across calls for the same (nside, lmax, m).
        # They are in RING ordering (healpy default).
        basis_cos, basis_sin = _get_basis_maps(nside, lmax, abs_m)
        _, lon_rad = hp.pix2ang(nside, np.arange(npix))

        for target_p in periods_to_reconstruct:
            idx = np.argmin(np.abs(period - target_p))
            actual_p = period[idx]

            # E/W separated spectral coefficients in A_lm/B_lm space
            # Ak, Bk, ak, bk have shape (n_scales, n_time, n_l)
            # After period selection → (n_time, n_l)
            cw_l = 0.5 * (Ak[idx] - bk[idx])  # westward A_lm
            sw_l = 0.5 * (Bk[idx] + ak[idx])  # westward B_lm
            ce_l = 0.5 * (Ak[idx] + bk[idx])  # eastward A_lm
            se_l = 0.5 * (Bk[idx] - ak[idx])  # eastward B_lm

            # Convert E/W A_lm/B_lm to Re(alm)/Im(alm) for synthesis.
            # Scale factor dt/scale is already applied to W_A/W_B.
            re_w = cw_l / 2.0  # (n_time, n_l)
            im_w = sw_l / 2.0
            re_e = ce_l / 2.0
            im_e = se_l / 2.0

            # Pre-allocate output arrays
            amp_w_sym = np.empty((n_time, npix), dtype=np.float64)
            amp_w_asy = np.empty((n_time, npix), dtype=np.float64)
            amp_e_sym = np.empty((n_time, npix), dtype=np.float64)
            amp_e_asy = np.empty((n_time, npix), dtype=np.float64)
            pha_w_sym = np.empty((n_time, npix), dtype=np.float64)
            pha_w_asy = np.empty((n_time, npix), dtype=np.float64)
            pha_e_sym = np.empty((n_time, npix), dtype=np.float64)
            pha_e_asy = np.empty((n_time, npix), dtype=np.float64)

            # Process in time-chunks to cap intermediate memory at
            # O(TIME_CHUNK × npix) instead of O(n_time × npix).
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

                amp_w_sym[sl] = np.sqrt(mw_s ** 2 + mwH_s ** 2)
                pha_w_sym[sl] = np.mod(np.arctan2(-mwH_s, mw_s) - abs_m * lon_rad[None, :] + np.pi,
                                       2 * np.pi) - np.pi

                amp_w_asy[sl] = np.sqrt(mw_a ** 2 + mwH_a ** 2)
                pha_w_asy[sl] = np.mod(np.arctan2(-mwH_a, mw_a) - abs_m * lon_rad[None, :] + np.pi,
                                       2 * np.pi) - np.pi

                me_s = 0.5 * (me + me[:, sym_idx])
                meH_s = 0.5 * (meH + meH[:, sym_idx])
                me_a = 0.5 * (me - me[:, sym_idx])
                meH_a = 0.5 * (meH - meH[:, sym_idx])

                amp_e_sym[sl] = np.sqrt(me_s ** 2 + meH_s ** 2)
                pha_e_sym[sl] = np.mod(np.arctan2(-meH_s, me_s) - abs_m * lon_rad[None, :] + np.pi,
                                       2 * np.pi) - np.pi
                amp_e_asy[sl] = np.sqrt(me_a ** 2 + meH_a ** 2)
                pha_e_asy[sl] = np.mod(np.arctan2(-meH_a, me_a) - abs_m * lon_rad[None, :] + np.pi,
                                       2 * np.pi) - np.pi

            p_str = f"{actual_p:.1f}"

            out_vars[f'amp_sym_westward_{p_str}'] = ([time_dim, cell_dim], amp_w_sym)
            out_vars[f'amp_asy_westward_{p_str}'] = ([time_dim, cell_dim], amp_w_asy)
            out_vars[f'amp_sym_eastward_{p_str}'] = ([time_dim, cell_dim], amp_e_sym)
            out_vars[f'amp_asy_eastward_{p_str}'] = ([time_dim, cell_dim], amp_e_asy)

            out_vars[f'pha_sym_westward_{p_str}'] = ([time_dim, cell_dim], pha_w_sym)
            out_vars[f'pha_asy_westward_{p_str}'] = ([time_dim, cell_dim], pha_w_asy)
            out_vars[f'pha_sym_eastward_{p_str}'] = ([time_dim, cell_dim], pha_e_sym)
            out_vars[f'pha_asy_eastward_{p_str}'] = ([time_dim, cell_dim], pha_e_asy)

    # Build output with proper HEALPix cell coordinates
    from .grid import get_healpix_coords, ensure_original_order
    target_lon, target_lat = get_healpix_coords(nside)
    out_coords[cell_dim] = np.arange(npix)

    if is_nested:
        # Reorder all reconstructed spatial maps back to nested ordering
        for k in out_vars:
            if isinstance(out_vars[k], tuple) and len(out_vars[k]) == 2:
                dims, arr = out_vars[k]
                if cell_dim in dims:
                    arr_nest = ensure_original_order(arr, 'nested')
                    out_vars[k] = (dims, arr_nest)
        target_lon = ensure_original_order(target_lon, 'nested')
        target_lat = ensure_original_order(target_lat, 'nested')

    ds = xr.Dataset(data_vars=out_vars, coords=out_coords)
    ds['lon'] = (cell_dim, target_lon)
    ds['lat'] = (cell_dim, target_lat)
    ds['lon'].attrs = {'standard_name': 'longitude', 'units': 'degrees_east'}
    ds['lat'].attrs = {'standard_name': 'latitude', 'units': 'degrees_north'}

    ds.attrs['zonal_wavenumber'] = zwn
    ds.attrs['dj'] = dj
    ds.attrs['dt'] = dt

    # Attach CF-compliant HEALPix grid mapping
    ds = add_healpix_grid_mapping(ds, nside, order=order_str)

    return ds


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
    from .grid import (get_cells_dim, get_healpix_order,
                       append_history, add_healpix_grid_mapping,
                       get_healpix_coords, ensure_original_order)

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
    if np.issubdtype(time_vals.dtype, np.datetime64):
        dt_hours = float(np.median(np.diff(time_vals))
                         / np.timedelta64(1, 'h'))
    elif np.issubdtype(time_vals.dtype, np.timedelta64):
        dt_hours = float(np.median(np.diff(time_vals))
                         / np.timedelta64(1, 'h'))
    else:
        dt_hours = float(np.median(np.diff(time_vals)))
    logger.info(f"Inferred dt = {dt_hours:.2f} hours from '{time_dim}' "
                f"coordinate.")

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

        def _func(data_np):
            # data_np: (time, cells) numpy array
            da_tmp = xr.DataArray(
                data_np, dims=[time_dim, cell_dim],
                coords={time_dim: ds[time_dim]},
            )
            ds_w = spherical_harmonic_wavelet_spectrum(
                da_tmp, zwn=zwn, dt=dt_hours, dj=dj,
                lmax=lmax, map2alm_iter=map2alm_iter,
                periods_to_reconstruct=periods_for_wavelet,
                order=hp_order,
            )

            outputs = []
            for mode in zwn_modes:
                prefix = f"amp_sym_{mode['dir']}_"
                match = [v for v in ds_w.data_vars if v.startswith(prefix)]
                best = min(match, key=lambda v: abs(
                    float(v.replace(prefix, '')) - mode['period_h']))
                p_str = best.replace(prefix, '')

                for comp in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy'):
                    arr = ds_w[f"{comp}_{mode['dir']}_{p_str}"].values
                    if temporal_mean:
                        base_comp = comp.split('_')[1]  # 'sym' or 'asy'
                        amp_vals = ds_w[f"amp_{base_comp}_{mode['dir']}_{p_str}"].values
                        pha_vals = ds_w[f"pha_{base_comp}_{mode['dir']}_{p_str}"].values

                        # ── Time and space demodulation to match LS method exactly ──
                        # 1. Infer t_hours_vals exactly as in compute_tidal_analysis
                        time_vals = ds[time_dim].values
                        if time_dim == 'lst':
                            if np.issubdtype(time_vals.dtype, np.timedelta64):
                                t_hours_vals = time_vals / np.timedelta64(1, 'h')
                            else:
                                t_hours_vals = time_vals
                        else:
                            if np.issubdtype(time_vals.dtype, np.datetime64) or np.issubdtype(
                                    time_vals.dtype, np.timedelta64):
                                t_hours_vals = (time_vals - time_vals[0]) / np.timedelta64(1, 'h')
                            else:
                                t_hours_vals = time_vals - time_vals[0]

                        from healicon.grid import get_healpix_coords, ensure_original_order
                        nside_local = hp.npix2nside(amp_vals.shape[1])
                        target_lon_deg, _ = get_healpix_coords(nside_local)
                        if hp_order == 'nested':
                            target_lon_deg = ensure_original_order(target_lon_deg, 'nested')
                        phi_da = np.radians(target_lon_deg)
                        target_m = mode['m']

                        omega = 2 * np.pi / float(mode['period_h'])
                        wt = omega * t_hours_vals[:, None]

                        # Reconstruct spatial wave (with negative phase convention of Morlet CWT)
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

                        if 'pha' in comp:
                            arr = np.arctan2(imag_part, real_part)
                        else:
                            arr = np.sqrt(real_part ** 2 + imag_part ** 2)
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

    # ── Build output Dataset ─────────────────────────────────────────
    m_vals = sorted(set(mode['m'] for mode in modes))
    period_td = np.array(
        [np.timedelta64(int(p * 3600), 's') for p in periods_hours])

    # Stack along m and period dimensions
    var_units = ds[var_name].attrs.get('units', '')
    data_vars = {}

    for comp in ('amp_sym', 'amp_asy', 'pha_sym', 'pha_asy'):
        rows = []
        for m in m_vals:
            cols = []
            for p in periods_hours:
                cols.append(assembled[(m, p, comp)])
            rows.append(xr.concat(cols, dim='period'))
        stacked = xr.concat(rows, dim='m')
        stacked = stacked.assign_coords(
            m=('m', np.array(m_vals)),
            period=('period', period_td),
        )
        data_vars[f'{var_name}_{comp}'] = stacked

    out_ds = xr.Dataset(data_vars)

    # Spatial coordinates
    target_lon, target_lat = get_healpix_coords(nside)
    if hp_order == 'nested':
        target_lon = ensure_original_order(target_lon, 'nested')
        target_lat = ensure_original_order(target_lat, 'nested')
    out_ds['lon'] = (cell_dim, target_lon)
    out_ds['lat'] = (cell_dim, target_lat)
    out_ds['lon'].attrs = {'standard_name': 'longitude',
                           'units': 'degrees_east'}
    out_ds['lat'].attrs = {'standard_name': 'latitude',
                           'units': 'degrees_north'}

    # Variable metadata
    for comp in ('sym', 'asy'):
        label = 'Symmetric' if comp == 'sym' else 'Antisymmetric'
        out_ds[f'{var_name}_amp_{comp}'].attrs = {
            'units': var_units, 'long_name': f'{label} Amplitude',
            'grid_mapping': 'healpix',
        }
        out_ds[f'{var_name}_pha_{comp}'].attrs = {
            'units': 'rad', 'long_name': f'{label} Phase',
            'grid_mapping': 'healpix',
        }

    out_ds['m'].attrs = {'long_name': 'Zonal Wavenumber',
                         'description': 'positive=westward, negative=eastward'}
    out_ds['period'].attrs = {'long_name': 'Tidal Period'}

    # Preserve dataset attributes and grid mapping
    out_ds.attrs = ds.attrs.copy()
    out_ds = add_healpix_grid_mapping(out_ds, nside, order=hp_order)
    out_ds.attrs = append_history(
        out_ds.attrs,
        f"Wavelet tidal analysis (periods: {periods_hours}h, "
        f"m: {m_filters}, temporal_mean={temporal_mean})."
    )

    return out_ds
