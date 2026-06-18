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
import scipy.fft as sp_fft
import xarray as xr
from scipy.optimize import fminbound
from scipy.special import gamma, gammainc

from healicon.grid import (get_cells_dim, get_healpix_order,
                           add_healpix_grid_mapping,
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

    Optimization (mmax truncation): only m = abs_m is ever nonzero in the alm array
    passed to alm2map, so we build *compact* arrays sized for mmax=abs_m
    (rather than the full mmax=lmax array implied by a bare
    hp.Alm.getsize(lmax)) and pass the explicit lmax/mmax through to
    alm2map. healpy's alm storage convention lays alms out in blocks of
    increasing m, and the index formula for a given (l, m) depends only on
    lmax, not mmax -- so an mmax=abs_m array is exactly the leading prefix
    of the full mmax=lmax array, and the idx_m positions computed below
    from the full-size (l, m) layout remain valid indices into the smaller
    compact array. This lets healpy's associated-Legendre-transform step
    (the dominant cost of alm2map for large lmax) skip all m > abs_m
    instead of wastefully evaluating them against all-zero coefficients.
    The synthesized maps are numerically unchanged.
    """
    npix = hp.nside2npix(nside)
    n_alm_compact = hp.Alm.getsize(lmax, abs_m)
    _, m_arr = hp.Alm.getlm(lmax)
    idx_m = np.where(m_arr == abs_m)[0]
    n_l = len(idx_m)

    basis_cos = np.zeros((n_l, npix))
    basis_sin = np.zeros((n_l, npix))

    def _synth_one(k):
        alm_c = np.zeros(n_alm_compact, dtype=complex)
        alm_c[idx_m[k]] = 1.0
        cos_map = hp.alm2map(alm_c, nside=nside, lmax=lmax, mmax=abs_m)

        alm_s = np.zeros(n_alm_compact, dtype=complex)
        alm_s[idx_m[k]] = 1j
        sin_map = hp.alm2map(alm_s, nside=nside, lmax=lmax, mmax=abs_m)
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

def wave_bases(
        mother: str,
        k: np.ndarray,
        scale: float | np.ndarray,
        param: float | None = None
) -> tuple[np.ndarray, float, float, int]:
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

    if param is None or param == -1:
        if mother == 'MORLET':
            param = 6.0
        elif mother == 'PAUL':
            param = 4.0
        elif mother == 'DOG':
            param = 2.0

    if mother == 'MORLET':
        k0 = np.copy(param)
        expnt = -(scale[:, None] * k[None, :] - k0) ** 2 / 2. * kplus[None, :]
        norm = np.sqrt(scale * k[1]) * (np.pi ** (-0.25)) * np.sqrt(n)
        daughter = norm[:, None] * np.exp(expnt) * kplus[None, :]
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
        coi = fourier_factor / np.sqrt(2)
        dofmin = 2
    elif mother == 'PAUL':
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
                mother='MORLET', param=None, gws=None, Y_is_var=False):
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
        if param is None or param == -1:
            param = 6.
            empir[1:] = ([0.776, 2.32, 0.60])
        fourier_factor = (4 * np.pi) / (param + np.sqrt(2 + param ** 2))
    elif mother == 'PAUL':
        empir = ([2, -1, -1, -1])
        if param is None or param == -1:
            param = 4
            empir[1:] = ([1.132, 1.17, 1.5])
        fourier_factor = (4 * np.pi) / (2 * param + 1)
    elif mother == 'DOG':
        empir = ([1., -1, -1, -1])
        if param is None or param == -1:
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


@lru_cache(maxsize=64)
def _wavelet_basis(
        n1: int,
        dt: float,
        pad: int,
        dj: float | None,
        s0: float | None,
        J1: int | None,
        mother: str,
        param: float | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Precompute the data-independent pieces of the wavelet transform.

    The wavelet basis ("daughter"), period, scale, and coi arrays depend
    only on the time-axis configuration (n1, dt, pad, dj, s0, J1, mother,
    param) -- never on the values of Y. wavelet() is called once per
    height level / zonal wavenumber (via _compute_cwt_coefficients), and
    those calls all share the same time-axis configuration within a given
    dataset, so this basis is built once per unique configuration and
    reused from cache thereafter.

    Returns:
        daughter: complex wavelet basis, shape (n_scales, n_padded).
        period: Fourier period for each scale.
        scale: wavelet scale array.
        coi_out: cone-of-influence array, length n1.
        n: padded length of the time series.
        nzeroes: number of zeros appended for padding (0 if pad != 1).
    """
    if s0 is None or s0 == -1:
        s0 = 2.0 * dt
    if dj is None or dj == -1:
        dj = 0.25
    if J1 is None or J1 == -1:
        J1 = int(np.fix((np.log(n1 * dt / s0) / np.log(2)) / dj))

    if pad == 1:
        base2 = np.fix(np.log(n1) / np.log(2) + 0.4999)
        nzeroes = int(2 ** (base2 + 1) - n1)
    else:
        nzeroes = 0

    n = n1 + nzeroes

    k_plus = np.arange(1, int(n / 2) + 1)
    k_plus = (k_plus * 2 * np.pi / (n * dt))
    k_minus = np.arange(1, int((n - 1) / 2) + 1)
    k_minus = np.sort((-k_minus * 2 * np.pi / (n * dt)))
    k = np.concatenate(([0.], k_plus, k_minus))

    if param is None or param == -1:
        if mother == 'MORLET':
            param = 6.0
        elif mother == 'PAUL':
            param = 4.0
        elif mother == 'DOG':
            param = 2.0

    if mother == 'MORLET':
        fourier_factor = 4 * np.pi / (param + np.sqrt(2 + param ** 2))
    elif mother == 'PAUL':
        fourier_factor = 4 * np.pi / (2 * param + 1)
    elif mother == 'DOG':
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * param + 1))
    else:
        raise ValueError("Invalid mother wavelet")

    j = np.arange(0, J1 + 1)
    scale = s0 * 2. ** (j * dj)
    period = scale * fourier_factor

    daughter, _, coi, _ = wave_bases(mother, k, scale, param)

    coi_out = coi * dt * np.concatenate((
        np.insert(np.arange(int((n1 + 1) / 2) - 1), [0], [1E-5]),
        np.insert(np.flipud(np.arange(0, int(n1 / 2) - 1)), [-1], [1E-5])
    ))

    return daughter, period, scale, coi_out, n, nzeroes


def wavelet(
        Y: np.ndarray,
        dt: float,
        pad: int = 1,
        dj: float | None = None,
        s0: float | None = None,
        J1: int | None = None,
        mother: str = 'MORLET',
        param: float | None = None,
        axis: int = -1
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the 1D Wavelet transform along a specified axis of a multidimensional array.
    
    Args:
        Y: Input data array.
        dt: Time step.
        pad: Whether to pad the data (1 for padding, 0 for no padding).
        dj: Scale parameter (spacing between discrete scales).
        s0: Initial scale.
        J1: Number of scales minus 1.
        mother: Name of the mother wavelet ('MORLET', 'PAUL', 'DOG').
        param: Additional parameter for the wavelet.
        axis: Axis along which to compute the transform.
    
    Returns:
        tuple containing (wave, period, scale, coi).

    Optimization (cached basis): the data-independent basis/period/scale/coi
    arrays are now built by the cached _wavelet_basis() helper above, so
    only the data-dependent demean/pad/FFT/multiply/IFFT steps run here on
    every call. Verified to reproduce the original output bit-for-bit.
    """
    n1 = Y.shape[axis]

    daughter, period, scale, coi_out, n, nzeroes = _wavelet_basis(
        int(n1), float(dt), int(pad),
        float(dj) if dj is not None else None,
        float(s0) if s0 is not None else None,
        int(J1) if J1 is not None else None,
        mother,
        float(param) if param is not None else None,
    )

    Y_work = np.moveaxis(Y, axis, -1)
    x = Y_work - np.mean(Y_work, axis=-1, keepdims=True)

    if nzeroes > 0:
        pad_shape = list(x.shape)
        pad_shape[-1] = nzeroes
        x = np.concatenate((x, np.zeros(pad_shape, dtype=x.dtype)), axis=-1)

    f = sp_fft.fft(x, axis=-1, workers=-1)
    wave = sp_fft.ifft(f[..., np.newaxis, :] * daughter, axis=-1, workers=-1)

    # Trim padding and move axes
    wave = wave[..., :n1]

    # Original order was (..., n_scales, time_dim) where time_dim was at -1
    # Move scales axis to front
    wave = np.moveaxis(wave, -2, 0)
    # Move time_dim back to its original axis (+1 because of new scales axis at front)
    wave = np.moveaxis(wave, -1, axis + 1)

    return wave, period, scale, coi_out


def _compute_cwt_coefficients(Ck: np.ndarray, Sk: np.ndarray, dt: float, dj: float, axis: int = 0):
    """Computes Morlet wavelet coefficients (Ak, Bk, ak, bk) for cosine (Ck) and sine (Sk) parts.

    Args:
        Ck: Cosine component array.
        Sk: Sine component array.
        dt: Time step.
        dj: Scale spacing.
        axis: Time dimension axis.

    Returns:
        Ak, Bk, ak, bk, period, scale, coi
    """
    param = 6
    pad = 1
    # Stack Ck and Sk to compute CWT in a single batch
    # Ck and Sk have shape (..., time_dim, ...). We stack them along a new first axis.
    Y_stacked = np.stack([Ck, Sk], axis=0)
    # The time dimension axis gets shifted by 1 due to the new first axis
    Y_w, period, scale, coi = wavelet(Y_stacked, dt, pad, dj, 2 * dt, mother='MORLET', param=param,
                                      axis=axis + 1)
    Ck_w0 = Y_w[:, 0, ...]
    Sk_w0 = Y_w[:, 1, ...]

    # Scale normalization: broadcast scale over all non-scale axes
    scale_factor = np.sqrt(dt / scale).reshape((-1,) + (1,) * (Ck_w0.ndim - 1))

    Ck_w = Ck_w0 * scale_factor
    Sk_w = Sk_w0 * scale_factor

    Ak = np.real(Ck_w)
    Bk = -np.imag(Ck_w)
    ak = np.real(Sk_w)
    bk = -np.imag(Sk_w)

    return Ak, Bk, ak, bk, period, scale, coi


def _calculate_amplitudes_and_phases(Ak: np.ndarray, Bk: np.ndarray, ak: np.ndarray,
                                     bk: np.ndarray, compute_phase: bool = True):
    """Computes westward/eastward amplitudes and phases from wavelet coefficients."""
    Rw = 0.5 * np.sqrt((Ak - bk) ** 2 + (Bk + ak) ** 2)
    Re = 0.5 * np.sqrt((Ak + bk) ** 2 + (Bk - ak) ** 2)
    if compute_phase:
        Pw = np.arctan2(Bk + ak, Ak - bk)
        Pe = np.arctan2(Bk - ak, Ak + bk)
    else:
        Pw, Pe = None, None
    return Rw, Re, Pw, Pe


# --------------------------------------------------------------------------
# XARRAY-NATIVE FOURIER-WAVELET API
# --------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _get_ring2nest(nside: int) -> np.ndarray:
    """Cached RING->NEST pixel permutation for a given nside.

    fourier_wavelet_spectrum previously called hp.ring2nest(nside, ...)
    fresh every invocation, and (inside the assign_to_cells closure) called
    hp.nest2ring(nside, ...) fresh on every one of up to ~10 calls per
    invocation, even though the permutation depends only on nside. Caching
    it removes that repeated work without changing any values.
    """
    return hp.ring2nest(nside, np.arange(hp.nside2npix(nside)))


@lru_cache(maxsize=32)
def _get_nest2ring(nside: int) -> np.ndarray:
    """Cached NEST->RING pixel permutation for a given nside (see
    _get_ring2nest above for rationale)."""
    return hp.nest2ring(nside, np.arange(hp.nside2npix(nside)))


# --------------------------------------------------------------------------
# RING-BASED FOURIER EXTRACTION HELPERS
# --------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _get_ring_fourier_weights(nside: int, zwn: int):
    """Cached per-ring Fourier-extraction weights and ring layout.

    Returns:
        startpix, ringpix: from hp.ringinfo(nside, rings).
        W: per-pixel weight array, shape (npix,) -- real for zwn == 0,
           complex for zwn != 0.
    """
    rings = np.arange(1, 4 * nside)
    startpix, ringpix, _, _, shifted = hp.ringinfo(nside, rings)

    if zwn == 0:
        W = 1.0 / np.repeat(ringpix, ringpix)
    else:
        lons = _get_lon_rad(nside)
        W = np.exp(-1j * zwn * lons) / np.repeat(ringpix, ringpix)

    return startpix, ringpix, W


def _extract_ring_fourier_coefs(data_np: np.ndarray, nside: int, zwn: int) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts the zwn-th Fourier coefficient from each HEALPix isolatitude ring.

    Args:
        data_np: Array of shape (n_time, npix) in RING ordering
        nside: HEALPix Nside
        zwn: Zonal wavenumber

    Returns:
        tuple containing (Ck, Sk, startpix, ringpix) where Ck and Sk
        have shape (n_time, n_rings).
    """
    startpix, ringpix, W = _get_ring_fourier_weights(nside, zwn)

    data_W = data_np * W[None, :]
    fm_reduced = np.add.reduceat(data_W, startpix, axis=-1)

    if zwn == 0:
        Ck = fm_reduced
        Sk = np.zeros_like(Ck)
    else:
        Ck = 2.0 * fm_reduced.real
        Sk = -2.0 * fm_reduced.imag

    return Ck, Sk, startpix, ringpix


def fourier_wavelet_spectrum(da: xr.DataArray, zwn: int,
                             time_dim: str = 'time',
                             dt: float = 1.0, dj: float = 0.1,
                             min_t: float = -1, max_t: float = np.inf,
                             periods_to_reconstruct: list[float] | None = None,
                             order: str | None = None) -> xr.Dataset:
    """
    Computes the Fourier-Wavelet spectrum of a HEALPix DataArray.
    
    This is the Fourier analogue of ``spherical_harmonic_wavelet_spectrum``.
    It extracts Fourier coefficients natively from the HEALPix isolatitude
    rings, computes the wavelet transform, decomposes into symmetric and
    antisymmetric modes, and maps the results back to the original cells.
    """
    if time_dim not in da.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in DataArray")

    cell_dim = get_cells_dim(da)
    if cell_dim is None:
        raise ValueError("No spatial HEALPix cell dimension found in DataArray.")

    # Determine pixel ordering
    if order is not None:
        _order = order.lower()
    else:
        _order = get_healpix_order(da)

    is_nested = _order == 'nested'
    logger.debug(f"HEALPix ordering: {'nested' if is_nested else 'ring'}")

    npix = da.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    data_np = da.values

    if is_nested:
        data_ring = data_np[..., _get_ring2nest(nside)]
    else:
        data_ring = data_np

    Ck, Sk, start_pix, ring_pix = _extract_ring_fourier_coefs(data_ring, nside, zwn)

    Ak, Bk, ak, bk, period, scale, coi = _compute_cwt_coefficients(
        Ck, Sk, dt, dj, axis=0
    )

    period_mask = (period >= min_t) & (period <= max_t)
    period = period[period_mask]

    Ak = Ak[period_mask, ...]
    Bk = Bk[period_mask, ...]
    ak = ak[period_mask, ...]
    bk = bk[period_mask, ...]

    Rw_rings, Re_rings, Pw_rings, Pe_rings = _calculate_amplitudes_and_phases(Ak, Bk, ak, bk)

    # Hoisted out of assign_to_cells: this permutation depends only on
    # (nside, is_nested), not on the values being mapped, yet the closure
    # used to recompute hp.nest2ring fresh on every one of its up to ~10
    # calls per invocation. _get_nest2ring caches it across nside values
    # as well, so repeated calls (e.g. across height levels) are free.
    nest2ring_idx = _get_nest2ring(nside) if is_nested else None

    def assign_to_cells(ring_vals):
        mapped = np.repeat(ring_vals, ring_pix, axis=-1)
        if is_nested:
            return mapped[..., nest2ring_idx]
        return mapped

    reconstructed_vars = {}
    if periods_to_reconstruct is not None:
        Ak_sym = 0.5 * (Ak + np.flip(Ak, axis=-1))
        Ak_asy = 0.5 * (Ak - np.flip(Ak, axis=-1))
        Bk_sym = 0.5 * (Bk + np.flip(Bk, axis=-1))
        Bk_asy = 0.5 * (Bk - np.flip(Bk, axis=-1))
        ak_sym = 0.5 * (ak + np.flip(ak, axis=-1))
        ak_asy = 0.5 * (ak - np.flip(ak, axis=-1))
        bk_sym = 0.5 * (bk + np.flip(bk, axis=-1))
        bk_asy = 0.5 * (bk - np.flip(bk, axis=-1))

        for target_p in periods_to_reconstruct:
            idx = np.argmin(np.abs(period - target_p))
            actual_p = period[idx]
            p_str = f"{actual_p:.1f}"

            Rw_sym_idx, Re_sym_idx, Pw_sym_idx, Pe_sym_idx = _calculate_amplitudes_and_phases(
                Ak_sym[idx], Bk_sym[idx], ak_sym[idx], bk_sym[idx]
            )
            Rw_asy_idx, Re_asy_idx, Pw_asy_idx, Pe_asy_idx = _calculate_amplitudes_and_phases(
                Ak_asy[idx], Bk_asy[idx], ak_asy[idx], bk_asy[idx]
            )

            reconstructed_vars[f"amp_sym_westward_{p_str}"] = ([time_dim, cell_dim],
                                                               assign_to_cells(Rw_sym_idx))
            reconstructed_vars[f"amp_sym_eastward_{p_str}"] = ([time_dim, cell_dim],
                                                               assign_to_cells(Re_sym_idx))
            reconstructed_vars[f"pha_sym_westward_{p_str}"] = ([time_dim, cell_dim],
                                                               assign_to_cells(Pw_sym_idx))
            reconstructed_vars[f"pha_sym_eastward_{p_str}"] = ([time_dim, cell_dim],
                                                               assign_to_cells(Pe_sym_idx))

            reconstructed_vars[f"amp_asy_westward_{p_str}"] = ([time_dim, cell_dim],
                                                               assign_to_cells(Rw_asy_idx))
            reconstructed_vars[f"amp_asy_eastward_{p_str}"] = ([time_dim, cell_dim],
                                                               assign_to_cells(Re_asy_idx))
            reconstructed_vars[f"pha_asy_westward_{p_str}"] = ([time_dim, cell_dim],
                                                               assign_to_cells(Pw_asy_idx))
            reconstructed_vars[f"pha_asy_eastward_{p_str}"] = ([time_dim, cell_dim],
                                                               assign_to_cells(Pe_asy_idx))

    out_coords = {k: v for k, v in da.coords.items() if k != cell_dim}
    out_coords['period'] = period

    target_lon, target_lat = get_healpix_coords(nside)
    out_coords[cell_dim] = np.arange(npix)
    if is_nested:
        target_lon = ensure_original_order(target_lon, 'nested')
        target_lat = ensure_original_order(target_lat, 'nested')

    # Global mean amplitude diagnostics
    global_amp_w = (Rw_rings * ring_pix).sum(axis=-1) / npix
    global_amp_e = (Re_rings * ring_pix).sum(axis=-1) / npix

    data_vars = {
        'global_amplitude_westward': (['period', time_dim], global_amp_w),
        'global_amplitude_eastward': (['period', time_dim], global_amp_e),
        'coi': ([time_dim], coi)
    }
    data_vars.update(reconstructed_vars)

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords=out_coords
    )

    lon_da = xr.DataArray(target_lon, dims=cell_dim,
                          attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    lat_da = xr.DataArray(target_lat, dims=cell_dim,
                          attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    ds_out = ds_out.assign_coords(lon=lon_da, lat=lat_da)

    ds_out.attrs = da.attrs
    ds_out.attrs['zonal_wavenumber'] = zwn
    ds_out.attrs['dj'] = dj
    ds_out.attrs['dt'] = dt

    ds_out = add_healpix_grid_mapping(ds_out, nside, order=_order)

    return ds_out


# --------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _get_lon_rad(nside: int) -> np.ndarray:
    """Cached per-pixel longitude (radians, RING ordering) for a given nside.

    hp.pix2ang's phi (longitude) output depends only on nside, not on zwn
    or any data, but was being computed independently in two places
    (spherical_harmonic_wavelet_spectrum's reconstruction step, and
    _get_ring_fourier_weights's zwn != 0 branch). Both now share this one
    cached array per nside.
    """
    _, lon_rad = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    return lon_rad


@lru_cache(maxsize=32)
def _get_m_mask(lmax: int, abs_m: int) -> tuple[np.ndarray, int]:
    """Cached boolean mask selecting the m = abs_m block of an alm array.

    spherical_harmonic_wavelet_spectrum previously called
    hp.Alm.getlm(lmax) -> (l_arr, m_arr) every invocation just to build
    this mask, but l_arr is never used anywhere in that function. This
    helper computes (and caches) only the m_mask / n_l that are actually
    needed, for the same (lmax, abs_m) pair recurring across height
    levels / apply_ufunc slices.
    """
    _, m_arr = hp.Alm.getlm(lmax)
    m_mask = (m_arr == abs_m)
    n_l = int(np.sum(m_mask))
    return m_mask, n_l


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
        is_nested = get_healpix_order(da) == 'nested'
    logger.debug(f"HEALPix ordering: {'nested' if is_nested else 'ring'}")
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
    abs_m = abs(zwn)
    # l_arr from hp.Alm.getlm(lmax) was computed here but never used; only
    # m_mask/n_l are needed, and they depend solely on (lmax, abs_m), so we
    # fetch them from the cached helper instead of recomputing on every call.
    m_mask, n_l = _get_m_mask(lmax, abs_m)
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

    logger.debug(f"Computing SH spectral coefficients for m={abs_m} "
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
            logger.debug("No NaN values detected. Setting map2alm_iter=0 for ~4x speedup.")
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
    logger.debug("Applying Continuous Wavelet Transform...")
    Ak, Bk, ak, bk, period, scale, coi = _compute_cwt_coefficients(
        A_lm, B_lm, dt, dj, axis=0
    )

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
    Rw_l, Re_l, _, _ = _calculate_amplitudes_and_phases(Ak, Bk, ak, bk, compute_phase=False)

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
        from .tides import _get_symmetric_pixels
        sym_idx = _get_symmetric_pixels(nside, is_nested=False)

        # Basis maps are cached across calls for the same (nside, lmax, m).
        # They are in RING ordering (healpy default).
        basis_cos, basis_sin = _get_basis_maps(nside, lmax, abs_m)
        basis_stacked = np.vstack([basis_cos, basis_sin])

        # Hoist lon_rad: pix2ang is the same for every period (item 4).
        # Also now cached across calls by nside via _get_lon_rad, instead
        # of calling hp.pix2ang fresh on every invocation.
        lon_rad = _get_lon_rad(nside)
        abs_m_lon = abs_m * lon_rad  # pre-scale; reused in every chunk

        for p_idx, target_p in enumerate(periods_to_reconstruct):
            idx = np.argmin(np.abs(period - target_p))
            actual_p = period[idx]

            # Convert E/W A_lm/B_lm to Re(alm)/Im(alm) for synthesis.
            # The factor-of-2 is absorbed: A_lm = 2·Re(alm), B_lm = -2·Im(alm)
            # cw_l = 0.5 * (Ak - bk) -> re_w = 0.25 * (Ak - bk)
            re_w = 0.25 * (Ak[idx] - bk[idx])  # (n_time, n_l)
            im_w = 0.25 * (Bk[idx] + ak[idx])
            re_e = 0.25 * (Ak[idx] + bk[idx])
            im_e = 0.25 * (Bk[idx] - ak[idx])

            # Pre-allocate eight output arrays in a dict
            shape = (n_time, npix)
            chunked = {
                ('westward', 'sym', 'amp'): np.empty(shape, dtype=np.float64),
                ('westward', 'sym', 'pha'): np.empty(shape, dtype=np.float64),
                ('westward', 'asy', 'amp'): np.empty(shape, dtype=np.float64),
                ('westward', 'asy', 'pha'): np.empty(shape, dtype=np.float64),
                ('eastward', 'sym', 'amp'): np.empty(shape, dtype=np.float64),
                ('eastward', 'sym', 'pha'): np.empty(shape, dtype=np.float64),
                ('eastward', 'asy', 'amp'): np.empty(shape, dtype=np.float64),
                ('eastward', 'asy', 'pha'): np.empty(shape, dtype=np.float64),
            }

            # Process in time-chunks to cap intermediate memory at O(TIME_CHUNK × npix)
            for t0 in range(0, n_time, TIME_CHUNK):
                t1 = min(t0 + TIME_CHUNK, n_time)
                sl = slice(t0, t1)

                # Vectorized spatial synthesis via stacked basis
                # 0: w, 1: wH, 2: e, 3: eH
                coeff_stack = np.array([
                    np.hstack([re_w[sl], im_w[sl]]),
                    np.hstack([-im_w[sl], re_w[sl]]),
                    np.hstack([re_e[sl], im_e[sl]]),
                    np.hstack([-im_e[sl], re_e[sl]])
                ])

                # Single fused matrix multiplication -> shape: (4, chunk_len, npix)
                maps_all = coeff_stack @ basis_stacked

                # Single fused symmetric / antisymmetric decomposition -> shape: (2, 4, chunk_len, npix)
                maps_sa = np.stack([
                    0.5 * (maps_all + maps_all[..., sym_idx]),  # symmetric
                    0.5 * (maps_all - maps_all[..., sym_idx])  # antisymmetric
                ], axis=0)

                # Amplitude = sqrt(map^2 + mapH^2) computed for sym and asy simultaneously
                amp_w = np.sqrt(maps_sa[:, 0] ** 2 + maps_sa[:, 1] ** 2)
                amp_e = np.sqrt(maps_sa[:, 2] ** 2 + maps_sa[:, 3] ** 2)

                # Phase: arctan2(-H, map) then remove zonal-wavenumber lon phase
                phase_offset = abs_m_lon[None, :]
                pha_w = np.mod(np.arctan2(-maps_sa[:, 1], maps_sa[:, 0]) - phase_offset + np.pi,
                               2 * np.pi) - np.pi
                pha_e = np.mod(np.arctan2(-maps_sa[:, 3], maps_sa[:, 2]) - phase_offset + np.pi,
                               2 * np.pi) - np.pi

                # Assign back to output arrays (index 0 is sym, index 1 is asy)
                chunked[('westward', 'sym', 'amp')][sl], chunked[('westward', 'asy', 'amp')][sl] = \
                    amp_w[0], amp_w[1]
                chunked[('westward', 'sym', 'pha')][sl], chunked[('westward', 'asy', 'pha')][sl] = \
                    pha_w[0], pha_w[1]
                chunked[('eastward', 'sym', 'amp')][sl], chunked[('eastward', 'asy', 'amp')][sl] = \
                    amp_e[0], amp_e[1]
                chunked[('eastward', 'sym', 'pha')][sl], chunked[('eastward', 'asy', 'pha')][sl] = \
                    pha_e[0], pha_e[1]

            # Collect into main dict
            for key, arr in chunked.items():
                spatial_arrays[(p_idx, *key)] = (actual_p, arr)

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
