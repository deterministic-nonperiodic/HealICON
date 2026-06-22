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
from ._common import get_progress_bar


@lru_cache(maxsize=4)
def _get_basis_maps(nside: int, lmax: int, abs_m: int) -> np.ndarray:
    """Precompute cos/sin spatial basis maps for a single zonal wavenumber.

    Returns a single contiguous array of shape ``(2 * n_l, npix)`` where the
    first ``n_l`` rows are cos-basis maps and the last ``n_l`` rows are
    sin-basis maps.  Storing them in one allocation avoids the ``np.vstack``
    copy that would otherwise be needed at the call site.

    The result is cached (``maxsize=4``) so repeated calls with the same
    ``(nside, lmax, abs_m)`` triple are free.
    """
    npix = hp.nside2npix(nside)
    n_alm_compact = hp.Alm.getsize(lmax, abs_m)
    _, m_arr = hp.Alm.getlm(lmax)
    idx_m = np.where(m_arr == abs_m)[0]
    n_l = len(idx_m)

    # Single allocation: rows 0..n_l-1 = cos, n_l..2*n_l-1 = sin.
    # Slices basis_cos / basis_sin are views — no copy on access.
    basis_stacked = np.zeros((2 * n_l, npix), dtype=np.float64)
    basis_cos = basis_stacked[:n_l]
    basis_sin = basis_stacked[n_l:]

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
        for k, cos_map, sin_map in get_progress_bar(pool.map(_synth_one, range(n_l)),
                                                    desc="Precomputing basis maps", total=n_l):
            basis_cos[k] = cos_map
            basis_sin[k] = sin_map

    return basis_stacked


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
        coi = fourier_factor * np.sqrt(2)  # T&C (1998) Table 1: COI factor = sqrt(2) * lambda
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


@lru_cache(maxsize=8)
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

    # Process IFFT in scale batches to cap peak intermediate memory.
    # Materialising all n_scales at once creates shape (…, n_scales, n_padded)
    # in complex128; SCALE_BATCH limits this to 16 planes at a time.
    SCALE_BATCH = 16
    n_scales = daughter.shape[0]
    wave = np.empty(x.shape[:-1] + (n_scales, n), dtype=np.complex128)
    for _s0 in range(0, n_scales, SCALE_BATCH):
        _s1 = min(_s0 + SCALE_BATCH, n_scales)
        wave[..., _s0:_s1, :] = sp_fft.ifft(
            f[..., np.newaxis, :] * daughter[_s0:_s1], axis=-1, workers=-1
        )

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
    del Y_w  # Free (n_scales, 2, n_time, n_spatial) complex array early

    # Scale normalization: broadcast scale over all non-scale axes
    scale_factor = np.sqrt(dt / scale).reshape((-1,) + (1,) * (Ck_w0.ndim - 1))

    Ck_w = Ck_w0 * scale_factor
    Sk_w = Sk_w0 * scale_factor
    del Ck_w0, Sk_w0  # Free scaled intermediates

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

@lru_cache(maxsize=8)
def _get_ring2nest(nside: int) -> np.ndarray:
    """Cached RING->NEST pixel permutation for a given nside.

    fourier_wavelet_spectrum previously called hp.ring2nest(nside, ...)
    fresh every invocation, and (inside the assign_to_cells closure) called
    hp.nest2ring(nside, ...) fresh on every one of up to ~10 calls per
    invocation, even though the permutation depends only on nside. Caching
    it removes that repeated work without changing any values.
    """
    return hp.ring2nest(nside, np.arange(hp.nside2npix(nside)))


@lru_cache(maxsize=8)
def _get_nest2ring(nside: int) -> np.ndarray:
    """Cached NEST->RING pixel permutation for a given nside (see
    _get_ring2nest above for rationale)."""
    return hp.nest2ring(nside, np.arange(hp.nside2npix(nside)))


# --------------------------------------------------------------------------
# RING-BASED FOURIER EXTRACTION HELPERS
# --------------------------------------------------------------------------

@lru_cache(maxsize=8)
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


def _fourier_precompute_ring_coefs(
        da: xr.DataArray,
        zwn_list: list[int],
        time_dim: str,
        order: str,
) -> dict[int, dict]:
    """Precompute ring Fourier coefficients for ALL levels and ZWNs.

    This is the Fourier analogue of :func:`_sh_precompute_alm`.  Rather than
    calling :func:`_extract_ring_fourier_coefs` inside each Dask block (which
    would trigger N_threads concurrent pixel loads), we extract ``(Ck, Sk)``
    for every level in one vectorised pass here, then store the compact ring
    series in ``spectrum_kwargs['_ring_cache']`` for the block workers.

    Memory cost: ``n_zwn × n_time × n_extra × n_rings × 2 × 8`` bytes.
    For ICON nside=64 (n_time=185, n_extra=41, n_rings=255):
    ≈ 3 ZWNs × 185 × 41 × 255 × 16 = **93 MB** — trivial.

    Args:
        da: DataArray with dims ``(time_dim, [*extra_dims], cell_dim)``.
        zwn_list: All zonal wavenumbers needed (e.g. ``[1, 2, 3]``).
        time_dim: Name of the time dimension.
        order: HEALPix ordering, ``'ring'`` or ``'nested'``.

    Returns:
        Dict ``{zwn: {'Ck': (n_time, n_extra, n_rings),
                      'Sk': (n_time, n_extra, n_rings),
                      'start_pix': ndarray,
                      'ring_pix': ndarray}}``.
    """
    cell_dim = get_cells_dim(da)
    npix = da.sizes[cell_dim]
    nside = hp.npix2nside(npix)
    is_nested = order.lower() == 'nested'

    extra_dims = [d for d in da.dims if d not in (time_dim, cell_dim)]
    n_time = da.sizes[time_dim]
    n_extra = int(np.prod([da.sizes[d] for d in extra_dims])) if extra_dims else 1
    order_dims = [time_dim] + extra_dims + [cell_dim]

    # Determine how many levels to materialise at once so we don't spike RAM.
    # Each batch loads (n_time, lev_batch, npix) float64 plus one real
    # intermediate of the same shape per ZWN.
    lev_batch = _compute_lev_batch(n_time, npix, n_extra=n_extra, target_fraction=0.25)
    logger.debug(f"Fourier precompute: lev_batch={lev_batch} of {n_extra} level(s)")

    # For zwn>0 decompose the complex weight exp(-i*zwn*phi)/ringpix into
    # real cos and sin components stored separately.  This replaces one
    # complex128 intermediate (16 bytes/pixel) with two float64 ones
    # (8 bytes each), with identical numerical result.
    ring_cos: dict[int, np.ndarray] = {}  # (npix,) per zwn
    ring_sin: dict[int, np.ndarray] = {}
    ring_meta: dict[int, tuple] = {}  # (startpix, ringpix)
    for zwn in zwn_list:
        startpix, ringpix, W = _get_ring_fourier_weights(nside, zwn)
        ring_meta[zwn] = (startpix, ringpix)
        ring_cos[zwn] = np.real(W).astype(np.float64)
        ring_sin[zwn] = np.imag(W).astype(np.float64)  # zero for zwn==0

    # Pre-allocate output arrays for all ZWNs.
    result: dict[int, dict] = {}
    for zwn in zwn_list:
        startpix, ringpix = ring_meta[zwn]
        n_rings = len(startpix)
        result[zwn] = {
            'Ck': np.empty((n_time, n_extra, n_rings), dtype=np.float64),
            'Sk': np.empty((n_time, n_extra, n_rings), dtype=np.float64),
            'start_pix': startpix,
            'ring_pix': ringpix,
        }

    # Iterate over level batches, materialising only one batch at a time.
    for lev_start in range(0, n_extra, lev_batch):
        lev_end = min(lev_start + lev_batch, n_extra)

        if extra_dims:
            # Build a flat-index selector for the current batch.
            # Works for a single extra dim; for multiple dims we flatten.
            flat_sel = {extra_dims[0]: slice(lev_start, lev_end)} \
                if len(extra_dims) == 1 else {}
            if flat_sel:
                batch_np = da.isel(flat_sel).transpose(
                    time_dim, extra_dims[0], cell_dim
                ).values  # (n_time, batch, npix)
            else:
                # Multi-extra-dim fallback: load all, slice flat
                batch_np = (
                    da.transpose(*order_dims).values
                    .reshape(n_time, n_extra, npix)[:, lev_start:lev_end, :]
                )
        else:
            batch_np = da.transpose(*order_dims).values.reshape(
                n_time, 1, npix
            )

        # RING ordering once per batch.
        if is_nested:
            ring2nest = _get_ring2nest(nside)
            batch_np = batch_np[:, :, ring2nest]

        for zwn in zwn_list:
            startpix, ringpix = ring_meta[zwn]
            cos_w = ring_cos[zwn]
            sin_w = ring_sin[zwn]

            # Real part: data × cos(zwn×phi)/N_ring
            fm_cos = np.add.reduceat(
                batch_np * cos_w[None, None, :], startpix, axis=-1
            )  # (n_time, batch, n_rings)

            if zwn == 0:
                result[zwn]['Ck'][:, lev_start:lev_end, :] = fm_cos
                result[zwn]['Sk'][:, lev_start:lev_end, :] = 0.0
            else:
                # Imag part separately to avoid complex128 intermediates.
                fm_sin = np.add.reduceat(
                    batch_np * sin_w[None, None, :], startpix, axis=-1
                )
                result[zwn]['Ck'][:, lev_start:lev_end, :] = 2.0 * fm_cos
                result[zwn]['Sk'][:, lev_start:lev_end, :] = -2.0 * fm_sin
                del fm_sin
            del fm_cos

        del batch_np  # release batch memory before next iteration

    return result


def _fourier_reconstruct_level(
        Ck_lev: np.ndarray,
        Sk_lev: np.ndarray,
        ring_pix: np.ndarray,
        dt: float,
        dj: float,
        nside: int,
        is_nested: bool = False,
        periods_to_reconstruct: list[float] | None = None,
        min_t: float = -1,
        max_t: float = np.inf,
) -> dict:
    """CWT + spatial reconstruction for ONE level's pre-computed ring Fourier series.

    This is the Fourier analogue of :func:`_sh_reconstruct_level`.  Both
    functions accept pre-computed spectral coefficients for a single level and
    return the same result dict format so that the block workers
    (:func:`~healicon.analysis.tides._wavelet_sh_analysis_block` /
    :func:`~healicon.analysis.tides._wavelet_fourier_analysis_block`) can
    share the same downstream result-processing code.

    Args:
        Ck_lev: ``(n_time, n_rings)`` cosine ring Fourier coefficients.
        Sk_lev: ``(n_time, n_rings)`` sine ring Fourier coefficients.
        ring_pix: ``(n_rings,)`` number of HEALPix pixels per ring.
        dt: Sampling interval in hours.
        dj: Wavelet scale spacing.
        nside: HEALPix nside.
        is_nested: Whether to output in NESTED pixel ordering.
        periods_to_reconstruct: Target periods for spatial maps; ``None``
            returns global diagnostics only.
        min_t / max_t: Period range filter (same units as *dt*).

    Returns:
        Dict with keys:

        * ``'global_amp_w'`` / ``'global_amp_e'`` — ``(n_p, n_time)``
        * ``'coi'`` — ``(n_time,)``
        * ``'period'`` — ``(n_p,)``
        * ``(actual_p, direction, sa, ap)`` — ``(n_time, npix)`` for each
          requested period, direction, sym/asy, amp/pha component.
    """
    npix = hp.nside2npix(nside)

    Ak, Bk, ak, bk, period_full, _, coi = _compute_cwt_coefficients(
        Ck_lev, Sk_lev, dt, dj, axis=0
    )
    pmask = (period_full >= min_t) & (period_full <= max_t)
    period = period_full[pmask]
    Ak, Bk, ak, bk = Ak[pmask], Bk[pmask], ak[pmask], bk[pmask]

    # Global amplitude: ring-area-weighted mean over all periods.
    Rw_rings, Re_rings, _, _ = _calculate_amplitudes_and_phases(
        Ak, Bk, ak, bk, compute_phase=False
    )
    result: dict = {
        'global_amp_w': (Rw_rings * ring_pix).sum(axis=-1) / npix,  # (n_p, n_time)
        'global_amp_e': (Re_rings * ring_pix).sum(axis=-1) / npix,
        'coi': coi,
        'period': period,
    }

    if periods_to_reconstruct is None:
        return result

    # Pixel permutation for nested output ordering (cached).
    nest2ring_idx = _get_nest2ring(nside) if is_nested else None

    def assign_to_cells(ring_vals):
        """``(..., n_rings)`` → ``(..., npix)``."""
        mapped = np.repeat(ring_vals, ring_pix, axis=-1)
        return mapped[..., nest2ring_idx] if is_nested else mapped

    # Symmetric / antisymmetric decomposition over the ring axis (last).
    Ak_sym = 0.5 * (Ak + np.flip(Ak, axis=-1))
    Ak_asy = 0.5 * (Ak - np.flip(Ak, axis=-1))
    Bk_sym = 0.5 * (Bk + np.flip(Bk, axis=-1))
    Bk_asy = 0.5 * (Bk - np.flip(Bk, axis=-1))
    ak_sym = 0.5 * (ak + np.flip(ak, axis=-1))
    ak_asy = 0.5 * (ak - np.flip(ak, axis=-1))
    bk_sym = 0.5 * (bk + np.flip(bk, axis=-1))
    bk_asy = 0.5 * (bk - np.flip(bk, axis=-1))

    for target_p in periods_to_reconstruct:
        idx = int(np.argmin(np.abs(period - target_p)))
        actual_p = float(period[idx])

        Rw_s, Re_s, Pw_s, Pe_s = _calculate_amplitudes_and_phases(
            Ak_sym[idx], Bk_sym[idx], ak_sym[idx], bk_sym[idx]
        )
        Rw_a, Re_a, Pw_a, Pe_a = _calculate_amplitudes_and_phases(
            Ak_asy[idx], Bk_asy[idx], ak_asy[idx], bk_asy[idx]
        )

        result[(actual_p, 'westward', 'sym', 'amp')] = assign_to_cells(Rw_s)
        result[(actual_p, 'westward', 'sym', 'pha')] = assign_to_cells(Pw_s)
        result[(actual_p, 'westward', 'asy', 'amp')] = assign_to_cells(Rw_a)
        result[(actual_p, 'westward', 'asy', 'pha')] = assign_to_cells(Pw_a)
        result[(actual_p, 'eastward', 'sym', 'amp')] = assign_to_cells(Re_s)
        result[(actual_p, 'eastward', 'sym', 'pha')] = assign_to_cells(Pe_s)
        result[(actual_p, 'eastward', 'asy', 'amp')] = assign_to_cells(Re_a)
        result[(actual_p, 'eastward', 'asy', 'pha')] = assign_to_cells(Pe_a)

    return result


def fourier_wavelet_spectrum(da: xr.DataArray, zwn: int,
                             time_dim: str = 'time',
                             dt: float = 1.0, dj: float = 0.1,
                             min_t: float = -1, max_t: float = np.inf,
                             periods_to_reconstruct: list[float] | None = None,
                             order: str | None = None) -> xr.Dataset:
    """Compute the Fourier-Wavelet spectrum of a HEALPix DataArray.

    This is the Fourier analogue of :func:`spherical_harmonic_wavelet_spectrum`.
    It extracts Fourier coefficients natively from the HEALPix isolatitude
    rings, computes the wavelet transform, decomposes into symmetric and
    antisymmetric modes, and maps the results back to the original cells.

    Internally this is a thin wrapper around
    :func:`_fourier_precompute_ring_coefs` + :func:`_fourier_reconstruct_level`,
    which are also used by the memory-safe block worker in the tidal pipeline.
    """
    if time_dim not in da.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in DataArray")

    cell_dim = get_cells_dim(da)
    if cell_dim is None:
        raise ValueError("No spatial HEALPix cell dimension found in DataArray.")

    _order = (order or get_healpix_order(da)).lower()
    is_nested = _order == 'nested'
    logger.debug(f"HEALPix ordering: {'nested' if is_nested else 'ring'}")

    npix = da.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    # ── Phase 1: precompute ring Fourier series ───────────────────────────
    ring_cache = _fourier_precompute_ring_coefs(da, [zwn], time_dim, _order)
    cache = ring_cache[zwn]
    # da has no extra dims → n_extra=1; squeeze the singleton axis.
    Ck_lev = cache['Ck'][:, 0, :]  # (n_time, n_rings)
    Sk_lev = cache['Sk'][:, 0, :]
    ring_pix = cache['ring_pix']

    # ── Phase 2: CWT + sym/asy + pixel reconstruction ─────────────────────
    rec = _fourier_reconstruct_level(
        Ck_lev, Sk_lev, ring_pix, dt, dj, nside, is_nested,
        periods_to_reconstruct=periods_to_reconstruct,
        min_t=min_t, max_t=max_t,
    )

    period = rec['period']
    coi = rec['coi']

    # ── Build output Dataset (same variable naming as before) ─────────────
    target_lon, target_lat = get_healpix_coords(nside)
    if is_nested:
        target_lon = ensure_original_order(target_lon, 'nested')
        target_lat = ensure_original_order(target_lat, 'nested')

    out_coords = {k: v for k, v in da.coords.items() if k != cell_dim}
    out_coords['period'] = period
    out_coords[cell_dim] = np.arange(npix)

    data_vars: dict = {
        'global_amplitude_westward': (['period', time_dim], rec['global_amp_w']),
        'global_amplitude_eastward': (['period', time_dim], rec['global_amp_e']),
        'coi': ([time_dim], coi),
    }

    for key, arr in rec.items():
        if not isinstance(key, tuple):
            continue
        actual_p, direction, sa, ap = key
        p_str = f"{actual_p:.1f}"
        data_vars[f"{ap}_{sa}_{direction}_{p_str}"] = ([time_dim, cell_dim], arr)

    ds_out = xr.Dataset(data_vars=data_vars, coords=out_coords)
    lon_da = xr.DataArray(target_lon, dims=cell_dim,
                          attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    lat_da = xr.DataArray(target_lat, dims=cell_dim,
                          attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    ds_out = ds_out.assign_coords(lon=lon_da, lat=lat_da)

    ds_out.attrs = da.attrs
    ds_out.attrs['zonal_wavenumber'] = zwn
    ds_out.attrs['dj'] = dj
    ds_out.attrs['dt'] = dt

    return add_healpix_grid_mapping(ds_out, nside, order=_order)


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


def _compute_lev_batch(n_time: int, n_cell: int,
                       n_extra: int = 1,
                       target_fraction: float = 0.40) -> int:
    """Estimate a memory-safe level batch size for :func:`_sh_precompute_alm`.

    Targets using *target_fraction* of available RAM for the pixel-data
    batch ``data_np`` of shape ``(n_time, lev_batch, npix)``.

    Args:
        n_time: Number of time steps.
        n_cell: Number of HEALPix pixels.
        n_extra: Total number of extra (e.g. vertical) levels — used to cap
            the result so we never request more levels than exist.
        target_fraction: Fraction of available RAM to target.

    Returns:
        Level batch size (≥ 1, ≤ n_extra).
    """
    bytes_per_level = n_time * n_cell * 8  # float64 pixel data per level

    try:
        import psutil
        available = psutil.virtual_memory().available
    except ImportError:
        available = 16 * 1024 ** 3  # conservative 16 GB fallback

    target_bytes = target_fraction * available
    lev_batch = max(1, int(target_bytes // bytes_per_level))
    lev_batch = min(lev_batch, n_extra)
    logger.debug(
        f"lev_batch={lev_batch} "
        f"(~{bytes_per_level * lev_batch / 1e9:.1f} GB pixel data per batch, "
        f"{target_fraction * 100:.0f}% of "
        f"{available / 1e9:.0f} GB available)"
    )
    return lev_batch


def _sh_precompute_alm(
        da: xr.DataArray,
        zwn_list: list,
        time_dim: str = 'time',
        lmax: int | None = None,
        map2alm_iter: int = 3,
        order: str | None = None,
        lev_batch: int = 8,
) -> dict:
    """Batch-compute A_lm/B_lm for ALL levels and ALL unique |m| values.

    Loads data in ``lev_batch`` level slices to bound memory, then
    parallelises ``map2alm`` across all ``(time x batch_lev)`` pairs in a
    single ``ThreadPoolExecutor`` call, extracting every requested |m|
    from each transform at no extra cost.

    Args:
        da: DataArray with dims ``(time_dim, [*extra_dims], cell_dim)``.
            May be a lazy Dask array; each level-batch is materialised with
            ``.values`` so at most ``lev_batch x n_time x npix x 8`` bytes
            are loaded at once.
        zwn_list: All zonal wavenumbers needed (e.g. ``[1, 2, 3]``).
        time_dim: Name of the time dimension.
        lmax: Maximum SH degree. Defaults to ``3*nside - 1``.
        map2alm_iter: ``hp.map2alm`` iteration count.
        order: HEALPix ordering (``'ring'``/``'nested'``).
        lev_batch: Number of levels loaded into RAM simultaneously.

    Returns:
        dict with keys ``'A_lm'``, ``'B_lm'`` (each ``{abs_m: ndarray
        (n_time, n_extra, n_l)}``), plus ``'nside'``, ``'lmax'``,
        ``'is_nested'``, ``'n_extra'``.
    """
    cell_dim = get_cells_dim(da)
    is_nested = (get_healpix_order(da) if order is None else order).lower() == 'nested'

    n_time = da.sizes[time_dim]
    n_cell = da.sizes[cell_dim]
    nside = hp.npix2nside(n_cell)
    if lmax is None:
        lmax = 3 * nside - 1

    extra_dims = [d for d in da.dims if d not in (time_dim, cell_dim)]
    extra_sizes = [da.sizes[d] for d in extra_dims]
    n_extra = int(np.prod(extra_sizes)) if extra_dims else 1

    # Build m-masks for every unique |m| — one map2alm call gives all
    abs_m_list = sorted({abs(int(z)) for z in zwn_list})
    _, m_arr = hp.Alm.getlm(lmax)
    m_masks = {m: (m_arr == m) for m in abs_m_list}
    n_ls = {m: int(mask.sum()) for m, mask in m_masks.items()}

    A_lm_all = {m: np.zeros((n_time, n_extra, n_ls[m])) for m in abs_m_list}
    B_lm_all = {m: np.zeros((n_time, n_extra, n_ls[m])) for m in abs_m_list}

    n_workers = min(32, os.cpu_count() or 4)
    reorder_dims = [time_dim] + extra_dims + [cell_dim]
    da_t = da.transpose(*reorder_dims)

    hp_logger = logging.getLogger('healpy')
    old_hp_level = hp_logger.level
    hp_logger.setLevel(logging.WARNING)

    try:
        for batch_start in range(0, n_extra, lev_batch):
            batch_end = min(batch_start + lev_batch, n_extra)
            batch_size = batch_end - batch_start

            # Materialise this level-batch: (n_time, batch_size, npix)
            if extra_dims:
                if len(extra_dims) == 1:
                    data_np = da_t.isel(
                        {extra_dims[0]: slice(batch_start, batch_end)}
                    ).values
                else:
                    full = da_t.values.reshape(n_time, n_extra, n_cell)
                    data_np = full[:, batch_start:batch_end, :]
            else:
                data_np = da_t.values[:, np.newaxis, :]  # (n_time, 1, npix)

            if is_nested:
                # hp.reorder only handles 1D/2D; flatten the batch dimension first
                orig_shape = data_np.shape  # (n_time, batch_size, npix)
                data_np = hp.reorder(data_np.reshape(-1, n_cell), n2r=True)
                data_np = data_np.reshape(orig_shape)

            has_nan = bool(np.isnan(np.sum(data_np)))
            if has_nan:
                filled = np.where(np.isnan(data_np), 0.0, data_np)
                all_nan = np.isnan(data_np).all(axis=-1)
                eff_iter = map2alm_iter

                def _alm_fn(args, _f=filled, _an=all_nan, _bs=batch_start, _it=eff_iter):
                    t, jr = args
                    if _an[t, jr]:
                        return t, _bs + jr, None
                    return t, _bs + jr, hp.map2alm(_f[t, jr], lmax=lmax, iter=_it)
            else:
                eff_iter = 0 if map2alm_iter == 3 else map2alm_iter

                def _alm_fn(args, _d=data_np, _bs=batch_start, _it=eff_iter):
                    t, jr = args
                    return t, _bs + jr, hp.map2alm(_d[t, jr], lmax=lmax, iter=_it)

            pairs = [(t, jr) for t in range(n_time) for jr in range(batch_size)]
            desc = f"SH transform (levels {batch_start}-{batch_end - 1})"

            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for t, j_abs, alm in get_progress_bar(
                        pool.map(_alm_fn, pairs), desc=desc, total=len(pairs)):
                    if alm is None:
                        continue
                    for abs_m in abs_m_list:
                        alm_m = alm[m_masks[abs_m]]
                        if abs_m == 0:
                            A_lm_all[abs_m][t, j_abs] = np.real(alm_m)
                        else:
                            A_lm_all[abs_m][t, j_abs] = 2.0 * np.real(alm_m)
                            B_lm_all[abs_m][t, j_abs] = -2.0 * np.imag(alm_m)

            del data_np
    finally:
        hp_logger.setLevel(old_hp_level)

    return {
        'A_lm': A_lm_all,
        'B_lm': B_lm_all,
        'nside': nside,
        'lmax': lmax,
        'is_nested': is_nested,
        'n_extra': n_extra,
    }


def _sh_reconstruct_level(
        A_lm_lev: np.ndarray,
        B_lm_lev: np.ndarray,
        dt: float,
        dj: float,
        nside: int,
        lmax: int,
        abs_m: int,
        periods_to_reconstruct: list | None,
        is_nested: bool = False,
        min_t: float = -1,
        max_t: float = np.inf,
) -> dict:
    """CWT + spatial reconstruction for ONE level's pre-computed A_lm/B_lm.

    Peak memory is bounded at ``O(TIME_CHUNK x npix)`` per period — no
    ``(n_time, npix)`` pre-allocation across all periods simultaneously.

    Args:
        A_lm_lev: ``(n_time, n_l)`` real SH coefficients.
        B_lm_lev: ``(n_time, n_l)`` imaginary SH coefficients.
        dt: Sampling interval.
        dj: Wavelet scale spacing.
        nside: HEALPix nside.
        lmax: Maximum SH degree.
        abs_m: Absolute zonal wavenumber.
        periods_to_reconstruct: Target periods. ``None`` = global spectrum only.
        is_nested: Output pixel ordering.
        min_t / max_t: Period range filter.

    Returns:
        dict with ``'global_amp_w'`` ``(n_p, n_time)``,
        ``'global_amp_e'``, ``'coi'``, ``'period'``, and for each
        requested period: ``(actual_p, direction, sa, ap)`` keys mapping to
        ``(n_time, npix)`` arrays.
    """
    from .tides import _get_symmetric_pixels

    npix = hp.nside2npix(nside)
    n_time = A_lm_lev.shape[0]

    Ak, Bk, ak, bk, period_full, _, coi = _compute_cwt_coefficients(
        A_lm_lev, B_lm_lev, dt, dj, axis=0
    )
    pmask = (period_full >= min_t) & (period_full <= max_t)
    period = period_full[pmask]
    Ak, Bk, ak, bk = Ak[pmask], Bk[pmask], ak[pmask], bk[pmask]

    Rw_l, Re_l, _, _ = _calculate_amplitudes_and_phases(Ak, Bk, ak, bk, compute_phase=False)
    norm = 1.0 / np.sqrt(2.0 * np.pi)
    result = {
        'global_amp_w': norm * np.sqrt(np.sum(Rw_l ** 2, axis=-1)),  # (n_p, n_time)
        'global_amp_e': norm * np.sqrt(np.sum(Re_l ** 2, axis=-1)),
        'coi': coi,
        'period': period,
    }

    if periods_to_reconstruct is None:
        return result

    basis_stacked = _get_basis_maps(nside, lmax, abs_m)
    abs_m_lon = abs_m * _get_lon_rad(nside)
    sym_idx = _get_symmetric_pixels(nside, is_nested=False)

    TIME_CHUNK = 32
    PIX_CHUNK = 65536

    for target_p in periods_to_reconstruct:
        idx = int(np.argmin(np.abs(period - target_p)))
        actual_p = float(period[idx])

        re_w = 0.25 * (Ak[idx] - bk[idx])
        im_w = 0.25 * (Bk[idx] + ak[idx])
        re_e = 0.25 * (Ak[idx] + bk[idx])
        im_e = 0.25 * (Bk[idx] - ak[idx])

        # Allocate 8 output arrays for this period — one level at a time
        aw_s = np.empty((n_time, npix))
        aw_a = np.empty((n_time, npix))
        pw_s = np.empty((n_time, npix))
        pw_a = np.empty((n_time, npix))
        ae_s = np.empty((n_time, npix))
        ae_a = np.empty((n_time, npix))
        pe_s = np.empty((n_time, npix))
        pe_a = np.empty((n_time, npix))

        for t0 in range(0, n_time, TIME_CHUNK):
            t1 = min(t0 + TIME_CHUNK, n_time)
            sl = slice(t0, t1)

            coeff = np.array([
                np.hstack([re_w[sl], im_w[sl]]),
                np.hstack([-im_w[sl], re_w[sl]]),
                np.hstack([re_e[sl], im_e[sl]]),
                np.hstack([-im_e[sl], re_e[sl]])
            ])  # (4, chunk x 2*n_l)

            maps = np.empty((4, t1 - t0, npix), dtype=np.float64)
            for p0 in range(0, npix, PIX_CHUNK):
                p1 = min(p0 + PIX_CHUNK, npix)
                maps[:, :, p0:p1] = coeff @ basis_stacked[:, p0:p1]

            sym = 0.5 * (maps + maps[..., sym_idx])
            asy = 0.5 * (maps - maps[..., sym_idx])
            del maps

            po = abs_m_lon[None, :]
            aw_s[sl] = np.sqrt(sym[0] ** 2 + sym[1] ** 2)
            aw_a[sl] = np.sqrt(asy[0] ** 2 + asy[1] ** 2)
            ae_s[sl] = np.sqrt(sym[2] ** 2 + sym[3] ** 2)
            ae_a[sl] = np.sqrt(asy[2] ** 2 + asy[3] ** 2)
            pw_s[sl] = np.mod(np.arctan2(-sym[1], sym[0]) - po + np.pi, 2 * np.pi) - np.pi
            pw_a[sl] = np.mod(np.arctan2(-asy[1], asy[0]) - po + np.pi, 2 * np.pi) - np.pi
            pe_s[sl] = np.mod(np.arctan2(-sym[3], sym[2]) - po + np.pi, 2 * np.pi) - np.pi
            pe_a[sl] = np.mod(np.arctan2(-asy[3], asy[2]) - po + np.pi, 2 * np.pi) - np.pi
            del sym, asy

        if is_nested:
            for arr in [aw_s, aw_a, pw_s, pw_a, ae_s, ae_a, pe_s, pe_a]:
                arr[:] = ensure_original_order(arr, 'nested')

        result[(actual_p, 'westward', 'sym', 'amp')] = aw_s
        result[(actual_p, 'westward', 'asy', 'amp')] = aw_a
        result[(actual_p, 'westward', 'sym', 'pha')] = pw_s
        result[(actual_p, 'westward', 'asy', 'pha')] = pw_a
        result[(actual_p, 'eastward', 'sym', 'amp')] = ae_s
        result[(actual_p, 'eastward', 'asy', 'amp')] = ae_a
        result[(actual_p, 'eastward', 'sym', 'pha')] = pe_s
        result[(actual_p, 'eastward', 'asy', 'pha')] = pe_a

    return result


def spherical_harmonic_wavelet_spectrum(da: xr.DataArray, zwn: int,
                                        time_dim: str = 'time',
                                        dt: float = 1.0, dj: float = 0.1,
                                        min_t: float = -1, max_t: float = np.inf,
                                        lmax: int | None = None,
                                        map2alm_iter: int = 3,
                                        periods_to_reconstruct: list | None = None,
                                        order: str | None = None,
                                        ) -> xr.Dataset:
    """Spherical-harmonic wavelet spectrum of a HEALPix DataArray.

    Computes the continuous wavelet transform (CWT) of the zonal-wavenumber-m
    component of *da*, returning global amplitudes and optionally full spatial
    reconstruction maps.

    This function is a convenient single-call wrapper around the lower-level
    :func:`_sh_precompute_alm` and :func:`_sh_reconstruct_level` primitives,
    which are used directly by the tidal analysis pipeline.

    Args:
        da: HEALPix DataArray with dims ``(time_dim, [*extra], cell_dim)``.
        zwn: Zonal wavenumber (positive integer for westward, negative for eastward).
        time_dim: Name of the time dimension.
        dt: Sampling interval in hours.
        dj: Wavelet scale spacing.
        min_t / max_t: Period range filter.
        lmax: Maximum SH degree. Defaults to ``3*nside - 1``.
        map2alm_iter: Iterations for ``hp.map2alm``.
        periods_to_reconstruct: Periods for which to return spatial amplitude/phase
            maps.  ``None`` returns global spectrum only.
        order: HEALPix ordering (``'ring'`` / ``'nested'``). Auto-detected if *None*.

    Returns:
        ``xr.Dataset`` with:

        * ``global_amplitude_westward`` / ``_eastward`` — ``(period, time[, *extra])``
        * ``coi`` — ``(time,)``
        * ``amp_sym_westward_<p>``, ``pha_sym_westward_<p>``, etc. for each period
          in *periods_to_reconstruct* — ``(time[, *extra], cells)``
    """
    if time_dim not in da.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in DataArray")

    cell_dim = get_cells_dim(da)

    if order is not None:
        is_nested = order.lower() == 'nested'
    else:
        is_nested = get_healpix_order(da) == 'nested'

    npix = da.sizes[cell_dim]
    nside = hp.npix2nside(npix)
    if lmax is None:
        lmax = 3 * nside - 1
    n_time = da.sizes[time_dim]

    extra_dims = [d for d in da.dims if d not in (time_dim, cell_dim)]
    extra_sizes = [da.sizes[d] for d in extra_dims]
    n_extra = int(np.prod(extra_sizes)) if extra_dims else 1
    order_str = 'nested' if is_nested else 'ring'

    # ── Phase 1: batched map2alm across all (time × extra) pairs ──────────
    alm_result = _sh_precompute_alm(
        da, zwn_list=[abs(zwn)], time_dim=time_dim,
        lmax=lmax, map2alm_iter=map2alm_iter,
        order=order_str, lev_batch=max(1, n_extra),  # one batch = all levels
    )
    abs_m = abs(zwn)
    A_lm_all = alm_result['A_lm'][abs_m]  # (n_time, n_extra, n_l)
    B_lm_all = alm_result['B_lm'][abs_m]

    # ── Phase 2: CWT + reconstruction, extra-dim by extra-dim ─────────────
    # Collect per-extra-index results then stack into output arrays.
    # period / coi come from the first level (same for all).
    period = None
    coi = None

    # Accumulators for global amplitudes: (n_p, n_time, n_extra)
    gap_w_list = []
    gap_e_list = []
    # Spatial maps: {(actual_p, dir, sa, ap): list of (n_time, npix) per extra}
    spatial_lists: dict = {}

    for j in range(n_extra):
        rec = _sh_reconstruct_level(
            A_lm_all[:, j, :], B_lm_all[:, j, :],
            dt=dt, dj=dj,
            nside=nside, lmax=lmax, abs_m=abs_m,
            periods_to_reconstruct=periods_to_reconstruct,
            is_nested=is_nested,
            min_t=min_t, max_t=max_t,
        )
        if period is None:
            period = rec['period']
            coi = rec['coi']
        gap_w_list.append(rec['global_amp_w'])  # (n_p, n_time)
        gap_e_list.append(rec['global_amp_e'])

        if periods_to_reconstruct is not None:
            for key, arr in rec.items():
                if not isinstance(key, tuple) or len(key) != 4:
                    continue  # skip 'period', 'coi', etc.
                if key not in spatial_lists:
                    spatial_lists[key] = []
                spatial_lists[key].append(arr)  # (n_time, npix)

    # Stack: (n_p, n_time, n_extra) → reshape to (n_p, n_time, *extra_sizes)
    gap_w = np.stack(gap_w_list, axis=-1)  # (n_p, n_time, n_extra)
    gap_e = np.stack(gap_e_list, axis=-1)

    # ── Build output Dataset ───────────────────────────────────────────────
    target_lon, target_lat = get_healpix_coords(nside)
    if is_nested:
        target_lon = ensure_original_order(target_lon, order_str)
        target_lat = ensure_original_order(target_lat, order_str)

    # Define output coordinates
    out_coords = {time_dim: da.coords[time_dim], 'period': period}
    for ed in extra_dims:
        out_coords[ed] = da.coords[ed]
    out_coords[cell_dim] = np.arange(npix)

    if extra_dims:
        extra_shape = tuple(extra_sizes)
        global_dims = ['period', time_dim] + extra_dims
        gap_w = gap_w.reshape((len(period), n_time) + extra_shape)
        gap_e = gap_e.reshape((len(period), n_time) + extra_shape)
    else:
        global_dims = ['period', time_dim]
        gap_w = gap_w[..., 0]
        gap_e = gap_e[..., 0]

    out_vars = {
        'global_amplitude_westward': (global_dims, gap_w),
        'global_amplitude_eastward': (global_dims, gap_e),
        'coi': ([time_dim], coi),
    }

    # Add spatial maps for each reconstructed period and mode
    if periods_to_reconstruct is not None:
        if extra_dims:
            spatial_dims = [time_dim] + extra_dims + [cell_dim]
        else:
            spatial_dims = [time_dim, cell_dim]

        for (actual_p, direction, sa, ap), arr_list in spatial_lists.items():
            p_str = f"{actual_p:.1f}"
            # arr_list: list of (n_time, npix), one per extra index
            if extra_dims:
                stacked = np.stack(arr_list, axis=1)  # (n_time, n_extra, npix)
                stacked = stacked.reshape((n_time,) + tuple(extra_sizes) + (npix,))
            else:
                stacked = arr_list[0]  # (n_time, npix)
            out_vars[f"{ap}_{sa}_{direction}_{p_str}"] = (spatial_dims, stacked)

    # Construct output dataset and add metadata
    ds = xr.Dataset(data_vars=out_vars, coords=out_coords)
    lon_da = xr.DataArray(target_lon, dims=cell_dim,
                          attrs={'standard_name': 'longitude', 'units': 'degrees_east'})
    lat_da = xr.DataArray(target_lat, dims=cell_dim,
                          attrs={'standard_name': 'latitude', 'units': 'degrees_north'})
    ds = ds.assign_coords(lon=lon_da, lat=lat_da)
    ds.attrs['zonal_wavenumber'] = zwn
    ds.attrs['dj'] = dj
    ds.attrs['dt'] = dt
    ds = add_healpix_grid_mapping(ds, nside, order=order_str)
    return ds
