"""
HealICON Wavelet Analysis Module
Contains an optimized, xarray-native implementation of the Fourier-Wavelet spectrum,
based on Torrence & Compo (1998) and Y. Yamazaki (2023).

References:
Torrence, C. and G. P. Compo, 1998: A Practical Guide to Wavelet Analysis. 
Bull. Amer. Meteor. Soc., 79, 61-78.

Yamazaki, Y., 2023: A method to derive Fourier–wavelet spectra for the 
characterization of global-scale waves in the mesosphere and lower thermosphere.
Geosci. Model Dev., 16, 4749–4766.
"""

import logging

import healpy as hp
import numpy as np
import xarray as xr
from scipy.optimize import fminbound
from scipy.special import gamma, gammainc

from healicon.analysis import ensure_ring, ensure_original_order, _get_symmetric_pixels
from healicon.grid import get_cells_dim, get_healpix_order

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# CORE WAVELET FUNCTIONS
# --------------------------------------------------------------------------

def wave_bases(mother, k, scale, param):
    """Computes the wavelet function as a function of Fourier frequency."""
    n = len(k)
    kplus = np.array(k > 0., dtype=float)

    if mother == 'MORLET':
        if param == -1: param = 6.
        k0 = np.copy(param)
        expnt = -(scale * k - k0) ** 2 / 2. * kplus
        norm = np.sqrt(scale * k[1]) * (np.pi ** (-0.25)) * np.sqrt(n)
        daughter = norm * np.exp(expnt) * kplus
        fourier_factor = (4 * np.pi) / (k0 + np.sqrt(2 + k0 ** 2))
        coi = fourier_factor / np.sqrt(2)
        dofmin = 2
    elif mother == 'PAUL':
        if param == -1: param = 4.
        m = param
        expnt = -scale * k * kplus
        norm_bottom = np.sqrt(m * np.prod(np.arange(1, (2 * m))))
        norm = np.sqrt(scale * k[1]) * (2 ** m / norm_bottom) * np.sqrt(n)
        daughter = norm * ((scale * k) ** m) * np.exp(expnt) * kplus
        fourier_factor = 4 * np.pi / (2 * m + 1)
        coi = fourier_factor * np.sqrt(2)
        dofmin = 2
    elif mother == 'DOG':
        if param == -1: param = 2.
        m = param
        expnt = -(scale * k) ** 2 / 2.0
        norm = np.sqrt(scale * k[1] / gamma(m + 0.5)) * np.sqrt(n)
        daughter = -norm * (1j ** m) * ((scale * k) ** m) * np.exp(expnt)
        fourier_factor = 2 * np.pi * np.sqrt(2. / (2 * m + 1))
        coi = fourier_factor / np.sqrt(2)
        dofmin = 1
    else:
        raise ValueError('Mother must be one of MORLET, PAUL, DOG')

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


def wave_signif(Y, dt, scale, sigtest=0, lag1=0.0, siglvl=0.95, dof=None, mother='MORLET', param=-1,
                gws=None):
    n1 = len(np.atleast_1d(Y))
    J1 = len(scale) - 1
    dj = np.log2(scale[1] / scale[0])

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
        fft_theor = gws
    else:
        fft_theor = (1 - lag1 ** 2) / (1 - 2 * lag1 * np.cos(freq * 2 * np.pi) + lag1 ** 2)
        fft_theor = variance * fft_theor

    signif = fft_theor.copy()
    if dof is None:
        dof = dofmin

    if sigtest == 0:
        dof = dofmin
        chisquare = chisquare_inv(siglvl, dof) / dof
        signif = fft_theor * chisquare
    else:
        raise NotImplementedError('Only sigtest=0 is fully ported for vectorized usage.')

    return signif


def wavelet(Y, dt, pad=1, dj=-1, s0=-1, J1=-1, mother='MORLET', param=-1, axis=-1):
    """
    Computes the 1D Wavelet transform along a specified axis of a multidimensional array.
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

    kplus = np.arange(1, int(n / 2) + 1)
    kplus = (kplus * 2 * np.pi / (n * dt))
    kminus = np.arange(1, int((n - 1) / 2) + 1)
    kminus = np.sort((-kminus * 2 * np.pi / (n * dt)))
    k = np.concatenate(([0.], kplus, kminus))

    f = np.fft.fft(x, axis=-1)

    # Pre-calculate mother parameters to get scales
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
        raise ValueError("Invalid mother")

    j = np.arange(0, J1 + 1)
    scale = s0 * 2. ** (j * dj)
    freq = 1. / (fourier_factor * scale)
    period = 1. / freq

    out_shape = list(Y_work.shape)
    out_shape[-1] = len(scale)
    out_shape.append(n)

    wave = np.zeros(out_shape, dtype=complex)

    for a1 in range(len(scale)):
        daughter, _, coi, _ = wave_bases(mother, k, scale[a1], param)
        wave[..., a1, :] = np.fft.ifft(f * daughter, axis=-1)

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

    # We evaluate wave_signif natively using loops over the non-time dimensions, or flatten it
    # wave_signif takes a 1D scalar or array for variance. If it takes scalar, it returns 1D (n_scales).
    # Since max_var has shape of (...,), we can loop over it and assemble the result

    flat_var = max_var.flatten()
    sig_list = []
    for v in flat_var:
        sig = wave_signif(v, dt, scale, lag1=0.72, siglvl=0.95, mother='MORLET', param=param)
        sig = 0.5 * np.sqrt(sig * np.sqrt(dt / scale))
        sig_list.append(sig)

    # shape is (n_flat, n_scales)
    signif_arr = np.array(sig_list)
    # Restore original spatial shape + scale at front
    # max_var shape is the spatial shape. We want (n_scales, *max_var.shape)
    signif_arr = signif_arr.T.reshape((len(scale), *max_var.shape))

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
                                        periods_to_reconstruct: list[
                                                                    float] | None = None) -> xr.Dataset:
    """
    Computes the Spherical-Harmonic Wavelet spectrum of a HEALPix DataArray.
    
    Args:
        da: Input xarray.DataArray on a HEALPix grid over time.
        zwn: Target zonal wavenumber (m).
        time_dim: Name of the time dimension.
        dt: Sampling time in your preferred time units (default 1.0)
        dj: Spacing between discrete scales (default 0.1)
        min_t: Minimum period to include
        max_t: Maximum period to include
        lmax: Maximum spherical harmonic degree. If None, uses 3 * nside - 1.
        periods_to_reconstruct: List of periods to reconstruct full spatial maps for.
            If None, only the global mean spectrum is returned.
            
    Returns:
        xr.Dataset containing the global mean spectrum (scale, time) and 
        spatially reconstructed amplitude and phase maps (if requested).
    """
    if time_dim not in da.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in DataArray")

    cell_dim = get_cells_dim(da)
    is_nested = get_healpix_order(da.to_dataset(name='var')) == 'nested'
    npix = da.sizes[cell_dim]
    nside = hp.npix2nside(npix)

    if lmax is None:
        lmax = 3 * nside - 1

    time_axis = list(da.dims).index(time_dim)
    n_time = da.sizes[time_dim]

    # We transpose to (time, cells) for processing
    data_2d = da.transpose(time_dim, cell_dim).values

    # 1. Spatial Decomposition
    l_arr, m_arr = hp.Alm.getlm(lmax)
    m_mask = (m_arr == zwn)

    A_lm = np.zeros((n_time, np.sum(m_mask)), dtype=float)
    B_lm = np.zeros((n_time, np.sum(m_mask)), dtype=float)

    logger.info(f"Computing spherical harmonics for m={zwn} over {n_time} time steps...")
    for i in range(n_time):
        ring_map = ensure_ring(data_2d[i], 'nested' if is_nested else 'ring')

        valid_mask = ~np.isnan(ring_map)
        if not np.any(valid_mask):
            continue

        filled_map = np.where(valid_mask, ring_map, 0.0)
        alm = hp.map2alm(filled_map, lmax=lmax, iter=3)
        alm_m = alm[m_mask]

        A_lm[i, :] = 2.0 * np.real(alm_m)
        B_lm[i, :] = -2.0 * np.imag(alm_m)

    # 2. Temporal Wavelet Transform
    param = 6
    pad = 1

    logger.info("Applying Continuous Wavelet Transform...")
    W_A, period, scale, coi = wavelet(A_lm, dt, pad, dj, 2 * dt, mother='MORLET', param=param,
                                      axis=0)
    W_B, _, _, _ = wavelet(B_lm, dt, pad, dj, 2 * dt, mother='MORLET', param=param, axis=0)

    scale_factor = np.sqrt(dt / scale)
    scale_factor_bcast = scale_factor[:, None, None]

    W_A *= scale_factor_bcast
    W_B *= scale_factor_bcast

    Ak = np.real(W_A)
    Bk = -np.imag(W_A)
    ak = np.real(W_B)
    bk = -np.imag(W_B)

    # 3. Analytic Signal Separation
    C_W = Ak - bk
    S_W = Bk + ak

    C_E = Ak + bk
    S_E = Bk - ak

    # Filter by period
    period_mask = (period >= min_t) & (period <= max_t)
    period = period[period_mask]
    scale = scale[period_mask]
    C_W = C_W[period_mask, ...]
    S_W = S_W[period_mask, ...]
    C_E = C_E[period_mask, ...]
    S_E = S_E[period_mask, ...]

    # Calculate Global Mean RMS Amplitude using Parseval's theorem equivalent
    global_power_w = np.sqrt(0.5 * np.sum(C_W ** 2 + S_W ** 2, axis=-1))
    global_power_e = np.sqrt(0.5 * np.sum(C_E ** 2 + S_E ** 2, axis=-1))

    out_coords = {k: v for k, v in da.coords.items() if k not in [cell_dim]}
    out_coords['period'] = period

    out_vars = {
        'global_amplitude_westward': (['period', time_dim], global_power_w),
        'global_amplitude_eastward': (['period', time_dim], global_power_e),
        'coi': ([time_dim], coi)
    }

    if periods_to_reconstruct is not None:
        logger.info(f"Reconstructing full spatial maps for periods: {periods_to_reconstruct}")
        sym_idx = _get_symmetric_pixels(nside, is_nested=is_nested)

        logger.info("Precomputing spatial basis maps for vectorized reconstruction...")
        theta, lon = hp.pix2ang(nside, np.arange(npix))
        n_l = np.sum(m_mask)
        idx_m = np.where(m_mask)[0]
        basis_cos = np.zeros((n_l, npix))
        basis_sin = np.zeros((n_l, npix))
        
        for k in range(n_l):
            alm_b_cos = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
            alm_b_cos[idx_m[k]] = 1.0
            basis_cos[k] = hp.alm2map(alm_b_cos, nside=nside)
            
            alm_b_sin = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
            alm_b_sin[idx_m[k]] = -1j
            basis_sin[k] = hp.alm2map(alm_b_sin, nside=nside)

        for target_p in periods_to_reconstruct:
            idx = np.argmin(np.abs(period - target_p))
            actual_p = period[idx]

            cw_t = C_W[idx] * np.sqrt(dt / actual_p)
            sw_t = S_W[idx] * np.sqrt(dt / actual_p)
            ce_t = C_E[idx] * np.sqrt(dt / actual_p)
            se_t = S_E[idx] * np.sqrt(dt / actual_p)

            # Vectorized spatial reconstruction of the complex envelope Z(theta, lambda)
            # The 0.5 factor is applied here to match the Fourier decomposition definition.
            Z_real_w = 0.5 * (cw_t @ basis_cos - sw_t @ basis_sin)
            Z_imag_w = 0.5 * (cw_t @ basis_sin + sw_t @ basis_cos)
            
            Z_real_e = 0.5 * (ce_t @ basis_cos - se_t @ basis_sin)
            Z_imag_e = 0.5 * (ce_t @ basis_sin + se_t @ basis_cos)

            # Vectorized Symmetric / Antisymmetric Separation
            Z_real_w_sym = 0.5 * (Z_real_w + Z_real_w[:, sym_idx])
            Z_imag_w_sym = 0.5 * (Z_imag_w + Z_imag_w[:, sym_idx])
            amp_w_sym = np.sqrt(Z_real_w_sym**2 + Z_imag_w_sym**2)
            pha_w_sym = np.mod(np.arctan2(Z_imag_w_sym, Z_real_w_sym) - zwn * lon[None, :] + np.pi, 2 * np.pi) - np.pi
            
            Z_real_w_asy = 0.5 * (Z_real_w - Z_real_w[:, sym_idx])
            Z_imag_w_asy = 0.5 * (Z_imag_w - Z_imag_w[:, sym_idx])
            amp_w_asy = np.sqrt(Z_real_w_asy**2 + Z_imag_w_asy**2)
            pha_w_asy = np.mod(np.arctan2(Z_imag_w_asy, Z_real_w_asy) - zwn * lon[None, :] + np.pi, 2 * np.pi) - np.pi

            Z_real_e_sym = 0.5 * (Z_real_e + Z_real_e[:, sym_idx])
            Z_imag_e_sym = 0.5 * (Z_imag_e + Z_imag_e[:, sym_idx])
            amp_e_sym = np.sqrt(Z_real_e_sym**2 + Z_imag_e_sym**2)
            pha_e_sym = np.mod(np.arctan2(Z_imag_e_sym, Z_real_e_sym) - zwn * lon[None, :] + np.pi, 2 * np.pi) - np.pi
            
            Z_real_e_asy = 0.5 * (Z_real_e - Z_real_e[:, sym_idx])
            Z_imag_e_asy = 0.5 * (Z_imag_e - Z_imag_e[:, sym_idx])
            amp_e_asy = np.sqrt(Z_real_e_asy**2 + Z_imag_e_asy**2)
            pha_e_asy = np.mod(np.arctan2(Z_imag_e_asy, Z_real_e_asy) - zwn * lon[None, :] + np.pi, 2 * np.pi) - np.pi

            # Revert to nested order if necessary
            amp_w_sym_map = ensure_original_order(amp_w_sym, 'nested' if is_nested else 'ring')
            amp_w_asy_map = ensure_original_order(amp_w_asy, 'nested' if is_nested else 'ring')
            amp_e_sym_map = ensure_original_order(amp_e_sym, 'nested' if is_nested else 'ring')
            amp_e_asy_map = ensure_original_order(amp_e_asy, 'nested' if is_nested else 'ring')

            pha_w_sym_map = ensure_original_order(pha_w_sym, 'nested' if is_nested else 'ring')
            pha_w_asy_map = ensure_original_order(pha_w_asy, 'nested' if is_nested else 'ring')
            pha_e_sym_map = ensure_original_order(pha_e_sym, 'nested' if is_nested else 'ring')
            pha_e_asy_map = ensure_original_order(pha_e_asy, 'nested' if is_nested else 'ring')

            p_str = f"{actual_p:.1f}"
            out_vars[f'amp_sym_westward_{p_str}'] = ([time_dim, cell_dim], amp_w_sym_map)
            out_vars[f'amp_asy_westward_{p_str}'] = ([time_dim, cell_dim], amp_w_asy_map)
            out_vars[f'amp_sym_eastward_{p_str}'] = ([time_dim, cell_dim], amp_e_sym_map)
            out_vars[f'amp_asy_eastward_{p_str}'] = ([time_dim, cell_dim], amp_e_asy_map)
            
            out_vars[f'pha_sym_westward_{p_str}'] = ([time_dim, cell_dim], pha_w_sym_map)
            out_vars[f'pha_asy_westward_{p_str}'] = ([time_dim, cell_dim], pha_w_asy_map)
            out_vars[f'pha_sym_eastward_{p_str}'] = ([time_dim, cell_dim], pha_e_sym_map)
            out_vars[f'pha_asy_eastward_{p_str}'] = ([time_dim, cell_dim], pha_e_asy_map)

            # Re-add cell coordinate for output
            if cell_dim in da.coords:
                out_coords[cell_dim] = da.coords[cell_dim]

    ds = xr.Dataset(data_vars=out_vars, coords=out_coords)
    ds.attrs = da.attrs
    ds.attrs['zonal_wavenumber'] = zwn
    ds.attrs['dj'] = dj
    ds.attrs['dt'] = dt

    return ds
