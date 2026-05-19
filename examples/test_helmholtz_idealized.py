"""
Idealized validation of the Helmholtz decomposition.

Constructs purely rotational and purely divergent wind fields from known
spherical harmonic modes, then verifies that compute_helmholtz recovers
them correctly with near-zero cross-contamination.
"""

import numpy as np
import healpy as hp
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from healicon.analysis import compute_helmholtz

# Private constant from analysis module
_EARTH_RADIUS_M = 6371.229e3

NSIDE = 32
LMAX  = 3 * NSIDE - 1


# ── Synthetic field constructors ─────────────────────────────────────────────

def _single_alm(lmax, l, m, amplitude=1.0):
    alm = np.zeros(hp.Alm.getsize(lmax), dtype=np.complex128)
    alm[hp.Alm.getidx(lmax, l, m)] = amplitude
    return alm


def synthetic_rotational(nside, lmax, l=5, m=3, amplitude=10.0):
    """Pure rotational wind from ψ = amplitude * Re(Y_l^m)."""
    l_arr, _ = hp.Alm.getlm(lmax)
    fl = np.sqrt(l_arr * (l_arr + 1.0))

    psi_alm = _single_alm(lmax, l, m, amplitude / _EARTH_RADIUS_M)
    almB = -fl * psi_alm
    zeros = np.zeros_like(almB)

    maps = hp.alm2map_spin([zeros, almB], nside, 1, lmax=lmax)
    u, v = maps[1], -maps[0]
    psi_true = hp.alm2map(psi_alm, nside, lmax=lmax) * _EARTH_RADIUS_M
    return u, v, psi_true


def synthetic_divergent(nside, lmax, l=4, m=2, amplitude=10.0):
    """Pure divergent wind from χ = amplitude * Re(Y_l^m)."""
    l_arr, _ = hp.Alm.getlm(lmax)
    fl = np.sqrt(l_arr * (l_arr + 1.0))

    chi_alm = _single_alm(lmax, l, m, amplitude / _EARTH_RADIUS_M)
    almE = fl * chi_alm
    zeros = np.zeros_like(almE)

    maps = hp.alm2map_spin([almE, zeros], nside, 1, lmax=lmax)
    u, v = maps[1], -maps[0]
    chi_true = hp.alm2map(chi_alm, nside, lmax=lmax) * _EARTH_RADIUS_M
    return u, v, chi_true


def to_ds(nside, u, v):
    npix = hp.nside2npix(nside)
    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    return xr.Dataset(
        {'u': ('cell', u.astype(np.float32)),
         'v': ('cell', v.astype(np.float32))},
        coords={'cell': np.arange(npix), 'lon': ('cell', lon), 'lat': ('cell', lat)},
        attrs={'healpix_nside': nside, 'healpix_scheme': 'RING'}
    )


# ── Plotting helpers ─────────────────────────────────────────────────────────

def hp2ll(m, nside, nlon=360, nlat=181):
    lons = np.linspace(0, 360, nlon, endpoint=False)
    lats = np.linspace(-90, 90, nlat)
    LON, LAT = np.meshgrid(lons, lats)
    vals = hp.get_interp_val(m, np.deg2rad(90 - LAT).ravel(),
                              np.deg2rad(LON).ravel()).reshape(nlat, nlon)
    return vals, lons, lats


def pmap(ax, data, lons, lats, title, vmax=None, cmap='RdBu_r'):
    if vmax is None:
        vmax = np.nanpercentile(np.abs(data), 99)
    vmax = max(vmax, 1e-12)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    ax.pcolormesh(lons, lats, data, cmap=cmap, norm=norm, rasterized=True)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
    ax.axhline(0, color='k', lw=0.4, ls='--', alpha=0.4)
    ax.set_xlabel('Lon °', fontsize=8); ax.set_ylabel('Lat °', fontsize=8)
    ax.tick_params(labelsize=7)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building idealized fields …")
    u_r, v_r, psi_t = synthetic_rotational(NSIDE, LMAX)
    u_d, v_d, chi_t = synthetic_divergent(NSIDE, LMAX)

    print("Running Helmholtz decomposition …")
    hrot = compute_helmholtz(to_ds(NSIDE, u_r, v_r), 'u', 'v', lmax=LMAX).compute()
    hdiv = compute_helmholtz(to_ds(NSIDE, u_d, v_d), 'u', 'v', lmax=LMAX).compute()
    hmix = compute_helmholtz(to_ds(NSIDE, u_r + u_d, v_r + v_d), 'u', 'v', lmax=LMAX).compute()

    # Error metrics
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a) ** 2)))
    leakage_rot = rms(hrot['u_div']) / rms(u_r)
    leakage_div = rms(hdiv['u_rot']) / rms(u_d)
    psi_err     = rms(hrot['psi'].values - psi_t) / rms(psi_t)
    chi_err     = rms(hdiv['chi'].values - chi_t) / rms(chi_t)

    print(f"\n=== Leakage / Recovery Errors (relative RMS) ===")
    print(f"  Rotational → divergent leakage : {leakage_rot:.2e}  (ideal: 0)")
    print(f"  Divergent  → rotational leakage: {leakage_div:.2e}  (ideal: 0)")
    print(f"  ψ recovery error               : {psi_err:.2e}     (ideal: 0)")
    print(f"  χ recovery error               : {chi_err:.2e}     (ideal: 0)")

    print("\nGenerating figure …")

    # Regrid everything
    LL = lambda v: hp2ll(np.asarray(v).ravel(), NSIDE)

    u_r_ll, lon, lat   = LL(u_r)
    ur_out_ll, *_      = LL(hrot['u_rot'])
    ur_leak_ll, *_     = LL(hrot['u_div'])   # should be ~0

    u_d_ll, *_         = LL(u_d)
    ud_out_ll, *_      = LL(hdiv['u_div'])
    ud_leak_ll, *_     = LL(hdiv['u_rot'])   # should be ~0

    psi_t_ll, *_       = LL(psi_t)
    psi_r_ll, *_       = LL(hrot['psi'])
    chi_t_ll, *_       = LL(chi_t)
    chi_r_ll, *_       = LL(hdiv['chi'])

    umix_ll, *_        = LL(u_r + u_d)
    umix_rot_ll, *_    = LL(hmix['u_rot'])
    umix_div_ll, *_    = LL(hmix['u_div'])

    fig, axes = plt.subplots(4, 3, figsize=(14, 13))
    fig.suptitle('Helmholtz Decomposition — Idealized Validation (nside=32)', fontsize=13)

    vr = np.nanpercentile(np.abs(u_r_ll), 99)
    vd = np.nanpercentile(np.abs(u_d_ll), 99)
    vm = np.nanpercentile(np.abs(umix_ll), 99)
    vp = np.nanpercentile(np.abs(psi_t_ll), 99)
    vc = np.nanpercentile(np.abs(chi_t_ll), 99)

    # Row 0 — pure rotational
    pmap(axes[0, 0], u_r_ll,    lon, lat, 'Input $u$ (pure rotational)',        vmax=vr)
    pmap(axes[0, 1], ur_out_ll, lon, lat, 'Recovered $u_{rot}$',                vmax=vr)
    pmap(axes[0, 2], ur_leak_ll,lon, lat, f'Leakage $u_{{div}}$ (RMS={leakage_rot:.1e}×signal)',
         vmax=vr * 0.05)

    # Row 1 — pure divergent
    pmap(axes[1, 0], u_d_ll,    lon, lat, 'Input $u$ (pure divergent)',         vmax=vd)
    pmap(axes[1, 1], ud_out_ll, lon, lat, 'Recovered $u_{div}$',                vmax=vd)
    pmap(axes[1, 2], ud_leak_ll,lon, lat, f'Leakage $u_{{rot}}$ (RMS={leakage_div:.1e}×signal)',
         vmax=vd * 0.05)

    # Row 2 — scalar potentials
    pmap(axes[2, 0], psi_t_ll,  lon, lat, 'True $\\psi$ [m² s⁻¹]',             vmax=vp, cmap='PRGn')
    pmap(axes[2, 1], psi_r_ll,  lon, lat, f'Recovered $\\psi$  (err={psi_err:.1e})', vmax=vp, cmap='PRGn')
    pmap(axes[2, 2], chi_t_ll,  lon, lat, 'True $\\chi$ [m² s⁻¹]',             vmax=vc, cmap='PuOr')

    # Row 3 — mixed field
    pmap(axes[3, 0], umix_ll,    lon, lat, 'Mixed $u$ (rot + div)',             vmax=vm)
    pmap(axes[3, 1], umix_rot_ll,lon, lat, 'Decomposed $u_{rot}$ from mixed',  vmax=vm)
    pmap(axes[3, 2], umix_div_ll,lon, lat, 'Decomposed $u_{div}$ from mixed',  vmax=vm)

    plt.tight_layout()
    out = 'helmholtz_idealized_test.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    print(f"Saved → {out}")


if __name__ == '__main__':
    main()
