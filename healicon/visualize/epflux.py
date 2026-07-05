import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from .common import set_publication_style
from ..analysis.ep_flux import _find_alt_name, _is_pressure_coord
from ..cf_coords import _coord_is_meter

logger = logging.getLogger(__name__)

# Constants
_A_EARTH = 6.371e6  # Earth radius [m]
_H_SCALE = 7.0  # log-pressure scale height [km]
_P0_HPA = 1013.25  # reference surface pressure [hPa]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _prep(da, alt_name, is_meters, is_pres, div_time=True):
    """Squeeze, time-mean, convert the vertical coordinate, and sort a field.
 
    Consolidates the boilerplate that was repeated for every DataArray. Unlike
    the original, sort failures are *not* silently swallowed -- a genuinely
    unsortable axis (e.g. a mis-named coordinate) will now raise instead of
    producing a silently unsorted plot.
    """
    da = da.squeeze()
    if div_time and 'time' in da.dims:
        da = da.mean('time')
    if alt_name in da.coords:
        if is_meters:
            da = da.assign_coords({alt_name: da[alt_name] / 1000.0})  # m -> km
        elif is_pres:
            da = da.assign_coords({alt_name: da[alt_name] / 100.0})  # Pa -> hPa
    for dim in ('lat', alt_name):
        if dim in da.dims:
            da = da.sortby(dim)
    return da


# ---------------------------------------------------------------------------
# Zonal-mean wind contours
# ---------------------------------------------------------------------------
def _add_wind_contours(ax, u_zm, alt_name):
    """Overlay zonal-mean zonal wind with a bold zero-wind line + labels."""
    cs = u_zm.plot.contour(
        ax=ax, x='lat', y=alt_name,
        levels=np.arange(-120, 121, 20),
        colors='k', linewidths=0.6, alpha=0.65,
        add_colorbar=False, add_labels=False,
    )
    u_zm.plot.contour(
        ax=ax, x='lat', y=alt_name,
        levels=[0], colors='k', linewidths=1.8,
        add_colorbar=False, add_labels=False,
    )
    try:
        ax.clabel(cs, fmt='%d m/s', fontsize=7, inline=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# EP flux quiver (magnitude-preserving, display-normalised)
# ---------------------------------------------------------------------------
def _ep_components(F_phi, F_z, lat, alt, alt_name, rho0, is_pres):
    """Return (Qx, Qz) as (lat, alt) arrays, scaled for visualization."""
    fp = F_phi.transpose('lat', alt_name).values
    fz = F_z.transpose('lat', alt_name).values

    lat_r = rho0['lat'].values if (rho0 is not None and 'lat' in rho0.coords) else lat
    cos_r = np.cos(np.deg2rad(lat_r))
    
    if rho0 is not None and rho0.ndim == 2:
        geom = np.outer(cos_r, np.ones(len(alt))) * _A_EARTH
    else:
        geom = np.outer(cos_r, np.ones(len(alt))) * _A_EARTH
    geom = np.where(np.abs(geom) < 1e-30, 1e-30, geom)

    if is_pres:
        # aostools standard scaling: sqrt(1000/p) instead of 1/rho0
        p_safe = np.maximum(alt, 1e-10)
        scale = np.sqrt(1000.0 / p_safe)
        scale_2d = np.tile(scale, (len(lat), 1))
        fp_scaled = (fp / geom) * scale_2d
        fz_scaled = (fz / geom) * scale_2d
    else:
        if rho0 is not None:
            if rho0.ndim == 2:
                r_val = rho0.transpose(alt_name, 'lat').values.T
            else:
                r_val = np.outer(np.ones(len(lat)), rho0.values)
            r_val = np.where(r_val < 1e-30, 1e-30, r_val)
            fp_scaled = (fp / geom) / r_val
            fz_scaled = (fz / geom) / r_val
        else:
            fp_scaled = fp
            fz_scaled = fz
            
    # Jucker (2021) aspect-ratio correction for visual group velocity.
    # To balance the visual dimensions of a global plot (x = 180 deg, y = 1000 hPa),
    # the vertical component is multiplied by (length of x in meters) / (length of y in Pa).
    # a * pi / 100000 Pa ≈ 200.0
    c_geom = (_A_EARTH * np.pi) / 100000.0 if is_pres else (_A_EARTH * np.pi) / max(alt[-1], 10000)
    return fp_scaled, fz_scaled * c_geom


def _add_ep_quiver(ax, F_phi, F_z, lat, alt, alt_name, *, is_pres, rho0=None,
                   n_lat=22, n_alt=18, max_frac=0.09, power=0.45, color='0.15'):
    """Draw EP-flux arrows whose *direction* is correct in display space and
    whose *length* encodes magnitude via a power-law compression (so the wide
    dynamic range of EP flux stays legible instead of a few arrows dominating).
 
    Returns the matplotlib Quiver, or None if there is nothing finite to draw.
    """
    Qx, Qz = _ep_components(F_phi, F_z, lat, alt, alt_name, rho0, is_pres)

    # --- subsample to a readable arrow grid ---
    sl = max(1, len(lat) // n_lat)
    sa = max(1, len(alt) // n_alt)
    lat_s, alt_s = lat[::sl], alt[::sa]
    LAT2, ALT2 = np.meshgrid(lat_s, alt_s)  # (n_alt_s, n_lat_s)
    Qx_s = Qx[::sl, ::sa].T  # (lat, alt) -> (alt, lat)
    Qz_s = Qz[::sl, ::sa].T
    assert Qx_s.shape == LAT2.shape, \
        f"quiver shape mismatch: {Qx_s.shape} vs {LAT2.shape}"

    # --- direction components in axis-fraction space (visually correct) ---
    x_range = (lat[-1] - lat[0]) or 1.0
    if is_pres:
        ylog = abs(np.log(max(alt[-1], 1e-10) / max(alt[0], 1e-10))) or 1.0
        p_safe = np.maximum(ALT2, 1e-10)
        fx = Qx_s / x_range
        fz = (Qz_s / p_safe) / ylog  # fraction of a log-p decade
    else:
        y_range = (alt[-1] - alt[0]) or 1.0
        fx = Qx_s / x_range
        fz = Qz_s / y_range

    # --- power-law magnitude compression, direction preserved ---
    with np.errstate(invalid='ignore', divide='ignore'):
        mag = np.hypot(fx, fz)
        mmax = np.nanmax(mag) if np.isfinite(mag).any() else 0.0
        if not np.isfinite(mmax) or mmax <= 0.0:
            return None
        length = (mag / mmax) ** power  # in [0, 1], boosts weak arrows
        dir_x = np.where(mag > 0, fx / mag, 0.0)
        dir_z = np.where(mag > 0, fz / mag, 0.0)

    Fx_frac = dir_x * length * max_frac
    Fz_frac = dir_z * length * max_frac

    # mask negligible / non-finite arrows to cut clutter
    tiny = (~np.isfinite(mag)) | (mag < mmax * 1e-3)
    Fx_frac = np.where(tiny, np.nan, Fx_frac)
    Fz_frac = np.where(tiny, np.nan, Fz_frac)

    # --- convert axis fractions back to data units for scale_units='xy' ---
    u_data = Fx_frac * x_range
    if is_pres:
        # move a fraction of a log-p decade: dp = p*(exp(-frac*ylog) - 1)
        # (negative dp = towards lower pressure = upward, as expected)
        v_data = ALT2 * (np.exp(-Fz_frac * ylog) - 1.0)
    else:
        v_data = Fz_frac * y_range

    q = ax.quiver(
        LAT2, ALT2, u_data, v_data,
        angles='xy', scale_units='xy', scale=1.0,
        color=color, alpha=0.85, zorder=5,
        width=0.0032, headwidth=4.5, headlength=5.0, headaxislength=4.2,
        minshaft=1.5, pivot='tail',
    )
    # reference arrow = the strongest (fully compressed) vector
    try:
        ax.quiverkey(q, 0.88, 1.035, max_frac * x_range,
                     'strongest EP flux', labelpos='E', coordinates='axes',
                     fontproperties={'size': 8})
    except Exception:
        pass
    return q


# ---------------------------------------------------------------------------
# Right-hand y-axis (pressure <-> altitude companion scale)
# ---------------------------------------------------------------------------
def _add_dual_yaxis(ax, alt, pres_hpa, is_pres):
    """Add the companion vertical scale on a twin axis."""
    ax2 = ax.twinx()
    if is_pres:
        p_min_hpa, p_max_hpa = float(pres_hpa.min()), float(pres_hpa.max())
        z_top = _H_SCALE * np.log(_P0_HPA / max(p_min_hpa, 1e-10))
        z_bot = max(0.0, _H_SCALE * np.log(_P0_HPA / p_max_hpa))
        z_range = z_top - z_bot
        # pick a round step giving ~6-10 ticks; falls through to 100 if the
        # range is enormous (intentional -- a coarse scale beats none)
        for step in (1, 2, 5, 10, 20, 25, 50, 100):
            if z_range / step <= 12:
                break
        km_ticks = np.arange(np.ceil(z_bot / step) * step,
                             np.floor(z_top / step) * step + step, step)
        p_at_km = _P0_HPA * np.exp(-km_ticks / _H_SCALE)
        in_range = (p_at_km >= p_min_hpa * 0.9) & (p_at_km <= p_max_hpa * 1.1)
        if in_range.any():
            ax2.set_yscale('log')
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(p_at_km[in_range])
            ax2.set_yticklabels([f"{int(z)}" for z in km_ticks[in_range]],
                                fontsize=8)
        ax2.set_ylabel('Approx. altitude (km)', fontsize=10)
    else:
        if len(alt) > 1 and len(pres_hpa) == len(alt):
            try:
                from scipy.interpolate import interp1d
                p_to_a = interp1d(pres_hpa[::-1], alt[::-1],
                                  bounds_error=False, fill_value='extrapolate')
                p_ticks_all = np.array(
                    [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1, 0.1, 0.01])
                p_min, p_max = pres_hpa.min(), pres_hpa.max()
                p_ticks = p_ticks_all[(p_ticks_all >= p_min * 0.5)
                                      & (p_ticks_all <= p_max * 2)]
                alt_at_p = p_to_a(p_ticks)
                ok = (alt_at_p >= alt[0]) & (alt_at_p <= alt[-1])
                if ok.any():
                    ax2.set_yticks(alt_at_p[ok])
                    ax2.set_yticklabels([f"{p:.3g}" for p in p_ticks[ok]],
                                        fontsize=8)
            except Exception:
                pass
        ax2.set_ylim(ax.get_ylim())
        ax2.set_ylabel('Pressure (hPa)', fontsize=10)
    ax2.tick_params(axis='y', labelsize=8)
    return ax2


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def plot_ep_flux(ds, out_dir=".", prefix="ep_flux",
                 vmax=120.0, rho_min=1e-6, quiver_power=0.45, quiver_frac=0.09):
    """Plot an Eliassen-Palm flux cross-section (lat x altitude/pressure).
 
    Parameters
    ----------
    vmax : float or None
        Fixed symmetric colour limit for a_EP in m/s/day. EP-flux accelerations
        are rarely more than a few tens of m/s/day even in the MLT, so a fixed
        physical ceiling keeps the stratosphere/mesosphere legible while the
        density-blow-up artifacts near the model top simply saturate the end
        colours. Pass None to fall back to a NaN-safe 97th-percentile estimate
        (the old behaviour).
    rho_min : float or None
        Background-density floor [kg/m3]. Where rho0 falls below this, a_EP is
        masked, because dividing the flux divergence by a near-zero density
        produces unphysical (~1e8) values near ~100 km. Pass None to disable.
    quiver_power, quiver_frac : float
        Magnitude-compression exponent and max arrow length (axis fraction) for
        the EP-flux arrows; forwarded to the quiver helper.
    """
    set_publication_style()

    # --- compute or time-average the EP diagnostics ---
    if 'a_EP' not in ds:
        logger.info("EP flux not pre-computed - running pipeline in-memory.")
        from ..analysis.ep_flux import eliassen_palm
        ds = eliassen_palm(ds, time_mean=True)
    elif 'time' in ds['a_EP'].dims:
        logger.info("Averaging pre-computed EP flux over time.")  # noqa: F821
        ds = ds.mean(dim='time', keep_attrs=True)

    alt_name = _find_alt_name(ds)

    # --- classify the vertical coordinate from the RAW coord (pre-conversion) ---
    a_EP_raw = ds['a_EP'].squeeze()
    is_meters = _coord_is_meter(a_EP_raw[alt_name])
    is_pres = (not is_meters) and _is_pressure_coord(alt_name, ds)
    if is_meters:
        alt_label = 'Altitude (km)'
    elif is_pres:
        alt_label = 'Pressure (hPa)'
    else:
        alt_label = alt_name

    # --- prepare every field through one consistent path ---
    def prep(name):
        return _prep(ds[name], alt_name, is_meters, is_pres)

    a_EP = prep('a_EP')
    F_phi = prep('F_phi')
    F_z = prep('F_z')
    u_zm = prep('u_zm') if 'u_zm' in ds else None
    rho0 = prep('rho0') if 'rho0' in ds else None

    # --- mask the density blow-up region -----------------------------------
    # Near ~100 km rho0 -> 1e-7 kg/m3, so a_EP = flux_div / (rho0 a cos phi)
    # explodes to unphysical (~1e8) values in height coordinates. In pressure
    # coordinates, a_EP does not divide by rho0, so it remains physically stable.
    if not is_pres and rho_min is not None and rho0 is not None:
        try:
            a_EP = a_EP.where(rho0.reindex_like(a_EP) > rho_min)
        except Exception:
            logger.warning("could not apply rho0 mask - shapes incompatible")  # noqa: F821

    lat = a_EP['lat'].values
    alt = a_EP[alt_name].values

    # --- pressure profile for the companion axis ---
    if is_pres:
        pres_hpa = a_EP[alt_name].values  # already hPa
    elif 'pres_zm' in ds:
        pz = _prep(ds['pres_zm'], alt_name, is_meters, is_pres)
        pres_hpa = (pz.mean('lat').values if 'lat' in pz.dims else pz.values) / 100.0
    else:
        pres_hpa = _P0_HPA * np.exp(-alt / _H_SCALE)  # standard atmosphere

    # --- figure ---
    fig, ax = plt.subplots(figsize=(11, 7))

    # filled contours: a_EP acceleration
    if vmax is None:
        # legacy behaviour: NaN-safe 97th percentile
        vlim = float(np.nanpercentile(np.abs(a_EP.values), 97))
        vlim = 0.1 if not np.isfinite(vlim) else max(vlim, 0.1)
    else:
        # fixed physical ceiling; artifacts saturate via extend='both'
        vlim = float(vmax)
    fill_levels = np.linspace(-vlim, vlim, 41)
    cf = a_EP.plot.contourf(
        ax=ax, x='lat', y=alt_name,
        levels=fill_levels, cmap='RdBu_r', extend='both',
        add_colorbar=False, add_labels=False,
    )
    cbar = fig.colorbar(cf, ax=ax, pad=0.12, fraction=0.03)
    cbar.set_label(r"EP-flux acceleration  $a_{EP}$  (m s$^{-1}$ day$^{-1}$)",
                   fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # zonal-mean wind contours
    if u_zm is not None:
        _add_wind_contours(ax, u_zm, alt_name)

    # EP flux quiver
    _add_ep_quiver(ax, F_phi, F_z, lat, alt, alt_name,
                   is_pres=is_pres, rho0=rho0,
                   power=quiver_power, max_frac=quiver_frac)

    # vertical scale
    if is_pres:
        ax.set_yscale('log')
        ax.invert_yaxis()
    _add_dual_yaxis(ax, alt, pres_hpa, is_pres)

    # decoration
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xticklabels(['90S', '60S', '30S', '0', '30N', '60N', '90N'])
    ax.set_xlabel('Latitude', fontsize=11)
    ax.set_ylabel(alt_label, fontsize=11)
    mode_lbl = ds.attrs.get('ep_flux_mode', '')
    title = "Eliassen-Palm Flux  (shading: forcing,  arrows: F,  contours: u-bar)"
    if mode_lbl:
        title += f"  [{mode_lbl.upper()}]"
    ax.set_title(title, fontweight='bold', fontsize=11, pad=10)
    ax.grid(True, linestyle='--', alpha=0.35, color='grey')

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_ep_flux.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved EP flux plot to {out_path}")  # noqa: F821
    plt.close(fig)
    return out_path
