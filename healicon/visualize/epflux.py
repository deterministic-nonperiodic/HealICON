import logging
import os

import matplotlib.pyplot as plt
import numpy as np

from .common import set_publication_style
from ..analysis.ep_flux import _find_alt_name, _is_pressure_coord
from ..cf_coords import _coord_is_meter, convert_units, equivalent_units

logger = logging.getLogger(__name__)

# Constants
_A_EARTH = 6.371e6  # Earth radius [m]
_H_SCALE = 7.0  # log-pressure scale height [km]
_P0_HPA = 1013.25  # reference surface pressure [hPa]
_G = 9.81 # gravity acceleration [m/s^2]
_KAPPA = 2/7 # ratio of specific heats

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def _prep(da, alt_name, is_meters, is_pres, div_time=True):
    """Squeeze, optionally time-mean, convert vertical coordinate to display units, and sort.

    Height coords → km; pressure coords → hPa.  A mis-named coordinate that
    cannot be sorted will raise rather than produce a silently unsorted plot.
    """
    da = da.squeeze()
    if div_time and 'time' in da.dims:
        da = da.mean('time')
    if alt_name in da.coords:
        coord = da[alt_name]
        coord_units = str(coord.attrs.get('units', '')) or None
        if is_meters:
            src = coord_units or 'm'
            if not equivalent_units(src, 'km'):
                da = da.assign_coords({alt_name: convert_units(coord, src, 'km')})
        elif is_pres:
            # Infer Pa from magnitude when units attribute is absent
            if coord_units is None:
                coord_units = 'Pa' if float(coord.max()) > 2000.0 else 'hPa'
            if not equivalent_units(coord_units, 'hPa'):
                da = da.assign_coords({alt_name: convert_units(coord, coord_units, 'hPa')})
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
# EP flux quiver — Jucker (2021) / Edmon, Hoskins & McIntyre (1980) scaling
# ---------------------------------------------------------------------------
def _ep_scale_jucker(F_phi, F_z, lat, alt, alt_name, rho0, is_pres):
    """Scale EP flux vectors for display (Edmon et al. 1980 / Jucker 2021).

    Pressure coords — matches aostools PlotEPfluxArrows exactly:
        ep1 = F_phi / (a cosφ)           [m² s⁻²]
        ep2 = F_z   / (100 a cosφ)       [hPa m s⁻²]
        Fphi  = (2π/g) cos²φ a²  ep1     [m³]
        Fvert = (2π/g) cos²φ a³  ep2     [m³ hPa]

    Height coords — adapted so that Fphi·dx and Fvert·dy have matching units
    after the inch-conversion in _add_ep_quiver:
        ep1 = ep2 = F / (ρ₀ a cosφ)     [m² s⁻²]
        Fphi  = cos²φ a²  ep1            [m⁴ s⁻²]
        Fvert = cos²φ a³  ep2            [m⁵ s⁻²]

    Returns (Fphi, Fvert) as (lat, alt) numpy arrays.
    """
    fp = F_phi.transpose('lat', alt_name).values
    fz = F_z.transpose('lat', alt_name).values

    cos_phi = np.cos(np.deg2rad(lat))                  # (lat,)
    cos2    = cos_phi ** 2
    geom    = np.outer(cos_phi, np.ones(len(alt))) * _A_EARTH   # a cosφ  (lat, alt)
    geom    = np.where(np.abs(geom) < 1e-30, 1e-30, geom)
    cos2_2d = np.outer(cos2, np.ones(len(alt)))

    if is_pres:
        # ep1_cart [m²/s²], ep2_hPa [hPa·m/s²]
        ep1 = fp / geom                    # [m²/s²]
        ep2_hpa = fz / geom / 100.0        # [hPa·m/s²]  (F_z in Pa·m²/s², /100 → hPa)
        Fphi  = (2*np.pi / _G) * cos2_2d * _A_EARTH**2 * ep1      # [m³]
        Fvert = (2*np.pi / _G) * cos2_2d * _A_EARTH**3 * ep2_hpa  # [m³·hPa]
    else:
        # Both components [m²/s²]; ep2 gets an extra a so units match after dy [in/m]
        if rho0 is not None:
            r_val = (rho0.transpose(alt_name, 'lat').values.T
                     if rho0.ndim == 2
                     else np.outer(np.ones(len(lat)), rho0.values))
            r_val = np.where(r_val < 1e-30, 1e-30, r_val)
        else:
            r_val = np.ones_like(geom)
        ep1 = fp / (geom * r_val)          # [m²/s²]
        ep2 = fz / (geom * r_val)          # [m²/s²]
        Fphi  = cos2_2d * _A_EARTH**2 * ep1   # [m⁴/s²]
        Fvert = cos2_2d * _A_EARTH**3 * ep2   # [m⁵/s²]  (→ m⁴·in/s² after ×in/m)

    return Fphi, Fvert


def _add_ep_quiver(ax, F_phi, F_z, lat, alt, alt_name, *, is_pres, rho0=None,
                   n_lat=22, n_alt=18, color='0.15', quiver_scale=1.0):
    """Draw EP-flux arrows scaled by :func:`_ep_scale_jucker`.

    Arrow direction is the group-velocity direction in physical (lat, z) space.
    For pressure coords the result is identical to aostools.PlotEPfluxArrows.

    Returns the matplotlib Quiver, or None if no finite values remain.
    """
    Fphi, Fvert = _ep_scale_jucker(F_phi, F_z, lat, alt, alt_name, rho0, is_pres)

    # --- subsample to a readable arrow grid --------------------------------
    sl = max(1, len(lat) // n_lat)
    sa = max(1, len(alt) // n_alt)
    lat_s = lat[::sl]
    alt_s = alt[::sa]
    LAT2, ALT2 = np.meshgrid(lat_s, alt_s)          # (n_alt_s, n_lat_s)
    Fphi_s  = Fphi[::sl, ::sa].T                     # (lat,alt) → (alt,lat)
    Fvert_s = Fvert[::sl, ::sa].T

    # Suppress polar region: cosφ → 0 amplifies residual noise by 1/cosφ.
    # Exclude |lat| > 85° completely; they must not influence the arrow scale.
    pole_mask = np.abs(LAT2) > 85.0
    Fphi_s  = np.where(pole_mask, np.nan, Fphi_s)
    Fvert_s = np.where(pole_mask, np.nan, Fvert_s)

    # --- Jucker (2021): convert to display-space (inches) ------------------
    # Axis physical dimensions in inches, independent of canvas draw state.
    fig   = ax.get_figure()
    fig_w, fig_h = fig.get_size_inches()
    pos   = ax.get_position()
    ax_w  = pos.width  * fig_w   # inches
    ax_h  = pos.height * fig_h   # inches

    delta_x   = (lat[-1] - lat[0]) or 1.0
    delta_x_r = delta_x * np.pi / 180.0             # radians

    # dx [in/rad ≡ in]: "distance occupied by 1 radian of latitude on diagram"
    dx = ax_w / delta_x_r

    if is_pres:
        # log-pressure y-axis; pressure is inverted (larger p at bottom)
        p_min   = float(np.nanmin(alt_s[alt_s > 0])) if np.any(alt_s > 0) else 1e-3
        p_max   = float(np.nanmax(alt_s))
        log_span = abs(np.log(p_max / max(p_min, 1e-10))) or 1.0
        # dy [in/hPa]: element-wise so it is correct for each pressure level
        dy = -ax_h / np.maximum(ALT2, 1e-10) / log_span   # negative: p increases downward
        u_arr = Fphi_s  * dx                 # [m³·in]
        v_arr = Fvert_s * dy                 # [m³·hPa·in/hPa] = [m³·in]
    else:
        # linear height y-axis; alt is in km (after _prep), convert to m for units
        delta_y_m = (alt[-1] - alt[0]) * 1000.0 or 1.0
        dy = ax_h / delta_y_m                # [in/m]
        u_arr = Fphi_s  * dx                 # [m⁴/s²·in]
        v_arr = Fvert_s * dy                 # [m⁵/s²·in/m] = [m⁴/s²·in]

    # --- quiver with aostools conventions: angles='uv', scale_units='inches'
    finite = np.isfinite(u_arr) & np.isfinite(v_arr)
    if not finite.any():
        return None

    # Explicit scale so quiver_scale actually controls arrow length.
    # With auto-scale (scale=None), matplotlib compensates for any pre-scaling
    # of u_arr/v_arr, making a divisor invisible. Instead we fix scale =
    # max_amplitude * quiver_scale / ax_w so the longest arrow spans
    # ax_w / (n_lat * quiver_scale) inches.  quiver_scale > 1 → shorter arrows.
    max_amp = float(np.nanmax(np.hypot(u_arr[finite], v_arr[finite])))
    ref_scale = max_amp * n_lat * quiver_scale / max(ax_w, 1e-6)

    q = ax.quiver(
        LAT2, ALT2, u_arr, v_arr,
        angles='uv', scale_units='inches', scale=ref_scale,
        pivot='tail', color=color, alpha=0.85, zorder=5,
        width=0.0028, headwidth=4.5, headlength=5.0, headaxislength=4.2,
    )

    # canvas.draw() is required to populate Q.scale in non-interactive mode
    try:
        fig.canvas.draw()
        U = q.scale
        if U is not None:
            label = (r'{:.1e}$\,\mathrm{{m}}^3$'.format(U / ax_w)
                     if is_pres
                     else r'{:.1e}$\,\mathrm{{m}}^4\mathrm{{s}}^{{-2}}$'.format(U / ax_w))
            ax.quiverkey(q, 0.88, 1.035, U / ax_w, label,
                         labelpos='E', coordinates='axes',
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
                 vmax=120.0, rho_min=None, quiver_scale=0.8):
    """Plot an Eliassen-Palm flux cross-section (lat x altitude/pressure).

    Parameters
    ----------
    vmax : float or None
        Fixed symmetric colour limit for a_EP in m/s/day.  Large values near
        the model top saturate the end colours rather than dominating the scale.
        Pass None to use a NaN-safe 97th-percentile estimate instead.
    rho_min : float or None
        Optional density floor [kg/m3].  Where rho0 falls below this, a_EP is
        blanked.  Default None (no masking): ``vmax`` already keeps the colour
        scale readable by saturation, and blanking hides physically valid high-
        altitude data.  Pass e.g. ``1e-6`` to restore the old behaviour of
        cutting off above ~100 km in height coordinates.

    Returns
    -------
    str
        Absolute path of the saved PNG.
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
    if not is_pres and rho_min is not None and rho0 is not None:
        try:
            dense_enough = rho0 > rho_min
            a_EP  = a_EP.where(dense_enough)
            F_phi = F_phi.where(dense_enough)
            F_z   = F_z.where(dense_enough)
        except Exception as e:
            logger.warning(f"rho0 mask could not be applied: {e}")

    # --- mask thermosphere where the EP framework itself breaks down --------
    alt_max_km = None
    if not is_pres:
        valid_max_m = ds['a_EP'].attrs.get('valid_height_max_m', None)
        if valid_max_m is not None:
            alt_max_km = float(valid_max_m) / 1000.0
            if rho_min is None:
                a_EP  = a_EP.where(a_EP[alt_name]   <= alt_max_km)
                F_phi = F_phi.where(F_phi[alt_name] <= alt_max_km)
                F_z   = F_z.where(F_z[alt_name]     <= alt_max_km)

    lat = a_EP['lat'].values
    alt = a_EP[alt_name].values

    # --- pressure profile for the companion axis ---
    if is_pres:
        pres_hpa = a_EP[alt_name].values  # already hPa
    elif 'pres_zm' in ds:
        pz = _prep(ds['pres_zm'], alt_name, is_meters, is_pres)
        pz_mean = pz.mean('lat') if 'lat' in pz.dims else pz
        pz_units = str(pz_mean.attrs.get('units', 'Pa')) or 'Pa'
        pres_hpa = convert_units(pz_mean, pz_units, 'hPa').values
    else:
        pres_hpa = _P0_HPA * np.exp(-alt / _H_SCALE)  # standard atmosphere

    # --- figure ---
    fig, ax = plt.subplots(figsize=(11, 7))

    # filled contours: a_EP acceleration
    if vmax is None:
        vlim = float(np.nanpercentile(np.abs(a_EP.values), 97))
        vlim = 0.1 if not np.isfinite(vlim) else max(vlim, 0.1)
    else:
        vlim = float(vmax)
    fill_levels = np.linspace(-vlim, vlim, 41)
    cf = a_EP.plot.contourf(
        ax=ax, x='lat', y=alt_name,
        levels=fill_levels, cmap='RdBu_r', extend='both',
        add_colorbar=False, add_labels=False,
    )
    cbar = fig.colorbar(cf, ax=ax, pad=0.075, fraction=0.024, shrink=0.90)
    cbar.set_label(r"EP-flux acceleration  $a_{EP}$  (m s$^{-1}$ day$^{-1}$)",
                   fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # zonal-mean wind contours
    if u_zm is not None:
        _add_wind_contours(ax, u_zm, alt_name)

    # EP flux quiver
    _add_ep_quiver(ax, F_phi, F_z, lat, alt, alt_name,
                   is_pres=is_pres, rho0=rho0, quiver_scale=quiver_scale)

    # --- hatched overlay above EP validity ceiling (height coords only) ----
    if alt_max_km is not None and alt_max_km < alt[-1]:
        ax.fill_between([-90, 90], alt_max_km, alt[-1],
                        hatch='////', facecolor='white', edgecolor='0.55',
                        linewidth=0.0, alpha=0.55, zorder=5)
        ax.axhline(alt_max_km, color='0.35', lw=0.9, ls='--', zorder=5)
        ax.text(0.02, alt_max_km, ' EP validity limit', transform=ax.get_yaxis_transform(),
                va='bottom', fontsize=7, color='0.35', zorder=6)

    # vertical scale
    if is_pres:
        ax.set_yscale('log')
        ax.invert_yaxis()
    _add_dual_yaxis(ax, alt, pres_hpa, is_pres)

    # decoration
    ax.set_xlim(-90, 90)
    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])
    ax.set_xticklabels(['90°S', '60°S', '30°S', '0°', '30°N', '60°N', '90°N'])
    ax.set_xlabel('Latitude', fontsize=11)
    ax.set_ylabel(alt_label, fontsize=11)
    mode_lbl = ds.attrs.get('ep_flux_mode', '')
    title = "Eliassen-Palm Flux"
    if mode_lbl:
        title += f"  [{mode_lbl.upper()}]"
    ax.set_title(title, fontweight='bold', fontsize=11, pad=10)
    ax.grid(True, linestyle='-', alpha=0.35, color='grey')

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_ep_flux.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved EP flux plot to {out_path}")  # noqa: F821
    plt.close(fig)
    return out_path
