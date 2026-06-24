import logging
import os
import re

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .cf_coords import _find_coordinate, _coord_is_meter

logger = logging.getLogger(__name__)

# Try to import cmasher, fallback to standard matplotlib cmaps
try:
    import cmasher as cmr

    wind_cm = cmr.fusion_r
except ImportError:
    wind_cm = 'RdYlBu_r'

# Simple temp colormap definition based on standard colors if file not present
# This mimics a standard diverging or sequential colormap
temp_cm = 'inferno'

# ---------------------------------------------------------------------------
# Variable display-name map (HealICON spectral conventions)
# Keys are dataset variable names produced by compute_spectrum(); values are
# short LaTeX labels used in legends and y-axis titles.
# ---------------------------------------------------------------------------
SPECTRAL_KEYMAP: dict[str, str] = {
    # --- Kinetic energy spectra ---
    'kinetic_energy_cl': r'$E_K$',
    # --- Wind-component power spectra (CF short names) ---
    'u_cl': r'$C_l^{u}$',
    'v_cl': r'$C_l^{v}$',
    'w_cl': r'$C_l^{w}$',
    'ua_cl': r'$C_l^{u}$',
    'va_cl': r'$C_l^{v}$',
    'wa_cl': r'$C_l^{w}$',
    # --- Scalar thermodynamic variables ---
    'temperature_cl': r'$C_l^{T}$',
    'temp_cl': r'$C_l^{T}$',
    'T_cl': r'$C_l^{T}$',
    'ta_cl': r'$C_l^{T}$',
    'theta_cl': r'$C_l^{\theta}$',
    'pt_cl': r'$C_l^{\theta}$',
    # --- Humidity ---
    'qv_cl': r'$C_l^{q_v}$',
    'hus_cl': r'$C_l^{q_v}$',
    'q_cl': r'$C_l^{q}$',
    # --- Divergence and vorticity ---
    'divergence_cl': r'$C_l^{D}$',
    'div_cl': r'$C_l^{D}$',
    'vorticity_cl': r'$C_l^{\zeta}$',
    'vor_cl': r'$C_l^{\zeta}$',
    'zeta_cl': r'$C_l^{\zeta}$',
    # --- Geopotential / pressure ---
    'zg_cl': r'$C_l^{\Phi}$',
    'geopot_cl': r'$C_l^{\Phi}$',
    'phi_cl': r'$C_l^{\Phi}$',
    'pres_cl': r'$C_l^{p}$',
    'ps_cl': r'$C_l^{p_s}$',
    # --- Density ---
    'rho_cl': r'$C_l^{\rho}$',
}

def cf_to_latex(unit_string: str) -> str:
    """Convert a CF/udunits unit string to a minimal LaTeX math-mode string.

    Verbose unit names are abbreviated (kelvin→K, meter→m, second→s, …)
    before exponents are wrapped in LaTeX braces.

    Examples::

        'kelvin ** 2'  →  '$K^{2}$'
        'm s-1'        →  '$m s^{-1}$'
        'm2 s-2'       →  '$m^{2} s^{-2}$'
        'meter second-1' → '$m s^{-1}$'
    """
    # 1. Normalise verbose CF/udunits names to SI abbreviations (whole words only)
    _ABBREV = {
        'kelvin':     'K',
        'meter':      'm',
        'second':     's',
        'kilogram':   'kg',
        'pascal':     'Pa',
        'joule':      'J',
        'watt':       'W',
        'radian':     'rad',
        'degree':     'deg',
        'kilometer':  'km',
    }
    for long, short in _ABBREV.items():
        unit_string = re.sub(rf'\b{long}\b', short, unit_string, flags=re.IGNORECASE)

    # 2. Normalise exponentiation: '**' → '^', then strip spaces around '^'
    unit_string = unit_string.replace('**', '^')
    unit_string = re.sub(r'\s*\^\s*', '^', unit_string)   # 'K ^ 2' → 'K^2'

    # 3. Wrap exponents in braces: 'K^2' → 'K^{2}', 'm s-1' → 'm s^{-1}'
    unit_string = re.sub(r'([a-zA-Z])\^?([\-]?\d+)', r'\1^{\2}', unit_string)

    return f'${unit_string}$'


def set_publication_style():
    """Apply publication-ready matplotlib parameters."""
    params = {
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small',
        'font.size': 13,
        'legend.title_fontsize': 13,
        'legend.fontsize': 13,
        'font.family': 'serif',
        'text.usetex': False,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'figure.titlesize': 16,
    }
    plt.rcParams.update(params)


def plot_tides(ds: xr.Dataset, out_dir: str = ".", prefix: str = "tides", max_amplitude=None):
    """
    Plot tidal amplitude and phase across latitudes and heights.

    Accepts either a pre-processed zonal-mean dataset (with a ``'lat'``
    coordinate) or a raw tidal analysis output (with a HEALPix cell dimension).
    In the latter case zonal mean is computed on-the-fly.  If the dataset still
    has a ``'time'`` dimension, a temporal mean is applied automatically.
    """
    set_publication_style()

    level_coord = _find_coordinate(ds, 'level', raise_notfound=False)
    if level_coord is None:
        logger.error("Dataset missing height coordinate for tidal cross-section.")
        return
    level_name = level_coord.name

    lat_coord = _find_coordinate(ds, 'lat', raise_notfound=False)
    lat_name = lat_coord.name if (
            lat_coord is not None and lat_coord.ndim == 1 and lat_coord.name in ds.dims) else 'lat'

    if lat_name not in ds.dims:
        # Raw HEALPix output — compute zonal mean on-the-fly.
        from .extract import zonal_mean
        logger.info(f"'{lat_name}' coordinate not found — computing zonal mean before plotting.")
        try:
            # If 'time' is still present, collapse it first.
            # amp variables: linear temporal mean; pha variables: circular mean.
            if 'time' in ds.dims:
                logger.info("Averaging over 'time' dimension before zonal mean.")
                amp_vars = [v for v in ds.data_vars if '_amp_' in v]
                pha_vars = [v for v in ds.data_vars if '_pha_' in v]
                parts = {}
                if amp_vars:
                    parts.update(ds[amp_vars].mean('time').data_vars)
                if pha_vars:
                    import numpy as _np
                    for v in pha_vars:
                        pha = ds[v]
                        # Amplitude-weighted circular mean: phase is meaningful
                        # only where amplitude is large; weight by amplitude so
                        # that low-amplitude (noisy) time steps barely contribute.
                        v_amp = v.replace('_pha_', '_amp_')
                        if v_amp in ds.data_vars:
                            w = ds[v_amp]
                        else:
                            w = xr.ones_like(pha)
                        parts[v] = _np.arctan2(
                            (w * _np.sin(pha)).mean('time'),
                            (w * _np.cos(pha)).mean('time'),
                        )
                # Preserve 0-D grid-mapping scalars (e.g. 'healpix') so that
                # zonal_mean can detect the pixel ordering (nested vs ring).
                for v in ds.data_vars:
                    if ds[v].dims == () and v not in parts:
                        parts[v] = ds[v]
                ds_new = xr.Dataset(
                    parts,
                    coords={k: v for k, v in ds.coords.items()
                            if 'time' not in ds[k].dims},
                )
                ds_new.attrs = ds.attrs
                ds = ds_new
            # Zonal mean: amplitude variables — arithmetic ring average.
            # Phase variables — must go via cos/sin so zonal_mean does
            # circular (not arithmetic) averaging.  Matches the explicit
            # cos/sin decomposition used in recreate_tides_wavelet.py.
            pha_vars_ds = [v for v in ds.data_vars if '_pha_' in v]
            cos_sin_map = {}  # original var → (cos_name, sin_name)
            ds_for_zm = ds
            for v in pha_vars_ds:
                cos_name = f'__cos_{v}'
                sin_name = f'__sin_{v}'
                ds_for_zm = ds_for_zm.assign({cos_name: np.cos(ds_for_zm[v]),
                                              sin_name: np.sin(ds_for_zm[v])})
                cos_sin_map[v] = (cos_name, sin_name)
            if cos_sin_map:
                ds_for_zm = ds_for_zm.drop_vars(list(cos_sin_map.keys()))

            ds = zonal_mean(ds_for_zm)
            lat_name = 'lat'

            # Reconstruct phase from zonal-mean cos/sin
            for v, (cos_name, sin_name) in cos_sin_map.items():
                ds[v] = np.arctan2(ds[sin_name], ds[cos_name])
                ds = ds.drop_vars([cos_name, sin_name])


        except Exception as exc:
            logger.error(f"Automatic zonal mean failed: {exc}")
            return

    if lat_name in ds.dims and level_name in ds.dims:
        ds = ds.sortby([lat_name, level_name])

    # Dynamic variable base name detection (detects e.g., 'temp' or 'u')
    amp_sym_vars = [v for v in ds.data_vars if v.endswith('_amp_sym')]
    if not amp_sym_vars:
        logger.error("No tidal amplitude variables (*_amp_sym) found in dataset.")
        return
    var_base = amp_sym_vars[0][:-8]  # Remove '_amp_sym'

    p_12 = np.timedelta64(12, 'h')
    p_24 = np.timedelta64(24, 'h')

    modes = {
        'DW1': {'period': p_24, 'm': 1, 'type': 'Symmetric'},
        'DW1_asy': {'period': p_24, 'm': 1, 'type': 'Antisymmetric'},
        'SW2': {'period': p_12, 'm': 2, 'type': 'Symmetric'},
        'SW2_asy': {'period': p_12, 'm': 2, 'type': 'Antisymmetric'},
        'SE2': {'period': p_12, 'm': -2, 'type': 'Symmetric'},
        'SE2_asy': {'period': p_12, 'm': -2, 'type': 'Antisymmetric'},
        'DE3': {'period': p_24, 'm': -3, 'type': 'Symmetric'},
        'DE3_asy': {'period': p_24, 'm': -3, 'type': 'Antisymmetric'},
    }

    # Identify available modes in the dataset
    available_modes = {}
    for name, meta in modes.items():
        var_name = f'{var_base}_amp_sym' if meta['type'] == 'Symmetric' else f'{var_base}_amp_asy'
        if var_name in ds and 'period' in ds.coords and 'm' in ds.coords:
            try:
                # We use nearest to just check if it's broadly there, or let sel handle it
                ds[var_name].sel(period=meta['period'], m=meta['m'], method='nearest')
                available_modes[name] = meta
            except Exception:
                pass

    if not available_modes:
        logger.error("No matching tidal modes found in dataset.")
        return

    n_modes = len(available_modes)
    n_cols = 2
    n_rows = int(np.ceil(n_modes / n_cols))

    # --- Amplitude Plot ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows + 1), sharex=True, sharey=True,
                             squeeze=False)
    axes = axes.flatten()

    lat_ticks = [-60, -30, 0, 30, 60]
    lat_labels = ['60°S', '30°S', '0°', '30°N', '60°N']
    vmax = 6.0

    # Try to find max amplitude for better scaling
    try:
        max_amp = float(ds[[f'{var_base}_amp_sym', f'{var_base}_amp_asy']].to_array().max())
        if np.isfinite(max_amp) and max_amp > 0:
            vmax = min(max_amp,
                       max_amplitude if max_amplitude is not None else 20.0)  # Cap at 20 for visibility
    except:
        pass

    levels = np.linspace(0, vmax, 13)

    cf = None
    for i, (name, meta) in enumerate(available_modes.items()):
        ax = axes[i]
        var_name = f'{var_base}_amp_sym' if meta['type'] == 'Symmetric' else f'{var_base}_amp_asy'
        data = ds[var_name].sel(period=meta['period'], m=meta['m'], method='nearest')

        # Ensure correct dimension order: (height, lat)
        if data.dims != (level_name, lat_name):
            data = data.transpose(level_name, lat_name)

        if _coord_is_meter(level_coord):
            data = data.assign_coords({level_name: data[level_name] / 1000.0})
            data[level_name].attrs['units'] = 'km'
            if 'long_name' not in data[level_name].attrs:
                data[level_name].attrs['long_name'] = 'Height'

        cf = data.plot.contourf(
            ax=ax,
            x=lat_name,
            y=level_name,
            levels=levels,
            cmap='inferno',
            add_colorbar=False,
            add_labels=False
        )
        ax.set_title(f"{name.split('_')[0]} ({meta['type']})", fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

    # Formatting
    for i, ax in enumerate(axes):
        if i >= n_modes:
            ax.remove()
        else:
            if i % n_cols == 0:
                y_label_name = data[level_name].attrs.get('long_name',
                                                          level_name.replace('_', ' ').title())
                y_units = data[level_name].attrs.get('units', '')
                ax.set_ylabel(f"{y_label_name} / {y_units}" if y_units else y_label_name)
            if i >= n_modes - n_cols:
                ax.set_xlabel("Latitude")
                ax.set_xlim(-60, 60)
                ax.set_xticks(lat_ticks)
                ax.set_xticklabels(lat_labels)
            ax.label_outer()

    if _coord_is_meter(level_coord):
        axes[0].set_ylim(60, 110)
    else:
        axes[0].invert_yaxis()

    fig.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.1)
    if cf:
        cbar_ax = fig.add_axes([0.26, 0.05, 0.52, 0.02])
        cbar = fig.colorbar(cf, cax=cbar_ax, orientation='horizontal')
        var_units = ds[f'{var_base}_amp_sym'].attrs.get('units', 'K')
        cbar.set_label(f'Amplitude / {var_units}')

    os.makedirs(out_dir, exist_ok=True)
    amp_out_path = os.path.join(out_dir, f"{prefix}_amplitude.png")
    plt.savefig(amp_out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved amplitude plot to {amp_out_path}")
    plt.close(fig)

    # --- Phase Plot ---
    fig_pha, axes_pha = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows + 1), sharex=True,
                                     sharey=True, squeeze=False)
    axes_pha = axes_pha.flatten()
    levels_pha = np.linspace(-np.pi, np.pi, 20)

    cf_pha = None
    for i, (name, meta) in enumerate(available_modes.items()):
        ax = axes_pha[i]
        var_name = f'{var_base}_pha_sym' if meta['type'] == 'Symmetric' else f'{var_base}_pha_asy'
        data = ds[var_name].sel(period=meta['period'], m=meta['m'], method='nearest')

        # Ensure correct dimension order: (height, lat)
        if data.dims != (level_name, lat_name):
            data = data.transpose(level_name, lat_name)

        if _coord_is_meter(level_coord):
            data = data.assign_coords({level_name: data[level_name] / 1000.0})
            data[level_name].attrs['units'] = 'km'
            if 'long_name' not in data[level_name].attrs:
                data[level_name].attrs['long_name'] = 'Height'

        cf_pha = data.plot.contourf(
            ax=ax,
            x=lat_name,
            y=level_name,
            levels=levels_pha,
            cmap='twilight_shifted',
            add_colorbar=False,
            add_labels=False
        )
        ax.set_title(f"{name.split('_')[0]} Phase ({meta['type']})", fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

    for i, ax in enumerate(axes_pha):
        if i >= n_modes:
            ax.remove()
        else:
            if i % n_cols == 0:
                y_label_name = data[level_name].attrs.get('long_name',
                                                          level_name.replace('_', ' ').title())
                y_units = data[level_name].attrs.get('units', '')
                ax.set_ylabel(f"{y_label_name} / {y_units}" if y_units else y_label_name)
            if i >= n_modes - n_cols:
                ax.set_xlabel("Latitude")
                ax.set_xlim(-60, 60)
                ax.set_xticks(lat_ticks)
                ax.set_xticklabels(lat_labels)
            ax.label_outer()

    if _coord_is_meter(level_coord):
        axes_pha[0].set_ylim(60, 110)
    else:
        axes_pha[0].invert_yaxis()

    fig_pha.subplots_adjust(bottom=0.15, hspace=0.1, wspace=0.1)
    if cf_pha:
        cbar_ax_pha = fig_pha.add_axes([0.26, 0.05, 0.52, 0.02])
        cbar_pha = fig_pha.colorbar(cf_pha, cax=cbar_ax_pha, orientation='horizontal',
                                    ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        cbar_pha.ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$'])
        var_units_pha = ds[f'{var_base}_pha_sym'].attrs.get('units', 'rad')
        cbar_pha.set_label(f'Phase / {var_units_pha}')

    pha_out_path = os.path.join(out_dir, f"{prefix}_phase.png")
    plt.savefig(pha_out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved phase plot to {pha_out_path}")
    plt.close(fig_pha)


def plot_section(ds: xr.Dataset, var_name: str, x_dim: str = 'lat', y_dim: str = 'z_mc',
                 out_dir: str = ".", prefix: str = "section"):
    """
    Plot a 2D cross-section (e.g. Latitude vs Height, or Time vs Height).
    """
    set_publication_style()

    if var_name not in ds:
        logger.error(f"Variable '{var_name}' not found in dataset.")
        return

    data = ds[var_name]

    # Squeeze out singleton dimensions
    data = data.squeeze()

    if x_dim not in data.coords or y_dim not in data.coords:
        logger.error(f"Dimensions '{x_dim}' or '{y_dim}' not available for plotting.")
        return

    # Average over remaining dimensions
    reduced_dims = [dim for dim in data.dims if dim not in [x_dim, y_dim]]
    if reduced_dims:
        logger.info(f"Averaging over additional dimensions: {reduced_dims}")
        data = data.mean(dim=reduced_dims)

    if y_dim in ds.coords and _coord_is_meter(ds[y_dim]):
        data = data.assign_coords({y_dim: data[y_dim] / 1000.0})
        data[y_dim].attrs['units'] = 'km'
        if 'long_name' not in data[y_dim].attrs:
            data[y_dim].attrs['long_name'] = 'Height'

    fig, ax = plt.subplots(figsize=(8, 5))

    # Determine colormap based on variable
    cmap = temp_cm if 'temp' in var_name else wind_cm

    # Use xarray's built-in contourf plotting
    cf = data.plot.contourf(
        ax=ax,
        x=x_dim,
        y=y_dim,
        levels=20,
        cmap=cmap,
        add_colorbar=True,
        cbar_kwargs={'pad': 0.02}
    )

    if x_dim == 'lat':
        ax.set_xlim(-90, 90)
        ax.set_xticks([-60, -30, 0, 30, 60])
        ax.set_xticklabels(['60°S', '30°S', '0°', '30°N', '60°N'])

    y_is_pressure = False
    if y_dim in ds.coords:
        units = str(ds[y_dim].attrs.get('units', '')).strip().lower()
        if any(u in units for u in ('pa', 'hpa', 'mb', 'millibar', 'bar')):
            y_is_pressure = True

    if y_dim == 'plev' or y_is_pressure:
        ax.invert_yaxis()
        ax.set_yscale('log')

    # Title & grid customization
    long_name = data.attrs.get('long_name', var_name)
    ax.set_title(f"{long_name} Cross-Section", fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_{var_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved cross-section plot to {out_path}")
    plt.close(fig)


def plot_map(ds: xr.Dataset, var_name: str, target_height: float | None = None, out_dir: str = ".",
             prefix: str = "map"):
    """
    Plot a 2D global map of HEALPix or regular grid data.
    """
    set_publication_style()

    if var_name not in ds:
        logger.error(f"Variable '{var_name}' not found in dataset.")
        return

    data = ds[var_name].squeeze()

    # Check if HEALPix
    is_healpix = False
    from .grid import get_cells_dim
    try:
        cell_dim = get_cells_dim(ds)
        if cell_dim in data.dims:
            is_healpix = True
    except ValueError:
        pass

    title_suffix = ""
    # Handle vertical level selection
    level_coord = _find_coordinate(ds, 'level', raise_notfound=False)
    if level_coord is not None:
        v_dim = level_coord.name
        is_meters = _coord_is_meter(level_coord)

        if target_height is not None:
            target_val = target_height * 1000.0 if is_meters else target_height
            logger.info(
                f"Selecting {v_dim} closest to {target_height} km (target val: {target_val}).")
            data = data.sel({v_dim: target_val}, method='nearest')
        elif v_dim in data.dims:
            logger.info(f"Selecting first level for {v_dim}.")
            data = data.isel({v_dim: 0})

        if v_dim in data.coords:
            val = data[v_dim].item()
            val_km = val / 1000.0 if is_meters else val
            title_suffix = f" at ~{val_km:.1f} km"

    reduced_dims = [dim for dim in data.dims if
                    dim not in (cell_dim if is_healpix else ['lat', 'lon'])]
    if reduced_dims:
        logger.info(f"Selecting first index for extra dimensions: {reduced_dims}")
        data = data.isel({dim: 0 for dim in reduced_dims})

    units = data.attrs.get('units', '')
    long_name = data.attrs.get('long_name', var_name)
    cmap = temp_cm if 'temp' in var_name else wind_cm

    fig = plt.figure(figsize=(10, 5))

    if is_healpix:
        # HEALPix mollview
        # Healpy handles plotting differently, directly acting on current figure
        arr = data.values
        from .grid import get_healpix_order
        is_nested = get_healpix_order(ds) == 'nested'

        hp.mollview(arr, fig=fig.number, nest=is_nested, cmap=cmap,
                    title=f"{long_name}{title_suffix}", unit=units)
    else:
        lat_coord = _find_coordinate(ds, 'lat', raise_notfound=False)
        lon_coord = _find_coordinate(ds, 'lon', raise_notfound=False)
        if lat_coord is None or lon_coord is None:
            logger.error("Dataset missing 'lat' or 'lon' coordinates for regular map.")
            return

        ax = plt.axes()
        # Use xarray's built-in contourf plotting
        cf = data.plot.contourf(
            ax=ax,
            x=lon_coord.name,
            y=lat_coord.name,
            levels=20,
            cmap=cmap,
            cbar_kwargs={'orientation': 'horizontal', 'pad': 0.1}
        )

        ax.set_title(f"{long_name} Map{title_suffix}", fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_{var_name}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved map plot to {out_path}")
    plt.close(fig)


def _add_reference_slopes(ax, l_arr, data_arr):
    """
    Overlay canonical atmospheric kinetic-energy reference slopes on a log-log spectral plot.

    Two slopes are drawn:
      • l⁻³   anchored in the synoptic range  (l ≤ 20)
      • l⁻⁵·³ anchored in the mesoscale range (20 < l ≤ min(lmax, 1000))

    The magnitude of each reference line is set so that it passes through the median
    of the plotted data inside the corresponding l-range, keeping the lines informative
    without obscuring the actual spectra.

    Args:
        ax:      Matplotlib axes object (must already be in log-log mode).
        l_arr:   1-D array of spherical harmonic degrees (positive integers, l ≥ 1).
        data_arr: 1-D array of power/energy values aligned with l_arr.
    """

    l_arr = np.asarray(l_arr, dtype=float)
    data_arr = np.asarray(data_arr, dtype=float)

    slopes = [
        (-3, (2, 20), r"$l^{-3}$"),
        (-5 / 3, (20, 1000), r"$l^{-5/3}$"),
    ]

    for exp, (l_lo, l_hi), label in slopes:
        l_hi = min(l_hi, l_arr[-1])
        if l_lo >= l_hi:
            continue

        # Build a dense l-grid for the reference line
        l_ref = np.geomspace(l_lo, l_hi, 120)

        # Anchor magnitude: median of the data in this l-range (ignore NaN)
        mask = (l_arr >= l_lo) & (l_arr <= l_hi)
        if mask.sum() == 0 or not np.any(np.isfinite(data_arr[mask])):
            continue
        l_anchor = np.sqrt(l_lo * l_hi)  # geometric-mean anchor point
        d_median = np.nanmedian(data_arr[mask])
        amplitude = 10 * d_median / (l_anchor ** exp)  # so ref(l_anchor) == d_median

        y_ref = amplitude * l_ref ** exp

        ax.plot(l_ref, y_ref, lw=1.2, ls='--', color='gray', zorder=1)

        # Label near the peak (highest-y) end of the slope line
        ax.annotate(
            label,
            xy=(l_ref[0], y_ref[0]),
            xytext=(-4, 4),
            textcoords='offset points',
            color='dimgray',
            fontsize=13,
            ha='right',
            va='bottom',
        )

    # Draw a faint vertical line separating the two slope regimes
    if l_arr[-1] > 20:
        ax.axvline(x=20, color='gray', lw=0.8, ls=':', alpha=0.5, zorder=0)


def plot_spectrum(ds: xr.Dataset, var_name: str = None, target_height: float | None = None,
                  out_dir: str = ".", prefix: str = "spectrum"):
    """
    Plot spectral energy or spherical harmonic power spectrum.

    Produces a publication-quality log-log plot styled after spectra_base_figure, with:
      - Constrained layout
      - Degree ticks [1, 10, 100, 1000] via ScalarFormatter (minor ticks suppressed)
      - Y-tick labels left-aligned with padding (seba style)
      - AnchoredText box for the plot title in the upper-right corner
      - Dual x-axis: spherical harmonic degree (bottom), equivalent wavelength km (top)
      - Reference slope lines l⁻³ (synoptic, l ≤ 20) and l⁻⁵·³ (mesoscale, 20 < l ≤ 1000),
        anchored to the median of the plotted data in each range

    Args:
        ds: Input xarray Dataset containing spectral variables (dimension 'l' or 'wavenumber').
        var_name: Name of the variable to plot (optional; defaults to all spectral variables).
        target_height: Vertical level to select, in km (optional).
        out_dir: Output directory for the PNG file.
        prefix: Filename prefix for the output PNG.
    """
    from matplotlib.offsetbox import AnchoredText
    from matplotlib.ticker import ScalarFormatter, NullFormatter

    set_publication_style()

    # Find spectral variables
    if var_name:
        vars_to_plot = [var_name]
    else:
        vars_to_plot = [v for v in ds.data_vars if
                        'l' in ds[v].coords or 'wavenumber' in ds[v].coords]

    if not vars_to_plot:
        logger.error("No spectral variables found to plot.")
        return

    # --- Build figure with constrained_layout (seba style) ---
    fig, ax = plt.subplots(figsize=(6, 5.5), constrained_layout=True)

    ax.set_xscale('log')
    ax.set_yscale('log')

    # --- Degree-axis tick formatting: show [1, 10, 100, 1000], hide minor labels ---
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Collect all plotted (l, value) pairs for reference-slope anchoring
    all_l = []
    all_vals = []
    title_suffix = ""
    lmin_observed = None  # set from first variable
    lmax_observed = 0  # updated from every variable; stays at data truncation
    ydata_min = np.inf  # finite positive extremes across all variables
    ydata_max = -np.inf
    all_units = []          # units string per variable (in plot order)
    all_long_names = []     # long_name per variable

    for var in vars_to_plot:
        data = ds[var].squeeze()

        x_dim = 'l' if 'l' in data.dims else ('wavenumber' if 'wavenumber' in data.dims else None)
        if not x_dim:
            continue

        # Handle vertical level selection
        # _find_coordinate expects a Dataset; wrap the DataArray temporarily.
        # Scoping to *data* (not ds) prevents picking up vertical coords that
        # belong to other variables and are not dims of this one.
        level_coord = _find_coordinate(data.to_dataset(name=var), 'level', raise_notfound=False)
        if level_coord is not None:
            v_dim = level_coord.name
            is_meters = _coord_is_meter(level_coord)

            if target_height is not None and v_dim in data.dims:
                target_val = target_height * 1000.0 if is_meters else target_height
                logger.info(
                    f"Selecting {v_dim} closest to {target_height} km (target val: {target_val}).")
                data = data.sel({v_dim: target_val}, method='nearest')
            elif v_dim in data.dims:
                logger.info(f"Selecting first level for {v_dim}.")
                data = data.isel({v_dim: 0})

            if v_dim in data.coords:
                val = data[v_dim].item()
                val_km = val / 1000.0 if is_meters else val
                title_suffix = f" at ~{val_km:.1f} km"


        reduced_dims = [dim for dim in data.dims if dim != x_dim]
        if reduced_dims:
            logger.info(f"Selecting first index for extra dimensions: {reduced_dims}")
            data = data.isel({dim: 0 for dim in reduced_dims})

        # Look up a concise display name; fall back to the short_name stripped of '_cl'.
        # The SPECTRAL_KEYMAP is also used for the y-axis label (single-variable case).
        attrs = ds[var].attrs
        long_name = attrs.get('long_name', attrs.get('standard_name', var))
        units = attrs.get('units', '')
        short_name = var.replace('_cl', '')
        display_name = SPECTRAL_KEYMAP.get(var, short_name)

        all_long_names.append(long_name)
        all_units.append(units)

        # Keep only l > 0 (l=0 is the global mean)
        data_valid = data.sel({x_dim: data[x_dim] > 0})

        l_vals = data_valid[x_dim].values.astype(float)
        v_vals = data_valid.values.ravel().astype(float)

        ax.plot(l_vals, v_vals, lw=1.8, label=display_name)

        # Accumulate for slope anchoring (first variable drives the reference)
        if not len(all_l):
            all_l = l_vals
            all_vals = v_vals

        if lmin_observed is None:
            lmin_observed = int(l_vals[0])
        lmax_observed = max(lmax_observed, int(l_vals[-1]))

        # Track finite positive extremes for y-limit snapping
        finite_pos = v_vals[np.isfinite(v_vals) & (v_vals > 0)]
        if finite_pos.size:
            ydata_min = min(ydata_min, finite_pos.min())
            ydata_max = max(ydata_max, finite_pos.max())

    # --- x-axis limits and major ticks (seba style) ---
    if len(all_l) and lmax_observed > 0:
        l_min = max(1, lmin_observed if lmin_observed is not None else 1)
        l_max = lmax_observed + 50
        ax.set_xlim(l_min, l_max)

        # Tick positions derived from the actual data range – no hard-coded 1000
        candidate_ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
        x_ticks = candidate_ticks[(candidate_ticks >= l_min) & (candidate_ticks <= l_max)]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)

    # --- Reference slopes ---
    if len(all_l) > 0:
        _add_reference_slopes(ax, all_l, all_vals)

    # --- Y-limits: one decade above/below data extremes ---
    if np.isfinite(ydata_min) and np.isfinite(ydata_max) and ydata_min > 0:
        y_lo = 10 ** (np.floor(np.log10(ydata_min)) - 1)
        y_hi = 10 ** (np.ceil(np.log10(ydata_max)) + 1)
        ax.set_ylim(y_lo, y_hi)

    # --- Y-tick labels: left-aligned with extra pad (seba style) ---
    for tick_label in ax.yaxis.get_ticklabels():
        tick_label.set_horizontalalignment('left')
    ax.yaxis.set_tick_params(pad=30)

    # --- Y-label, title and legend (single vs. multi-variable) ---
    _fs = 12
    is_single = (len(vars_to_plot) == 1)

    if is_single:
        # Descriptive y-label: "<long_name> / <units in LaTeX>"
        _ln = all_long_names[0] if all_long_names else vars_to_plot[0]
        _u  = all_units[0] if all_units else ''
        y_label = f"{_ln} / {cf_to_latex(_u)}" if _u else _ln

        # AnchoredText: "$E_K$ at ~50.0 km" (math symbol + altitude)
        _solo_var = vars_to_plot[0]
        _solo_display = SPECTRAL_KEYMAP.get(_solo_var, _solo_var.replace('_cl', ''))
        if title_suffix:
            at = AnchoredText(
                f"{_solo_display} {title_suffix.strip()}", prop=dict(size=_fs - 1),
                frameon=True, loc='upper right',
            )
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
    else:
        # Generic y-label with the list of unique units
        unique_units = list(dict.fromkeys(u for u in all_units if u))
        units_str = ', '.join(cf_to_latex(u) for u in unique_units) if unique_units else ''
        y_label = f"Power spectra / {units_str}" if units_str else "Power spectra"

        # Legend with math-symbol labels; altitude as legend title
        legend_title = title_suffix.strip() if title_suffix else None
        ax.legend(fontsize=11, framealpha=0.9, title=legend_title,
                  title_fontsize=11)

    ax.set_xlabel("Spherical harmonic degree $l$", fontsize=_fs, labelpad=3)
    ax.set_ylabel(y_label, fontsize=_fs)

    # --- Secondary x-axis: wavelength in km (top, seba style) ---
    _R_KM = 6371.229

    def _l_to_wav(l):
        return (2 * np.pi * _R_KM) / np.maximum(l, 1e-10)

    def _wav_to_l(wav):
        return (2 * np.pi * _R_KM) / np.maximum(wav, 1e-10)

    secax = ax.secondary_xaxis('top', functions=(_l_to_wav, _wav_to_l))
    secax.xaxis.set_major_formatter(ScalarFormatter())
    secax.set_xlabel(r'wavelength / km', fontsize=_fs, labelpad=6)

    # Choose wavelength ticks that correspond to degrees within the plotted range
    wav_candidates = np.array([20000, 10000, 5000, 2000, 1000, 500, 250, 100, 50])
    if len(all_l):
        wav_at_lmin = _l_to_wav(all_l[0])
        wav_at_lmax = _l_to_wav(all_l[-1])
        wav_ticks = wav_candidates[
            (wav_candidates <= wav_at_lmin) & (wav_candidates >= wav_at_lmax)
            ]
        if len(wav_ticks):
            secax.set_xticks(wav_ticks)

    os.makedirs(out_dir, exist_ok=True)
    height_tag = f"_{target_height:.1f}km" if target_height is not None else ""
    out_path = os.path.join(out_dir, f"{prefix}{height_tag}_spectrum.png")

    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved spectrum plot to {out_path}")
    plt.close(fig)
