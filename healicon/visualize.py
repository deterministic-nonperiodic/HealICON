import logging
import os

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

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

    vert_dims = [d for d in ['height', 'z_mc', 'altitude', 'plev'] if
                 d in ds.coords or d in ds.dims]
    if not vert_dims:
        logger.error("Dataset missing height coordinate for tidal cross-section.")
        return
    level_name = vert_dims[0]

    if 'lat' not in ds.dims:
        # Raw HEALPix output — compute zonal mean on-the-fly.
        from .extract import zonal_mean
        logger.info("'lat' coordinate not found — computing zonal mean before plotting.")
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

            # Reconstruct phase from zonal-mean cos/sin
            for v, (cos_name, sin_name) in cos_sin_map.items():
                ds[v] = np.arctan2(ds[sin_name], ds[cos_name])
                ds = ds.drop_vars([cos_name, sin_name])


        except Exception as exc:
            logger.error(f"Automatic zonal mean failed: {exc}")
            return

    if 'lat' in ds.dims and level_name in ds.dims:
        ds = ds.sortby(['lat', level_name])


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
        if data.dims != (level_name, 'lat'):
            data = data.transpose(level_name, 'lat')

        if level_name in ['z_mc', 'height', 'altitude']:
            data = data.assign_coords({level_name: data[level_name] / 1000.0})
            data[level_name].attrs['units'] = 'km'
            if 'long_name' not in data[level_name].attrs:
                data[level_name].attrs['long_name'] = 'Height'

        cf = data.plot.contourf(
            ax=ax,
            x='lat',
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

    if level_name in ['z_mc', 'height', 'altitude']:
        axes[0].set_ylim(60, 110)
    elif level_name == 'plev':
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
        if data.dims != (level_name, 'lat'):
            data = data.transpose(level_name, 'lat')

        if level_name in ['z_mc', 'height', 'altitude']:
            data = data.assign_coords({level_name: data[level_name] / 1000.0})
            data[level_name].attrs['units'] = 'km'
            if 'long_name' not in data[level_name].attrs:
                data[level_name].attrs['long_name'] = 'Height'

        cf_pha = data.plot.contourf(
            ax=ax,
            x='lat',
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

    if level_name in ['z_mc', 'height', 'altitude']:
        axes_pha[0].set_ylim(60, 110)
    elif level_name == 'plev':
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

    if y_dim in ['z_mc', 'height', 'altitude']:
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

    if y_dim == 'plev':
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
    vert_dims = [d for d in ['height', 'z_mc', 'altitude', 'plev'] if
                 d in data.dims or d in ds.coords]
    if vert_dims:
        v_dim = vert_dims[0]
        max_val_global = ds[v_dim].max().item() if v_dim in ds else 1000
        is_meters = max_val_global > 500 and v_dim != 'plev'

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
        if 'lat' not in data.coords or 'lon' not in data.coords:
            logger.error("Dataset missing 'lat' or 'lon' coordinates for regular map.")
            return

        ax = plt.axes()
        # Use xarray's built-in contourf plotting
        cf = data.plot.contourf(
            ax=ax,
            x='lon',
            y='lat',
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


def plot_spectrum(ds: xr.Dataset, var_name: str = None, target_height: float | None = None,
                  out_dir: str = ".", prefix: str = "spectrum"):
    """
    Plot spectral energy or spherical harmonic power.
    """
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

    fig, ax = plt.subplots(figsize=(8, 6))

    for var in vars_to_plot:
        data = ds[var].squeeze()

        x_dim = 'l' if 'l' in data.dims else ('wavenumber' if 'wavenumber' in data.dims else None)
        if not x_dim:
            continue

        title_suffix = ""
        # Handle vertical level selection
        vert_dims = [d for d in ['height', 'z_mc', 'altitude', 'plev'] if
                     d in data.dims or d in ds.coords]
        if vert_dims:
            v_dim = vert_dims[0]
            max_val_global = ds[v_dim].max().item() if v_dim in ds else 1000
            is_meters = max_val_global > 500 and v_dim != 'plev'

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

        reduced_dims = [dim for dim in data.dims if dim != x_dim]
        if reduced_dims:
            logger.info(f"Selecting first index for extra dimensions: {reduced_dims}")
            data = data.isel({dim: 0 for dim in reduced_dims})

        # Get metadata
        var_base = var.replace('_cl', '')
        long_name = ds[var].attrs.get('long_name',
                                      ds.get(var_base, {}).attrs.get('long_name', var)) if hasattr(
            ds, var_base) else var
        units = ds[var].attrs.get('units', ds.get(var_base, {}).attrs.get('units', '')) if hasattr(
            ds, var_base) else ''
        label = f"{long_name} ({units})" if units else long_name

        # Slice data to only keep valid (positive) coordinates
        data_valid = data.sel({x_dim: data[x_dim] > 0})

        # Use xarray's built-in line plotting
        data_valid.plot.line(
            ax=ax,
            x=x_dim,
            xscale='log',
            yscale='log',
            label=label,
            linewidth=2,
            add_legend=False
        )

    ax.set_xlabel("Spherical Harmonic Degree ($l$)", fontsize=12)
    ax.set_ylabel("Power / Energy", fontsize=12)

    # Use title_suffix from the last variable (assuming they share the same grid)
    full_title = f"Spectral Distribution{title_suffix}" if 'title_suffix' in locals() else "Spectral Distribution"
    ax.set_title(full_title, fontweight='bold', fontsize=14)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=11)

    # Add secondary axis for wavelength
    def l_to_wav(l):
        l = np.maximum(l, 1e-10)
        return (2 * np.pi * 6371.229) / l

    def wav_to_l(wav):
        wav = np.maximum(wav, 1e-10)
        return (2 * np.pi * 6371.229) / wav

    secax = ax.secondary_xaxis('top', functions=(l_to_wav, wav_to_l))
    secax.set_xlabel('Equivalent Wavelength (km)', fontsize=12)
    secax.set_xticks([10000, 5000, 2000, 1000, 500, 250, 100])
    secax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{prefix}_spectrum.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved spectrum plot to {out_path}")
    plt.close(fig)
