import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .common import set_publication_style
from ..cf_coords import _find_coordinate, _coord_is_meter

logger = logging.getLogger(__name__)


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
        from ..extract import zonal_mean
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

    # ── Detect decomposition mode ─────────────────────────────────────────
    amp_sym_vars = [v for v in ds.data_vars if v.endswith('_amp_sym')]
    amp_total_vars = [v for v in ds.data_vars if v.endswith('_amp_total')]

    if amp_sym_vars:
        # Standard sym/asy output
        decompose_sym_asy = True
        var_base = amp_sym_vars[0][:-8]  # strip '_amp_sym'
    elif amp_total_vars:
        # --no-sym-asy output
        decompose_sym_asy = False
        var_base = amp_total_vars[0][:-10]  # strip '_amp_total'
    else:
        logger.error("No tidal amplitude variables (*_amp_sym or *_amp_total) found in dataset.")
        return

    # ── Determine amp/pha suffixes ────────────────────────────────────────
    # In sym/asy mode we keep the symmetric and antisymmetric panels separate.
    # In total mode every mode maps to the single 'total' suffix.
    def _amp_var(sa: str) -> str:
        return f'{var_base}_amp_{sa}'

    def _pha_var(sa: str) -> str:
        return f'{var_base}_pha_{sa}'

    p_12 = np.timedelta64(12, 'h')
    p_24 = np.timedelta64(24, 'h')

    if decompose_sym_asy:
        modes = {
            'DW1': {'period': p_24, 'm': 1, 'sa': 'sym'},
            'DW1_asy': {'period': p_24, 'm': 1, 'sa': 'asy'},
            'SW2': {'period': p_12, 'm': 2, 'sa': 'sym'},
            'SW2_asy': {'period': p_12, 'm': 2, 'sa': 'asy'},
            'SE2': {'period': p_12, 'm': -2, 'sa': 'sym'},
            'SE2_asy': {'period': p_12, 'm': -2, 'sa': 'asy'},
            'DE3': {'period': p_24, 'm': -3, 'sa': 'sym'},
            'DE3_asy': {'period': p_24, 'm': -3, 'sa': 'asy'},
        }
    else:
        # Total mode: one panel per mode name, no sym/asy split
        modes = {
            'DW1': {'period': p_24, 'm': 1, 'sa': 'total'},
            'SW2': {'period': p_12, 'm': 2, 'sa': 'total'},
            'SE2': {'period': p_12, 'm': -2, 'sa': 'total'},
            'DE3': {'period': p_24, 'm': -3, 'sa': 'total'},
        }

    # Identify available modes in the dataset
    available_modes = {}
    for name, meta in modes.items():
        av = _amp_var(meta['sa'])
        if av in ds and 'period' in ds.coords and 'm' in ds.coords:
            try:
                ds[av].sel(period=meta['period'], m=meta['m'], method='nearest')
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
    amp_data_vars = [v for v in ds.data_vars if '_amp_' in v
                     and not any(d in v for d in ('_cos_', '_sin_'))]
    try:
        max_amp = float(ds[amp_data_vars].to_array().max()) if amp_data_vars else 0.0
        if np.isfinite(max_amp) and max_amp > 0:
            vmax = min(max_amp,
                       max_amplitude if max_amplitude is not None else 20.0)
    except Exception:
        pass

    levels = np.linspace(0, vmax, 13)

    cf = None
    for i, (name, meta) in enumerate(available_modes.items()):
        ax = axes[i]
        data = ds[_amp_var(meta['sa'])].sel(
            period=meta['period'], m=meta['m'], method='nearest')

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
        label_suffix = {'sym': 'Symmetric', 'asy': 'Antisymmetric', 'total': 'Total'}.get(
            meta['sa'], meta['sa'].title())
        ax.set_title(f"{name.split('_')[0]} ({label_suffix})", fontweight='bold')
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
        amp_ref = _amp_var(next(iter(available_modes.values()))['sa'])
        var_units = ds[amp_ref].attrs.get('units', 'K')
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
        data = ds[_pha_var(meta['sa'])].sel(
            period=meta['period'], m=meta['m'], method='nearest')

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
        label_suffix = {'sym': 'Symmetric', 'asy': 'Antisymmetric', 'total': 'Total'}.get(
            meta['sa'], meta['sa'].title())
        ax.set_title(f"{name.split('_')[0]} Phase ({label_suffix})", fontweight='bold')
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
        pha_ref = _pha_var(next(iter(available_modes.values()))['sa'])
        var_units_pha = ds[pha_ref].attrs.get('units', 'rad')
        cbar_pha.set_label(f'Phase / {var_units_pha}')

    pha_out_path = os.path.join(out_dir, f"{prefix}_phase.png")
    plt.savefig(pha_out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved phase plot to {pha_out_path}")
    plt.close(fig_pha)
