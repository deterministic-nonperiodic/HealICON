"""Keogram (time x height) cross-section plots."""
import logging
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .common import VARIABLE_ATTRS, set_publication_style, wind_cm
from ..cf_coords import _find_coordinate, _coord_is_meter, _is_pressure_coord

logger = logging.getLogger(__name__)


def _resolve_alt_dim(da: xr.DataArray, time_dim: str, var: str) -> str | None:
    """Return the altitude / level dimension name in *da*."""
    coord = _find_coordinate(da.to_dataset(name=var), 'level', raise_notfound=False)
    if coord is not None and coord.name in da.dims:
        return coord.name
    for d in da.dims:
        dl = d.lower()
        if d == time_dim or 'lat' in dl or 'lon' in dl or 'cell' in dl:
            continue
        return d
    return None


def _prep_keogram_data(
        ds: xr.Dataset,
        var: str,
        time_dim: str,
        factor: float,
        lat: float | None = None,
        lon: float | None = None,
) -> tuple[xr.DataArray, str, bool]:
    """Prepare a (time, alt) DataArray for keogram plotting.

    For HEALPix datasets:
    - If *lat* and *lon* are both given, extracts a single grid point via
      :func:`~healicon.extract.extract_point` — no spatial averaging.
    - Otherwise computes a zonal mean then a cos-latitude-weighted average.

    The altitude coordinate is converted: m -> km, Pa -> hPa.

    Returns (da, alt_dim, is_pres).
    """
    from ..grid import get_cells_dim
    try:
        cell_dim = get_cells_dim(ds)
        if cell_dim in ds.dims:
            if lat is not None and lon is not None:
                from ..extract import extract_point
                logger.info(f"Extracting point lat={lat}, lon={lon} for keogram.")
                ds = extract_point(ds, lat=lat, lon=lon)
            else:
                logger.info("Unstructured grid detected: computing zonal mean for keogram.")
                from ..extract import zonal_mean
                ds = zonal_mean(ds)
    except Exception:
        pass

    da = ds[var].squeeze()
    alt_dim = _resolve_alt_dim(da, time_dim, var)
    if alt_dim is None:
        raise ValueError(
            f"Cannot detect altitude dimension for '{var}'. Dims: {list(da.dims)}"
        )

    extra = [d for d in da.dims if d not in (time_dim, alt_dim)]
    lat_dims = [d for d in extra if 'lat' in d.lower()]
    other_dims = [d for d in extra if d not in lat_dims]

    for d in lat_dims:
        w = np.cos(np.deg2rad(da[d]))
        da = da.weighted(w).mean(d)
    if other_dims:
        da = da.mean(other_dims)

    is_pres = False
    if alt_dim in da.coords:
        coord = da[alt_dim]
        if _coord_is_meter(coord):
            da = da.assign_coords({alt_dim: coord / 1000.0})
            da[alt_dim].attrs.update({'units': 'km', 'long_name': 'Altitude'})
        elif _is_pressure_coord(alt_dim, da.coords):
            is_pres = True
            units = str(coord.attrs.get('units', 'Pa')) or 'Pa'
            if units.lower() not in ('hpa', 'hectopascal', 'hectopascals',
                                     'mb', 'mbar', 'millibar', 'millibars'):
                from ..cf_coords import convert_units, equivalent_units
                if not equivalent_units(units, 'hPa'):
                    da = da.assign_coords(
                        {alt_dim: convert_units(coord, units, 'hPa')}
                    )
                    da[alt_dim].attrs.update({'units': 'hPa'})

    try:
        da = da.sortby(time_dim).sortby(alt_dim)
    except Exception:
        pass

    return da * factor, alt_dim, is_pres


def plot_keogram(
        datasets,
        variables,
        time_dim: str = 'time',
        lat: float | None = None,
        lon: float | None = None,
        y_limits=None,
        v_range=None,
        cmap=None,
        share_cbar: bool = True,
        location_label: str | None = None,
        start_label: str = 'a',
        out_dir: str = '.',
        prefix: str = 'keogram',
):
    """Plot time x height keograms for one or more variables.

    Parameters
    ----------
    datasets : xr.Dataset or dict[str, xr.Dataset]
        Input data. HEALPix datasets are reduced to (time, height) automatically.
        A bare Dataset is wrapped under the label ``'Data'``.
    variables : str or list of str
        Variable names. Each becomes one row of panels.
    time_dim : str
        Name of the time dimension.
    lat : float, optional
        Latitude of the extraction point (degrees). Requires *lon* as well.
        When both are given, a single HEALPix grid point is extracted instead
        of computing a spatial average.
    lon : float, optional
        Longitude of the extraction point (degrees). Requires *lat* as well.
    y_limits : [float, float], optional
        Vertical-axis limits in km (height) or hPa (pressure).
    v_range : [vmin, vmax, vstep] or dict[str, list], optional
        Colour-scale bounds. A dict maps variable names to individual bounds.
    cmap : colormap, optional
        Override the default per-variable colormap.
    share_cbar : bool
        Share one vertical colorbar per variable row (default ``True``).
    location_label : str, optional
        Explicit annotation in the upper-right corner of each panel.
        Auto-populated from *lat*/*lon* when those are given and this is ``None``.
    start_label : str
        First letter for panel annotations ``(a)``, ``(b)``, ... (default ``'a'``).
    out_dir : str
        Output directory.
    prefix : str
        Filename prefix.

    Returns
    -------
    str
        Absolute path of the saved PNG.
    """
    from matplotlib.offsetbox import AnchoredText

    set_publication_style()

    if isinstance(datasets, xr.Dataset):
        datasets = {'Data': datasets}
    variables = [variables] if isinstance(variables, str) else list(variables)

    n_vars = len(variables)
    n_cols = len(datasets)

    fw = 9.0 if n_cols == 1 else 6.2 * n_cols
    fig, axes = plt.subplots(
        nrows=n_vars, ncols=n_cols,
        figsize=(fw, 4.5 * n_vars),
        constrained_layout=True,
        sharex=True,
    )
    axes = np.atleast_2d(np.array(axes).reshape(n_vars, n_cols))

    _letters = 'abcdefghijklmnopqrstuvwxyz'
    start_idx = _letters.index(start_label) if start_label in _letters else 0
    panel_idx = 0

    for var_idx, var in enumerate(variables):
        attrs = VARIABLE_ATTRS.get(var, {
            'label': var, 'units': '', 'factor': 1.0,
            'v_range': [-10., 10., 2.], 'colormap': wind_cm,
        })
        colormap = cmap or attrs['colormap']

        if isinstance(v_range, dict):
            vr = v_range.get(var, attrs['v_range'])
        else:
            vr = v_range if v_range is not None else attrs['v_range']
        v_min, v_max, v_inc = float(vr[0]), float(vr[1]), float(vr[2])

        num_cn = max(50, int((v_max - v_min) / 1.25))
        cn_levels = np.linspace(v_min, v_max, num_cn)
        cbar_ticks = np.arange(v_min, v_max + v_inc * 0.5, v_inc)

        if var == 'temp':
            cc_levels = np.append(
                np.arange(v_min, 300., 20.),
                np.arange(300., 501., 50.),
            )
        else:
            cc_levels = np.arange(v_min, v_max + v_inc * 0.5, v_inc)

        cb_label = f"{attrs['label']} / {attrs['units']}"
        cn_last = None

        for col_idx, (label, ds) in enumerate(datasets.items()):
            ax = axes[var_idx, col_idx]

            if var not in ds:
                ax.set_visible(False)
                panel_idx += 1
                continue

            try:
                da, alt_dim, is_pres = _prep_keogram_data(
                    ds, var, time_dim, attrs['factor'], lat=lat, lon=lon
                )
            except Exception as exc:
                logger.warning(f"Skipping '{var}' in '{label}': {exc}")
                ax.set_visible(False)
                panel_idx += 1
                continue

            cn = da.plot.contourf(
                ax=ax, x=time_dim, y=alt_dim,
                levels=cn_levels, cmap=colormap, extend='both',
                add_colorbar=False, add_labels=False,
            )
            cn.axes.set_title("")
            try:
                for c in cn.collections:
                    c.set_edgecolor("face")
            except AttributeError:
                cn.set_edgecolor("face")

            cc = da.plot.contour(
                ax=ax, x=time_dim, y=alt_dim,
                levels=cc_levels, colors=['black'], linewidths=0.5,
                add_colorbar=False,
            )
            cc.axes.set_title("")
            ax.clabel(cc, inline=True, colors='white', fontsize=9, fmt='%1.0f')

            cn_last = cn

            if col_idx == 0:
                y_label = "Pressure / hPa" if is_pres else "Altitude / km"
                ax.set_ylabel(y_label, fontsize=11)
                if not is_pres:
                    ylim = y_limits or [
                        float(da[alt_dim].min()), float(da[alt_dim].max())
                    ]
                    y_span = ylim[1] - ylim[0]
                    y_step = 20 if y_span > 60 else 10 if y_span > 30 else 5
                    yticks = np.arange(ylim[0], ylim[1] + y_step, y_step)
                    ax.set_yticks(
                        yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])]
                    )
            else:
                ax.set_ylabel('')
                ax.set_yticks([])

            if y_limits:
                ax.set_ylim(*y_limits)
            if is_pres:
                ax.set_yscale('log')
                ax.invert_yaxis()

            ax.set_xlabel("")
            ax.xaxis.set_major_formatter(
                mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
            )
            if var_idx == n_vars - 1:
                plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=10)

            letter = _letters[min(start_idx + panel_idx, len(_letters) - 1)]
            ax.set_title(
                f'({letter}) {label.replace("_", " ").upper()}',
                fontsize=12, loc='left', fontweight='bold',
            )

            _loc = location_label
            if _loc is None and lat is not None and lon is not None:
                _loc = (f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, "
                        f"{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}")
            if _loc:
                at = AnchoredText(
                    _loc, loc='upper right',
                    prop=dict(size=9), frameon=True,
                )
                at.patch.set_boxstyle("round,pad=0.3")
                at.patch.set_alpha(0.6)
                ax.add_artist(at)

            panel_idx += 1

        if share_cbar and cn_last is not None:
            cb = fig.colorbar(
                cn_last, ax=axes[var_idx, :].tolist(),
                ticks=cbar_ticks, shrink=1.0, orientation='vertical',
                pad=0.02, extend='both',
            )
            cb.set_label(cb_label, fontsize=11, fontweight='bold')
            cb.ax.tick_params(which='minor', length=0)
            cb.ax.set_yticklabels(
                [str(int(np.round(t))) for t in cbar_ticks], fontsize=11
            )

    os.makedirs(out_dir, exist_ok=True)
    var_str = '_'.join(variables)
    loc_tag = f"_lat{int(lat)}_lon{int(lon)}" if (lat is not None and lon is not None) else ""
    out_path = os.path.join(out_dir, f"{prefix}_keogram_{var_str}{loc_tag}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved keogram to {out_path}")
    plt.close(fig)
    return out_path
