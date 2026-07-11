import logging
import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from .common import wind_cm, temp_cm, set_publication_style, VARIABLE_ATTRS
from ..cf_coords import _find_coordinate, _coord_is_meter, _is_pressure_coord

logger = logging.getLogger(__name__)


def plot_section(
    ds: xr.Dataset,
    var_name: str,
    x_dim: str = 'lat',
    y_dim: str = 'z_mc',
    out_dir: str = ".",
    prefix: str = "section",
    v_range=None,
    y_limits=None,
    lon: float | None = None,
):
    """Plot a 2D cross-section (latitude x height or time x height).

    For HEALPix datasets (spatial dimension ``'cells'``):
    - If *lon* is given, extracts data along that meridian via
      :func:`~healicon.extract.extract_along_longitude`.
    - Otherwise computes a zonal mean.

    The *x_dim* / *y_dim* arguments are used as hints; CF coordinate detection
    is used as a fallback so that ``'altitude'``, ``'z'``, ``'height'``, ...
    are all resolved transparently.

    Parameters
    ----------
    ds : xr.Dataset
    var_name : str
    x_dim : str
        Hint for the horizontal dimension (default ``'lat'``).
    y_dim : str
        Hint for the vertical dimension (default ``'z_mc'``).
    out_dir : str
    prefix : str
    v_range : [vmin, vmax, vstep], optional
        Colour-scale bounds. Defaults to the entry in ``VARIABLE_ATTRS``.
    y_limits : [z_min, z_max], optional
        Vertical-axis limits in km (height) or hPa (pressure).
    lon : float, optional
        If given, extract the section at this longitude (degrees) instead of
        computing a zonal mean.

    Returns
    -------
    str or None
        Path of the saved PNG, or ``None`` if the variable was not found.
    """
    set_publication_style()

    if var_name not in ds:
        logger.error(f"Variable '{var_name}' not found in dataset.")
        return None

    data = ds[var_name].squeeze()

    # Step 1: HEALPix -> meridional slice or zonal mean
    from ..grid import get_cells_dim
    try:
        cell_dim = get_cells_dim(ds)
        if cell_dim in data.dims:
            if lon is not None:
                from ..extract import extract_along_longitude
                logger.info(f"Extracting section along lon={lon:.1f}.")
                ds_use = extract_along_longitude(ds, lon=lon)
                data = ds_use[var_name].squeeze()
            else:
                logger.info(
                    f"HEALPix dataset detected (dim='{cell_dim}'). "
                    "Computing zonal mean before section plot."
                )
                from ..extract import zonal_mean
                ds_zm = zonal_mean(ds)
                data = ds_zm[var_name].squeeze()
    except Exception:
        pass

    # Step 2: Resolve x/y dims via CF detection
    def _resolve_dim(hint: str, cf_type: str):
        if hint in data.dims and hint in data.coords:
            return hint
        ds_for_search = data.to_dataset(name=var_name)
        coord = _find_coordinate(ds_for_search, cf_type, raise_notfound=False)
        if coord is not None and coord.name in data.dims:
            logger.info(f"Resolved '{hint}' -> '{coord.name}' via CF detection.")
            return coord.name
        return None

    resolved_x = _resolve_dim(x_dim, 'lat')
    resolved_y = _resolve_dim(y_dim, 'level')

    if resolved_x is None or resolved_y is None:
        missing = []
        if resolved_x is None:
            missing.append(f"x='{x_dim}'")
        if resolved_y is None:
            missing.append(f"y='{y_dim}'")
        logger.error(
            f"Could not resolve dimension(s) {', '.join(missing)} for section plot. "
            f"Available dims: {list(data.dims)}, coords: {list(data.coords)}"
        )
        return None

    x_dim, y_dim = resolved_x, resolved_y

    # Step 3: Average over remaining dims
    reduced_dims = [d for d in data.dims if d not in (x_dim, y_dim)]
    if reduced_dims:
        logger.info(f"Averaging over additional dimensions: {reduced_dims}")
        data = data.mean(dim=reduced_dims)

    # Step 4: m -> km for height coords
    if y_dim in data.coords and _coord_is_meter(data[y_dim]):
        data = data.assign_coords({y_dim: data[y_dim] / 1000.0})
        data[y_dim].attrs.update({'units': 'km', 'long_name': 'Height'})

    try:
        data = data.sortby(x_dim).sortby(y_dim)
    except Exception:
        pass

    # Step 5: Variable display attributes
    default_cmap = temp_cm if 'temp' in var_name else wind_cm
    attrs = VARIABLE_ATTRS.get(var_name, {
        'label': data.attrs.get('long_name', var_name),
        'units': data.attrs.get('units', ''),
        'factor': 1.0,
        'v_range': [
            float(data.min()), float(data.max()),
            (float(data.max()) - float(data.min())) / 8.0,
        ],
        'colormap': default_cmap,
    })

    data = data * attrs['factor']

    vr = v_range if v_range is not None else attrs['v_range']
    v_min, v_max, v_inc = float(vr[0]), float(vr[1]), float(vr[2])

    num_cn = max(50, int((v_max - v_min) / 1.25))
    cn_levels = np.linspace(v_min, v_max, num_cn)
    cbar_ticks = np.arange(v_min, v_max + v_inc * 0.5, v_inc)

    if var_name == 'temp':
        cc_levels = np.append(np.arange(v_min, 300., 20.), np.arange(300., 501., 50.))
    else:
        cc_levels = np.arange(v_min, v_max + v_inc * 0.5, v_inc)

    # Step 6: Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    cn = data.plot.contourf(
        ax=ax, x=x_dim, y=y_dim,
        levels=cn_levels, cmap=attrs['colormap'], extend='both',
        add_colorbar=False, add_labels=False,
    )
    cn.axes.set_title("")
    try:
        for c in cn.collections:
            c.set_edgecolor("face")
    except AttributeError:
        cn.set_edgecolor("face")

    cc = data.plot.contour(
        ax=ax, x=x_dim, y=y_dim,
        levels=cc_levels, colors=['black'], linewidths=0.5,
        add_colorbar=False,
    )
    cc.axes.set_title("")
    ax.clabel(cc, inline=True, colors='white', fontsize=9, fmt='%1.0f')

    cb = fig.colorbar(
        cn, ax=ax, ticks=cbar_ticks, orientation='vertical',
        pad=0.02, extend='both', shrink=0.92,
    )
    cb.set_label(
        f"{attrs['label']} / {attrs['units']}", fontsize=11, fontweight='bold'
    )
    cb.ax.tick_params(which='minor', length=0)
    cb.ax.set_yticklabels(
        [str(int(np.round(t))) for t in cbar_ticks], fontsize=10
    )

    # x-axis
    if x_dim == 'lat':
        ax.set_xlim(-90, 90)
        ax.set_xticks([-60, -30, 0, 30, 60])
        ax.set_xticklabels(['60S', '30S', '0', '30N', '60N'])
        ax.set_xlabel('Latitude', fontsize=11)
    elif x_dim == 'time':
        ax.set_xlabel("")
        ax.xaxis.set_major_formatter(
            mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # y-axis
    is_pres = _is_pressure_coord(y_dim, data.coords)
    if is_pres:
        ax.set_yscale('log')
        ax.invert_yaxis()
        ax.set_ylabel('Pressure / hPa', fontsize=11)
    else:
        if y_limits is not None:
            ax.set_ylim(*y_limits)
            ylim = list(y_limits)
        else:
            ylim = [float(data[y_dim].min()), float(data[y_dim].max())]
        y_span = ylim[1] - ylim[0]
        y_step = 20 if y_span > 60 else 10 if y_span > 30 else 5
        yticks = np.arange(ylim[0], ylim[1] + y_step, y_step)
        ax.set_yticks(yticks[(yticks >= ylim[0]) & (yticks <= ylim[1])])
        ax.set_ylabel('Altitude / km', fontsize=11)

    long_name = data.attrs.get('long_name', var_name)
    lon_suffix = f" (lon={lon:.1f}°)" if lon is not None else ""
    ax.set_title(f"{long_name} Cross-Section{lon_suffix}", fontweight='bold')

    os.makedirs(out_dir, exist_ok=True)
    lon_tag = f"_lon{int(lon)}" if lon is not None else ""
    out_path = os.path.join(out_dir, f"{prefix}_section_{var_name}{lon_tag}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved cross-section plot to {out_path}")
    plt.close(fig)
    return out_path
