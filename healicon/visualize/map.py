import logging
import os

import healpy as hp
import matplotlib.pyplot as plt
import xarray as xr

from .common import wind_cm, temp_cm, set_publication_style
from ..cf_coords import _find_coordinate, _coord_is_meter

logger = logging.getLogger(__name__)


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
    from ..grid import get_cells_dim
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
        from ..grid import get_healpix_order
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
