import logging
import os

import matplotlib.pyplot as plt
import xarray as xr

from .common import wind_cm, temp_cm, set_publication_style
from ..cf_coords import _find_coordinate, _coord_is_meter

logger = logging.getLogger(__name__)


def plot_section(ds: xr.Dataset, var_name: str, x_dim: str = 'lat', y_dim: str = 'z_mc',
                 out_dir: str = ".", prefix: str = "section"):
    """
    Plot a 2D cross-section (e.g. Latitude vs Height, or Time vs Height).

    For HEALPix datasets (spatial dimension 'cells') a zonal mean is computed
    automatically so that a proper latitude axis is available.  The x_dim /
    y_dim arguments are treated as *hints*; if the exact name is absent the
    function falls back to CF coordinate detection so that names like
    'altitude', 'z', 'height', … are all handled transparently.
    """
    set_publication_style()

    if var_name not in ds:
        logger.error(f"Variable '{var_name}' not found in dataset.")
        return

    data = ds[var_name]

    # Squeeze out singleton dimensions
    data = data.squeeze()

    # ------------------------------------------------------------------
    # Step 1: If this is a HEALPix dataset, compute a zonal mean first
    # so we get a proper lat dimension to plot against.
    # ------------------------------------------------------------------
    from ..grid import get_cells_dim
    try:
        cell_dim = get_cells_dim(ds)
        if cell_dim in data.dims:
            logger.info(
                f"HEALPix dataset detected (dim='{cell_dim}'). "
                "Computing zonal mean before section plot."
            )
            from ..extract import zonal_mean
            # Pass the full dataset so that all dimension coordinates
            # (e.g. altitude) are present in the output and get properly
            # attached to the target DataArray.  Slice to the target
            # variable afterward to keep memory usage small.
            ds_zm = zonal_mean(ds)
            data = ds_zm[var_name].squeeze()
    except (ValueError, Exception):
        pass  # not HEALPix — proceed with original data

    # ------------------------------------------------------------------
    # Step 2: Resolve the x / y dimension names via CF conventions.
    # The user-supplied names (default 'lat', 'z_mc') are used as
    # hints; if they are not present we try CF detection.
    # ------------------------------------------------------------------
    def _resolve_dim(hint: str, cf_type: str):
        """Return the actual dim name for *hint*, falling back to CF detection."""
        # Exact match first (must be both a coord and a dimension)
        if hint in data.dims and hint in data.coords:
            return hint
        # Try CF coordinate detection on the current data slice
        ds_for_search = data.to_dataset(name=var_name)
        coord = _find_coordinate(ds_for_search, cf_type, raise_notfound=False)
        if coord is not None and coord.name in data.dims:
            logger.info(
                f"Resolved '{hint}' \u2192 '{coord.name}' via CF detection."
            )
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
        return

    x_dim, y_dim = resolved_x, resolved_y

    # Average over remaining dimensions
    reduced_dims = [dim for dim in data.dims if dim not in [x_dim, y_dim]]
    if reduced_dims:
        logger.info(f"Averaging over additional dimensions: {reduced_dims}")
        data = data.mean(dim=reduced_dims)

    if y_dim in data.coords and _coord_is_meter(data[y_dim]):
        data = data.assign_coords({y_dim: data[y_dim] / 1000.0})
        data[y_dim].attrs['units'] = 'km'
        if 'long_name' not in data[y_dim].attrs:
            data[y_dim].attrs['long_name'] = 'Height'

    # Ensure both axes are monotonically increasing so that contourf
    # renders correctly (ICON stores altitude top-down, i.e. descending).
    try:
        data = data.sortby(x_dim).sortby(y_dim)
    except Exception:
        pass  # non-fatal — proceed with original ordering

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
    if y_dim in data.coords:
        units = str(data[y_dim].attrs.get('units', '')).strip().lower()
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
