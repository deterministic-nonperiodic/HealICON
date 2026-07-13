import logging

import healpy as hp
import numpy as np
import xarray as xr

from .cf_coords import _find_coordinate

logger = logging.getLogger(__name__)


def parse_saber(ds: xr.Dataset, nside: int = None, ut_bins: int = None,
                order: str = 'ring') -> xr.Dataset:
    """
    Parse SABER satellite data and bin it to a HEALPix grid.
    The input dataset is expected to have dimensions (event, altitude)
    with tangent-point lat/lon coordinates.
    """
    if nside is None:
        nside = 32
        logger.info(f"No nside provided for SABER parsing. Defaulting to nside={nside}.")

    npix = hp.nside2npix(nside)
    cell_area = 4 * np.pi / npix

    # Ensure required coordinates exist
    lat_coord = _find_coordinate(ds, 'lat', raise_notfound=False)
    lon_coord = _find_coordinate(ds, 'lon', raise_notfound=False)
    level_coord = _find_coordinate(ds, 'level', raise_notfound=False)

    if lat_coord is None or lon_coord is None:
        raise ValueError("SABER parser requires latitude and longitude coordinates.")
    alt_name = None
    if level_coord is not None:
        alt_name = level_coord.name
    else:
        # Fallback: check if any of the dimensions look like a level coordinate
        for dim in ds.dims:
            dim_lower = str(dim).lower()
            if any(pat in dim_lower for pat in
                   ('altitude', 'height', 'z_mc', 'level', 'lev', 'plev')):
                alt_name = str(dim)
                break
        if alt_name is None:
            raise ValueError("SABER parser requires altitude/level coordinate or dimension.")

    # Convert coordinates to numpy arrays and handle missing values
    lat = lat_coord.values
    lon = lon_coord.values

    # NetCDF definition uses -999.f for missing values. Also check for NaN.
    missing_val = lat_coord.attrs.get('missing_value', -999.0)

    # Valid mask for coordinates
    valid_coords = (lat != missing_val) & (lon != missing_val) & (~np.isnan(lat)) & (~np.isnan(lon))

    # Convert to radians
    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)

    # Map to pixels. Invalid coordinates will be mapped to pixel 0 temporarily, but we'll mask them out.
    theta_safe = np.where(valid_coords, theta, 0.0)
    phi_safe = np.where(valid_coords, phi, 0.0)
    pix = hp.ang2pix(nside, theta_safe, phi_safe, nest=(order.lower() == 'nested'))

    n_alt = ds.sizes[alt_name]

    # We will build a new dataset with dimensions (altitude, cells)
    # Handle UT extraction and binning
    ut_indices = None
    if ut_bins is not None:
        if 'time' not in ds:
            raise ValueError("SABER parser requires 'time' to bin by UT.")
        ut_msec = ds['time'].values
        ut_missing = ds['time'].attrs.get('missing_value', missing_val)
        valid_coords = valid_coords & (ut_msec != ut_missing) & (~np.isnan(ut_msec))

        # Convert msec to hours
        ut_hours = ut_msec / 3600000.0
        ut_hours = ut_hours % 24.0

        # Calculate indices [0, ut_bins - 1]
        ut_indices_raw = np.floor(ut_hours / 24.0 * ut_bins).astype(int)
        ut_indices = np.clip(np.where(valid_coords, ut_indices_raw, 0), 0, ut_bins - 1)

        out_coords = {
            alt_name: level_coord.values if alt_name in ds.coords or alt_name in ds else np.arange(
                n_alt),
            'cells': np.arange(npix),
            'ut': np.linspace(0, 24, ut_bins, endpoint=False) + (12.0 / ut_bins)  # Bin centers
        }
    else:
        out_coords = {
            alt_name: level_coord.values if alt_name in ds.coords or alt_name in ds else np.arange(
                n_alt),
            'cells': np.arange(npix)
        }

    out_ds = xr.Dataset(coords=out_coords)
    if ut_bins is not None:
        out_ds['ut'].attrs = {"standard_name": "time", "long_name": "Universal Time",
                              "units": "hours"}

    # Helper to bin a single 2D array (event, altitude)
    def bin_var(data, missing, is_valid):
        # We process altitude by altitude to avoid huge memory spikes, though the dataset is relatively small.
        if ut_bins is None:
            out = np.full((n_alt, npix), np.nan, dtype=data.dtype)
        else:
            out = np.full((n_alt, npix, ut_bins), np.nan, dtype=data.dtype)

        for i in range(n_alt):
            d = data[:, i]
            p = pix[:, i]
            v = is_valid[:, i]

            var_valid = v & (d != missing) & (~np.isnan(d))
            if not np.any(var_valid):
                continue

            d_valid = d[var_valid]
            p_valid = p[var_valid]

            if ut_bins is None:
                sums = np.bincount(p_valid, weights=d_valid, minlength=npix)
                counts = np.bincount(p_valid, minlength=npix)
                mask = counts > 0
                out[i, mask] = sums[mask] / counts[mask]
            else:
                l_valid = ut_indices[:, i][var_valid]
                # Flatten spatial and UT indices: 1D index = p * ut_bins + l
                flat_idx = p_valid * ut_bins + l_valid
                total_bins = npix * ut_bins

                sums = np.bincount(flat_idx, weights=d_valid, minlength=total_bins)
                counts = np.bincount(flat_idx, minlength=total_bins)
                mask = counts > 0

                sums_valid = sums[mask] / counts[mask]

                # Assign back to 2D shape (npix, ut_bins)
                flat_out = np.full(total_bins, np.nan, dtype=data.dtype)
                flat_out[mask] = sums_valid
                out[i] = flat_out.reshape(npix, ut_bins)

        return out

    # Iterate over data variables
    for var in ds.data_vars:
        if var in ['tplatitude', 'tplongitude', 'orbit', 'date', 'time', 'tpaltitude',
                   'tpgpaltitude', 'tpSolarLT', 'tpSolarZen']:
            # We skip coordinate-like variables and metadata for the binned product,
            # unless we specifically want them. 
            # Often, we might want the binned time or solar zenith angle.
            # Let's bin pressure, density, temperature, and mixing ratios.
            # We can also bin tpSolarZen if useful.
            pass

        if 'event' in ds[var].dims and alt_name in ds[var].dims:
            logger.info(f"Binning SABER variable {var} to HEALPix...")
            d_val = ds[var].values
            m_val = ds[var].attrs.get('missing_value', missing_val)

            binned_data = bin_var(d_val, m_val, valid_coords)

            # Some variables might be integer (like time), convert out to float to support NaN
            dims_out = [alt_name, 'cells'] if ut_bins is None else [alt_name, 'cells', 'ut']
            out_ds[var] = xr.DataArray(
                binned_data,
                dims=dims_out,
                attrs=ds[var].attrs
            )
            # Ensure _FillValue / missing_value is consistent, but xarray uses NaN.
            if 'missing_value' in out_ds[var].attrs:
                del out_ds[var].attrs['missing_value']

    # Copy global attributes
    out_ds.attrs = ds.attrs.copy()

    # Add HEALPix metadata
    out_ds.attrs['healpix_nside'] = nside
    out_ds.attrs['healpix_npix'] = npix
    out_ds.attrs['healpix_scheme'] = order.upper()
    out_ds.attrs['healpix_cell_area_sr'] = f"{cell_area:.6e}"

    out_ds["healpix"] = xr.DataArray(
        np.int32(0),
        attrs={
            "grid_mapping_name": "healpix",
            "healpix_nside": np.int32(nside),
            "healpix_order": order.lower()
        }
    )

    for var in out_ds.data_vars:
        if var != 'healpix':
            out_ds[var].attrs['grid_mapping'] = 'healpix'

    # Add lon/lat coordinates for cells (optional but standard for HealICON)
    from .grid import get_healpix_coords
    target_lon, target_lat = get_healpix_coords(nside, nest=(order.lower() == 'nested'))
    out_ds.coords["lon"] = ("cells", target_lon)
    out_ds.coords["lat"] = ("cells", target_lat)
    out_ds.coords["lon"].attrs = {"standard_name": "longitude", "units": "degrees_east"}
    out_ds.coords["lat"].attrs = {"standard_name": "latitude", "units": "degrees_north"}

    history_msg = f"Parsed SABER Level2A dataset and binned to HEALPix grid (nside={nside}, {order.upper()}) using HealICON."
    out_ds.attrs['history'] = out_ds.attrs.get('history',
                                               '') + '\n' + history_msg if 'history' in out_ds.attrs else history_msg

    return out_ds
