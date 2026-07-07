import re
import warnings
from typing import Tuple, Union, Dict, Any, List

import numpy as np
import xarray as xr
import logging
import pint

logger = logging.getLogger(__name__)

# --- Public API for external access ---
__all__: List[str] = [
    "_cf_guess",
    "_coord_is_degrees",
    "_is_geographic",
    "_is_z",
    "_coord_is_meter",
    "is_geographic_grid",
    "get_spatial_dims",
    "check_convert_units",
    "convert_units",
    "get_conversion_components",
    "equivalent_units",
    "compatible_units",
]

# -----------------------------
# --- CF Convention Lookups ---
# -----------------------------
_CF_COORDS_LOOKUP: Dict[str, Dict[str, Any]] = {
    "lon": {
        "names": ("lon", "long", "longitude", "clon"),
        "units_hints": ("east", "degree", "degrees", "deg", "degree_east", "rad", "radian"),
        "standard_name": "longitude",
        "axis": "X",
    },
    "lat": {
        "names": ("lat", "latitude", "clat"),
        "units_hints": ("north", "degree", "degrees", "deg", "degree_north", "rad", "radian"),
        "standard_name": "latitude",
        "axis": "Y",
    },
    "level": {
        "standard_name": {
            'altitude', 'height', 'depth', 'geopotential_height',
            'height_above_geopotential_datum',
            'height_above_mean_sea_level',
            'height_above_reference_ellipsoid',
            'atmosphere_hybrid_height_coordinate',
            'atmosphere_sigma_coordinate',
            'atmosphere_sleve_coordinate'
        },
        "units": ('meter', 'm', 'gpm', 'Pa', 'hPa', 'mb', 'millibar', '~'),
        "axis": ('Z', 'vertical')
    }
}

_CF_VARS_LOOKUP: Dict[str, Dict[str, Any]] = {
    "u": {"standard_names": {"eastward_wind"}, "units": {"m s-1", "m/s"}},
    "v": {"standard_names": {"northward_wind"}, "units": {"m s-1", "m/s"}},
    "w": {"standard_names": {"upward_air_velocity", "vertical_velocity_in_air"},
          "units": {"m s-1", "Pa s-1"}},
    "pressure": {"standard_names": {"air_pressure"}, "units": {"Pa", "pascal"}},
    "temperature": {"standard_names": {"air_temperature"}, "units": {"K", "kelvin"}},
    "density": {"standard_names": {"air_density"}, "units": {"kg / m**3", "kg m-3"}},
    "theta": {"standard_names": {"air_potential_temperature"}, "units": {"K", "kelvin"}},
    "divergence": {"standard_names": {"divergence_of_wind"}, "units": {"s-1"}},
    "vorticity": {"standard_names": {"relative_vorticity"}, "units": {"s-1"}},
}

ALLOWED_UNITS: List[str] = ["deg", "degrees", "degrees_north", "degrees_east",
                            "m", "meters", "km", "kilometers"]

_METER_UNITS: set = {"m", "meter", "meters", "metre", "metres"}

# --------------------------
# Unit and coordinate checks
# --------------------------
expected_units = {
    "u": "m/s",
    "v": "m/s",
    "w": "m/s",
    "divergence": "1/s",
    "vorticity": "1/s",
    "temperature": "K",
    "temp": "K",
    "theta": "K",
    "pressure": "Pa",
    "pres": "Pa",
    "ps": "Pa",
    "ts": "K",
    "level": "Pa",
    "omega": "Pa/s",
    "geopotential": "m**2 s**-2",
    "u_wind": "m/s",
    "v_wind": "m/s",
    "w_wind": "m/s",
    'lat': 'degrees_north',
    'lon': 'degrees_east',
}

expected_range = {
    "u": [-350., 350.],
    "v": [-350., 350.],
    "w": [-100., 100.],
    "temp": [120, 2500.],    # up to ~1500 K in active thermosphere
    "theta": [120, 1e8],     # no practical upper bound in the MLT/thermosphere
    "pres": [0.0, 2000e2],
    "u_wind": [-350., 350.],
    "v_wind": [-350., 350.],
    "w_wind": [-100., 100.],
    "divergence": [-10., 10.],
    "vorticity": [-10., 10.],
    "temperature": [120, 2500.],
    "pressure": [0.0, 2000e2],
    "ts": [120, 350.],
    "ps": [0.0, 2000e2],
    "omega": [-100, 100],
    "geopotential": [0., 1e8],
}

# --------------------------
# Unit Registry & Converters
# --------------------------
UNITS_REG = pint.UnitRegistry()
_unit_cmd = re.compile(r"(?<=[A-Za-z)])(?![A-Za-z)])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])")

def _parse_units(unit_str):
    if isinstance(unit_str, (pint.Quantity, pint.Unit)):
        return unit_str
    else:
        unit_str = unit_str.replace('degrees_east', 'degree').replace('degrees_north', 'degree')
        unit_str = unit_str.replace('degree_east', 'degree').replace('degree_north', 'degree')
        unit_str = unit_str.replace('degrees_E', 'degree').replace('degrees_N', 'degree')
        return UNITS_REG(_unit_cmd.sub('**', unit_str))

def equivalent_units(unit_1, unit_2):
    ratio = (_parse_units(unit_1) / _parse_units(unit_2)).to_base_units()
    return ratio.dimensionless and np.isclose(ratio.magnitude, 1.0)

def compatible_units(unit_1, unit_2):
    return _parse_units(unit_1).is_compatible_with(_parse_units(unit_2))

def _get_units_str(da: xr.DataArray) -> str:
    """Extract and normalise the units string from a DataArray (empty string if absent)."""
    return str(da.attrs.get("units", "")).strip()

def get_conversion_components(from_units: str, to_units: str) -> tuple[float, float]:
    """Affine conversion coefficients for Y = X · multiplier + offset.

    Derived by querying pint with the values 0 and 1 of *from_units*,
    so offset units (e.g. °C → K, offset = 273.15) are handled correctly.
    Raises ``pint.DimensionalityError`` for incompatible unit pairs.
    """
    q0 = pint.Quantity(0.0, _parse_units(from_units))
    offset = float(q0.to(_parse_units(to_units)).magnitude)
    q1 = pint.Quantity(1.0, _parse_units(from_units))
    multiplier = float(q1.to(_parse_units(to_units)).magnitude) - offset
    return multiplier, offset

def convert_units(da: xr.DataArray, from_units: str, to_units: str) -> xr.DataArray:
    """Apply affine unit conversion Y = X · m + b, preserving Dask chunks.

    The units attribute on *da* takes precedence over *from_units*; use
    *from_units* only when the attribute is absent.  Raises ``ValueError``
    for incompatible unit pairs (e.g. Pa → m).
    """
    source_units = _get_units_str(da) or from_units
    if equivalent_units(source_units, to_units):
        return da
    if compatible_units(source_units, to_units):
        multiplier, offset = get_conversion_components(source_units, to_units)
        new_da = da.copy(data=(multiplier * da + offset).data)
        new_da.attrs.update({"units": to_units})
        return new_da
    raise ValueError(
        f"Cannot convert '{da.name or 'variable'}': "
        f"incompatible units '{source_units}' → '{to_units}'."
    )

def check_convert_units(dataset_or_array: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Validate and convert all recognised variables to canonical SI units.

    For each variable whose name appears in ``expected_units``:
    - If units are present and compatible, apply :func:`convert_units`.
    - If units are absent, infer them from ``expected_range`` (range check)
      and assign the expected unit before converting.

    Raises ``ValueError`` when units are present but incompatible, or when
    missing units cannot be resolved from the value range.
    """
    is_array = isinstance(dataset_or_array, xr.DataArray)
    if isinstance(dataset_or_array, xr.Dataset):
        ds = dataset_or_array.copy()
    elif is_array:
        ds = dataset_or_array.to_dataset(
            name=dataset_or_array.name or '_tmp', promote_attrs=True)
    else:
        raise ValueError("Illegal type for parameter 'dataset_or_array'")

    for varname in list(ds.data_vars.keys()) + list(ds.coords.keys()):
        if varname not in expected_units:
            continue
        var = ds[varname]
        var_units = _get_units_str(var)
        expected_unit = expected_units[varname]

        if not var_units:
            # Range-based unit inference: assign expected unit if values are in bounds.
            var_min, var_max = float(var.min()), float(var.max())
            r_min, r_max = expected_range.get(varname, [-np.inf, np.inf])
            if r_min <= var_min and var_max <= r_max:
                logger.info(
                    f"Variable '{varname}' missing units; assigned '{expected_unit}' "
                    f"based on value range [{var_min:.4g}, {var_max:.4g}]."
                )
                ds[varname].attrs["units"] = expected_unit
                var_units = expected_unit
            else:
                raise ValueError(
                    f"Variable '{varname}' has no units and values "
                    f"[{var_min:.4g}, {var_max:.4g}] are outside the admitted range "
                    f"{expected_range.get(varname)}."
                )

        if not equivalent_units(var_units, expected_unit):
            logger.info(f"Converting '{varname}' units: '{var_units}' → '{expected_unit}'.")
            ds[varname] = convert_units(var, var_units, expected_unit)

    if is_array:
        return ds.to_array().squeeze('variable').drop_vars('variable')
    return ds


# ----------------------
# CF-based var guessing
# ----------------------
def _cf_guess(ds: xr.Dataset, target: str) -> str | None:
    """
    Very light CF-based guess for a logical variable name.

    Looks at ``standard_name`` first (reliable), then falls back to common
    units only when a ``standard_name`` is absent.  This avoids false positives
    where e.g. temperature and theta share the same unit (K).
    """
    rule = _CF_VARS_LOOKUP.get(target)
    if rule is None:
        return None

    # Pass 1: standard_name match (most reliable)
    for name, da in ds.data_vars.items():
        std = str(da.attrs.get("standard_name", "")).strip()
        if std and std in rule["standard_names"]:
            return name

    # Pass 2: units-only match — but only for variables that have NO
    # standard_name, to avoid misidentifying a variable whose standard_name
    # belongs to a *different* physical quantity.
    # Additionally, check long_name to break ties: if long_name clearly
    # refers to a different physical quantity, skip the match.
    _CONFLICTING_LONG_NAMES: dict[str, set[str]] = {
        'theta': {'temperature', 'temp'},       # if long_name says "temperature", it's not theta
        'temperature': {'potential temperature', 'theta'},  # vice versa
    }
    conflicts = _CONFLICTING_LONG_NAMES.get(target, set())
    for name, da in ds.data_vars.items():
        std = str(da.attrs.get("standard_name", "")).strip()
        if std:
            continue
        units = str(da.attrs.get("units", "")).strip()
        if units and any(u == units for u in rule["units"]):
            long_name = str(da.attrs.get("long_name", "")).lower()
            if long_name and any(c in long_name for c in conflicts):
                continue
            return name

    return None


def _find_coordinate(ds: xr.Dataset, name: str,
                     raise_notfound: bool = True,
                     check_duplicates: bool = False) -> xr.DataArray | None:
    """Find a coordinate in *ds* by type name ('lat', 'lon', 'level').

    Search order: exact name match, then CF standard_name / axis / units.
    Returns None when *raise_notfound* is False and nothing is found.
    """
    # Try exact name match first
    if name in ds.variables:
        return ds[name]

    if name not in _CF_COORDS_LOOKUP:
        raise ValueError(
            f"Unknown coordinate type: {name}. Must be one of {list(_CF_COORDS_LOOKUP.keys())}")

    criteria = _CF_COORDS_LOOKUP[name]

    # Well-known coordinate names for the 'level' (vertical) type.
    # Used as a last-resort name-pattern fallback when CF attributes are
    # absent or incomplete (e.g. after a NetCDF round-trip strips axis/units).
    _LEVEL_NAME_PATTERNS = (
        'altitude', 'alt', 'height', 'depth', 'z_mc', 'z_ifc', 'zlev',
        'lev', 'level', 'z', 'plev', 'pressure',
    )

    # Build predicate function based on available criteria
    def matches_criteria(c: xr.DataArray) -> bool:
        if name in ('lat', 'lon'):
            return _is_geographic(c, name)

        # Check name
        if 'names' in criteria and c.name in criteria['names']:
            return True

        # Check standard_name attribute
        # NOTE: criteria['standard_name'] may be a str, tuple, or set.
        if 'standard_name' in criteria:
            std_name = c.attrs.get('standard_name', '').strip().lower()
            if std_name:
                expected = criteria['standard_name']
                if isinstance(expected, str):
                    if std_name == expected:
                        return True
                elif isinstance(expected, (tuple, set, frozenset)):
                    if std_name in expected:
                        return True

        # Check axis attribute
        if 'axis' in criteria:
            axis = c.attrs.get('axis', '').strip().upper()
            if axis:
                expected = criteria['axis']
                if isinstance(expected, str):
                    if axis == expected:
                        return True
                elif isinstance(expected, (tuple, set, frozenset)):
                    if axis in expected:
                        return True

        # Check units hints
        if 'units_hints' in criteria:
            units = c.attrs.get('units', '').strip().lower()
            if units and any(hint in units for hint in criteria['units_hints']):
                return True

        # Check units (for level coordinate)
        if 'units' in criteria:
            units = c.attrs.get('units', '').strip().lower()
            expected = criteria['units']
            if isinstance(expected, (tuple, set, frozenset)):
                if units in expected:
                    return True

        # Last resort for 'level': match by well-known coordinate name patterns.
        # This handles cases where CF attributes were stripped during I/O.
        if name == 'level':
            cname_lower = (c.name or '').lower()
            if any(pat == cname_lower or cname_lower.startswith(pat)
                   for pat in _LEVEL_NAME_PATTERNS):
                # Only return True when the coordinate is actually a dimension
                # of size > 1 (avoids misidentifying scalar variables).
                if c.ndim >= 1:
                    return True

        return False

    # Search through all variables, not just coordinates
    candidates = [ds[var_name] for var_name in ds.variables if matches_criteria(ds[var_name])]

    if check_duplicates and len(candidates) > 1:
        raise ValueError(f"Multiple {name} coordinates found: {[c.name for c in candidates]}")

    if not candidates:
        if raise_notfound:
            raise ValueError(
                f"The coordinate '{name}' is not in the dataset or is "
                f"inconsistent with CF conventions. Available dims: {list(ds.dims)}"
            )
        return None

    return candidates[0]


def _get_units_str(c: xr.DataArray) -> str:
    """Extracts and normalizes the units string from a DataArray."""
    units = c.attrs.get("units", "").strip()
    return units


# ----------------------
# Compact CF-aware utils
# ----------------------
def _infer_coordinate_units(coord: xr.DataArray, name: str) -> str:
    """Infers and validates coordinate units against ALLOWED_UNITS."""
    units = coord.attrs.get("units", "").lower()
    if not units:
        raise ValueError(f"Missing 'units' attribute for {name} coordinate.")
    if units not in ALLOWED_UNITS:
        raise ValueError(f"Invalid units for {name}: '{units}'. Allowed: {ALLOWED_UNITS}")
    return units


def _coord_is_degrees(
        coord: xr.DataArray,
        allow_infer: bool = True,
        tol: float = 1e-12,
) -> bool:
    """
    True if `coord` uses degrees (CF-compliant).

    If units are absent/ambiguous, and we need to infer, treat as degrees
    when |values| exceed 2π (cannot be radians).
    """
    units = _get_units_str(coord)

    # Explicit units
    if "radian" in units:
        return False
    if units == "deg" or units.startswith("degree") or units.startswith("degrees"):
        return True

    # Heuristic inference when units missing/unknown
    if allow_infer:
        vals = np.asarray(coord.values)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            if float(np.nanmax(np.abs(vals))) > (2.0 * np.pi + tol):
                return True

    return False


def _coord_is_meter(c: xr.DataArray) -> bool:
    """Checks if the coordinate units are meter-like."""
    u = _get_units_str(c)
    return (u in _METER_UNITS) or any(tok in u for tok in ("metre", "meter"))


def _is_z(cname: str, coords: Union[xr.Dataset, xr.DataArray, Any]) -> bool:
    """True if *cname* is a height-in-metres vertical coordinate.

    Returns False for pressure/isobaric coordinates.  Detection order:
    1. axis='Z' + metre units  (most reliable)
    2. CF ``standard_name`` in the altitude/height set
    3. Name pattern (height, altitude, z_*) + metre units

    Parameters
    ----------
    cname : str
        Coordinate name to check.
    coords : Dataset, DataArray, or coordinate dict
        Container that holds the coordinate.
    """
    if cname not in coords:
        return False

    coord = coords[cname]
    name = cname.lower()
    units = _get_units_str(coord).lower()
    standard_name = (coord.attrs.get("standard_name", "") or "").strip().lower()
    axis = (coord.attrs.get("axis", "") or "").strip().upper()

    # Exclude pressure coordinates explicitly
    pressure_units = ('pa', 'hpa', 'mb', 'millibar', 'bar')
    pressure_names = ('plev', 'pressure', 'pres', 'isobaric')
    pressure_std_names = ('air_pressure', 'atmosphere_ln_pressure_coordinate')

    # If it's clearly a pressure coordinate, return False
    if any(unit in units for unit in pressure_units):
        return False
    if any(pname in name for pname in pressure_names):
        return False
    if standard_name in pressure_std_names:
        return False

    # Check for height-based coordinates
    # Check axis='Z' with meter units (most reliable)
    meter_units = ('m', 'meter', 'meters', 'metre', 'metres', 'gpm')
    if axis == "Z" and any(unit in units for unit in meter_units):
        return True

    # Check standard_name (CF-compliant, height-based only)
    if standard_name in _CF_COORDS_LOOKUP['level']['standard_name']:
        return True

    # Check name patterns (height-related only)
    height_name_patterns = ('z', 'height', 'altitude', 'depth', 'zlev', 'z_')
    if any(pattern in name for pattern in height_name_patterns):
        # Verify it has meter units to avoid false positives
        if any(unit in units for unit in meter_units):
            return True

    # Check for generic 'lev' or 'level' with meter units
    if ('lev' in name or 'level' in name) and any(unit in units for unit in meter_units):
        return True

    return False


def _is_geographic(coord: xr.DataArray, coord_type: str) -> bool:
    """True if *coord* matches the expected geographic type ('lat' or 'lon').

    Detection order: axis + units, then CF standard_name, then name pattern +
    units.  Value-range sanity checks (|lat| ≤ 90, |lon| ≤ 400) prevent false
    positives from bare dimension indices.
    """
    if coord_type not in ('lat', 'lon'):
        raise ValueError(f"coord_type must be 'lat' or 'lon', got {coord_type}")

    lookup = _CF_COORDS_LOOKUP.get(coord_type, {})
    if not lookup:
        return False

    name = (coord.name or "").lower()
    units = str(coord.attrs.get("units", "")).lower()
    standard_name = str(coord.attrs.get("standard_name", "")).lower()
    axis = str(coord.attrs.get("axis", "")).upper()

    # Check axis (most reliable for CF compliance)
    expected_axis = lookup.get("axis")
    if axis and axis == expected_axis:
        # Verify units are degree-like or radian-like to avoid false positives
        if any(hint in units for hint in ('degree', 'deg', 'rad', 'radian')):
            return True

    # Check standard_name (CF-compliant)
    expected_std = lookup.get("standard_name")
    if standard_name:
        if standard_name == expected_std:
            return True
        elif ("longitude" in standard_name and coord_type == "lat") or \
                ("latitude" in standard_name and coord_type == "lon"):
            # If it explicitly claims to be the other geographic coordinate, reject it immediately
            return False

    # Check name patterns
    name_ok = any(name == n or name.endswith(n) for n in lookup.get("names", ()))

    # Check units with direction-specific validation
    units_hints = lookup.get("units_hints", ())
    units_ok = any(hint in units for hint in units_hints)

    # Generic units ('degree', 'rad') aren't enough to distinguish lat from lon on their own
    is_generic_unit = units in ('degree', 'degrees', 'deg', 'rad', 'radian')
    if is_generic_unit and not name_ok and not standard_name:
        return False
    if coord_type == "lon":
        if "north" in units or "degree_north" in units:
            return False  # This is latitude, not longitude
        if units_ok or name_ok:
            # Additional check: longitude values should be in reasonable range
            if coord.size > 0:
                vals = coord.values
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    # Longitude typically in [-180, 360] range
                    if np.abs(vals).max() > 400:
                        return False
            return True

    if coord_type == "lat":
        if "east" in units or "degree_east" in units:
            return False  # This is longitude, not latitude
        if units_ok or name_ok:
            # Additional check: latitude values should be in [-90, 90]
            if coord.size > 0:
                vals = coord.values
                vals = vals[np.isfinite(vals)]
                if vals.size > 0:
                    if np.abs(vals).max() > 90.5:  # Small tolerance
                        return False
            return True

    return False


def is_geographic_grid(coord_x: xr.DataArray, coord_y: xr.DataArray) -> bool:
    """True if *coord_x* is longitude-like and *coord_y* is latitude-like."""
    # Check if the X coordinate is Longitude-like
    is_x_lon = _is_geographic(coord_x, "lon")

    # Check if the Y coordinate is Latitude-like
    is_y_lat = _is_geographic(coord_y, "lat")

    # The grid is geographic if and only if both components are identified
    return np.logical_and(is_x_lon, is_y_lat)


def _is_global_longitude(x_coord: xr.DataArray) -> bool:
    """
    Determine if a longitude coordinate covers (nearly) the full globe.

    Strategy (handles wrap-around and irregular spacing):
    - Normalize longitudes to [0, 360).
    - Sort along the x-direction and compute circular gaps between consecutive points,
      including the wrap gap (last→first + 360).
    - If the largest gap is no bigger than a small tolerance (~a few grid spacings),
      then coverage is effectively global.

    Ignores NaNs and duplicate endpoints (e.g., both 0 and 360 present).
    """

    def _normalize_deg(x):
        """Map to [0, 360) in degrees; ignore NaNs."""
        return np.asarray(x, dtype=np.float32) % 360.0

    lon = x_coord.values
    lon = _normalize_deg(lon[np.isfinite(lon)])

    # Sort unique longitudes (avoid duplicate endpoints like 0 and 360)
    lon = np.unique(np.sort(lon))
    if lon.size < 2:
        return False

    # Estimate a representative spacing (median nearest-neighbor gap on the circle)
    diffs = np.diff(lon)
    wrap_gap = (lon[0] + 360.0) - lon[-1]
    all_gaps = np.concatenate([diffs, [wrap_gap]])
    # If there are large holes, the largest gap will reflect that.
    max_gap = float(np.max(all_gaps))

    # Tolerance: allow a few grid spacings worth of slack (handles uneven grids)
    # Use the median gap as a spacing proxy; fall back to 360/N if needed.
    spacing = float(np.median(all_gaps)) if all_gaps.size else 360.0 / lon.size

    # “Global” if there is no big uncovered arc: i.e., largest gap ≲ tol=1.5 * spacing
    return max_gap <= 1.5 * spacing


# ----------------------
# Spatial dim resolution
# ----------------------
def get_spatial_dims(obj: Union[xr.Dataset, xr.DataArray]) -> Tuple[str, str]:
    """Return (y_dim, x_dim) horizontal dimension names using CF conventions.

    Priority: (1) CF lat/lon 1-D dimensions, (2) projected y/x with auxiliary
    lat/lon, (3) plain y/x, (4) last two non-time/non-vertical dimensions.
    Raises ``ValueError`` when no spatial dimensions can be identified.
    """
    ds = obj if isinstance(obj, xr.Dataset) else obj.to_dataset(name="_tmp")
    dims = set(ds.dims)

    # Case A: Try to find CF-compliant lat/lon as 1-D dimensions
    try:
        lat_coord = _find_coordinate(ds, 'lat', raise_notfound=False)
        lon_coord = _find_coordinate(ds, 'lon', raise_notfound=False)

        if lat_coord is not None and lon_coord is not None:
            # Check if they are 1-D dimension coordinates
            if (lat_coord.name in dims and lon_coord.name in dims and
                    lat_coord.ndim == 1 and lon_coord.ndim == 1):
                return str(lat_coord.name), str(lon_coord.name)
    except ValueError:
        pass

    # Case B: Check for standard lat/lon names as 1-D dims (fallback)
    if "lat" in dims and "lon" in dims:
        if ds["lat"].ndim == 1 and ds["lon"].ndim == 1:
            # Verify they look geographic
            if _is_geographic(ds["lat"], "lat") and _is_geographic(ds["lon"], "lon"):
                return "lat", "lon"

    # Case C: Projected axes with 2-D auxiliary lat/lon(y,x)
    if {"y", "x"} <= dims:
        # Check if there are 2-D lat/lon coordinates
        if "lat" in ds.coords and "lon" in ds.coords:
            if ds["lat"].dims == ("y", "x") and ds["lon"].dims == ("y", "x"):
                return "y", "x"
        # Plain y/x without auxiliary coords
        return "y", "x"

    # Case D: Fallback - use last two dimensions if they look spatial
    # (not time, not vertical)
    if len(ds.dims) >= 2:
        # Get all dims, filter out known non-spatial dims
        spatial_candidates = []
        for dim in ds.dims:
            # Skip if it's clearly time
            if str(dim).lower() in ('time', 't', 'date'):
                continue
            # Skip if it's clearly vertical
            if _is_z(str(dim), ds.coords):
                continue
            spatial_candidates.append(dim)

        # If we have at least 2 spatial candidates, use the last two
        if len(spatial_candidates) >= 2:
            # Convention: last two are (y, x) or (lat, lon)
            return spatial_candidates[-2], spatial_candidates[-1]

    raise ValueError(
        "get_spatial_dims: Could not determine horizontal dimensions. "
        "Expected CF-compliant lat/lon or projected y/x coordinates. "
        f"Available dims: {tuple(ds.dims)}, coords: {tuple(ds.coords)}"
    )
