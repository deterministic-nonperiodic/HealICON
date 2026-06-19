"""
Shared constants, utilities, and threading infrastructure for the analysis subpackage.
"""
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import healpy as hp
import numpy as np
import xarray as xr

from ..grid import (
    get_healpix_order, get_cells_dim, ensure_ring, ensure_original_order,
    append_history, add_healpix_grid_mapping, is_healpix,
)

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.229

# Earth radius in metres (used to scale streamfunction / velocity potential to m²/s)
_EARTH_RADIUS_M = EARTH_RADIUS_KM * 1e3

# --- Lazy Pint setup (avoids ~100ms import cost at module load) ---
_UNITS_REG = None
_unit_cmd = re.compile(r"(?<=[A-Za-z)])(?![A-Za-z)])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])")

def _parse_units(unit_str):
    import pint
    global _UNITS_REG
    if _UNITS_REG is None:
        _UNITS_REG = pint.UnitRegistry()
    if isinstance(unit_str, (pint.Quantity, pint.Unit)):
        return unit_str
    return _UNITS_REG(_unit_cmd.sub('**', unit_str))


# --- Thread pool sizing: cap at 8 to keep memory in check for large maps ---
_MAX_WORKERS = min(os.cpu_count() or 4, 8)


def degree_to_wavelength(l, radius=EARTH_RADIUS_KM):
    """
    Converts spherical harmonic degree l to characteristic wavelength (scale).
    
    Args:
        l: Spherical harmonic degree (scalar or array)
        radius: Radius of the sphere (defaults to Earth radius 6371.229 km)
        
    Returns:
        Wavelength matching the units of radius (km).
    """
    # Avoid division by zero for l=0
    l_safe = np.maximum(l, 1e-10)
    return (2 * np.pi * radius) / np.sqrt(l_safe * (l_safe + 1))


def wavelength_to_degree(wavelength, radius=EARTH_RADIUS_KM):
    """
    Converts characteristic wavelength (scale) to spherical harmonic degree l.
    
    Args:
        wavelength: Characteristic wavelength in same units as radius
        radius: Radius of the sphere (defaults to Earth radius 6371.229 km)
        
    Returns:
        Spherical harmonic degree l (float)
    """
    val = (2 * np.pi * radius) / wavelength
    # Solve l^2 + l - val^2 = 0
    return (-1.0 + np.sqrt(1.0 + 4.0 * val ** 2)) / 2.0


def get_progress_bar(iterable, desc=None, total=None, leave=False):
    """
    Returns a tqdm progress bar if called from the main thread,
    otherwise returns a quiet/silent iterator.
    """
    import threading
    is_main_thread = threading.current_thread() == threading.main_thread()
    env_disable = os.environ.get("HEALICON_DISABLE_PROGRESS", "0") == "1"
    if not is_main_thread or env_disable:
        return iterable
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, leave=leave)
    except ImportError:
        return iterable
