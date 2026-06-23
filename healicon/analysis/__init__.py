"""
HealICON Analysis Package

Provides spectral analysis, spatial filtering, resolution regrading,
Helmholtz decomposition, and tidal analysis on HEALPix grids.

All public symbols are re-exported here for backward compatibility:
    from healicon.analysis import compute_spectrum   # still works
"""

# --- Shared constants and utilities ---
from ._common import (
    EARTH_RADIUS_KM,
    degree_to_wavelength,
    wavelength_to_degree,
    _parse_units,
    _MAX_WORKERS,
)

# --- Spectral analysis ---
from .spectral import (
    _anafast_block,
    compute_spectrum,
)

# --- Spatial filtering ---
from .filtering import (
    _filter_block,
    filter_spatial,
)

# --- Resolution regrading ---
from .regrade import (
    _regrade_block,
    regrade_resolution,
)

# --- Helmholtz decomposition and wind <-> div/vor ---
from .helmholtz import (
    _helmholtz_block,
    compute_helmholtz,
    _vorticity_divergence_block,
    compute_vorticity_divergence,
    _uv_from_vorticity_divergence_block,
    compute_uv_from_vorticity_divergence,
)

# --- Tidal analysis ---
from .tides import (
    _directional_filter_block,
    _extract_spatial_tide_components,
    compute_leastsquares_tidal_analysis,
    compute_wavelet_tidal_analysis,
)

# --- Wavelet analysis ---
from .wavelet import (
    _get_symmetric_pixels,
    fourier_wavelet_spectrum,
    spherical_harmonic_wavelet_spectrum,
)

# Re-export grid utilities that were historically importable from analysis
from ..grid import ensure_ring, ensure_original_order

__all__ = [
    "EARTH_RADIUS_KM",
    "degree_to_wavelength",
    "wavelength_to_degree",
    "compute_spectrum",
    "filter_spatial",
    "regrade_resolution",
    "compute_helmholtz",
    "compute_vorticity_divergence",
    "compute_uv_from_vorticity_divergence",
    "compute_leastsquares_tidal_analysis",
    "fourier_wavelet_spectrum",
    "spherical_harmonic_wavelet_spectrum",
    "compute_wavelet_tidal_analysis",
    # Internal but used by wavelet module
    "ensure_ring",
    "ensure_original_order",
    "_get_symmetric_pixels",
]
