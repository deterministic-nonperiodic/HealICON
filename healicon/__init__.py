"""
HealICON: Interpolate atmospheric model outputs to HEALPix grid.
"""

__version__ = "0.1.0"

from .interpolate import interpolate_to_healpix
from . import core, grid, interpolate, analysis, extract, visualize
from .analysis import (
    compute_spectrum,
    filter_spatial,
    regrade_resolution,
    compute_vorticity_divergence,
    compute_helmholtz,
    degree_to_wavelength,
    wavelength_to_degree,
    EARTH_RADIUS_KM,
)
from .extract import (
    extract_along_latitude,
    extract_along_longitude,
    extract_point,
    zonal_mean,
)

__all__ = [
    "interpolate_to_healpix",
    "compute_spectrum",
    "filter_spatial",
    "regrade_resolution",
    "compute_vorticity_divergence",
    "compute_helmholtz",
    "degree_to_wavelength",
    "wavelength_to_degree",
    "EARTH_RADIUS_KM",
    "extract_along_latitude",
    "extract_along_longitude",
    "extract_point",
    "zonal_mean",
    "visualize",
]
