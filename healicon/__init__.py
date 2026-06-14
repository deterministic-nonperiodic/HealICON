"""
HealICON: Interpolate atmospheric model outputs to HEALPix grid.
"""

__version__ = "0.1.1"

from .interpolate import interpolate_to_healpix
from . import core, grid, interpolate, analysis, extract, visualize, wavelet
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
from .wavelet import (fourier_wavelet_spectrum,
                      spherical_harmonic_wavelet_spectrum,
                      compute_wavelet_tidal_analysis)

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
    "fourier_wavelet_spectrum",
    "spherical_harmonic_wavelet_spectrum",
    "compute_wavelet_tidal_analysis",
    "visualize",
    "wavelet",
]
