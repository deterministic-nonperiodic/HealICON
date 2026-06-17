"""
HealICON: Interpolate atmospheric model outputs to HEALPix grid.
"""

__version__ = "0.1.1"

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
    compute_tidal_analysis,
    compute_leastsquares_tidal_analysis,
    fourier_wavelet_spectrum,
    spherical_harmonic_wavelet_spectrum,
    compute_wavelet_tidal_analysis,
    compute_fourier_tidal_analysis,
    wavelet,
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
    "fourier_wavelet_spectrum",
    "spherical_harmonic_wavelet_spectrum",
    "compute_wavelet_tidal_analysis",
    "compute_fourier_tidal_analysis",
    "compute_tidal_analysis",
    "compute_leastsquares_tidal_analysis",
    "visualize",
    "wavelet",
]
