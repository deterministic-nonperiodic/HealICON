import healpy as hp
import numpy as np
import xarray as xr


def get_healpix_coords(nside: int):
    """
    Generate longitude and latitude coordinates for a HEALPix grid.
    Returns:
        lon: numpy array of longitudes in degrees [0, 360]
        lat: numpy array of latitudes in degrees [-90, 90]
    """
    npix = hp.nside2npix(nside)

    # healpy pix2ang returns colatitude (theta) and longitude (phi) in radians
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # Convert colatitude to latitude (-90 to 90)
    lat = 90.0 - np.rad2deg(theta)

    # Convert longitude to degrees (0 to 360)
    lon = np.rad2deg(phi)

    return lon, lat


def create_healpix_dataset(nside: int) -> xr.Dataset:
    """
    Creates an empty xarray Dataset with HEALPix coordinates.
    """
    lon, lat = get_healpix_coords(nside)
    npix = len(lon)

    ds = xr.Dataset(
        coords={
            "cells": np.arange(npix),
        }
    )
    ds["lon"] = ("cells", lon)
    ds["lat"] = ("cells", lat)
    ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}

    return ds


def get_healpix_order(ds: xr.Dataset) -> str:
    """
    Get the HEALPix ordering from the dataset attributes.
    Returns 'nested' or 'ring'.
    """
    for name, var in ds.variables.items():
        if var.attrs.get('grid_mapping_name') == 'healpix':
            return var.attrs.get('healpix_order', 'ring').lower()
    return ds.attrs.get('healpix_scheme', 'ring').lower()


def ensure_ring(data: np.ndarray, order: str) -> np.ndarray:
    """
    Ensure the data array is in RING ordering.
    If order is 'nested', converts to 'ring'.
    Assumes HEALPix dimension is the last axis.
    """
    if order == 'nested':
        return hp.reorder(data, n2r=True)
    return data


def ensure_original_order(data: np.ndarray, original_order: str) -> np.ndarray:
    """
    Convert RING array back to its original order if it was 'nested'.
    Assumes HEALPix dimension is the last axis.
    """
    if original_order == 'nested':
        return hp.reorder(data, r2n=True)
    return data


def get_cells_dim(ds: xr.Dataset) -> str:
    """
    Return the spatial dimension name. Tries 'cells' then 'cell'.
    Raises ValueError if neither found.
    """
    for dim in ['cells', 'cell', 'ncells', 'x']:
        if dim in ds.dims:
            return dim
    raise ValueError(
        "Dataset must have a HEALPix spatial dimension (e.g. 'cell', 'cells', 'ncells', or 'x').")
