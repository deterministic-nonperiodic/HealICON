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
            "cell": np.arange(npix),
            "lon": ("cell", lon),
            "lat": ("cell", lat)
        }
    )
    ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
    ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}

    return ds
