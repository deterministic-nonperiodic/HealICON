import pytest
import xarray as xr
import numpy as np

from healicon.cf_coords import _find_coordinate, get_spatial_dims, _is_z, _is_geographic

def test_find_coordinate_lon_lat():
    # Test finding standard CF lon/lat
    ds = xr.Dataset(
        coords={
            'longitude': ('longitude', np.linspace(0, 360, 10), {'standard_name': 'longitude', 'units': 'degrees_east'}),
            'latitude': ('latitude', np.linspace(-90, 90, 10), {'standard_name': 'latitude', 'units': 'degrees_north'})
        }
    )
    
    lon = _find_coordinate(ds, 'lon')
    assert lon is not None
    assert lon.name == 'longitude'
    
    lat = _find_coordinate(ds, 'lat')
    assert lat is not None
    assert lat.name == 'latitude'


def test_find_coordinate_level():
    # Test finding vertical level coordinate
    ds = xr.Dataset(
        coords={
            'height': ('height', np.linspace(0, 100, 5), {'standard_name': 'altitude', 'units': 'm', 'axis': 'Z'})
        }
    )
    
    level = _find_coordinate(ds, 'level')
    assert level is not None
    assert level.name == 'height'


def test_get_spatial_dims():
    # Test extracting the spatial dimensions
    ds = xr.Dataset(
        coords={
            'x': ('x', np.arange(10), {'standard_name': 'projection_x_coordinate', 'units': 'm'}),
            'y': ('y', np.arange(10), {'standard_name': 'projection_y_coordinate', 'units': 'm'}),
            'lat': (('y', 'x'), np.zeros((10, 10))),
            'lon': (('y', 'x'), np.zeros((10, 10))),
        }
    )
    
    # In this case it should return ('y', 'x') because lat/lon are 2D auxiliary coordinates
    y_dim, x_dim = get_spatial_dims(ds)
    assert y_dim == 'y'
    assert x_dim == 'x'


def test_is_z():
    # Test checking if coordinate is Z
    ds = xr.Dataset(
        coords={
            'z_mc': ('z_mc', np.arange(10), {'standard_name': 'altitude', 'units': 'm'}),
            'plev': ('plev', np.arange(10), {'standard_name': 'air_pressure', 'units': 'Pa'})
        }
    )
    
    assert _is_z('z_mc', ds.coords) == True
    assert _is_z('plev', ds.coords) == False


def test_is_geographic():
    ds = xr.Dataset(
        coords={
            'lon': ('lon', np.linspace(0, 360, 10), {'units': 'degrees_east'}),
            'lat': ('lat', np.linspace(-90, 90, 10), {'units': 'degrees_north'})
        }
    )
    assert _is_geographic(ds['lon'], 'lon') == True
    assert _is_geographic(ds['lat'], 'lat') == True
    assert _is_geographic(ds['lon'], 'lat') == False
    assert _is_geographic(ds['lat'], 'lon') == False
