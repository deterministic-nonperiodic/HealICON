import pytest
import xarray as xr
import numpy as np
import healpy as hp
from healicon.interpolate import interpolate_to_healpix

def test_interpolate_regular():
    # Create a synthetic 2D regular grid dataset
    lon = np.linspace(0, 360, 36, endpoint=False)
    lat = np.linspace(-90, 90, 18)
    
    # Simple smooth function: f(lon, lat) = cos(lat)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    data = np.cos(np.deg2rad(lat_grid))
    
    ds = xr.Dataset(
        data_vars={
            'test_var': (('lat', 'lon'), data)
        },
        coords={
            'lon': lon,
            'lat': lat
        }
    )
    ds.lon.attrs['units'] = 'degrees_east'
    ds.lat.attrs['units'] = 'degrees_north'
    
    # Interpolate
    nside = 4
    out_ds = interpolate_to_healpix(ds, nside=nside)
    
    assert 'cells' in out_ds.dims
    assert 'test_var' in out_ds.data_vars
    
    print("max:", np.nanmax(out_ds['test_var'].values))
    print("min:", np.nanmin(out_ds['test_var'].values))
    print("nan count:", np.isnan(out_ds['test_var'].values).sum())
    
    # Check that max value is close to 1 (equator) and min close to cos(78.28) ~ 0.2 for nside=4
    assert np.nanmax(out_ds['test_var'].values) > 0.9
    assert np.nanmin(out_ds['test_var'].values) < 0.25

def test_interpolate_unstructured():
    # Create synthetic unstructured grid (random points)
    np.random.seed(42)
    ncells = 1000
    lon = np.random.uniform(0, 360, ncells)
    lat = np.random.uniform(-90, 90, ncells)
    
    # f(lat) = cos(lat)
    data = np.cos(np.deg2rad(lat))
    
    ds = xr.Dataset(
        data_vars={
            'test_var': (('ncells',), data)
        },
        coords={
            'clon': (('ncells',), lon),
            'clat': (('ncells',), lat)
        }
    )
    ds.clon.attrs['units'] = 'degrees_east'
    ds.clat.attrs['units'] = 'degrees_north'
    
    # Interpolate
    nside = 4
    out_ds = interpolate_to_healpix(ds, nside=nside)
    
    assert 'cells' in out_ds.dims
    assert 'test_var' in out_ds.data_vars
    
    # The random unstructured grid might not perfectly cover the poles, 
    # but the general smooth profile should hold.
    assert np.nanmax(out_ds['test_var'].values) > 0.8
    assert np.nanmin(out_ds['test_var'].values) < 0.3
    
def test_metadata_propagation():
    # Create simple dataset
    lon = np.array([0, 10, 20])
    lat = np.array([0, 10, 20])
    data = np.random.rand(3)
    
    ds = xr.Dataset(
        data_vars={'test_var': (('cell',), data)},
        coords={'clon': (('cell',), lon), 'clat': (('cell',), lat)}
    )
    ds.clon.attrs['units'] = 'degrees_east'
    ds.clat.attrs['units'] = 'degrees_north'
    ds.attrs['model_name'] = 'SYNTHETIC_MODEL'
    ds.attrs['uuid'] = '12345'
    
    out_ds = interpolate_to_healpix(ds, nside=2)
    
    # Check original metadata
    assert out_ds.attrs.get('model_name') == 'SYNTHETIC_MODEL'
    assert out_ds.attrs.get('uuid') == '12345'
    
    # Check new HEALPix metadata
    assert out_ds.attrs.get('healpix_nside') == 2
    assert out_ds.attrs.get('healpix_npix') == hp.nside2npix(2)
    assert out_ds.attrs.get('healpix_scheme') == 'RING'
    assert 'healpix_cell_area_sr' in out_ds.attrs
    assert 'history' in out_ds.attrs

def test_radian_conversion():
    # Create dataset with coordinates in radians
    lon_rad = np.array([0, np.pi/2, np.pi])
    lat_rad = np.array([0, np.pi/4, np.pi/2])
    data = np.array([1.0, 2.0, 3.0])
    
    ds = xr.Dataset(
        data_vars={'test_var': (('cell',), data)},
        coords={'clon': (('cell',), lon_rad), 'clat': (('cell',), lat_rad)}
    )
    ds.clon.attrs['units'] = 'radian'
    ds.clat.attrs['units'] = 'radian'
    
    # Should convert automatically
    out_ds = interpolate_to_healpix(ds, nside=2)
    
    assert 'cells' in out_ds.dims
    assert 'test_var' in out_ds.data_vars
