import pytest
import xarray as xr
import numpy as np
import healpy as hp
from healicon.extract import extract_along_latitude, zonal_mean
from healicon.grid import create_healpix_dataset, get_healpix_coords, add_healpix_grid_mapping

def test_extract_along_latitude():
    # Create a synthetic HEALPix dataset
    nside = 4
    npix = hp.nside2npix(nside)
    ds = create_healpix_dataset(nside)
    
    # Simple data: latitude itself
    # So if we extract along latitude=45, the data should be close to 45.
    ds['test_var'] = (('cells',), ds.lat.values)
    
    # Extract at 45 N
    target_lat = 45.0
    out_ds = extract_along_latitude(ds, lat=target_lat, num_lons=10)
    
    assert 'lon' in out_ds.dims
    assert len(out_ds.lon) == 10
    
    # Verify the extracted data
    extracted_data = out_ds['test_var'].values
    assert np.allclose(extracted_data, target_lat, atol=10.0) # HEALPix nside=4 is very coarse, so atol is relaxed


def create_test_healpix_dataset(nside, order='ring'):
    ds = create_healpix_dataset(nside)
    lon, lat = get_healpix_coords(nside, nest=(order == 'nested'))
    ds['lon'] = ("cells", lon)
    ds['lat'] = ("cells", lat)
    ds = add_healpix_grid_mapping(ds, nside, order=order)
    return ds


def test_zonal_mean_healpix():
    nside = 4
    npix = hp.nside2npix(nside)
    
    # --- RING ordering ---
    ds_ring = create_test_healpix_dataset(nside, order='ring')
    # Simple data: latitude itself
    ds_ring['temp'] = (('cells',), ds_ring.lat.values)
    
    # Clean zonal mean
    zm_ring = zonal_mean(ds_ring)
    assert 'lat' in zm_ring.dims
    assert not np.isnan(zm_ring['temp'].values).any()
    
    # With NaNs (regional simulation)
    # Mask out southern hemisphere (lat < 0)
    temp_with_nans = ds_ring.lat.values.copy()
    temp_with_nans[temp_with_nans < 0] = np.nan
    ds_ring['temp_nan'] = (('cells',), temp_with_nans)
    
    zm_ring_nan = zonal_mean(ds_ring)
    # The northern rings should have valid values, southern rings should be NaN
    lat_vals = zm_ring_nan.lat.values
    temp_nan_vals = zm_ring_nan['temp_nan'].values
    for lat_val, val in zip(lat_vals, temp_nan_vals):
        if lat_val > 5.0:  # clearly northern hemisphere
            assert not np.isnan(val)
            assert np.allclose(val, lat_val, atol=15.0)
        elif lat_val < -5.0:  # clearly southern hemisphere
            assert np.isnan(val)
            
    # --- NESTED ordering ---
    ds_nest = create_test_healpix_dataset(nside, order='nested')
    ds_nest['temp'] = (('cells',), ds_nest.lat.values)
    
    # Clean zonal mean
    zm_nest = zonal_mean(ds_nest)
    assert 'lat' in zm_nest.dims
    assert not np.isnan(zm_nest['temp'].values).any()
    
    # With NaNs
    temp_with_nans_nest = ds_nest.lat.values.copy()
    temp_with_nans_nest[temp_with_nans_nest < 0] = np.nan
    ds_nest['temp_nan'] = (('cells',), temp_with_nans_nest)
    
    zm_nest_nan = zonal_mean(ds_nest)
    lat_vals_nest = zm_nest_nan.lat.values
    temp_nan_vals_nest = zm_nest_nan['temp_nan'].values
    for lat_val, val in zip(lat_vals_nest, temp_nan_vals_nest):
        if lat_val > 5.0:
            assert not np.isnan(val)
            assert np.allclose(val, lat_val, atol=15.0)
        elif lat_val < -5.0:
            assert np.isnan(val)


def test_zonal_mean_unstructured():
    # Create a generic unstructured dataset (not HEALPix)
    ncells = 100
    lats = np.linspace(-80, 80, ncells)
    lons = np.linspace(0, 360, ncells)
    
    ds = xr.Dataset(
        coords={
            'cells': np.arange(ncells),
        }
    )
    # geographic coordinate variables in data_vars
    ds['lat'] = (('cells',), lats)
    ds['lon'] = (('cells',), lons)
    ds['lat'].attrs = {'units': 'degrees_north'}
    ds['lon'].attrs = {'units': 'degrees_east'}
    
    # Data variable: just latitude
    ds['temp'] = (('cells',), lats.copy())
    
    # With NaNs
    temp_nans = lats.copy()
    temp_nans[temp_nans < 0] = np.nan
    ds['temp_nan'] = (('cells',), temp_nans)
    
    zm = zonal_mean(ds)
    assert 'lat' in zm.dims
    
    lat_bands = zm.lat.values
    temp_vals = zm['temp'].values
    temp_nan_vals = zm['temp_nan'].values
    
    for i, lat_val in enumerate(lat_bands):
        # If the clean mean is valid (non-NaN), check how temp_nan behaves
        if not np.isnan(temp_vals[i]):
            if lat_val > 10.0:
                assert not np.isnan(temp_nan_vals[i])
                assert np.allclose(temp_nan_vals[i], lat_val, atol=10.0)
            elif lat_val < -10.0:
                assert np.isnan(temp_nan_vals[i])
