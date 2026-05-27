import pytest
import xarray as xr
import numpy as np
import healpy as hp
from healicon.extract import extract_along_latitude
from healicon.grid import create_healpix_dataset

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
