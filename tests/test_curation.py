import numpy as np
import xarray as xr
import healpy as hp
from healicon.curation import fill_healpix_gaps

def test_fill_healpix_gaps_spatial():
    nside = 2
    npix = hp.nside2npix(nside)
    
    # Create dataset with one variable
    # We will set pixel 0 to NaN, and the rest to 10.0
    data = np.full(npix, 10.0)
    data[0] = np.nan
    
    ds = xr.Dataset(
        data_vars=dict(
            temperature=(["cells"], data)
        ),
        coords=dict(
            cells=np.arange(npix)
        ),
        attrs=dict(healpix_nside=nside)
    )
    
    filled_ds = fill_healpix_gaps(ds, spatial_dim="cells")
    
    assert not np.isnan(filled_ds["temperature"].values).any()
    # It should have filled pixel 0 with the value of its nearest neighbor (which is 10.0)
    assert filled_ds["temperature"].values[0] == 10.0

def test_fill_healpix_gaps_temporal():
    nside = 1
    npix = hp.nside2npix(nside)
    time_size = 3
    
    # Create 2D data (cells, time)
    # We will create a linear temporal trend 0.0, 10.0, 20.0
    # And we will set the middle time step to NaN for pixel 0
    # The linear interpolation should recover 10.0
    
    data = np.zeros((npix, time_size))
    for t in range(time_size):
        data[:, t] = t * 10.0
        
    # Introduce spatial+temporal gap
    data[0, 1] = np.nan
    
    # Introduce a purely spatial gap (time step 0, pixel 1 is missing)
    # But pixel 2 is valid and has value 0.0. Spatial KDTree will fill it with 0.0.
    data[1, 0] = np.nan
    
    ds = xr.Dataset(
        data_vars=dict(
            temperature=(["cells", "time"], data)
        ),
        coords=dict(
            cells=np.arange(npix),
            time=np.arange(time_size)
        ),
        attrs=dict(healpix_nside=nside)
    )
    
    filled_ds = fill_healpix_gaps(ds, spatial_dim="cells", time_dim="time")
    
    assert not np.isnan(filled_ds["temperature"].values).any()
    
    # Check temporal filling: pixel 0 at time 1 should be 10.0
    assert np.isclose(filled_ds["temperature"].values[0, 1], 10.0)
    
    # Check spatial filling: pixel 1 at time 0 should be filled by another pixel's value at time 0 (0.0)
    assert np.isclose(filled_ds["temperature"].values[1, 0], 0.0)
