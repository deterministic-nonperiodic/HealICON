import numpy as np
import xarray as xr
import healpy as hp
import pytest
from healicon.analysis import (
    compute_spectrum,
    filter_spatial,
    regrade_resolution,
    compute_vorticity_divergence,
    compute_uv_from_vorticity_divergence,
    compute_helmholtz
)
from healicon.extract import (
    extract_along_longitude,
    zonal_mean,
    extract_point
)

@pytest.fixture
def synthetic_healpix_ds():
    nside = 4
    npix = hp.nside2npix(nside)
    
    # Create a synthetic dataset with a dipole
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    # Y_1^0 dipole
    data = np.cos(theta)
    
    # Vector field for vorticity/divergence
    # Pure divergence: u = 0, v = sin(theta) (actually v_theta = cos(theta) => div ~ something)
    u = np.zeros(npix)
    v = np.sin(theta)
    
    ds = xr.Dataset(
        data_vars={
            'temp': (['time', 'cells'], data[np.newaxis, :]),
            'u': (['time', 'cells'], u[np.newaxis, :]),
            'v': (['time', 'cells'], v[np.newaxis, :])
        },
        coords={
            'time': [1],
            'cells': np.arange(npix)
        }
    )
    return ds

def test_compute_spectrum(synthetic_healpix_ds):
    ds_cl = compute_spectrum(synthetic_healpix_ds, 'temp', lmax=3)
    assert 'temp_cl' in ds_cl
    assert 'l' in ds_cl.dims
    assert ds_cl.sizes['l'] == 4
    
    # Since it's a pure Y_1^0 dipole, C_1 should be dominant, C_0 and C_2 should be near 0
    cls = ds_cl['temp_cl'].values[0]
    assert cls[1] > 1e-2
    assert cls[0] < 1e-10
    assert cls[2] < 1e-10

def test_filter_spatial(synthetic_healpix_ds):
    # Test fwhm
    ds_fwhm = filter_spatial(synthetic_healpix_ds, fwhm_deg=10.0)
    assert 'temp' in ds_fwhm
    
    # Test lmax hard cutoff
    ds_lmax = filter_spatial(synthetic_healpix_ds, lmax=1)
    assert 'temp' in ds_lmax

def test_regrade_resolution(synthetic_healpix_ds):
    ds_regraded = regrade_resolution(synthetic_healpix_ds, new_nside=8)
    assert ds_regraded.sizes['cells'] == hp.nside2npix(8)
    assert 'temp' in ds_regraded
    assert ds_regraded['temp'].shape == (1, hp.nside2npix(8))

def test_compute_vorticity_divergence(synthetic_healpix_ds):
    ds_vd = compute_vorticity_divergence(synthetic_healpix_ds, 'u', 'v', lmax=3)
    assert 'vorticity' in ds_vd
    assert 'divergence' in ds_vd
    assert ds_vd['vorticity'].shape == (1, hp.nside2npix(4))

def test_compute_uv_from_vorticity_divergence(synthetic_healpix_ds):
    ds_vd = compute_vorticity_divergence(synthetic_healpix_ds, 'u', 'v', lmax=3)
    ds_uv = compute_uv_from_vorticity_divergence(ds_vd, 'divergence', 'vorticity', lmax=3)
    assert 'u' in ds_uv
    assert 'v' in ds_uv
    assert ds_uv['u'].shape == (1, hp.nside2npix(4))

def test_compute_helmholtz(synthetic_healpix_ds):
    ds_helm = compute_helmholtz(synthetic_healpix_ds, 'u', 'v', lmax=3)
    assert 'u_rot' in ds_helm
    assert 'v_rot' in ds_helm
    assert 'u_div' in ds_helm
    assert 'v_div' in ds_helm
    assert 'psi' in ds_helm
    assert 'chi' in ds_helm

def test_extract_along_longitude(synthetic_healpix_ds):
    ds_lon = extract_along_longitude(synthetic_healpix_ds, lon=45.0, num_lats=10)
    assert 'lat' in ds_lon.dims
    assert ds_lon.sizes['lat'] == 10
    assert 'temp' in ds_lon

def test_zonal_mean(synthetic_healpix_ds):
    ds_zonal = zonal_mean(synthetic_healpix_ds)
    assert 'lat' in ds_zonal.dims
    n_rings = 4 * 4 - 1
    assert ds_zonal.sizes['lat'] == n_rings
    assert 'temp' in ds_zonal

def test_extract_point(synthetic_healpix_ds):
    ds_pt = extract_point(synthetic_healpix_ds, lat=0.0, lon=0.0)
    # The cell dimension is gone or size 1? isel drops the dimension if scalar
    assert 'cells' not in ds_pt.dims
    assert 'temp' in ds_pt
    assert 'time' in ds_pt.dims
