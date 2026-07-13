import pytest
import numpy as np
import pandas as pd
import xarray as xr
import healpy as hp
from healicon.compare import compare, print_table, _to_latex

def test_latex_table_generation():
    # Create test dataframe
    records = [
        {'Variable': 'temp_phys', 'Units': 'K', 'is_pres': False, 'is_meter': True,
         'Level': 100.1, 'N': 168, 'Bias': 1.0, 'RMSE': 5.0, 'cRMSE': 4.9,
         'MAE': 3.5, 'r': 0.95, 'σ_ref': 8.0, 'σ_cmp': 7.5, 'Skill': 0.9},
        {'Variable': 'temp_phys', 'Units': 'K', 'is_pres': False, 'is_meter': True,
         'Level': None, 'N': 168, 'Bias': 1.0, 'RMSE': 5.0, 'cRMSE': 4.9,
         'MAE': 3.5, 'r': 0.95, 'σ_ref': 8.0, 'σ_cmp': 7.5, 'Skill': 0.9},
        {'Variable': 'wind_speed', 'Units': 'm s⁻¹', 'is_pres': False, 'is_meter': True,
         'Level': 100.1, 'N': 168, 'Bias': 2.0, 'RMSE': 20.0, 'cRMSE': 19.9,
         'MAE': 15.0, 'r': 0.45, 'σ_ref': 15.0, 'σ_cmp': 20.0, 'Skill': -0.5},
        {'Variable': 'wind_speed_hat', 'Units': 'm s$^{-1}$', 'is_pres': False, 'is_meter': True,
         'Level': 100.1, 'N': 168, 'Bias': 2.0, 'RMSE': 20.0, 'cRMSE': 19.9,
         'MAE': 15.0, 'r': 0.45, 'σ_ref': 15.0, 'σ_cmp': 20.0, 'Skill': -0.5},
        {'Variable': 'wind_speed_star', 'Units': 'm s^-1', 'is_pres': False, 'is_meter': True,
         'Level': 100.1, 'N': 168, 'Bias': 2.0, 'RMSE': 20.0, 'cRMSE': 19.9,
         'MAE': 15.0, 'r': 0.45, 'σ_ref': 15.0, 'σ_cmp': 20.0, 'Skill': -0.5},
        {'Variable': 'wind_speed', 'Units': 'm s⁻¹', 'is_pres': False, 'is_meter': True,
         'Level': None, 'N': 168, 'Bias': 2.0, 'RMSE': 20.0, 'cRMSE': 19.9,
         'MAE': 15.0, 'r': 0.45, 'σ_ref': 15.0, 'σ_cmp': 20.0, 'Skill': -0.5}
    ]
    df = pd.DataFrame(records)
    
    latex_out = _to_latex(df, 3)
    
    # Assert tabular setup
    assert '\\begin{tabular}{llrrrrrrrrr}' in latex_out
    assert '\\toprule' in latex_out
    assert '\\bottomrule' in latex_out
    assert '\\end{tabular}' in latex_out
    
    # Assert special LaTeX escapes
    assert 'temp\\_phys' in latex_out
    assert 'wind\\_speed' in latex_out
    assert 'm s\\textsuperscript{-1}' in latex_out
    assert '$\\sigma_{\\text{ref}}$ (K)' in latex_out
    assert '$\\sigma_{\\text{cmp}}$ (K)' in latex_out
    assert '$\\sigma_{\\text{ref}}$ (m s\\textsuperscript{-1})' in latex_out
    assert '$\\sigma_{\\text{cmp}}$ (m s\\textsuperscript{-1})' in latex_out
    assert 'wind\\_speed\\_hat & 100.1 & 168 & +2.000 & 20.000 & 19.900 & 15.000 & 0.450 & 15.000 & 20.000 & -0.500' in latex_out
    assert 'm s$^{-1}$' in latex_out  # Ensure existing latex block ($) is kept as is
    
    # Assert duplicate variable names are blanked out in subsequent rows
    assert ' & GLOBAL & 168 & +1.000 & 5.000 & 4.900 & 3.500 & 0.950 & 8.000 & 7.500 & 0.900 \\\\' in latex_out
    assert ' & GLOBAL & 168 & +2.000 & 20.000 & 19.900 & 15.000 & 0.450 & 15.000 & 20.000 & -0.500 \\\\' in latex_out

def test_compare_with_filtering():
    np.random.seed(42)
    nside = 4
    npix = hp.nside2npix(nside)
    
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    lat = 90.0 - np.rad2deg(theta)
    smooth_field = 250.0 + 30.0 * np.cos(np.deg2rad(lat))
    
    levels = np.array([50000], dtype=float)
    # Add random noise to both datasets
    ref_data = np.stack([smooth_field + np.random.randn(npix) * 5 for _ in levels])
    cmp_data = np.stack([smooth_field + np.random.randn(npix) * 5 for _ in levels])
    
    ds_ref = xr.Dataset(
        {'temp': (['z_mc', 'cells'], ref_data.astype(np.float32))},
        coords={'z_mc': levels, 'cells': np.arange(npix)},
        attrs={'healpix_nside': nside, 'healpix_order': 'ring'}
    )
    ds_ref['z_mc'].attrs = {'units': 'm', 'axis': 'Z'}
    
    ds_cmp = xr.Dataset(
        {'temp': (['z_mc', 'cells'], cmp_data.astype(np.float32))},
        coords={'z_mc': levels, 'cells': np.arange(npix)},
        attrs={'healpix_nside': nside, 'healpix_order': 'ring'}
    )
    ds_cmp['z_mc'].attrs = {'units': 'm', 'axis': 'Z'}
    
    # Without filter, correlation will be lower due to noise
    df_no_filt = compare(ds_ref, ds_cmp, by_level=True)
    r_no_filt = df_no_filt.iloc[0]['r']
    assert df_no_filt.iloc[0]['Variable'] == 'Temperature'
    
    # Verify long_name attribute from dataset is respected if short
    ds_ref['temp'].attrs['long_name'] = 'Short Temp'
    # We temporarily rename the variable to something not in DISPLAY_MAPPING to test fallback
    ds_ref_renamed = ds_ref.rename({'temp': 'temp_custom'})
    ds_cmp_renamed = ds_cmp.rename({'temp': 'temp_custom'})
    df_custom = compare(ds_ref_renamed, ds_cmp_renamed, by_level=True)
    assert df_custom.iloc[0]['Variable'] == 'Short Temp'
    
    # With lmax filter, correlation should be very high
    df_lmax = compare(ds_ref, ds_cmp, by_level=True, lmax=1)
    r_lmax = df_lmax.iloc[0]['r']
    
    assert r_lmax > r_no_filt
    assert r_lmax > 0.99

def test_compare_zonal_mean_nested():
    from healicon.grid import add_healpix_grid_mapping
    np.random.seed(42)
    nside = 4
    npix = hp.nside2npix(nside)
    
    # Generate data using nested ordering
    theta, phi = hp.pix2ang(nside, np.arange(npix), nest=True)
    lat = 90.0 - np.rad2deg(theta)
    smooth_field = 250.0 + 30.0 * np.cos(np.deg2rad(lat))
    
    levels = np.array([50000], dtype=float)
    ref_data = np.stack([smooth_field + np.random.randn(npix) * 0.1 for _ in levels])
    cmp_data = np.stack([smooth_field + np.random.randn(npix) * 0.1 for _ in levels])
    
    ds_ref = xr.Dataset(
        {'temp': (['z_mc', 'cells'], ref_data.astype(np.float32))},
        coords={'z_mc': levels, 'cells': np.arange(npix)}
    )
    ds_ref['z_mc'].attrs = {'units': 'm', 'axis': 'Z'}
    ds_ref = add_healpix_grid_mapping(ds_ref, nside=nside, order='nested')
    
    # Intentionally remove global healpix attributes to force relying on grid mapping variable
    if 'healpix_scheme' in ds_ref.attrs:
        del ds_ref.attrs['healpix_scheme']
    if 'healpix_nside' in ds_ref.attrs:
        del ds_ref.attrs['healpix_nside']
        
    ds_cmp = xr.Dataset(
        {'temp': (['z_mc', 'cells'], cmp_data.astype(np.float32))},
        coords={'z_mc': levels, 'cells': np.arange(npix)}
    )
    ds_cmp['z_mc'].attrs = {'units': 'm', 'axis': 'Z'}
    ds_cmp = add_healpix_grid_mapping(ds_cmp, nside=nside, order='nested')
    
    if 'healpix_scheme' in ds_cmp.attrs:
        del ds_cmp.attrs['healpix_scheme']
    if 'healpix_nside' in ds_cmp.attrs:
        del ds_cmp.attrs['healpix_nside']
        
    # Call compare with reduce='zonal-mean'
    df = compare(ds_ref, ds_cmp, variables=['temp'], reduce='zonal-mean', by_level=True)
    assert len(df) > 0
    assert df.iloc[0]['Variable'] == 'Temperature'
