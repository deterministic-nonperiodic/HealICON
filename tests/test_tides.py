import pytest
import numpy as np
import healpy as hp
import xarray as xr
from click.testing import CliRunner

from healicon import (
    compute_leastsquares_tidal_analysis,
    compute_wavelet_tidal_analysis,
    compute_fourier_tidal_analysis,
)
from healicon.cli import tides


@pytest.fixture
def synthetic_healpix_ds():
    nside = 4
    npix = hp.nside2npix(nside)
    n_time = 48  # 48 hours
    
    # DW1 tide: m=1, period=24h
    amp_true = 3.0
    phase_true = 0.5
    omega = 2 * np.pi / 24.0
    
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    t_hours = np.arange(n_time, dtype=float)
    
    # signal: cos(phi - omega*t + phase)
    signal = amp_true * np.cos(phi[None, :] - omega * t_hours[:, None] + phase_true)
    
    ds = xr.Dataset(
        {'temp': (['lst', 'cells'], signal.astype(np.float64))},
        coords={
            'lst': t_hours,
            'cells': np.arange(npix),
        }
    )
    ds['temp'].attrs = {'units': 'K'}
    ds.attrs['healpix_scheme'] = 'RING'
    return ds


def test_tides_methods(synthetic_healpix_ds):
    ds = synthetic_healpix_ds
    periods = [24.0]
    m_filters = [1]
    
    # 1. Least-Squares
    ds_ls = compute_leastsquares_tidal_analysis(ds, 'temp', periods, m_filters, time_dim='lst')
    assert 'temp_amp_sym' in ds_ls
    assert 'temp_pha_sym' in ds_ls
    assert 'temp_amp_asy' in ds_ls
    assert 'temp_pha_asy' in ds_ls
    
    assert ds_ls['temp_amp_sym'].shape == (1, 1, ds.sizes['cells'])

    # 2. Wavelet
    ds_wav = compute_wavelet_tidal_analysis(
        ds, 'temp', periods, m_filters, dj=0.1, temporal_mean=True, time_dim='lst'
    )
    assert 'temp_amp_sym' in ds_wav
    assert ds_wav['temp_amp_sym'].shape == (1, 1, ds.sizes['cells'])
    
    # 3. Fourier
    ds_four = compute_fourier_tidal_analysis(
        ds, 'temp', periods, m_filters, dj=0.1, temporal_mean=True, time_dim='lst'
    )
    assert 'temp_amp_sym' in ds_four
    assert ds_four['temp_amp_sym'].shape == (1, 1, ds.sizes['cells'])


def test_tides_cli(tmp_path, synthetic_healpix_ds):
    ifile = tmp_path / "input.nc"
    ofile = tmp_path / "output.nc"
    
    synthetic_healpix_ds.to_netcdf(ifile)
    
    runner = CliRunner()
    
    # Test LS CLI
    result = runner.invoke(tides, [
        str(ifile), str(ofile),
        '-v', 'temp',
        '-p', '24.0',
        '-m', '1',
        '--time-dim', 'lst',
        '--method', 'ls'
    ])
    assert result.exit_code == 0
    assert ofile.exists()
    
    ds_out = xr.open_dataset(ofile)
    assert 'temp_amp_sym' in ds_out
    ofile.unlink()
    
    # Test Fourier CLI
    result = runner.invoke(tides, [
        str(ifile), str(ofile),
        '-v', 'temp',
        '-p', '24.0',
        '-m', '1',
        '--time-dim', 'lst',
        '--method', 'fourier',
        '--temporal-mean'
    ])
    assert result.exit_code == 0
    assert ofile.exists()
    ofile.unlink()
    
    # Test Wavelet CLI
    result = runner.invoke(tides, [
        str(ifile), str(ofile),
        '-v', 'temp',
        '-p', '24.0',
        '-m', '1',
        '--time-dim', 'lst',
        '--method', 'wavelet',
        '--temporal-mean'
    ])
    assert result.exit_code == 0
    assert ofile.exists()
    ofile.unlink()

    # Test --modes option for all three methods
    for meth in ('ls', 'fourier', 'wavelet'):
        result = runner.invoke(tides, [
            str(ifile), str(ofile),
            '-v', 'temp',
            '--modes', 'DW1',
            '--time-dim', 'lst',
            '--method', meth,
            '--temporal-mean'
        ])
        assert result.exit_code == 0
        assert ofile.exists()
        ds_out = xr.open_dataset(ofile)
        # For all unified methods, DW1 (westward) maps to m=1.0
        assert float(ds_out['m'].values[0]) == 1.0
        ofile.unlink()
