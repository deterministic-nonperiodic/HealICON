import pytest
import numpy as np
import healpy as hp
import xarray as xr
from click.testing import CliRunner

from healicon import (
    compute_leastsquares_tidal_analysis,
    compute_wavelet_tidal_analysis,
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
    ds_four = compute_wavelet_tidal_analysis(
        ds, 'temp', periods, m_filters, dj=0.1, temporal_mean=True, time_dim='lst', method='fourier'
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
    
    # Test Wavelet CLI (sh)
    result = runner.invoke(tides, [
        str(ifile), str(ofile),
        '-v', 'temp',
        '-p', '24.0',
        '-m', '1',
        '--time-dim', 'lst',
        '--method', 'sh',
        '--temporal-mean'
    ])
    assert result.exit_code == 0
    assert ofile.exists()
    ofile.unlink()

    # Test --modes option for all three methods
    for meth in ('ls', 'fourier', 'sh'):
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



@pytest.fixture
def westward_dw1_ds():
    """A single westward diurnal tide, in the package's own convention.

    `_directional_filter_block` defines positive m as westward, cos(m*lambda +
    omega*t). `synthetic_healpix_ds` builds cos(phi - omega*t), which is
    eastward, and then asks for m=+1; the filter correctly returns nothing and
    the assertions never notice because they only check array shapes. A tide
    the analysis is supposed to find is needed to test what it returns.
    """
    nside = 8
    npix = hp.nside2npix(nside)
    amp, phase, m = 3.0, 0.5, 1
    omega = 2 * np.pi / 24.0
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    t_hours = np.arange(48, dtype=float)
    # a hemispheric asymmetry, so that the antisymmetric part is not
    # identically zero and recombining the two is a real test
    lat_shape = 1.0 + 0.6 * np.cos(theta)
    signal = (amp * lat_shape[None, :]
              * np.cos(m * phi[None, :] + omega * t_hours[:, None] + phase))
    ds = xr.Dataset({'temp': (['lst', 'cells'], signal)},
                    coords={'lst': t_hours, 'cells': np.arange(npix)})
    ds['temp'].attrs = {'units': 'K'}
    ds.attrs['healpix_scheme'] = 'RING'
    ds.attrs['_amp_true'] = amp
    return ds


def test_westward_tide_is_recovered(westward_dw1_ds):
    """The amplitude is right and the wrong direction is empty.

    Neither is asserted anywhere else: the suite checks shapes only, so a
    filter returning zeros passes it.
    """
    ds = westward_dw1_ds
    out = compute_leastsquares_tidal_analysis(
        ds, 'temp', [24.0], [1, -1], time_dim='lst', decompose_sym_asy=False)
    amp = out['temp_amp_total'].isel(period=0)
    west = amp.sel(m=1).values
    east = amp.sel(m=-1).values
    np.testing.assert_allclose(np.median(west), ds.attrs['_amp_true'], rtol=0.05)
    assert np.median(east) < 0.01 * ds.attrs['_amp_true']


def test_no_sym_asy_is_honoured_by_the_default_method(westward_dw1_ds):
    """`--no-sym-asy` was accepted and ignored on the `ls` path.

    The flag was only ever forwarded to the wavelet analysis, while the CLI
    defaults to least squares, so the output kept its symmetric/antisymmetric
    split and nothing said so. Reading `amp_sym` as the field is then
    silently wrong.
    """
    out = compute_leastsquares_tidal_analysis(
        westward_dw1_ds, 'temp', [24.0], [1], time_dim='lst',
        decompose_sym_asy=False)
    assert 'temp_amp_total' in out and 'temp_pha_total' in out
    assert 'temp_amp_sym' not in out
    assert out['temp_amp_total'].attrs['long_name'] == 'Total Amplitude'


def test_sym_and_asy_sum_to_the_total(westward_dw1_ds):
    """The two parts recombine as complex numbers, which is what makes
    `amp_sym` alone the wrong thing to read."""
    ds = westward_dw1_ds
    split = compute_leastsquares_tidal_analysis(
        ds, 'temp', [24.0], [1], time_dim='lst')
    whole = compute_leastsquares_tidal_analysis(
        ds, 'temp', [24.0], [1], time_dim='lst', decompose_sym_asy=False)

    recombined = (split['temp_amp_sym'] * np.exp(1j * split['temp_pha_sym'])
                  + split['temp_amp_asy'] * np.exp(1j * split['temp_pha_asy']))
    reference = whole['temp_amp_total'] * np.exp(1j * whole['temp_pha_total'])
    np.testing.assert_allclose(recombined.values, reference.values, atol=1e-10)

    # and the part is not the whole, so the distinction is worth making
    assert not np.allclose(split['temp_amp_sym'].values,
                           whole['temp_amp_total'].values, atol=1e-3)


def test_mode_phase_carries_minus_2m_lambda(westward_dw1_ds):
    """The rotation in `get_phase` has the wrong sign, and this pins it.

    Rotating by exp(-i*m*phi) is meant to leave a mode phase that does not
    depend on longitude. For every field the filter actually produces,
    arg(cos + i sin) already goes as -m*phi, so the rotation doubles the
    dependence instead of removing it: d(phase)/d(lambda) = -2m. The stored
    phase is therefore not a Greenwich-referenced mode phase, and a mode is
    not constant along a latitude circle.

    Changing the sign would make it one, and would also change every file the
    package has written, so the behaviour is pinned here rather than altered.
    `test_local_coefficient_is_recovered` covers what callers actually need.
    """
    ds = westward_dw1_ds
    m = 1
    out = compute_leastsquares_tidal_analysis(
        ds, 'temp', [24.0], [m], time_dim='lst', decompose_sym_asy=False)

    nside = hp.npix2nside(ds.sizes['cells'])
    theta, phi = hp.pix2ang(nside, np.arange(ds.sizes['cells']))
    # one ring, not a band: a band repeats longitudes across rings and the
    # spacing between neighbours is then zero
    ring = theta == theta[np.argmin(np.abs(theta - np.pi / 2))]
    order = np.argsort(phi[ring])
    pha = out['temp_pha_total'].isel(period=0, m=0).values[ring][order]
    lam = phi[ring][order]

    # the median wrapped increment; summing them around a closed circle
    # cancels the wraps and reports zero whatever the slope really is
    slope = (np.median(np.angle(np.exp(1j * np.diff(pha))))
             / np.median(np.diff(lam)))
    np.testing.assert_allclose(slope, -2 * m, atol=1e-6)


def test_local_coefficient_is_recovered(westward_dw1_ds):
    """What a caller needs: amp * exp(i*(phase + m*lambda)) is the field.

    Checked against the analytic coefficients of the input rather than
    against another output of the same code.
    """
    ds = westward_dw1_ds
    m, amp_true, phase_true = 1, ds.attrs['_amp_true'], 0.5
    out = compute_leastsquares_tidal_analysis(
        ds, 'temp', [24.0], [m], time_dim='lst', decompose_sym_asy=False)

    nside = hp.npix2nside(ds.sizes['cells'])
    theta, phi = hp.pix2ang(nside, np.arange(ds.sizes['cells']))
    pha = out['temp_pha_total'].isel(period=0, m=0).values
    amp = out['temp_amp_total'].isel(period=0, m=0).values

    # the phase of a vanishing amplitude is arbitrary, so judge it only where
    # there is something to judge
    lat_shape = 1.0 + 0.6 * np.cos(theta)
    strong = amp > 0.2 * amp.max()

    # signal = A(lat) cos(m*phi + omega*t + p) fits as
    # cos coefficient  A cos(m*phi + p), sin coefficient  -A sin(m*phi + p)
    arg_true = -(m * phi + phase_true)
    err = np.abs(np.angle(np.exp(1j * (pha + m * phi - arg_true))))[strong]
    assert err.max() < 1e-6, f"max phase error {err.max():.2e} rad"

    # 8% rather than 5%: the directional filter works through spherical
    # harmonics truncated at lmax, which does not reproduce the analytic
    # latitude shape exactly at the few most extreme pixels
    np.testing.assert_allclose(amp[strong], (amp_true * lat_shape)[strong],
                               rtol=0.08)

    assert 'exp(+i*m*lambda)' in out['temp_pha_total'].attrs['comment']
