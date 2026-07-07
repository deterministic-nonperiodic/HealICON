"""Tests for Eliassen-Palm flux analysis (healicon/analysis/ep_flux.py).

Unit conversion is exercised via healicon.cf_coords.convert_units /
check_convert_units so that the tests are realistic end-to-end: input data are
produced in non-standard units and then normalised before (or implicitly during)
the EP flux computation via the check_convert_units call inside
compute_eddy_fluxes.
"""
import numpy as np
import xarray as xr
import healpy as hp
import pytest

from healicon.analysis.ep_flux import eliassen_palm
from healicon.cf_coords import convert_units, check_convert_units, equivalent_units

# ---------------------------------------------------------------------------
# Shared synthetic dataset
# ---------------------------------------------------------------------------

_NSIDE = 8
_NPIX  = hp.nside2npix(_NSIDE)
_NLEV  = 20
_Z     = np.linspace(10e3, 100e3, _NLEV)        # geometric height [m], sfc→top
_PRES  = np.exp(-_Z / 7e3) * 1e5               # reference pressure profile [Pa]


def _lats() -> np.ndarray:
    theta_ring, _ = hp.pix2ang(_NSIDE, np.arange(_NPIX))
    return 90.0 - np.rad2deg(theta_ring)


def _make_ds(pres_units: str = 'Pa', seed: int = 0) -> xr.Dataset:
    """Synthetic HEALPix + geometric-height dataset for EP flux testing.

    Parameters
    ----------
    pres_units : 'Pa' | 'hPa'
        Unit in which the ``pres`` variable is delivered.  The hPa variant is
        produced from the canonical Pa array via :func:`~healicon.cf_coords.convert_units`,
        guaranteeing bit-exact consistency between the two forms.
    seed :
        RNG seed for reproducibility.
    """
    rng  = np.random.default_rng(seed)
    lats = _lats()

    pres_2d  = (_PRES[:, None] * np.ones(_NPIX)[None, :]).astype('f4')  # (nlev, npix)
    temp_2d  = (250 + rng.standard_normal((_NLEV, _NPIX)) * 5).astype('f4')
    theta_2d = (300 + 5 * np.sin(np.deg2rad(lats))[None, :]
                + 2 * _Z[:, None] / 1e3).astype('f4')

    pres_da = xr.DataArray(
        pres_2d, dims=['height', 'cell'],
        attrs={'units': 'Pa', 'long_name': 'Pressure'},
    )
    if pres_units == 'hPa':
        pres_da = convert_units(pres_da, 'Pa', 'hPa')

    return xr.Dataset(
        {
            'u': xr.DataArray(
                rng.standard_normal((_NLEV, _NPIX)).astype('f4'),
                dims=['height', 'cell'],
                attrs={'units': 'm s-1', 'long_name': 'Zonal wind'}),
            'v': xr.DataArray(
                rng.standard_normal((_NLEV, _NPIX)).astype('f4'),
                dims=['height', 'cell'],
                attrs={'units': 'm s-1', 'long_name': 'Meridional wind'}),
            'w': xr.DataArray(
                (rng.standard_normal((_NLEV, _NPIX)) * 0.01).astype('f4'),
                dims=['height', 'cell'],
                attrs={'units': 'm s-1', 'long_name': 'Vertical velocity'}),
            'theta': xr.DataArray(
                theta_2d, dims=['height', 'cell'],
                attrs={'units': 'K', 'long_name': 'Potential temperature'}),
            'temp': xr.DataArray(
                temp_2d, dims=['height', 'cell'],
                attrs={'units': 'K', 'long_name': 'Temperature'}),
            'pres': pres_da,
        },
        coords={
            'height': xr.DataArray(
                _Z, dims=['height'],
                attrs={'units': 'm', 'axis': 'Z', 'standard_name': 'altitude'}),
            'cell': xr.DataArray(np.arange(_NPIX), dims=['cell']),
        },
    )


# ---------------------------------------------------------------------------
# 1. convert_units / check_convert_units correctness
# ---------------------------------------------------------------------------

class TestConvertUnits:
    def test_hpa_to_pa_multiplier(self):
        da = xr.DataArray([1.0, 10.0, 1000.0], attrs={'units': 'hPa'})
        result = convert_units(da, 'hPa', 'Pa')
        np.testing.assert_allclose(result.values, [100.0, 1000.0, 100_000.0])
        assert equivalent_units(result.attrs['units'], 'Pa')

    def test_pa_to_hpa_multiplier(self):
        da = xr.DataArray([100.0, 1000.0, 100_000.0], attrs={'units': 'Pa'})
        result = convert_units(da, 'Pa', 'hPa')
        np.testing.assert_allclose(result.values, [1.0, 10.0, 1000.0])
        assert equivalent_units(result.attrs['units'], 'hPa')

    def test_equivalent_units_passthrough(self):
        da = xr.DataArray([1.0, 2.0], attrs={'units': 'Pa'})
        result = convert_units(da, 'Pa', 'Pa')
        assert result is da, "convert_units should return the same object when units are equivalent"

    def test_check_convert_units_normalises_hpa_pres(self):
        """check_convert_units should auto-convert pres from hPa → Pa."""
        ds = _make_ds(pres_units='hPa')
        assert equivalent_units(ds['pres'].attrs['units'], 'hPa')

        ds_conv = check_convert_units(ds)

        assert equivalent_units(ds_conv['pres'].attrs['units'], 'Pa')
        np.testing.assert_allclose(
            ds_conv['pres'].values,
            ds['pres'].values * 100.0,
            rtol=1e-5,
        )

    def test_check_convert_units_pa_passthrough(self):
        """check_convert_units must not alter variables that are already in standard units."""
        ds     = _make_ds(pres_units='Pa')
        pres_0 = ds['pres'].values.copy()
        ds_conv = check_convert_units(ds)
        np.testing.assert_array_equal(ds_conv['pres'].values, pres_0)

    def test_incompatible_units_raises(self):
        """convert_units must raise ValueError for incompatible unit pairs."""
        da = xr.DataArray([1.0], attrs={'units': 'Pa'})
        with pytest.raises(ValueError, match="incompatible"):
            convert_units(da, 'Pa', 'm s-1')


# ---------------------------------------------------------------------------
# 2. Structural / physical sanity of eliassen_palm output
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def ep_height():
    return eliassen_palm(_make_ds(), mode='full', vertical='native')


class TestEliassenPalmStructure:
    REQUIRED_VARS = ('F_phi', 'F_z', 'div_F', 'a_EP', 'rho0', 'Psi', 'v_star', 'w_star')

    def test_output_variables_present(self, ep_height):
        for var in self.REQUIRED_VARS:
            assert var in ep_height, f"Missing output variable: {var}"

    def test_2d_output_dims(self, ep_height):
        for var in ('F_phi', 'F_z', 'a_EP', 'Psi'):
            assert set(ep_height[var].dims) == {'lat', 'height'}, (
                f"'{var}' has unexpected dims: {ep_height[var].dims}")

    def test_rho0_everywhere_positive(self, ep_height):
        assert float(ep_height['rho0'].min()) > 0.0

    def test_units_attrs(self, ep_height):
        assert ep_height['Psi'].attrs.get('units')    == 'kg s-1'
        assert ep_height['v_star'].attrs.get('units') == 'm s-1'
        assert ep_height['w_star'].attrs.get('units') == 'm s-1'

    def test_no_nans_in_core_region(self, ep_height):
        """F_phi must be finite at mid-latitudes below 80 km."""
        lat_ok = ep_height['lat'].where(np.abs(ep_height['lat']) < 80, drop=True)
        hgt_ok = ep_height['height'].where(ep_height['height'] < 80e3, drop=True)
        core   = ep_height['F_phi'].sel(lat=lat_ok, height=hgt_ok)
        assert not np.any(np.isnan(core.values)), \
            "NaNs found in F_phi core region (|lat| < 80°, z < 80 km)"

    def test_valid_height_max_attr(self, ep_height):
        """a_EP should carry the validity-ceiling attribute for the plot layer."""
        assert 'valid_height_max_m' in ep_height['a_EP'].attrs


# ---------------------------------------------------------------------------
# 3. Unit-conversion invariance: Pa input ↔ hPa input
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def ep_pa():
    return eliassen_palm(_make_ds(pres_units='Pa'), mode='full', vertical='native')


@pytest.fixture(scope='module')
def ep_hpa():
    return eliassen_palm(_make_ds(pres_units='hPa'), mode='full', vertical='native')


class TestUnitConversionInvariance:
    """eliassen_palm must produce identical output regardless of the pressure
    unit in the input, because check_convert_units normalises it internally."""

    def test_F_phi_invariant(self, ep_pa, ep_hpa):
        np.testing.assert_allclose(
            ep_pa['F_phi'].values, ep_hpa['F_phi'].values,
            rtol=1e-4, atol=1e-30,
            err_msg="F_phi differs between Pa and hPa inputs",
        )

    def test_Psi_invariant(self, ep_pa, ep_hpa):
        np.testing.assert_allclose(
            ep_pa['Psi'].values, ep_hpa['Psi'].values,
            rtol=1e-4, atol=1e-30,
            err_msg="Psi (stream function) differs between Pa and hPa inputs",
        )

    def test_rho0_invariant(self, ep_pa, ep_hpa):
        np.testing.assert_allclose(
            ep_pa['rho0'].values, ep_hpa['rho0'].values,
            rtol=1e-4,
            err_msg="rho0 differs between Pa and hPa inputs",
        )

    def test_div_F_invariant(self, ep_pa, ep_hpa):
        np.testing.assert_allclose(
            ep_pa['div_F'].values, ep_hpa['div_F'].values,
            rtol=1e-4, atol=1e-30,
            err_msg="div_F differs between Pa and hPa inputs",
        )
