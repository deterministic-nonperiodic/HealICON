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


def test_gravity_falls_with_height():
    """g(z) must decline, and match the closed form where a height is given."""
    import numpy as np
    import xarray as xr
    from healicon.analysis.ep_flux import _resolve_gravity, _G, _RE

    z = np.array([0.0, 30e3, 70e3, 90e3])
    ds = xr.Dataset({"z_geom": ("height", z)},
                    coords={"height": z})
    ds.height.attrs.update(standard_name="height", units="m")
    g = np.asarray(_resolve_gravity(ds))

    expect = _G * (_RE / (_RE + z)) ** 2
    assert np.allclose(g, expect, rtol=1e-12)
    assert g[0] > g[-1]
    # the corrections the constant hides
    assert 0.020 < 1 - g[2] / _G < 0.022          # ~2.1% at 70 km
    assert 0.027 < 1 - g[3] / _G < 0.029          # ~2.8% at 90 km


def test_gravity_falls_back_to_standard_without_a_height():
    """No height resolvable: behaviour is exactly what it was before."""
    import numpy as np
    import xarray as xr
    from healicon.analysis.ep_flux import _resolve_gravity, _G

    ds = xr.Dataset({"u_zm": ("lat", np.zeros(8))},
                    coords={"lat": np.linspace(-80, 80, 8)})
    assert float(np.asarray(_resolve_gravity(ds))) == _G


def test_eddy_covariances_reach_the_output():
    """F_phi and F_z are each a difference of two terms; keep the terms.

    A difference in div_F says nothing on its own about which physics moved:
    [u'v'] is meridional convergence of momentum, [v'theta'] is vertical
    propagation. Dropping them left the output undecomposable, and [u'v'] is
    a momentum flux worth having in its own right.
    """
    import numpy as np
    import xarray as xr
    import healpy as hp
    from healicon.analysis.ep_flux import eliassen_palm
    from healicon.grid import create_healpix_dataset

    nside = 8
    ds = create_healpix_dataset(nside)
    npix, nlev = hp.nside2npix(nside), 6
    rng = np.random.default_rng(0)
    plev = np.array([9e4, 7e4, 5e4, 3e4, 1e4, 1e3])
    for name, scale in (("u", 20.0), ("v", 5.0), ("temp", 250.0)):
        ds[name] = (("plev", "cells"),
                    scale + rng.normal(0, 1.0, (nlev, npix)))
    ds = ds.assign_coords(plev=("plev", plev))
    ds.plev.attrs.update(standard_name="air_pressure", units="Pa", axis="Z")

    out = eliassen_palm(ds, mode="qg")
    for v in ("upvp_zm", "vptp_zm", "theta_zm"):
        assert v in out, f"{v} missing from the EP flux output"
    assert out["upvp_zm"].attrs["units"] == "m2 s-2"
    assert out["vptp_zm"].attrs["units"] == "K m s-1"


def test_omega_without_w():
    """A reanalysis on pressure levels carries omega and no vertical velocity.

    The two are independent inputs. Bundling them raised UnboundLocalError on
    w_prime the first time a dataset supplied omega alone -- which is the usual
    shape of the data this path exists for.
    """
    import numpy as np
    import xarray as xr
    import healpy as hp
    from healicon.analysis.ep_flux import eliassen_palm
    from healicon.grid import create_healpix_dataset

    nside = 8
    ds = create_healpix_dataset(nside)
    npix, nlev = hp.nside2npix(nside), 6
    rng = np.random.default_rng(1)
    plev = np.array([9e4, 7e4, 5e4, 3e4, 1e4, 1e3])
    for name, scale in (("u", 20.0), ("v", 5.0), ("temp", 250.0)):
        ds[name] = (("plev", "cells"), scale + rng.normal(0, 1.0, (nlev, npix)))
    ds["omega"] = (("plev", "cells"), rng.normal(0, 1e-3, (nlev, npix)))
    ds = ds.assign_coords(plev=("plev", plev))
    ds.plev.attrs.update(standard_name="air_pressure", units="Pa", axis="Z")

    out = eliassen_palm(ds, mode="full")          # omega alone must enable full
    assert "upomega_zm" in out
    assert "w_zm" not in out                       # there is no w to report
    assert np.isfinite(out["F_z"]).any()


def test_vertical_axis_must_be_a_dimension():
    """An auxiliary pressure coordinate is not the vertical axis.

    ICON-like output on height levels carries `z_mc` as the dimension in metres
    and each level's pressure as an auxiliary coordinate alongside it. Picking
    the auxiliary one made the height path unusable: everything downstream
    differentiates and chunks along the name returned here, and an auxiliary
    coordinate can carry neither.
    """
    import numpy as np
    import xarray as xr
    from healicon.analysis.ep_flux import _find_alt_name

    z = np.linspace(1e3, 1.4e5, 20)
    ds = xr.Dataset(
        {"u": (("z_mc", "lat"), np.zeros((20, 8)))},
        coords={"z_mc": ("z_mc", z),
                "plev": ("z_mc", 1e5 * np.exp(-z / 7e3)),   # auxiliary
                "lat": np.linspace(-80, 80, 8)},
    )
    ds.z_mc.attrs.update(standard_name="height", units="m", axis="Z", positive="up")

    got = _find_alt_name(ds)
    assert got == "z_mc", f"expected the height dimension, got {got!r}"
    assert got in ds.dims


def test_tem_mode_without_w_pressure_and_height():
    """mode='tem' retains absolute vorticity and shear while dropping vertical eddy flux.

    It must execute cleanly without w in both pressure and height coordinates,
    and differ from mode='qg' by retaining the vorticity factor and shear term.
    """
    from healicon.grid import create_healpix_dataset

    nside = 8
    npix, nlev = hp.nside2npix(nside), 6
    rng = np.random.default_rng(42)

    # 1. Pressure coordinates without w
    ds_pres = create_healpix_dataset(nside)
    plev = np.array([9e4, 7e4, 5e4, 3e4, 1e4, 1e3])
    lats = 90.0 - np.rad2deg(hp.pix2ang(nside, np.arange(npix))[0])
    # Build u with strong meridional shear so f_hat != f
    u_shear_pattern = 30.0 * np.sin(np.deg2rad(lats))[None, :]
    for name, scale in (("v", 5.0), ("temp", 250.0)):
        ds_pres[name] = (("plev", "cells"), scale + rng.normal(0, 1.0, (nlev, npix)))
    ds_pres["u"] = (("plev", "cells"), u_shear_pattern + rng.normal(0, 1.0, (nlev, npix)))
    ds_pres = ds_pres.assign_coords(plev=("plev", plev))
    ds_pres.plev.attrs.update(standard_name="air_pressure", units="Pa", axis="Z")

    out_tem = eliassen_palm(ds_pres, mode="tem")
    out_qg = eliassen_palm(ds_pres, mode="qg")

    assert out_tem.attrs["ep_flux_mode"] == "tem"
    assert out_qg.attrs["ep_flux_mode"] == "qg"
    assert "upwp_zm" not in out_tem
    assert "upomega_zm" not in out_tem
    assert np.isfinite(out_tem["F_phi"]).all()
    assert np.isfinite(out_tem["F_z"]).all()
    # In tem mode, f_hat includes vorticity and F_phi includes shear, so values differ from QG
    assert not np.allclose(out_tem["F_z"].values, out_qg["F_z"].values)
    assert not np.allclose(out_tem["F_phi"].values, out_qg["F_phi"].values)

    # 2. Height coordinates without w (w dropped from _make_ds)
    ds_hgt = _make_ds(seed=42).drop_vars("w")
    out_hgt_tem = eliassen_palm(ds_hgt, mode="tem", vertical="native")
    assert out_hgt_tem.attrs["ep_flux_mode"] == "tem"
    assert "Psi" in out_hgt_tem
    assert np.isfinite(out_hgt_tem["Psi"]).all()
    assert np.isfinite(out_hgt_tem["F_z"]).all()


def test_auto_mode_selection():
    """mode='auto' picks 'full' when w is present, and 'tem' when w is absent."""
    ds_with_w = _make_ds(seed=10)
    out_full = eliassen_palm(ds_with_w, mode="auto", vertical="native")
    assert out_full.attrs["ep_flux_mode"] == "full"

    ds_no_w = ds_with_w.drop_vars("w")
    out_tem = eliassen_palm(ds_no_w, mode="auto", vertical="native")
    assert out_tem.attrs["ep_flux_mode"] == "tem"

    # Explicit full must reject dataset without w or omega
    with pytest.raises(ValueError, match="mode='full' requires 'upwp_zm'"):
        eliassen_palm(ds_no_w, mode="full", vertical="native")


def test_all_eddy_covariances_and_vorticity_passthrough():
    """When w, vor, and thermodynamics are present, all covariances pass through."""
    ds = _make_ds(seed=20)
    rng = np.random.default_rng(20)
    ds["vor"] = xr.DataArray(
        rng.standard_normal((_NLEV, _NPIX)).astype("f4") * 1e-5,
        dims=["height", "cell"],
        attrs={"units": "s-1", "standard_name": "relative_vorticity"},
    )
    out = eliassen_palm(ds, mode="full", vertical="native")

    expected_vars = [
        "u_zm", "v_zm", "w_zm", "pres_zm", "temp_zm", "theta_zm",
        "upvp_zm", "vptp_zm", "upwp_zm", "wptp_zm", "vor_zm",
    ]
    for var in expected_vars:
        assert var in out, f"Expected {var} in EP flux output"

    assert out["upvp_zm"].attrs["units"] == "m2 s-2"
    assert out["vptp_zm"].attrs["units"] == "K m s-1"
    assert out["upwp_zm"].attrs["units"] == "m2 s-2"
    assert out["wptp_zm"].attrs["units"] == "K m s-1"
    assert out["vor_zm"].attrs["units"] == "s-1"


def test_gravity_geopotential_and_log_pressure():
    """_resolve_gravity correctly resolves geopotential height and log-p heights."""
    from healicon.analysis.ep_flux import _resolve_gravity, _G, _RE

    # 1. Geopotential height in m2 s-2
    zg_m = 50e3
    phi_val = zg_m * _G
    ds_phi = xr.Dataset({"geopotential": ("level", [phi_val])},
                        coords={"level": [1]})
    ds_phi.geopotential.attrs["units"] = "m2 s-2"
    g_phi = float(np.asarray(_resolve_gravity(ds_phi)).flat[0])
    z_geom = _RE * zg_m / (_RE - zg_m)
    expected_g = _G * (_RE / (_RE + z_geom)) ** 2
    assert np.isclose(g_phi, expected_g, rtol=1e-6)

    # 2. Geopotential height in metres (zg / gpm)
    ds_zg = xr.Dataset({"zg": ("level", [zg_m])},
                       coords={"level": [1]})
    ds_zg.zg.attrs["units"] = "m"
    g_zg = float(np.asarray(_resolve_gravity(ds_zg)).flat[0])
    assert np.isclose(g_zg, expected_g, rtol=1e-6)

    # 3. Log-pressure fallback from pressure and temperature
    p = np.array([1e5, 1e4, 1e2, 1e0])  # Pa
    ds_logp = xr.Dataset(
        {"temp_zm": (("plev", "lat"), 250.0 * np.ones((4, 6)))},
        coords={"plev": ("plev", p), "lat": np.linspace(-60, 60, 6)},
    )
    ds_logp.plev.attrs.update(standard_name="air_pressure", units="Pa", axis="Z")
    g_logp = np.asarray(_resolve_gravity(ds_logp))
    assert (g_logp[0] > g_logp[-1]).all()
    assert g_logp[0, 0] == pytest.approx(_G, rel=1e-4)


def test_ep_flux_auxiliary_plev_end_to_end():
    """ICON-like vertical coordinate with height dimension and auxiliary plev."""
    from healicon.grid import create_healpix_dataset

    nside = 8
    ds = create_healpix_dataset(nside)
    npix, nlev = hp.nside2npix(nside), 10
    z = np.linspace(1e3, 5e4, nlev)
    plev_aux = 1e5 * np.exp(-z / 7e3)
    rng = np.random.default_rng(33)

    for name, scale in (("u", 25.0), ("v", 5.0), ("w", 0.02), ("temp", 240.0)):
        ds[name] = (("z_mc", "cells"), scale + rng.normal(0, 1.0, (nlev, npix)))

    ds = ds.assign_coords(
        z_mc=("z_mc", z),
        plev=("z_mc", plev_aux),
    )
    ds.z_mc.attrs.update(standard_name="height", units="m", axis="Z", positive="up")
    ds.plev.attrs.update(standard_name="air_pressure", units="Pa")

    out = eliassen_palm(ds, mode="full", vertical="native")
    assert "z_mc" in out.dims
    assert "div_F" in out
    assert "a_EP" in out
    assert np.isfinite(out["a_EP"]).any()


def test_ep_flux_cli(tmp_path):
    """Test CLI commands 'ep-flux' and 'epflux' end-to-end."""
    from click.testing import CliRunner
    from healicon.cli import cli

    ds = _make_ds(seed=99)
    ifile = tmp_path / "ep_input.nc"
    ds.to_netcdf(ifile)

    runner = CliRunner()

    # 1. Test ep-flux with --mode tem
    ofile_tem = tmp_path / "ep_out_tem.nc"
    res_tem = runner.invoke(cli, [
        "ep-flux",
        str(ifile),
        str(ofile_tem),
        "--mode", "tem",
    ])
    assert res_tem.exit_code == 0, res_tem.output
    assert ofile_tem.exists()
    ds_tem = xr.open_dataset(ofile_tem)
    assert ds_tem.attrs.get("ep_flux_mode") == "tem"
    assert "F_phi" in ds_tem and "F_z" in ds_tem and "a_EP" in ds_tem
    ds_tem.close()

    # 2. Test epflux alias with --mode full and --time-mean
    ofile_full = tmp_path / "ep_out_full.nc"
    res_full = runner.invoke(cli, [
        "epflux",
        str(ifile),
        str(ofile_full),
        "--mode", "full",
        "--time-mean",
    ])
    assert res_full.exit_code == 0, res_full.output
    assert ofile_full.exists()
    ds_full = xr.open_dataset(ofile_full)
    assert ds_full.attrs.get("ep_flux_mode") == "full"
    ds_full.close()



# ---------------------------------------------------------------------------
# Analytic verification of the divergence discretisation
# ---------------------------------------------------------------------------
# The tests above check plumbing: units, dimensions, which variables survive.
# None of them would notice a dropped cosφ or a flipped sign in
#
#     div F = (a cosφ)⁻¹ ∂(F_φ cosφ)/∂φ + ∂F_z/∂{p,z}
#
# These do. Each case prescribes zonal-mean covariances for which F, div F and
# a_EP are exactly integrable, and compares. The inputs are not a balanced
# atmosphere and do not need to be: what is under test is the discretisation,
# not the physics that would produce such covariances.
#
# Two regimes. A term that is linear or quadratic in the differentiated
# variable is integrated exactly by a three-point stencil on any spacing, so
# it must come back at machine precision. A term in cosⁿφ is not, so it must
# converge at second order. Both are asserted; a change that silently drops
# to first order fails here rather than in a paper three years later.
#
# Constants come from the module under test. Using independent values turns a
# constant mismatch into an apparent truncation error -- Ω differing in the
# sixth digit shows up as a uniform 2e-6, which reads exactly like a
# discretisation failure and is not one.

_THETA0, _BETA = 300.0, 3.0e-4      # θ = θ₀ + βp; β stays above the 1e-10 floor
_GAMMA, _T_ISO, _RHO_C = 5.0e-3, 250.0, 1.0e-3
_P_REF, _Z_REF = 1.0e5, 1.0e5


def _gauss_lats(n):
    """Latitudes of an n-point Gaussian grid, as the reanalysis archives use."""
    x, _ = np.polynomial.legendre.leggauss(n)
    return np.rad2deg(np.arcsin(x))


def _zm_dataset(lat, vert, vert_name, is_pres, **fields):
    """Assemble the zonal-mean dataset compute_ep_flux consumes."""
    coords = {
        "lat": ("lat", lat, {"units": "degrees_north", "axis": "Y"}),
        vert_name: (vert_name, vert, {
            "units": "Pa" if is_pres else "m",
            "standard_name": "air_pressure" if is_pres else "height",
            "axis": "Z", "positive": "down" if is_pres else "up"}),
    }
    ds = xr.Dataset(coords=coords)
    LAT, VERT = np.meshgrid(lat, vert, indexing="ij")
    for name, fn in fields.items():
        ds[name] = (("lat", vert_name), np.asarray(fn(LAT, VERT), float))
    ds.attrs["ep_flux_coord"] = "pressure" if is_pres else "height"
    return ds


def _run(ds):
    from healicon.analysis.ep_flux import compute_ep_divergence, compute_ep_flux
    return compute_ep_divergence(compute_ep_flux(ds), mask_invalid=False)


def _rel_err(got, want, lat, trim=2, latmax=85.0):
    """Max and RMS error over the interior, normalised by the analytic scale.

    The first and last rows use a one-sided stencil, and a_EP carries a 1/cosφ
    that amplifies any error towards the pole without saying anything about
    the scheme, so both are excluded.
    """
    sl = slice(trim, -trim or None)
    g, w, la = np.asarray(got)[sl], np.asarray(want)[sl], np.asarray(lat)[sl]
    keep = np.abs(la) <= latmax
    g, w = g[keep], w[keep]
    scale = np.nanmax(np.abs(w))
    d = np.abs(g - w) / (scale if scale > 0 else 1.0)
    return float(np.nanmax(d)), float(np.sqrt(np.nanmean(d ** 2)))


def _case_meridional(lat, plev, upvp=12.0):
    """[u'v'] = C alone.  F_φ = -a C cosφ,  a_EP = (2C/a) tanφ · 86400.

    The cos²φ derivative is not exact under a three-point stencil: second order.
    """
    from healicon.analysis.ep_flux import _A, _SECS_PER_DAY
    ds = _zm_dataset(
        lat, plev, "plev", True,
        upvp_zm=lambda L, P: np.full_like(L, upvp),
        vptp_zm=lambda L, P: np.zeros_like(L),
        upomega_zm=lambda L, P: np.zeros_like(L),
        u_zm=lambda L, P: np.zeros_like(L),
        theta_zm=lambda L, P: _THETA0 + _BETA * P)
    phi = np.deg2rad(lat)
    ones = np.ones(plev.size)
    return ds, dict(
        F_phi=(-_A * upvp * np.cos(phi))[:, None] * ones,
        F_z=np.zeros((lat.size, plev.size)),
        a_EP=((2.0 * upvp / _A) * np.tan(phi) * _SECS_PER_DAY)[:, None] * ones)


def _case_vertical(lat, plev):
    """ū = 0 so f̂ = f; [v'θ']/θ_p = p²/2p_ref.  a_EP = f (p/p_ref) · 86400.

    Quadratic in p, so the stencil is exact on any spacing.
    """
    from healicon.analysis.ep_flux import _A, _OMEGA, _SECS_PER_DAY
    G = lambda P: P ** 2 / (2.0 * _P_REF)
    ds = _zm_dataset(
        lat, plev, "plev", True,
        upvp_zm=lambda L, P: np.zeros_like(L),
        vptp_zm=lambda L, P: _BETA * G(P),
        upomega_zm=lambda L, P: np.zeros_like(L),
        u_zm=lambda L, P: np.zeros_like(L),
        theta_zm=lambda L, P: _THETA0 + _BETA * P)
    phi = np.deg2rad(lat)
    f = 2.0 * _OMEGA * np.sin(phi)
    _, P = np.meshgrid(lat, plev, indexing="ij")
    return ds, dict(
        F_phi=np.zeros((lat.size, plev.size)),
        F_z=_A * np.cos(phi)[:, None] * f[:, None] * G(P),
        a_EP=f[:, None] * (P / _P_REF) * _SECS_PER_DAY)


def _case_all_terms(lat, plev, upvp=12.0, u0=40.0, v0=5.0):
    """Momentum flux, shear x heat flux, relative vorticity and dF_z/dp at once.

    ū = U₀ cosφ (p/p_ref), [u'v'] = C, [v'θ']/θ_p = V₀, so
        f̂    = 2Ω sinφ + (2U₀p/(a p_ref)) sinφ
        a_EP = [ -U₀V₀ sinφ/(a p_ref) + 2C tanφ/a ] · 86400
    """
    from healicon.analysis.ep_flux import _A, _OMEGA, _SECS_PER_DAY
    ds = _zm_dataset(
        lat, plev, "plev", True,
        upvp_zm=lambda L, P: np.full_like(L, upvp),
        vptp_zm=lambda L, P: np.full_like(L, _BETA * v0),
        upomega_zm=lambda L, P: np.zeros_like(L),
        u_zm=lambda L, P: u0 * np.cos(np.deg2rad(L)) * P / _P_REF,
        theta_zm=lambda L, P: _THETA0 + _BETA * P)
    LAT, P = np.meshgrid(lat, plev, indexing="ij")
    PHI = np.deg2rad(LAT)
    return ds, dict(
        F_phi=_A * np.cos(PHI) * (u0 * np.cos(PHI) * v0 / _P_REF - upvp),
        F_z=_A * np.cos(PHI) * v0 * (2 * _OMEGA * np.sin(PHI)
                                     + 2 * u0 * P * np.sin(PHI) / (_A * _P_REF)),
        a_EP=(-u0 * v0 * np.sin(PHI) / (_A * _P_REF)
              + 2 * upvp * np.tan(PHI) / _A) * _SECS_PER_DAY)


def _case_stream_function(lat, z):
    """Height branch through Ψ. θ depends on z only, so θ_φ = 0 and

        Ψ → a cosφ ρ₀ [v'θ']/θ_z        (Iwasaki 1989 / AHL87 eq. 3.5.3)

    With ū = [u'v'] = [u'w'] = [w'θ'] = 0 and [v'θ']/θ_z = z²/2z_ref,
    a_EP = f (z/z_ref) · 86400 -- quadratic in z, so exact.
    """
    from healicon.analysis.ep_flux import _A, _OMEGA, _RD, _SECS_PER_DAY
    H = lambda Z: Z ** 2 / (2.0 * _Z_REF)
    ds = _zm_dataset(
        lat, z, "height", False,
        upvp_zm=lambda L, Z: np.zeros_like(L),
        upwp_zm=lambda L, Z: np.zeros_like(L),
        wptp_zm=lambda L, Z: np.zeros_like(L),
        vptp_zm=lambda L, Z: _GAMMA * H(Z),
        u_zm=lambda L, Z: np.zeros_like(L),
        theta_zm=lambda L, Z: _THETA0 + _GAMMA * Z,
        temp_zm=lambda L, Z: np.full_like(L, _T_ISO),
        pres_zm=lambda L, Z: np.full_like(L, _RHO_C * _RD * _T_ISO))
    phi = np.deg2rad(lat)
    f = 2.0 * _OMEGA * np.sin(phi)
    _, Z = np.meshgrid(lat, z, indexing="ij")
    return ds, dict(
        F_phi=np.zeros((lat.size, z.size)),
        F_z=_A * np.cos(phi)[:, None] * _RHO_C * f[:, None] * H(Z),
        a_EP=f[:, None] * (Z / _Z_REF) * _SECS_PER_DAY)


@pytest.mark.parametrize("name,case,coord", [
    ("vertical term, pressure", _case_vertical, "plev"),
    ("stream function, height", _case_stream_function, "height"),
])
def test_exactly_integrable_terms_are_exact(name, case, coord):
    """Terms quadratic in the differentiated variable must return exactly.

    This is the strongest statement available: no tolerance to tune, and it
    pins dF_z/d{p,z}, f̂, Ψ and the a_EP normalisation -- div F/(a cosφ) on
    pressure, div F/(ρ₀ a cosφ) on height -- all at once. A missing ρ₀ in the
    height denominator, or a cosφ applied twice, moves this off zero.
    """
    lat = _gauss_lats(48)
    vert = (np.exp(np.linspace(np.log(1e5), np.log(1e-2), 40)) if coord == "plev"
            else np.linspace(0.0, 1.4e5, 40))
    ds, exact = case(lat, vert)
    out = _run(ds)
    for var in ("F_phi", "F_z", "a_EP"):
        mx, _ = _rel_err(out[var], exact[var], lat)
        assert mx < 1e-10, f"{name}: {var} off by {mx:.2e}, expected machine precision"


@pytest.mark.parametrize("name,case", [
    ("meridional term", _case_meridional),
    ("all terms", _case_all_terms),
])
def test_cos_phi_terms_converge_at_second_order(name, case):
    """Terms in cosⁿφ must converge at the order of the stencil.

    _gradient_1d is xarray's differentiate(edge_order=2), i.e. second-order
    centred. Dropping to first order -- a one-sided difference slipped into
    the interior, say -- halves nothing visibly on one grid but shows up
    immediately in the refinement rate.

    A uniform pressure grid is used deliberately. On a 7-decade log grid the
    all-terms error floors at ~1e-6 in the top few levels, where the
    synthetic F_z is a large constant plus a part 1e-8 of it and the dp
    derivative cancels catastrophically. That is a property of these
    contrived inputs, not of the scheme, and it would make the refinement
    rate meaningless.
    """
    plev = np.linspace(2e4, 1e5, 40)
    rms = []
    for nlat in (64, 128, 256):
        lat = _gauss_lats(nlat)
        ds, exact = case(lat, plev)
        rms.append(_rel_err(_run(ds)["a_EP"], exact["a_EP"], lat)[1])
    orders = [np.log2(a / b) for a, b in zip(rms, rms[1:])]
    assert all(o > 1.8 for o in orders), f"{name}: convergence orders {orders}"
    assert rms[-1] < 1e-4, f"{name}: rms {rms[-1]:.2e} at the finest grid"


def test_discretisation_error_is_negligible_on_a_reanalysis_grid():
    """The error that reaches a published band mean, in physical units.

    Convergence order says the scheme is right; this says the residual does
    not matter at the resolution the archives are actually on. The setup is
    the 127 x 124 grid of the JAWARA comparison, with [u'v'] scaled so a_EP
    reaches 15 m s⁻¹ d⁻¹, and the reduction is the cosine-weighted 45-75°N
    mean that the numbers are quoted from.
    """
    from healicon.analysis.ep_flux import _A, _SECS_PER_DAY
    lat = _gauss_lats(127)
    plev = np.exp(np.linspace(np.log(9.97e4), np.log(1.02e-4), 124))
    upvp = 15.0 * _A / (2 * np.tan(np.deg2rad(60.0)) * _SECS_PER_DAY)

    ds, exact = _case_meridional(lat, plev, upvp=upvp)
    got, want = np.asarray(_run(ds)["a_EP"]), exact["a_EP"]

    band = (lat >= 45.0) & (lat <= 75.0)
    weights = np.cos(np.deg2rad(lat[band]))
    mean = lambda x: np.average(x[band], axis=0, weights=weights)
    band_error = np.nanmax(np.abs(mean(got) - mean(want)))

    assert np.nanmax(np.abs(got - want)[band]) < 0.05
    # two orders below the smallest model-reanalysis difference reported,
    # and three below its confidence interval
    assert band_error < 0.02, f"band-mean error {band_error:.3e} m s-1 d-1"
