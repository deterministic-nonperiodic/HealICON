import os
import tempfile
import numpy as np
import xarray as xr
from healicon.core import run_sequential

def create_mock_saber_file(filename, events, altitudes):
    # Create mock coordinates
    # shape: (event, altitude)
    lat = np.full((events, altitudes), 45.0)
    lon = np.full((events, altitudes), 0.0)
    ut_time = np.full((events, altitudes), 12.0 * 3600000.0) # noon in msec
    
    # Create some mock data
    temp = np.random.rand(events, altitudes)
    
    ds = xr.Dataset(
        data_vars=dict(
            tplatitude=(["event", "altitude"], lat),
            tplongitude=(["event", "altitude"], lon),
            time=(["event", "altitude"], ut_time),
            ktemp=(["event", "altitude"], temp)
        ),
        attrs=dict(
            Mission="TIMED",
            Title="SABER mock dataset"
        )
    )
    ds.to_netcdf(filename)
    return ds

def test_run_sequential_cat():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "SABER_1.nc")
        file2 = os.path.join(tmpdir, "SABER_2.nc")
        
        # We will test the SABER pad logic. File 1 has 10 altitudes, File 2 has 8 altitudes.
        create_mock_saber_file(file1, events=5, altitudes=10)
        create_mock_saber_file(file2, events=3, altitudes=8)
        
        pattern = os.path.join(tmpdir, "SABER_*.nc")
        out_file = os.path.join(tmpdir, "combined.nc")
        
        run_sequential(
            input_pattern=pattern,
            output_template=out_file,
            nside=2,
            cat=True,
            ut_bins=24
        )
        
        assert os.path.exists(out_file)
        
        # Load the combined output and check its dimensions
        out_ds = xr.open_dataset(out_file)
        
        assert "cells" in out_ds.dims
        assert 'ut' in out_ds.dims
        assert "altitude" in out_ds.dims
        
        # The output altitude dimension should be the maximum of the input datasets (10)
        assert out_ds.sizes["altitude"] == 10
        
        out_ds.close()


# --- write_dataset tests ---
import pytest
from healicon.io_utils import write_dataset, _resolve_store_and_path
from pathlib import Path


def _make_simple_ds():
    """Create a small in-memory Dataset for I/O testing."""
    data = np.random.rand(5, 10).astype("float32")
    return xr.Dataset({"temperature": (["z", "cells"], data)})


def test_resolve_store_and_path_netcdf():
    path, store = _resolve_store_and_path("/tmp/out.nc")
    assert store == "netcdf"
    assert str(path).endswith(".nc")


def test_resolve_store_and_path_zarr():
    path, store = _resolve_store_and_path("/tmp/out.zarr")
    assert store == "zarr"
    assert str(path).endswith(".zarr")


def test_resolve_store_and_path_unknown_extension_defaults_netcdf():
    path, store = _resolve_store_and_path("/tmp/out.bin")
    assert store == "netcdf"


def test_resolve_store_and_path_override():
    # Extension says .nc but store_type forces zarr
    path, store = _resolve_store_and_path("/tmp/out.nc", store_type="zarr")
    assert store == "zarr"
    assert str(path).endswith(".zarr")


def test_write_dataset_netcdf():
    ds = _make_simple_ds()
    with tempfile.TemporaryDirectory() as tmpdir:
        ofile = os.path.join(tmpdir, "output.nc")
        write_dataset(ds, ofile)
        assert os.path.exists(ofile)
        out = xr.open_dataset(ofile)
        assert "temperature" in out.data_vars
        out.close()


def test_write_dataset_zarr():
    from healicon.io_utils import _ZARR_AVAILABLE
    if not _ZARR_AVAILABLE:
        pytest.skip("zarr not installed")
    ds = _make_simple_ds()
    with tempfile.TemporaryDirectory() as tmpdir:
        ofile = os.path.join(tmpdir, "output.zarr")
        write_dataset(ds, ofile)
        assert os.path.isdir(ofile)
        out = xr.open_zarr(ofile, consolidated=False)
        assert "temperature" in out.data_vars
        out.close()


def test_write_dataset_overwrite_false_raises():
    ds = _make_simple_ds()
    with tempfile.TemporaryDirectory() as tmpdir:
        ofile = os.path.join(tmpdir, "output.nc")
        write_dataset(ds, ofile)
        # Write again without overwrite: should raise
        with pytest.raises(FileExistsError):
            write_dataset(ds, ofile, overwrite=False)


def test_write_dataset_overwrite_true_replaces():
    ds = _make_simple_ds()
    with tempfile.TemporaryDirectory() as tmpdir:
        ofile = os.path.join(tmpdir, "output.nc")
        write_dataset(ds, ofile)
        # Modify dataset and overwrite
        ds2 = _make_simple_ds()
        ds2["pressure"] = xr.DataArray(np.ones((5, 10), dtype="float32"), dims=["z", "cells"])
        write_dataset(ds2, ofile, overwrite=True)
        out = xr.open_dataset(ofile)
        assert "pressure" in out.data_vars
        out.close()


def test_write_dataset_dask_backed_netcdf():
    """Test write with a dask-backed dataset (simulating large file workflow)."""
    ds = _make_simple_ds()
    ds_dask = ds.chunk({"z": 2, "cells": 5})
    with tempfile.TemporaryDirectory() as tmpdir:
        ofile = os.path.join(tmpdir, "output.nc")
        write_dataset(ds_dask, ofile)
        assert os.path.exists(ofile)
        out = xr.open_dataset(ofile)
        assert "temperature" in out.data_vars
        out.close()


def test_run_sequential_order_options():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple unstructured dataset (like mock ICON)
        lat = np.array([45.0, -45.0, 0.0])
        lon = np.array([0.0, 180.0, 90.0])
        data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
        
        ds = xr.Dataset(
            {'temp': (['cells'], data)},
            coords={'lat': (['cells'], lat), 'lon': (['cells'], lon)}
        )
        ds.lon.attrs = {"standard_name": "longitude", "units": "degrees_east"}
        ds.lat.attrs = {"standard_name": "latitude", "units": "degrees_north"}
        
        ifile = os.path.join(tmpdir, "input.nc")
        ds.to_netcdf(ifile)
        ds.close()
        
        # 1. Convert to RING
        ofile_ring = os.path.join(tmpdir, "output_ring.nc")
        run_sequential(
            input_pattern=ifile,
            output_template=ofile_ring,
            nside=2,
            order='ring'
        )
        
        ds_ring = xr.open_dataset(ofile_ring)
        assert ds_ring.attrs.get('healpix_scheme') == 'RING'
        assert ds_ring['healpix'].attrs.get('healpix_order') == 'ring'
        ds_ring.close()
        
        # 2. Convert to NESTED
        ofile_nested = os.path.join(tmpdir, "output_nested.nc")
        run_sequential(
            input_pattern=ifile,
            output_template=ofile_nested,
            nside=2,
            order='nested'
        )
        
        ds_nested = xr.open_dataset(ofile_nested)
        assert ds_nested.attrs.get('healpix_scheme') == 'NESTED'
        assert ds_nested['healpix'].attrs.get('healpix_order') == 'nested'
        ds_nested.close()
        
        # 3. Convert from RING output to NESTED (reordering)
        ofile_reordered = os.path.join(tmpdir, "output_reordered.nc")
        run_sequential(
            input_pattern=ofile_ring,
            output_template=ofile_reordered,
            nside=2,
            order='nested'
        )
        ds_reordered = xr.open_dataset(ofile_reordered)
        assert ds_reordered.attrs.get('healpix_scheme') == 'NESTED'
        assert ds_reordered['healpix'].attrs.get('healpix_order') == 'nested'
        ds_reordered.close()
