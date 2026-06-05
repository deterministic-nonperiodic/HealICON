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
