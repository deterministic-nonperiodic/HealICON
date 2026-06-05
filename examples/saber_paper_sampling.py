import xarray as xr
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_dd
from scipy.spatial import cKDTree
import glob
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    pattern = "/media/deterministic-nonperiodic/DATA/ORIGIN/SABER_Temp_O3_H2O_*2025_v2.0.nc"
    files = sorted(glob.glob(pattern))
    if not files:
        logger.error("No SABER files found.")
        sys.exit(1)
        
    logger.info(f"Loading {len(files)} files via mfdataset...")
    # Because of the altitude dim mismatch we handled earlier in core.py,
    # let's write a quick preprocessor here or just concatenate events manually to avoid crashing.
    max_alt = 0
    for f in files:
        with xr.open_dataset(f) as d:
            max_alt = max(max_alt, d.sizes.get('altitude', 0))
            
    def preprocess_saber(d):
        alt_size = d.sizes.get('altitude', 0)
        if alt_size < max_alt:
            return d.pad(altitude=(0, max_alt - alt_size), constant_values=np.nan)
        return d

    ds = xr.open_mfdataset(files, combine='nested', concat_dim='event', preprocess=preprocess_saber)
    
    logger.info("Extracting coordinates and variables...")
    # time is in Msec Since Midnight. We want hourly UT bins [0, 24)
    # wait, time in SABER is milliseconds since start of day. Yes.
    time_ms = ds['time'].values
    ut_hours = (time_ms / 3600000.0) % 24.0
    
    alt = ds['tpaltitude'].values
    lat = ds['tplatitude'].values
    lon = ds['tplongitude'].values
    ktemp = ds['ktemp'].values
    
    # Flatten arrays
    logger.info("Flattening arrays...")
    ut_flat = ut_hours.ravel()
    alt_flat = alt.ravel()
    lat_flat = lat.ravel()
    lon_flat = lon.ravel()
    ktemp_flat = ktemp.ravel()
    
    # Mask out invalid data
    logger.info("Masking invalid data...")
    mask = (
        (lat_flat != -999.0) & 
        (lon_flat != -999.0) & 
        (alt_flat != -999.0) & 
        (~np.isnan(lat_flat)) & 
        (~np.isnan(lon_flat)) & 
        (~np.isnan(alt_flat)) & 
        (~np.isnan(ut_flat)) & 
        (~np.isnan(ktemp_flat))
    )
    
    ut_valid = ut_flat[mask]
    alt_valid = alt_flat[mask]
    lat_valid = lat_flat[mask]
    lon_valid = lon_flat[mask]
    ktemp_valid = ktemp_flat[mask]
    
    logger.info(f"Total valid points: {len(ktemp_valid)}")
    
    # Define bins
    logger.info("Defining bins...")
    ut_bins = np.linspace(0, 24, 25) # 24 bins
    alt_bins = np.linspace(60, 110, 26) # 25 bins
    lat_bins = np.linspace(-90, 90, 37) # 36 bins
    lon_bins = np.linspace(0, 360, 25) # 24 bins
    
    logger.info("Running binned_statistic_dd...")
    ret = binned_statistic_dd(
        (ut_valid, alt_valid, lat_valid, lon_valid),
        ktemp_valid,
        statistic='mean',
        bins=[ut_bins, alt_bins, lat_bins, lon_bins]
    )
    
    binned_data = ret.statistic # shape: (24, 25, 36, 24)
    
    # Gap filling on regular grid
    logger.info("Gap filling empty bins...")
    # Coordinates of bin centers
    ut_cents = 0.5 * (ut_bins[:-1] + ut_bins[1:])
    alt_cents = 0.5 * (alt_bins[:-1] + alt_bins[1:])
    lat_cents = 0.5 * (lat_bins[:-1] + lat_bins[1:])
    lon_cents = 0.5 * (lon_bins[:-1] + lon_bins[1:])
    
    # Spatial KDTree gap filling per UT and Alt
    # Convert lat/lon to 3D Cartesian for KDTree
    lat_mesh, lon_mesh = np.meshgrid(np.deg2rad(lat_cents), np.deg2rad(lon_cents), indexing='ij')
    x = np.cos(lat_mesh) * np.cos(lon_mesh)
    y = np.cos(lat_mesh) * np.sin(lon_mesh)
    z = np.sin(lat_mesh)
    spatial_coords = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    for u in range(len(ut_cents)):
        for a in range(len(alt_cents)):
            grid_2d = binned_data[u, a, :, :] # (lat, lon)
            valid = ~np.isnan(grid_2d)
            if not np.any(valid) or np.all(valid):
                continue
                
            valid_flat = valid.ravel()
            missing_flat = ~valid_flat
            
            tree = cKDTree(spatial_coords[valid_flat])
            _, idx = tree.query(spatial_coords[missing_flat], workers=-1)
            
            grid_flat = grid_2d.ravel()
            grid_flat[missing_flat] = grid_flat[valid_flat][idx]
            binned_data[u, a, :, :] = grid_flat.reshape((36, 24))
            
    # Temporal linear interpolation
    logger.info("Temporal gap filling...")
    for a in range(len(alt_cents)):
        for la in range(len(lat_cents)):
            for lo in range(len(lon_cents)):
                ts = binned_data[:, a, la, lo]
                valid = ~np.isnan(ts)
                if not np.any(valid) or np.all(valid):
                    continue
                x_valid = np.where(valid)[0]
                x_missing = np.where(~valid)[0]
                binned_data[x_missing, a, la, lo] = np.interp(x_missing, x_valid, ts[valid])
                
    logger.info("Creating xarray Dataset...")
    out_ds = xr.Dataset(
        data_vars=dict(
            ktemp=(["ut", "altitude", "lat", "lon"], binned_data)
        ),
        coords=dict(
            ut=ut_cents,
            altitude=alt_cents,
            lat=lat_cents,
            lon=lon_cents
        )
    )
    out_ds['ktemp'].attrs = ds['ktemp'].attrs
    out_ds['ut'].attrs = {'standard_name': 'time', 'long_name': 'Universal Time', 'units': 'hours'}
    out_ds['lat'].attrs = {'standard_name': 'latitude', 'units': 'degrees_north'}
    out_ds['lon'].attrs = {'standard_name': 'longitude', 'units': 'degrees_east'}
    out_ds['altitude'].attrs = {'standard_name': 'altitude', 'units': 'km'}
    
    logger.info("Saving to saber_ut_grid.nc")
    out_ds.to_netcdf("saber_ut_grid.nc")
    logger.info("Done.")

if __name__ == "__main__":
    main()
